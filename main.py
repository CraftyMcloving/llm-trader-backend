import os, time, json, hmac, hashlib, threading, sqlite3, math, random, string
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
import numpy as np

import ccxt
from fastapi import FastAPI, Query, HTTPException, Depends, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# =========================
# Config
# =========================
API_KEY = os.getenv("API_KEY", "changeme")  # used by WP proxy as Bearer
HMAC_SECRET = os.getenv("HMAC_SECRET", "supersecret")

TOP_N = 20
MIN_CONFIDENCE = 0.12    # default scanning threshold
VOL_MIN = 0.001          # 0.1% min ATR%
VOL_CAP = 0.20           # 20% max ATR%

# =========================
# FastAPI app
# =========================
app = FastAPI(title="AI Trade Advisor", version="2.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

def require_key(request: Request):
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(401, detail="missing bearer")
    token = auth.split(" ", 1)[1].strip()
    if token != API_KEY:
        raise HTTPException(401, detail="bad token")

# =========================
# Kraken + OHLCV helpers
# =========================
kraken = ccxt.kraken()

# Spot USD universe on Kraken (curated; adjust as you like)
UNIVERSE = [
    "BTC/USD","ETH/USD","SOL/USD","XRP/USD","ADA/USD","LINK/USD","DOGE/USD","TRX/USD",
    "AVAX/USD","MATIC/USD","DOT/USD","ATOM/USD","ARB/USD","OP/USD","NEAR/USD","APT/USD",
    "SUI/USD","INJ/USD","AAVE/USD","LTC/USD"
]

TF_MAP = {
    "1h": "1h",
    "1d": "1d",
}

def get_universe(limit: int = TOP_N) -> List[Dict[str, str]]:
    pairs = UNIVERSE[:max(1, min(limit, len(UNIVERSE)))]
    return [{"symbol": s, "market": "kraken"} for s in pairs]

def fetch_ohlcv(symbol: str, tf: str, bars: int = 400) -> pd.DataFrame:
    tf2 = TF_MAP.get(tf, "1h")
    try:
        ohlcv = kraken.fetch_ohlcv(symbol, timeframe=tf2, limit=bars)
    except Exception as e:
        raise HTTPException(502, detail=f"kraken fetch failed: {e}")
    if not ohlcv:
        raise HTTPException(502, detail="no data")
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("ts", inplace=True)
    return df

# =========================
# TA features
# =========================
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    high, low = out["high"], out["low"]

    # RSI(14)
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/14, adjust=False).mean()
    roll_down = down.ewm(alpha=1/14, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    out["rsi14"] = 100.0 - (100.0 / (1.0 + rs))

    # SMAs
    out["sma20"] = close.rolling(20).mean()
    out["sma50"] = close.rolling(50).mean()
    out["sma200"] = close.rolling(200).mean()

    # Bollinger position (-1..+1 approx)
    bb_mid = out["sma20"]
    bb_std = close.rolling(20).std()
    bb_up = bb_mid + 2*bb_std
    bb_dn = bb_mid - 2*bb_std
    out["bb_pos"] = np.where((bb_up - bb_dn) == 0, 0.0, (close - bb_mid) / ((bb_up - bb_dn) / 2.0))

    # Donchian position (0..1)
    d_high = high.rolling(20).max()
    d_low = low.rolling(20).min()
    out["donch20"] = np.where((d_high - d_low) == 0, 0.5, (close - d_low) / (d_high - d_low))

    # ATR(14) + ATR%
    tr1 = (high - low)
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    out["atr14"] = tr.rolling(14).mean()
    out["atr_pct"] = out["atr14"] / (close.replace(0, np.nan))

    # slope of sma20
    out["slope20"] = out["sma20"].diff(3)

    # short return
    out["ret5"] = close.pct_change(5)

    return out

# =========================
# Signal inference + trade building
# =========================
def infer_signal(feats: pd.DataFrame, thresh: float) -> Tuple[str, float, Dict[str, bool], List[str]]:
    last = feats.iloc[-1]
    px = float(last["close"])
    rsi = float(last["rsi14"])
    slope = float(last["slope20"])
    bbpos = float(last["bb_pos"])
    atrp = float(last["atr_pct"])
    sma20 = float(last["sma20"])
    sma50 = float(last["sma50"])
    sma200 = float(last.get("sma200") or np.nan)
    macd_line = ema(feats["close"], 12) - ema(feats["close"], 26)
    macd_sig = ema(macd_line, 9)
    macd_hist = float((macd_line - macd_sig).iloc[-1])
    don = float(last["donch20"])

    # trend/vol filters
    trend_ok = (sma20 >= sma50) or (slope >= 0) or (px >= sma200 if not math.isnan(sma200) else False)
    vol_ok = (atrp is not None) and (atrp >= VOL_MIN) and (atrp <= VOL_CAP)

    # direction score
    bull_score = 0.0
    bull_score += (rsi - 50.0)/50.0 * 0.35
    bull_score += (bbpos) * 0.25
    bull_score += (1.0 if slope > 0 else -1.0) * 0.15
    bull_score += np.tanh(macd_hist*50.0) * 0.15
    bull_score += (don - 0.5) * 0.10

    confidence = float(max(0.0, min(1.0, abs(bull_score))))
    direction = "Bullish" if bull_score >= 0 else "Bearish"
    if confidence < 0.06:
        direction = "Neutral"

    reasons = []
    if not trend_ok: reasons.append("Trend filter blocked (sma20 < sma50 & slope < 0)")
    if not vol_ok:   reasons.append(f"ATR% {atrp:.2%} out of bounds")

    return direction, confidence, {"trend_ok": trend_ok, "vol_ok": vol_ok, "confidence_ok": confidence >= thresh}, reasons

def build_trade(symbol: str, feats: pd.DataFrame, direction: str,
                risk_pct: float, equity: Optional[float], leverage: float) -> Dict[str, Any]:
    last = feats.iloc[-1]
    px = float(last["close"])
    atr = float(last["atr14"])
    if not math.isfinite(atr) or atr <= 0:
        atr = px * 0.01

    k = 1.5  # RR target multiplier base
    if direction == "Long":
        entry = px
        stop = px - 2.0 * atr
        tp1, tp2, tp3 = px + 1.0*atr, px + 1.8*atr, px + 2.5*atr
    else:
        entry = px
        stop = px + 2.0 * atr
        tp1, tp2, tp3 = px - 1.0*atr, px - 1.8*atr, px - 2.5*atr

    rr1 = abs((tp1 - entry) / (entry - stop)) if (entry != stop) else 0.0
    rr2 = abs((tp2 - entry) / (entry - stop)) if (entry != stop) else 0.0
    rr3 = abs((tp3 - entry) / (entry - stop)) if (entry != stop) else 0.0

    qty = None
    notional = None
    if equity and equity > 0:
        risk_cash = equity * (risk_pct/100.0)
        per_unit_risk = abs(entry - stop)
        if per_unit_risk > 0:
            qty = (risk_cash / per_unit_risk) * float(leverage)
            notional = qty * entry

    trade = {
        "direction": direction,
        "entry": round(entry, 6),
        "stop": round(stop, 6),
        "targets": [round(tp1, 6), round(tp2, 6), round(tp3, 6)],
        "rr": [round(rr1, 2), round(rr2, 2), round(rr3, 2)],
        "risk_suggestions": {"breakeven_after_tp": 1, "trail_after_tp": 2, "trail_method": "ATR", "trail_multiple": 1},
        "position_size": {"qty": qty, "notional": notional} if qty else None,
    }
    return trade

# =========================
# Learning: weights + bias
# =========================
FEEDBACK_DB_PATH = os.getenv("FEEDBACK_DB", "/tmp/ai_trade_feedback.db")
_fb_lock = threading.Lock()

def fb_db():
    conn = sqlite3.connect(FEEDBACK_DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_feedback_db():
    with _fb_lock:
        conn = fb_db()
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS feedback(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts REAL NOT NULL,
          symbol TEXT, tf TEXT, market TEXT,
          direction TEXT, entry REAL, stop REAL, targets TEXT,
          confidence REAL, advice TEXT, outcome INTEGER,
          features TEXT, edge TEXT, composite TEXT,
          equity REAL, risk_pct REAL, leverage REAL,
          ua TEXT, ip TEXT
        );
        CREATE TABLE IF NOT EXISTS weights(
          feature TEXT PRIMARY KEY,
          w REAL NOT NULL
        );
        """)
        conn.commit()
        conn.close()

def _normalize_features(features: Dict[str, Any]) -> Dict[str, float]:
    norm: Dict[str, float] = {}
    for k, v in (features or {}).items():
        try:
            x = float(v)
        except Exception:
            continue
        if "rsi" in k.lower():
            norm[k] = (x - 50.0)/50.0
        elif "bb" in k.lower() or "donch" in k.lower():
            norm[k] = float(max(-1.0, min(1.0, x if abs(x) <= 2 else (x-0.5)*2)))
        elif "atr_pct" in k.lower():
            norm[k] = float(max(-1.0, min(1.0, (x - 0.02)/0.02)))  # center ~2%
        elif "sma" in k.lower() and "diff" in k.lower():
            norm[k] = float(max(-1.0, min(1.0, x)))
        elif "macd" in k.lower():
            norm[k] = float(max(-1.0, min(1.0, np.tanh(x*50.0))))
        elif "ret" in k.lower():
            norm[k] = float(max(-1.0, min(1.0, x*5)))
        else:
            norm[k] = float(max(-1.0, min(1.0, x)))
    return norm

def get_weights() -> Dict[str, float]:
    with _fb_lock:
        conn = fb_db()
        cur = conn.execute("SELECT feature, w FROM weights")
        d = {k: float(v) for k, v in cur.fetchall()}
        conn.close()
        return d

def update_weights(features: Dict[str, Any], outcome: int, lr: float = 0.05):
    norm = _normalize_features(features)
    if not norm:
        return
    with _fb_lock:
        conn = fb_db()
        for k, x in norm.items():
            cur = conn.execute("SELECT w FROM weights WHERE feature=?", (k,))
            row = cur.fetchone()
            w = float(row[0]) if row else 0.0
            w = w + lr * outcome * x
            conn.execute(
                "INSERT INTO weights(feature, w) VALUES(?,?) "
                "ON CONFLICT(feature) DO UPDATE SET w=excluded.w",
                (k, w)
            )
        conn.commit()
        conn.close()

def apply_feedback_bias(confidence: float, features: Dict[str, Any], cap: float = 0.15) -> float:
    W = get_weights()
    x = _normalize_features(features)
    score = sum(W.get(k, 0.0) * v for k, v in x.items())
    bias = max(-cap, min(cap, score))
    out = max(0.0, min(1.0, confidence + bias))
    return float(out)

# =========================
# Signals DB (signed snapshots) + voting
# =========================
SIG_DB_PATH = os.getenv("SIG_DB", "/tmp/ai_trade_signals.db")
_db_lock = threading.Lock()

def _sigdb():
    conn = sqlite3.connect(SIG_DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

with _db_lock:
    conn = _sigdb()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS signals(
      id TEXT PRIMARY KEY,
      created_at INTEGER NOT NULL,
      payload TEXT NOT NULL,
      outcome TEXT,
      resolved_at INTEGER
    );
    CREATE TABLE IF NOT EXISTS votes(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      sid TEXT, ts INTEGER, ip TEXT, ua TEXT, vote TEXT
    );
    """)
    conn.commit()
    conn.close()

def _rand_id(n=10) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(random.choice(alphabet) for _ in range(n))

def record_signal(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    sid = _rand_id(12)
    created_at = int(time.time())
    payload = json.dumps(snapshot, separators=(",", ":"), ensure_ascii=False)
    sig = hmac.new(HMAC_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()[:24]

    with _db_lock:
        conn = _sigdb()
        conn.execute("INSERT INTO signals (id, created_at, payload, outcome) VALUES (?,?,?,NULL)",
                     (sid, created_at, payload))
        conn.commit()
        conn.close()

    out = dict(snapshot)
    out["sid"] = sid
    out["sig"] = sig
    out["created_at"] = created_at
    return out

def _bars_since(df: pd.DataFrame, unix_start: int) -> pd.DataFrame:
    return df[df.index >= pd.to_datetime(unix_start, unit="s", utc=True)]

def resolve_outcome(sym: str, tf: str, direction: str, entry: float, stop: float,
                    targets: List[float], created_at_unix: int, horizon_bars: int = 12) -> str:
    df = fetch_ohlcv(sym, tf, bars=400)
    df2 = _bars_since(df, created_at_unix).iloc[1: 1 + horizon_bars].copy()
    if df2.empty:
        return "no_touch"
    # examine highs/lows per bar
    for i, row in df2.iterrows():
        hi = float(row["high"]); lo = float(row["low"])
        if direction == "Long":
            if lo <= stop: return "stop"
            for idx, tp in enumerate(targets, 1):
                if hi >= tp:
                    # continue to try higher TP in later bars
                    pass
        else:
            if hi >= stop: return "stop"
            for idx, tp in enumerate(targets, 1):
                if lo <= tp:
                    pass
    # if we saw any TP met, find the highest one hit
    tps_hit = []
    for _, row in df2.iterrows():
        hi = float(row["high"]); lo = float(row["low"])
        if direction == "Long":
            tps_hit.append(max([i+1 for i, tp in enumerate(targets) if hi >= tp] or [0]))
        else:
            tps_hit.append(max([i+1 for i, tp in enumerate(targets) if lo <= tp] or [0]))
    best = max(tps_hit or [0])
    if best > 0:
        return f"tp{best}"
    return "no_touch"

# =========================
# Pydantic models (feedback)
# =========================
class FeedbackIn(BaseModel):
    symbol: str
    tf: str
    market: Optional[str] = None
    direction: Optional[str] = None
    entry: Optional[float] = None
    stop: Optional[float] = None
    targets: Optional[List[float]] = None
    confidence: Optional[float] = None
    advice: Optional[str] = None
    features: Optional[Dict[str, Any]] = None
    edge: Optional[Dict[str, Any]] = None
    composite: Optional[Dict[str, Any]] = None
    equity: Optional[float] = None
    risk_pct: Optional[float] = None
    leverage: Optional[float] = None
    outcome: int = Field(..., description="+1 good, -1 bad")

class FeedbackAck(BaseModel):
    ok: bool
    stored_id: Optional[int] = None

# =========================
# API
# =========================
@app.get("/health")
def health():
    return {"ok": True, "ts": int(time.time()), "universe": len(UNIVERSE)}

@app.get("/instruments")
def instruments(limit: int = Query(TOP_N, ge=1, le=len(UNIVERSE)), _: None = Depends(require_key)):
    return get_universe(limit=limit)

@app.get("/chart")
def chart(symbol: str, tf: str = "1h", n: int = Query(120, ge=20, le=400), _: None = Depends(require_key)):
    df = fetch_ohlcv(symbol, tf, bars=max(160, n))
    c = df["close"].tail(n).astype(float).tolist()
    return {"closes": c}

def evaluate_signal(symbol: str, tf: str, risk_pct: float,
                    equity: Optional[float], leverage: float,
                    min_confidence: Optional[float],
                    ignore_trend: bool = False, ignore_vol: bool = False,
                    allow_neutral: bool = False) -> Dict[str, Any]:
    df = fetch_ohlcv(symbol, tf, bars=400)
    feats = compute_features(df).dropna().iloc[-220:]
    if len(feats) < 60:
        raise HTTPException(502, detail="insufficient window")
    last = feats.iloc[-1]
    px = float(last["close"])

    # construct feature pack for learning
    macd_line = ema(feats["close"], 12) - ema(feats["close"], 26)
    macd_sig  = ema(macd_line, 9)
    macd_hist = float((macd_line - macd_sig).iloc[-1])
    hh = feats["high"].rolling(20).max().iloc[-1]
    ll = feats["low"].rolling(20).min().iloc[-1]
    don = 0.0 if hh == ll else float((px - ll) / (hh - ll))

    f_pack = {
        "rsi14": float(last["rsi14"]),
        "sma20_50_diff": float((last["sma20"] - last["sma50"]) / max(px, 1e-9)),
        "bb_pos": float(last["bb_pos"]),
        "ret5": float(last["ret5"]),
        "atr14": float(last["atr14"]),
        "atr14_pct": float(last["atr_pct"]),
        "sma200": float(last["sma200"]),
        "macd_hist": macd_hist,
        "donch20": don,
    }

    thresh = min_confidence if (min_confidence is not None) else MIN_CONFIDENCE
    sig, conf, filters, reasons = infer_signal(feats, thresh)

    if ignore_trend:
        filters["trend_ok"] = True
        reasons = [r for r in reasons if "trend" not in r.lower()]
    if ignore_vol:
        filters["vol_ok"] = True
        reasons = [r for r in reasons if "atr%" not in r.lower() and "vol" not in r.lower()]

    advice = "Skip"
    trade = None
    directional_ok = (sig in ("Bullish", "Bearish")) or allow_neutral
    if conf >= thresh and filters["vol_ok"] and directional_ok:
        direction = "Long" if sig == "Bullish" else ("Short" if sig == "Bearish" else "Long")
        if sig == "Neutral":
            slope = float(feats["slope20"].iloc[-1]); rsi = float(last["rsi14"]); bbp = float(last["bb_pos"])
            direction = "Long" if (slope >= 0 or rsi >= 50 or bbp >= 0) else "Short"
        trade = build_trade(symbol, feats, direction, risk_pct, equity, leverage)
        advice = "Consider"

    return {
        "symbol": symbol,
        "timeframe": tf,
        "signal": sig,
        "confidence": conf,
        "updated": pd.Timestamp.utcnow().isoformat(),
        "trade": trade,
        "filters": {**filters, "reasons": reasons},
        "advice": advice,
        "features": f_pack,
    }

@app.get("/signals")
def signals(
    symbol: str,
    tf: str = "1h",
    risk_pct: float = Query(1.0, ge=0.1, le=5.0),
    equity: Optional[float] = Query(None, ge=0.0),
    leverage: float = Query(1.0, ge=1.0, le=100.0),
    min_confidence: Optional[float] = Query(None),
    allow_neutral: int = Query(1, ge=0, le=1),
    ignore_trend: int = Query(0, ge=0, le=1),
    ignore_vol: int = Query(0, ge=0, le=1),
    _: None = Depends(require_key)
):
    s = evaluate_signal(
        symbol=symbol, tf=tf, risk_pct=risk_pct, equity=equity, leverage=leverage,
        min_confidence=min_confidence,
        ignore_trend=bool(ignore_trend), ignore_vol=bool(ignore_vol),
        allow_neutral=bool(allow_neutral),
    )
    # apply learned bias
    base = float(s.get("confidence", 0.0))
    s["confidence"] = apply_feedback_bias(base, s.get("features") or {})
    if isinstance(s.get("filters"), dict) and min_confidence is not None:
        s["filters"]["confidence_ok"] = (s["confidence"] >= float(min_confidence))
    return s

@app.get("/scan")
def scan(
    tf: str = "1h",
    limit: int = Query(TOP_N, ge=3, le=50),
    top: int = Query(6, ge=1, le=12),
    min_confidence: Optional[float] = Query(None),
    auto_relax: int = Query(1, ge=0, le=1),
    allow_neutral: int = Query(1, ge=0, le=1),
    ignore_trend: int = Query(0, ge=0, le=1),
    ignore_vol: int = Query(0, ge=0, le=1),
    risk_pct: float = Query(1.0, ge=0.1, le=5.0),
    equity: Optional[float] = Query(None, ge=0.0),
    leverage: float = Query(1.0, ge=1.0, le=100.0),
    include_chart: int = Query(1, ge=0, le=1),
    _: None = Depends(require_key),
):
    uni = get_universe(limit=limit)
    results, ok = [], []

    def run_pass(thr: float, ign_trend: bool, ign_vol: bool, allow_neu: bool):
        local_results, local_ok = [], []
        for it in uni:
            try:
                s = evaluate_signal(
                    symbol=it["symbol"], tf=tf,
                    risk_pct=risk_pct, equity=equity, leverage=leverage,
                    min_confidence=thr,
                    ignore_trend=ign_trend, ignore_vol=ign_vol,
                    allow_neutral=allow_neu,
                )
                # learning bias
                base = float(s.get("confidence", 0.0))
                s["confidence"] = apply_feedback_bias(base, s.get("features") or {})
                if thr is not None:
                    s["filters"]["confidence_ok"] = (s["confidence"] >= float(thr))
                s["market"] = it["market"]
                s.setdefault("tf", tf)
                if s.get("advice") == "Consider":
                    local_ok.append(s)
                local_results.append(s)
            except HTTPException as he:
                local_results.append({
                    "symbol": it["symbol"], "tf": tf, "timeframe": tf, "signal": "Neutral",
                    "confidence": 0.0, "updated": pd.Timestamp.utcnow().isoformat(),
                    "trade": None,
                    "filters": {"trend_ok": False, "vol_ok": False, "confidence_ok": False, "reasons": [he.detail]},
                    "advice": "Skip", "market": it["market"],
                })
            except Exception as e:
                local_results.append({
                    "symbol": it["symbol"], "tf": tf, "timeframe": tf, "signal": "Neutral",
                    "confidence": 0.0, "updated": pd.Timestamp.utcnow().isoformat(),
                    "trade": None,
                    "filters": {"trend_ok": False, "vol_ok": False, "confidence_ok": False, "reasons": [f"exception: {e}"]},
                    "advice": "Skip", "market": it["market"],
                })
        return local_results, local_ok

    # pass 1: as requested
    thr = min_confidence if (min_confidence is not None) else MIN_CONFIDENCE
    results, ok = run_pass(thr, bool(ignore_trend), bool(ignore_vol), bool(allow_neutral))

    # optional relax ladder if nothing found
    note = None
    if not ok and auto_relax:
        ladder = [
            (thr*0.8 if thr else 0.08, bool(ignore_trend), bool(ignore_vol), True),
            (thr*0.7 if thr else 0.06, True, bool(ignore_vol), True),
            (thr*0.6 if thr else 0.05, True, True, True),
        ]
        for thr2, itf, ivf, an in ladder:
            results, ok = run_pass(thr2, itf, ivf, an)
            if ok:
                note = f"Relaxed filters to min_conf={thr2:.2f} (trend={'off' if itf else 'on'}, vol={'off' if ivf else 'on'})"
                break

    pool = ok if ok else results
    pool_sorted = sorted(pool, key=lambda s: abs(s.get("confidence") or 0.0), reverse=True)
    topK = pool_sorted[:top]

    out: List[Dict[str, Any]] = []
    for s in topK:
        signed = record_signal(dict(s))  # BEFORE adding chart
        if include_chart:
            try:
                df = fetch_ohlcv(signed["symbol"], tf, bars=160)
                signed["chart"] = {"closes": df["close"].tail(120).astype(float).tolist()}
            except Exception:
                signed["chart"] = None
        out.append(signed)

    if not ok and allow_neutral and note is None:
        note = "No high-confidence setups; returning best candidates by confidence."

    return {"universe": len(uni), "note": note, "results": out}

@app.post("/feedback", response_model=FeedbackAck)
def post_feedback(fb: FeedbackIn, request: Request, _: None = Depends(require_key)):
    ua = request.headers.get("user-agent", "")
    ip = request.client.host if request.client else ""
    with _fb_lock:
        conn = fb_db()
        cur = conn.execute(
            """INSERT INTO feedback
               (ts, symbol, tf, market, direction, entry, stop, targets, confidence, advice,
                outcome, features, edge, composite, equity, risk_pct, leverage, ua, ip)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                time.time(),
                fb.symbol, fb.tf, fb.market, fb.direction,
                fb.entry, fb.stop, json.dumps(fb.targets or []),
                fb.confidence, fb.advice, fb.outcome,
                json.dumps(fb.features or {}), json.dumps(fb.edge or {}),
                json.dumps(fb.composite or {}), fb.equity, fb.risk_pct,
                fb.leverage, ua[:300], ip
            )
        )
        rid = cur.lastrowid
        conn.commit()
        conn.close()
    try:
        update_weights(fb.features or {}, fb.outcome)
    except Exception:
        pass
    return FeedbackAck(ok=True, stored_id=rid)

@app.get("/feedback/stats")
def feedback_stats():
    with _fb_lock:
        conn = fb_db()
        total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        good  = conn.execute("SELECT COUNT(*) FROM feedback WHERE outcome=1").fetchone()[0]
        bad   = conn.execute("SELECT COUNT(*) FROM feedback WHERE outcome=-1").fetchone()[0]
        W = {k: v for k, v in conn.execute("SELECT feature, w FROM weights ORDER BY ABS(w) DESC LIMIT 24")}
        conn.close()
    return {"total": total, "good": good, "bad": bad, "weights": W}

@app.post("/vote")
def vote(payload: Dict[str, Any] = Body(...), request: Request = None, _: None = Depends(require_key)):
    sid = (payload.get("sid") or "").strip()
    vote = (payload.get("vote") or "").strip().lower()
    if vote not in ("up", "down") or not sid:
        raise HTTPException(400, detail="bad payload")

    ip = request.client.host if request else ""
    ua = request.headers.get("user-agent", "")
    now = int(time.time())

    with _db_lock:
        conn = _sigdb()
        row = conn.execute("SELECT 1 FROM votes WHERE sid=? AND ip=?", (sid, ip)).fetchone()
        if not row:
            conn.execute("INSERT INTO votes (sid, ts, ip, ua, vote) VALUES (?,?,?,?,?)", (sid, now, ip, ua, vote))
            conn.commit()

        row = conn.execute("SELECT created_at, payload, outcome FROM signals WHERE id=?", (sid,)).fetchone()
        if not row:
            conn.close()
            raise HTTPException(404, detail="unknown sid")
        created_at, payload_json, outcome = row
        snap = json.loads(payload_json)

        if not outcome:
            t = (snap.get("trade") or {})
            direction = t.get("direction") or snap.get("signal") or "Neutral"
            dl = str(direction).lower()
            if dl in ("bullish","bearish","neutral"):
                direction = "Long" if dl == "bullish" else ("Short" if dl == "bearish" else "Neutral")
            if direction in ("Long","Short") and t.get("entry") and t.get("stop") and t.get("targets"):
                horizon = int((snap.get("edge") or {}).get("horizon") or 12)
                outc = resolve_outcome(
                    sym=snap["symbol"], tf=snap.get("tf","1h"),
                    direction=direction, entry=float(t["entry"]),
                    stop=float(t["stop"]), targets=[float(x) for x in t["targets"]],
                    created_at_unix=int(created_at), horizon_bars=horizon
                )
            else:
                outc = "no_touch"
            conn.execute("UPDATE signals SET outcome=?, resolved_at=? WHERE id=?", (outc, int(time.time()), sid))
            conn.commit()
            outcome = outc
        conn.close()

    truth_good = str(outcome).startswith("tp")
    truth_bad  = (outcome == "stop")
    truthful = (vote == "up" and truth_good) or (vote == "down" and truth_bad)
    return {"ok": True, "sid": sid, "outcome": outcome, "vote": vote, "truthful": truthful}

@app.on_event("startup")
def _boot():
    init_feedback_db()