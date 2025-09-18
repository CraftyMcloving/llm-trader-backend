# main.py — AI Trade Advisor backend (Kraken/USD)
# FastAPI + ccxt + pandas (py3.12 recommended; see requirements.txt)
from fastapi import FastAPI, HTTPException, Depends, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
import os, time, math
import pandas as pd
import numpy as np
import ccxt

# ---- FEEDBACK: imports ----
import os, json, sqlite3, threading, time
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from fastapi import Request
# ---------------------------

import os, json, time, uuid, hmac, hashlib, sqlite3
from typing import Any, Dict, Optional
import ccxt  # you already use this

# ----- Security -----
API_KEY = os.getenv("API_KEY", "change-me")
def require_key(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    token = authorization.split(" ", 1)[1].strip()
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Bad token")
    return None
    
FEEDBACK_SECRET = os.getenv("FEEDBACK_SECRET", "change-me")  # set in Render env
DB_PATH = os.getenv("DB_PATH", "signals.sqlite3")

def _db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS signals (
      id TEXT PRIMARY KEY,
      created_at INTEGER,
      symbol TEXT,
      tf TEXT,
      payload TEXT,         -- JSON of the full server snapshot
      outcome TEXT,         -- 'stop' | 'tp1' | 'tp2' | 'tp3' | 'no_touch'
      resolved_at INTEGER
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS votes (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      sid TEXT,
      ts INTEGER,
      ip TEXT,
      ua TEXT,
      vote TEXT             -- 'up' | 'down'
    )""")
    return conn

DB = _db()

def plan_signature(snapshot: Dict[str, Any]) -> str:
    # canonical minimal payload to sign (prevents client tampering)
    msg = json.dumps({
        "symbol":    snapshot.get("symbol"),
        "tf":        snapshot.get("tf"),
        "direction": (snapshot.get("trade") or {}).get("direction") or snapshot.get("signal"),
        "entry":     (snapshot.get("trade") or {}).get("entry"),
        "stop":      (snapshot.get("trade") or {}).get("stop"),
        "targets":   (snapshot.get("trade") or {}).get("targets"),
        "updated":   snapshot.get("updated"),    # iso8601 from your pipeline
    }, sort_keys=True, separators=(",", ":")).encode()
    return hmac.new(FEEDBACK_SECRET.encode(), msg, hashlib.sha256).hexdigest()

def record_signal(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    sid = str(uuid.uuid4())
    created_ts = int(time.time())
    DB.execute("INSERT OR REPLACE INTO signals (id, created_at, symbol, tf, payload) VALUES (?,?,?,?,?)",
               (sid, created_ts, snapshot.get("symbol"), snapshot.get("tf","1h"), json.dumps(snapshot)))
    DB.commit()
    # attach id/signature/timestamps for the client
    snapshot["sid"] = sid
    snapshot["sig"] = plan_signature(snapshot)
    snapshot["created_at_unix"] = created_ts
    return snapshot
    
def tf_to_ms(tf: str) -> int:
    tf = tf.lower().strip()
    if tf.endswith("m"): return int(tf[:-1]) * 60_000
    if tf.endswith("h"): return int(tf[:-1]) * 3_600_000
    if tf.endswith("d"): return int(tf[:-1]) * 86_400_000
    return 3_600_000

def resolve_outcome(sym: str, tf: str, direction: str, entry: float, stop: float,
                    targets: list[float], created_at_unix: int, horizon_bars: int = 12) -> str:
    # Pull future candles since signal creation
    ex = ccxt.kraken()
    since_ms = created_at_unix * 1000
    limit = max(horizon_bars + 2, 20)
    ohlcv = ex.fetch_ohlcv(sym, timeframe=tf, since=since_ms, limit=limit)
    if not ohlcv:  # no data… be conservative
        return "no_touch"

    # First-touch-wins within the horizon window
    # Candle: [ts, open, high, low, close, vol]
    tps = sorted(targets or [])
    if not tps:  # if no targets, treat as no-touch (or compute PnL at horizon)
        return "no_touch"

    bars = ohlcv[:horizon_bars]
    is_long = (str(direction).lower() == "long")

    # Walk forward; if in one bar both TP and SL are inside, count STOP first (conservative)
    for ts, o, h, l, c, v in bars:
        if is_long:
            hit_stop = (l <= stop)
            hit_tp   = any(h >= tp for tp in tps)
        else:
            hit_stop = (h >= stop)
            hit_tp   = any(l <= tp for tp in tps)

        if hit_stop and hit_tp:
            return "stop"
        if hit_stop:
            return "stop"
        if hit_tp:
            # which TP? highest reached for long, lowest reached for short
            if is_long:
                for level in reversed(tps):
                    if h >= level: 
                        return f"tp{tps.index(level)+1}"
            else:
                for level in tps:
                    if l <= level:
                        return f"tp{tps.index(level)+1}"
    return "no_touch"

# ----- Exchange (Kraken) -----
EXCHANGE_ID = os.getenv("EXCHANGE", "kraken")
QUOTE = os.getenv("QUOTE", "USD")
TOP_N = int(os.getenv("TOP_N", "30"))

_ex = None
def get_exchange():
    global _ex
    if _ex is None:
        _ex = getattr(ccxt, EXCHANGE_ID)({
            "enableRateLimit": True,
            "timeout": 20000,
        })
    return _ex

def load_markets():
    ex = get_exchange()
    if not getattr(ex, "markets", None):
        ex.load_markets()
    return ex.markets

# ----- App -----
app = FastAPI(title="AI Trade Advisor API", version="2025.09")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ========= FEEDBACK STORAGE & ONLINE WEIGHTS =========
DB_PATH = os.getenv("FEEDBACK_DB", "/tmp/ai_trade_feedback.db")
_db_lock = threading.Lock()

def _db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_feedback_db():
    with _db_lock:
        conn = _db()
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS feedback(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts REAL NOT NULL,
          symbol TEXT, tf TEXT, market TEXT,
          direction TEXT, entry REAL, stop REAL, targets TEXT,
          confidence REAL, advice TEXT, outcome INTEGER,  -- +1 good, -1 bad
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

def get_weights() -> Dict[str, float]:
    with _db_lock:
        conn = _db()
        cur = conn.execute("SELECT feature, w FROM weights")
        d = {k: float(v) for k, v in cur.fetchall()}
        conn.close()
        return d

def update_weights(features: Dict[str, Any], outcome: int, lr: float = 0.05):
    """
    Super-light online learning: w_i += lr * outcome * norm(feature_i)
    outcome: +1 good (TP), -1 bad (SL). Features loosely normalized.
    """
    if not isinstance(features, dict):
        return
    # normalize a few known numeric features; ignore non-numerics
    norm: Dict[str, float] = {}
    for k, v in features.items():
        if v is None: 
            continue
        try:
            x = float(v)
        except Exception:
            continue
        # coarse normalization per feature name
        if "rsi" in k:          # RSI ~ [0..100]
            x = (x - 50.0) / 50.0
        elif "macd" in k or "diff" in k or "ret" in k or "bb_pos" in k or "donch" in k:
            # typically around [-1..+1] already-ish
            x = max(-2.0, min(2.0, x))
        elif "atr" in k:        # ATR or atr% → keep small
            x = max(-1.0, min(1.0, x))
        else:
            x = max(-3.0, min(3.0, x))
        norm[k] = x

    if not norm:
        return

    with _db_lock:
        conn = _db()
        for k, x in norm.items():
            cur = conn.execute("SELECT w FROM weights WHERE feature=?", (k,))
            row = cur.fetchone()
            w = float(row[0]) if row else 0.0
            w = w + lr * outcome * x
            conn.execute("INSERT INTO weights(feature, w) VALUES(?,?) ON CONFLICT(feature) DO UPDATE SET w=excluded.w",
                         (k, w))
        conn.commit()
        conn.close()

def feedback_bias(features: Dict[str, Any], alpha: float = 0.15) -> float:
    """
    Compute a bias adjustment from learned weights; clamp to [-1..1] then scale by alpha.
    """
    ws = get_weights()
    s = 0.0
    for k, v in (features or {}).items():
        try:
            x = float(v)
        except Exception:
            continue
        # re-use the same crude normalization as in update_weights
        if "rsi" in k:          x = (x - 50.0) / 50.0
        elif "macd" in k or "diff" in k or "ret" in k or "bb_pos" in k or "donch" in k:
            x = max(-2.0, min(2.0, x))
        elif "atr" in k:        x = max(-1.0, min(1.0, x))
        else:                   x = max(-3.0, min(3.0, x))
        w = ws.get(k, 0.0)
        s += w * x
    # squish
    s = max(-1.0, min(1.0, s))
    return alpha * s

@app.on_event("startup")
def _fb_startup():
    init_feedback_db()
# =====================================================

def apply_feedback_bias(confidence: float, features: Dict[str, Any]) -> float:
    """Return confidence nudged by learned weights; always safe."""
    try:
        return max(-1.0, min(1.0, float(confidence) + feedback_bias(features)))
    except Exception:
        return float(confidence)


# ----- Cache -----
CACHE: Dict[str, Tuple[float, Any]] = {}
def cache_get(key, ttl):
    v = CACHE.get(key)
    if not v: return None
    ts, data = v
    return data if (time.time() - ts) <= ttl else None
def cache_set(key, data): CACHE[key] = (time.time(), data)

# ----- Universe -----
CURATED = [
    "BTC","ETH","XRP","SOL","ADA","DOGE","LINK","LTC","BCH","TRX",
    "DOT","ATOM","XLM","ETC","MATIC","UNI","APT","ARB","OP","AVAX",
    "NEAR","ALGO","FIL","SUI","SHIB","USDC","USDT","XMR","AAVE"
    ,"PAXG","ONDO","PEPE","SEI","IMX","FIL","TIA"
]
def get_universe(quote=QUOTE, limit=TOP_N) -> List[Dict[str, Any]]:
    key = f"uni:{EXCHANGE_ID}:{quote}:{limit}"
    u = cache_get(key, 1800)
    if u is not None: return u
    markets = load_markets()

    available: List[str] = []
    for base in CURATED:
        sym = f"{base}/{quote}"
        if sym in markets and markets[sym].get("active"):
            available.append(sym)

    if len(available) < limit:
        # Fallback: fill with other active {quote} tickers
        for m, info in markets.items():
            if info.get("quote") == quote and info.get("active"):
                if m not in available:
                    available.append(m)
                if len(available) >= limit:
                    break

    out = [{"symbol": s, "name": s.split('/')[0], "market": EXCHANGE_ID, "tf_supported": ["1h","1d"]} for s in available[:limit]]
    cache_set(key, out)
    return out

# ----- Market data -----
TF_MAP = {"1h": "1h", "1d": "1d"}

def fetch_ohlcv(symbol: str, tf: str, bars: int = 720) -> pd.DataFrame:
    ex = get_exchange()
    tf_ex = TF_MAP.get(tf)
    if tf_ex is None:
        raise HTTPException(400, detail=f"Unsupported timeframe: {tf}")
    key = f"ohlcv:{symbol}:{tf_ex}:{bars}"
    cached = cache_get(key, 900)
    if cached is not None:
        return cached.copy()
    try:
        data = ex.fetch_ohlcv(symbol, timeframe=tf_ex, limit=bars)
    except ccxt.BadSymbol:
        raise HTTPException(502, detail=f"Symbol not available on {EXCHANGE_ID}: {symbol}")
    except Exception as e:
        raise HTTPException(502, detail=f"fetch_ohlcv failed: {e}")
    if not data or len(data) < 50:
        raise HTTPException(502, detail="insufficient candles")
    df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    cache_set(key, df)
    return df.copy()

# ----- Indicators -----
def ema(s: pd.Series, n: int) -> pd.Series: return s.ewm(span=n, adjust=False).mean()
def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff(); up = d.clip(lower=0); dn = -d.clip(upper=0)
    rs = ema(up,n) / (ema(dn,n) + 1e-12)
    return 100 - (100/(1+rs))
def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h,l,c = df["high"], df["low"], df["close"]; pc = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return ema(tr,n)

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sma20"]   = out["close"].rolling(20).mean()
    out["sma50"]   = out["close"].rolling(50).mean()
    out["sma200"]  = out["close"].rolling(200).mean()
    out["rsi14"]   = rsi(out["close"], 14)
    out["atr14"]   = atr(out, 14)
    out["atr_pct"] = (out["atr14"]/out["close"]).clip(lower=0, upper=2.0)
    out["ret5"]    = out["close"].pct_change(5)
    out["slope20"] = out["sma20"] - out["sma20"].shift(5)
    out["bb_mid"]  = out["close"].rolling(20).mean()
    out["bb_std"]  = out["close"].rolling(20).std().replace(0, np.nan)
    out["bb_pos"]  = ((out["close"] - out["bb_mid"]) / (2*out["bb_std"])).clip(-1,1)
    return out

# ----- Signal logic -----
MIN_CONFIDENCE     = float(os.getenv("MIN_CONFIDENCE", "0.14"))
VOL_CAP_ATR_PCT    = float(os.getenv("VOL_CAP_ATR_PCT", "0.25"))
VOL_MIN_ATR_PCT    = float(os.getenv("VOL_MIN_ATR_PCT", "0.001"))

def build_trade(symbol: str, df: pd.DataFrame, direction: str,
                risk_pct: float = 1.0, equity: Optional[float] = None, leverage: float = 1.0) -> Dict[str, Any]:
    ex = get_exchange()
    markets = load_markets()
    mkt = markets.get(symbol, {})

    last  = df.iloc[-1]
    price = float(last["close"])
    a     = float(last["atr14"])
    if not math.isfinite(a) or a <= 0:
        raise ValueError("ATR not finite")

    mult = 2.2
    if direction == "Long":
        stop_raw = price - mult * a
        targets_raw = [price + k * a for k in (1.5, 2.5, 3.5)]
    else:
        stop_raw = price + mult * a
        targets_raw = [price - k * a for k in (1.5, 2.5, 3.5)]

    price_p = float(ex.price_to_precision(symbol, price))
    stop    = float(ex.price_to_precision(symbol, stop_raw))
    targets = [float(ex.price_to_precision(symbol, t)) for t in targets_raw]

    denom = (price_p - stop)
    rr = [round(abs((t - price_p) / denom), 2) if denom else 0.0 for t in targets]

    pos = None
    risk_amt = None
    if equity and risk_pct:
        risk_amt = float(equity) * (float(risk_pct) / 100.0)
        risk_per_unit = abs(denom) / max(float(leverage), 1.0)
        qty_raw = risk_amt / max(risk_per_unit, 1e-8)
        qty = float(ex.amount_to_precision(symbol, qty_raw))
        notional = qty * price_p
        pos = {"qty": qty, "notional": notional}

    return {
        "direction": direction,
        "entry": price_p,
        "stop": stop,
        "targets": targets,
        "rr": rr,
        "position_size": pos,
        "risk_amount": risk_amt,         # NEW: shows $ at risk
        "leverage": leverage,            # handy for UI
        "risk_suggestions": {
            "breakeven_after_tp": 1,
            "trail_after_tp": 2,
            "trail_method": "ATR",
            "trail_multiple": 1.0
        },
        "precision": mkt.get("precision", {})
    }

def infer_signal(feats: pd.DataFrame, min_conf: float) -> Tuple[str,float,Dict[str,bool],List[str]]:
    last = feats.iloc[-1]; reasons=[]
    trend_up = bool(last["sma20"]>last["sma50"] and feats["slope20"].iloc[-1]>0)
    trend_dn = bool(last["sma20"]<last["sma50"] and feats["slope20"].iloc[-1]<0)
    trend_ok = trend_up or trend_dn
    if not trend_ok: reasons.append("no clear trend")

    atr_pct = float(last["atr_pct"])
    vol_ok = bool((atr_pct>=VOL_MIN_ATR_PCT) and (atr_pct<=VOL_CAP_ATR_PCT))
    if not vol_ok: reasons.append("ATR% outside bounds")

    rsi14 = float(last["rsi14"])
    bias_up = rsi14>=52; bias_dn = rsi14<=48

    conf=0.0
    if trend_ok: conf += 0.45
    if vol_ok:   conf += 0.25
    if bias_up or bias_dn: conf += 0.15
    conf += min(abs(float(feats["ret5"].iloc[-1] or 0.0))*2.0, 0.15)
    conf = float(max(0.0, min(conf,1.0)))

    if trend_up and bias_up:   sig="Bullish"
    elif trend_dn and bias_dn: sig="Bearish"
    else:                      sig="Neutral"

    # explicit reason for threshold gate
    if conf < min_conf:
        reasons.append(f"Composite conf {conf:.2f} < {min_conf:.2f}")

    return sig, conf, {"trend_ok":trend_ok,"vol_ok":vol_ok,"confidence_ok":conf>=min_conf}, reasons

# ----- Pure helper used by both endpoints -----
def evaluate_signal(
    symbol: str,
    tf: str,
    risk_pct: float,
    equity: Optional[float],
    leverage: float,
    min_confidence: Optional[float],
    ignore_trend: bool = False,
    ignore_vol: bool = False,
    allow_neutral: bool = False,
) -> Dict[str, Any]:
    df = fetch_ohlcv(symbol, tf, bars=400)
    feats = compute_features(df).dropna().iloc[-200:]
    if len(feats) < 50:
        raise HTTPException(502, detail="insufficient features window")

    thresh = min_confidence if (min_confidence is not None) else MIN_CONFIDENCE
    sig, conf, filt, reasons = infer_signal(feats, thresh)

    # Apply client relax flags to filters/reasons
    if ignore_trend:
        filt["trend_ok"] = True
        reasons = [r for r in reasons if "trend" not in r.lower()]
    if ignore_vol:
        filt["vol_ok"] = True
        reasons = [r for r in reasons if "atr%" not in r.lower() and "vol" not in r.lower()]

    trade = None
    advice = "Skip"

    # Direction gate: allow neutral if requested (choose side from slope/RSI/bb_pos)
    directional_ok = (sig in ("Bullish", "Bearish")) or allow_neutral
    if conf >= thresh and filt["vol_ok"] and directional_ok:
        if sig == "Bullish":
            direction = "Long"
        elif sig == "Bearish":
            direction = "Short"
        else:
            # pick a side for neutral
            slope = float(feats["slope20"].iloc[-1])
            rsi14 = float(feats["rsi14"].iloc[-1])
            bbpos = float(feats["bb_pos"].iloc[-1])
            direction = "Long" if (slope >= 0 or rsi14 >= 50 or bbpos >= 0) else "Short"

        trade = build_trade(symbol, feats, direction, risk_pct, equity, leverage)
        advice = "Consider"

    return {
        "symbol": symbol,
        "timeframe": tf,
        "signal": sig,
        "confidence": conf,
        "updated": pd.Timestamp.utcnow().isoformat(),
        "trade": trade,
        "filters": {**filt, "reasons": reasons},
        "advice": advice,
    }
    

# ----- Schemas (light) -----
class Instrument(BaseModel):
    symbol: str; name: str; market: str; tf_supported: List[str]

# ----- Endpoints -----
@app.get("/health")
def health():
    try: name = get_exchange().id
    except Exception as e: name = f"error: {e}"
    return {"ok": True, "exchange": name, "quote": QUOTE}

@app.get("/instruments", response_model=List[Instrument])
def instruments(_: None = Depends(require_key)):
    return get_universe()

@app.get("/chart")
def chart(symbol: str, tf: str = "1h", n: int = 120, _: None = Depends(require_key)):
    df = fetch_ohlcv(symbol, tf, bars=max(200, n+20))
    closes = df["close"].tail(n).astype(float).tolist()
    return {"symbol": symbol, "tf": tf, "closes": closes}

from typing import Optional, Dict, Any
from fastapi import Query, Depends, HTTPException

@app.get("/signals")
def signals(
    symbol: str,
    tf: str = "1h",
    risk_pct: float = Query(1.0, ge=0.1, le=5.0),
    equity: Optional[float] = Query(None, ge=0),
    leverage: float = Query(1.0, ge=1.0, le=100.0),
    min_confidence: Optional[float] = Query(None),
    _: None = Depends(require_key),
):
    try:
        # 1) get your base result (your existing function)
        res = evaluate_signal(
            symbol=symbol,
            tf=tf,
            risk_pct=risk_pct,
            equity=equity,
            leverage=leverage,
            min_confidence=min_confidence,
        )

        # 2) gently nudge confidence using learned weights
        base_conf = float(res.get("confidence", 0.0))
        feats: Dict[str, Any] = res.get("features") or {}
        res["confidence"] = apply_feedback_bias(base_conf, feats)

        # 3) keep the filter flag consistent, if present
        if isinstance(res.get("filters"), dict) and min_confidence is not None:
            res["filters"]["confidence_ok"] = (res["confidence"] >= float(min_confidence))
            
        # example inside evaluate_signal(...) just before `return res`
        res["tf"] = tf
        res = record_signal(res)
        return res

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"signals failed: {e}")

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
    """
    Universe scan → pick topK, attach server-signed IDs for feedback learning.
    NOTE: We call record_signal() *after* selection (topK) and *before* adding chart,
          so DB payload stays lean but the client still gets sid/sig.
    """
    uni = get_universe(limit=limit)
    results, ok = [], []

    for it in uni:
        try:
            s = evaluate_signal(
                symbol=it["symbol"], tf=tf,
                risk_pct=risk_pct, equity=equity, leverage=leverage,
                min_confidence=min_confidence,
                ignore_trend=bool(ignore_trend),
                ignore_vol=bool(ignore_vol),
                allow_neutral=bool(allow_neutral),
            )
            s["market"] = it["market"]
            s.setdefault("tf", tf)  # make sure TF is present for signing
            if s.get("advice") == "Consider":
                ok.append(s)
            results.append(s)
        except HTTPException as he:
            results.append({
                "symbol": it["symbol"], "tf": tf, "timeframe": tf, "signal": "Neutral",
                "confidence": 0.0, "updated": pd.Timestamp.utcnow().isoformat(),
                "trade": None,
                "filters": {"trend_ok": False, "vol_ok": False, "confidence_ok": False,
                            "reasons": [he.detail]},
                "advice": "Skip", "market": it["market"],
            })
        except Exception as e:
            results.append({
                "symbol": it["symbol"], "tf": tf, "timeframe": tf, "signal": "Neutral",
                "confidence": 0.0, "updated": pd.Timestamp.utcnow().isoformat(),
                "trade": None,
                "filters": {"trend_ok": False, "vol_ok": False, "confidence_ok": False,
                            "reasons": [f"exception: {e}"]},
                "advice": "Skip", "market": it["market"],
            })

    # pick pool (prefer 'Consider', else everything), sort by |confidence|
    pool = ok if ok else results
    pool_sorted = sorted(pool, key=lambda s: abs(s.get("confidence") or 0.0), reverse=True)
    topK = pool_sorted[:top]

    out: list[dict[str, Any]] = []
    for s in topK:
        # 1) sign & store snapshot (adds sid/sig/created_at_unix)
        #    IMPORTANT: do this BEFORE adding heavy extras like 'chart'
        signed = record_signal(dict(s))  # copy to avoid side-effects if you reuse 's'

        # 2) optional chart for the frontend
        if include_chart:
            try:
                df = fetch_ohlcv(signed["symbol"], tf, bars=160)
                signed["chart"] = {"closes": df["close"].tail(120).astype(float).tolist()}
            except Exception:
                signed["chart"] = None

        out.append(signed)

    note = None
    if not ok and allow_neutral:
        note = "No high-confidence setups; returning best candidates by confidence."

    return {"universe": len(uni), "note": note, "results": out}

# ========= FEEDBACK API =========
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
    outcome: int = Field(..., description="+1 good (TP/favorable), -1 bad (SL hit)")

class FeedbackAck(BaseModel):
    ok: bool
    stored_id: Optional[int] = None

@app.post("/feedback", response_model=FeedbackAck)
def post_feedback(fb: FeedbackIn, request: Request):
    ua = request.headers.get("user-agent", "")
    ip = request.client.host if request.client else ""
    with _db_lock:
        conn = _db()
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
    # update model weights
    try:
        update_weights(fb.features or {}, fb.outcome)
    except Exception:
        pass
    return FeedbackAck(ok=True, stored_id=rid)

@app.get("/feedback/stats")
def feedback_stats():
    with _db_lock:
        conn = _db()
        total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        good  = conn.execute("SELECT COUNT(*) FROM feedback WHERE outcome=1").fetchone()[0]
        bad   = conn.execute("SELECT COUNT(*) FROM feedback WHERE outcome=-1").fetchone()[0]
        W = {k: v for k, v in conn.execute("SELECT feature, w FROM weights ORDER BY ABS(w) DESC LIMIT 24")}
        conn.close()
    return {"total": total, "good": good, "bad": bad, "weights": W}
# =================================

from fastapi import Body, Request

@app.post("/feedback")
def feedback(
    payload: Dict[str, Any] = Body(...),
    request: Request = None,
    _: None = Depends(require_key)  # keep your auth
):
    sid = (payload.get("sid") or "").strip()
    vote = (payload.get("vote") or "").strip().lower()  # 'up' or 'down'
    if vote not in ("up","down") or not sid:
        raise HTTPException(400, detail="bad payload")

    # store raw vote (rate-limit: 1 per sid per IP)
    ip = request.client.host if request else ""
    ua = request.headers.get("user-agent","")
    now = int(time.time())
    # dedupe per sid/ip
    row = DB.execute("SELECT 1 FROM votes WHERE sid=? AND ip=?", (sid, ip)).fetchone()
    if not row:
        DB.execute("INSERT INTO votes (sid, ts, ip, ua, vote) VALUES (?,?,?,?,?)", (sid, now, ip, ua, vote))
        DB.commit()

    # load snapshot
    row = DB.execute("SELECT created_at, payload, outcome FROM signals WHERE id=?", (sid,)).fetchone()
    if not row:
        raise HTTPException(404, detail="unknown sid")

    created_at, payload_json, outcome = row
    snap = json.loads(payload_json)

    # already resolved? return cached truth
    if not outcome:
        t = (snap.get("trade") or {})
        direction = t.get("direction") or snap.get("signal") or "Neutral"
        if str(direction).lower() in ("bullish","bearish","neutral"):
            direction = "Long" if direction.lower()=="bullish" else "Short" if direction.lower()=="bearish" else "Neutral"

        if direction in ("Long","Short") and t.get("entry") and t.get("stop") and t.get("targets"):
            horizon = int((snap.get("edge") or {}).get("horizon") or 12)
            outcome = resolve_outcome(
                sym=snap["symbol"], tf=snap.get("tf","1h"),
                direction=direction, entry=float(t["entry"]),
                stop=float(t["stop"]), targets=[float(x) for x in t["targets"]],
                created_at_unix=int(created_at), horizon_bars=horizon
            )
        else:
            outcome = "no_touch"

        DB.execute("UPDATE signals SET outcome=?, resolved_at=? WHERE id=?", (outcome, int(time.time()), sid))
        DB.commit()

    # grade the vote vs truth (optional: include in response)
    # simple rule: TPx -> 'good', stop -> 'bad', no_touch -> neutral
    truth_good = outcome.startswith("tp")
    truth_bad  = (outcome == "stop")
    truthful   = (vote == "up" and truth_good) or (vote == "down" and truth_bad)

    return {
        "ok": True,
        "sid": sid,
        "outcome": outcome,       # server-truth
        "vote": vote,
        "truthful": truthful      # whether the click agrees with truth
    }