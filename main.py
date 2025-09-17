# app/main.py
from fastapi import FastAPI, HTTPException, Depends, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os, time, copy
import pandas as pd
import numpy as np

# ---- Config (normalize) ----
API_KEY  = os.getenv("API_KEY", "change-me")
EXCHANGE = os.getenv("EXCHANGE", "kraken").lower()
QUOTE    = os.getenv("QUOTE", "USD").upper()
TOP_N    = int(os.getenv("TOP_N", "20"))
CACHE_TTL_SEC = int(os.getenv("CACHE_TTL_SEC", "1800"))

# Curated & quote prefs
CURATED_BASES = [b.strip().upper() for b in os.getenv(
    "CURATED_BASES",
    "BTC,XRP,ETH,SOL,ADA,HYPE,USDT,BNB,USDC,DOGE,STETH,TRX,LINK,WBTC,SUI,AVAX,XLM,BCH,LTC,CRO,TON,USDS,SHIB,DOT,XMR,MNT,UNI"
).split(",") if b.strip()]
QUOTE_FALLBACKS = [q.strip().upper() for q in os.getenv(
    "QUOTE_FALLBACKS",
    f"{QUOTE},USD,USDT,USDC,EUR"
).split(",") if q.strip()]

# Filters / risk defaults
MIN_CONFIDENCE   = float(os.getenv("MIN_CONFIDENCE", "0.18"))   # abs(score) threshold
VOL_CAP_ATR_PCT  = float(os.getenv("VOL_CAP_ATR_PCT", "0.10"))  # 10% ATR cap
VOL_MIN_ATR_PCT  = float(os.getenv("VOL_MIN_ATR_PCT", "0.002")) # 0.2% ATR min

# ---- Lazy imports ----
_requests = None
_ccxt = None
def requests():
    global _requests
    if _requests is None:
        import requests as r
        _requests = r
    return _requests

def ccxt():
    global _ccxt
    if _ccxt is None:
        import ccxt as c
        _ccxt = c
    return _ccxt

# ---- Security ----
def require_key(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    token = authorization.split(" ", 1)[1].strip()
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid token")

# ---- App ----
app = FastAPI(title="AI Trade Advisor", version="1.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- In-memory state ----
STATE: Dict[str, Any] = {
    "universe": [],
    "universe_fetched_at": 0,
    "ohlcv_cache": {},   # (symbol, tf) -> {"at": ts, "df": DataFrame}
    "signal_cache": {},  # (symbol, tf) -> {"at": ts, "payload": dict}
    "universe_debug": {
        "path": None,
        "supported_count": 0,
        "last_error": None,
        "exchange": EXCHANGE,
        "quote": QUOTE,
        "top_n": TOP_N,
    }
}

# ---------------- TA features ----------------
def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = pd.Series(np.where(delta > 0, delta, 0.0), index=series.index)
    dn = pd.Series(np.where(delta < 0, -delta, 0.0), index=series.index)
    rs = up.rolling(n, min_periods=n).mean() / (dn.rolling(n, min_periods=n).mean() + 1e-12)
    return 100 - (100/(1+rs))

def bollinger(series: pd.Series, n: int = 20, k: float = 2.0):
    ma = series.rolling(n, min_periods=n).mean()
    sd = series.rolling(n, min_periods=n).std()
    return ma + k*sd, ma - k*sd

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def recent_extrema(series: pd.Series, lookback: int = 10) -> tuple[float, float]:
    window = series.iloc[-(lookback+1):-1]
    return float(window.min()), float(window.max())

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret1"]  = df["close"].pct_change()
    df["ret5"]  = df["close"].pct_change(5)

    df["sma20"]  = df["close"].rolling(20,  min_periods=20).mean()
    df["sma50"]  = df["close"].rolling(50,  min_periods=50).mean()
    df["sma200"] = df["close"].rolling(200, min_periods=200).mean()
    df["sma20_50_diff"] = (df["sma20"] - df["sma50"]) / (df["sma50"] + 1e-12)

    df["rsi14"] = rsi(df["close"], 14)
    up, dn = bollinger(df["close"], 20, 2)
    df["bb_up"], df["bb_dn"] = up, dn
    df["bb_pos"] = (df["close"] - df["bb_dn"]) / ((df["bb_up"] - df["bb_dn"]) + 1e-12)

    df["atr14"] = atr(df, 14)
    df["atr14_pct"] = df["atr14"] / (df["close"].abs() + 1e-12)
    return df

def rule_inference(df: pd.DataFrame) -> Dict[str, Any]:
    last = df.iloc[-1]
    slope = (df["sma20"].iloc[-1] - df["sma20"].iloc[-5]) if df["sma20"].notna().tail(5).all() else 0.0
    score = 0.0
    if last["rsi14"] < 30: score += 0.45
    if last["rsi14"] > 70: score -= 0.45
    base = abs(df["sma20"].iloc[-5]) + 1e-6 if df["sma20"].notna().iloc[-5] else 1.0
    score += np.tanh(slope / base) * 0.3
    score += np.clip(last["bb_pos"] - 0.5, -0.5, 0.5) * 0.25
    label = "Bullish" if score > 0.15 else "Bearish" if score < -0.15 else "Neutral"
    conf = float(np.clip(abs(score), 0, 1))
    return {"signal": label, "confidence": conf, "score": float(score)}

def trend_ok(direction: str, df: pd.DataFrame) -> tuple[bool, str]:
    if direction not in ("Long","Short"):
        return False, "Neutral signal"
    last = df.iloc[-1]
    px = float(last["close"])
    sma200 = float(last.get("sma200") or np.nan)
    if not np.isfinite(sma200):
        return True, "No SMA200 yet"
    if direction == "Long" and px <= sma200:
        return False, "Price below SMA200"
    if direction == "Short" and px >= sma200:
        return False, "Price above SMA200"
    return True, "Trend aligned"

def build_trade(df: pd.DataFrame, direction: str, risk_pct: float = 1.0,
                equity: Optional[float] = None, leverage: float = 1.0) -> Optional[Dict[str, Any]]:
    if direction not in ("Long", "Short"):
        return None
    last = df.iloc[-1]
    px   = float(last["close"])
    a    = float(last.get("atr14", np.nan))
    if not np.isfinite(a) or a <= 0:
        return None

    recent_low, _ = recent_extrema(df["low"], 10)
    _, recent_high = recent_extrema(df["high"], 10)

    if direction == "Long":
        base_sl   = px - 1.5*a
        struct_sl = recent_low - 0.2*a
        stop      = min(base_sl, struct_sl)
        R         = px - stop
        tps       = [px + k*R for k in (1,2,3)]
    else:
        base_sl   = px + 1.5*a
        struct_sl = recent_high + 0.2*a
        stop      = max(base_sl, struct_sl)
        R         = stop - px
        tps       = [px - k*R for k in (1,2,3)]

    if R <= 0 or not np.isfinite(R):
        return None

    trade = {
        "direction": direction,
        "entry": px,
        "stop": float(stop),
        "targets": [float(t) for t in tps],
        "rr": [1.0, 2.0, 3.0],
        "volatility": {"atr": a, "atr_pct": float(last.get("atr14_pct", np.nan))},
        "risk_model": {"suggested_risk_pct": risk_pct, "leverage": leverage},
        "risk_suggestions": {
            "breakeven_after_tp": 1,
            "trail_after_tp": 2,
            "trail_method": "ATR",
            "trail_multiple": 1.0,
            "scale_out": [0.5, 0.3, 0.2]
        }
    }

    if equity and equity > 0:
        risk_amt = float(equity) * (risk_pct/100.0)
        qty = max(risk_amt / (R + 1e-12), 0.0)
        notional = qty * px / max(leverage, 1.0)
        trade["position_size"] = {
            "equity": float(equity),
            "risk_amount": risk_amt,
            "qty": float(qty),
            "notional": float(notional),
        }
    return trade

# ---------------- Exchange / universe ----------------
def get_exchange():
    klass = getattr(ccxt(), EXCHANGE)
    return klass({"enableRateLimit": True, "timeout": 20000})

def first_supported_symbol(markets: dict, base: str, quote_priority: List[str]) -> Optional[str]:
    for q in quote_priority:
        sym = f"{base}/{q}"
        m = markets.get(sym)
        if m and m.get("spot", True) and not m.get("contract"):
            return sym
    return None

def curated_universe(ex, markets, bases: List[str], quote_priority: List[str]) -> List[Dict[str, Any]]:
    out = []
    for b in bases:
        sym = first_supported_symbol(markets, b, quote_priority)
        if not sym:
            continue
        m = markets[sym]
        out.append({
            "symbol": sym,
            "name": m.get("base") or b,
            "market": EXCHANGE,
            "tf_supported": ["1h", "1d"]
        })
    return out[:TOP_N]

def tier_b_universe(ex, markets, limit: int, quote_priority: List[str]) -> List[Dict[str, Any]]:
    rank = {b: i for i, b in enumerate(CURATED_BASES)}
    wanted_quotes = {q.upper() for q in quote_priority}
    cand = []
    for sym, m in markets.items():
        if m.get("contract") or not m.get("spot", True):
            continue
        q = (m.get("quote") or "").upper()
        b = (m.get("base") or "").upper()
        if not b or not q or q not in wanted_quotes:
            continue
        cand.append({
            "symbol": sym,
            "name": b,
            "market": EXCHANGE,
            "tf_supported": ["1h","1d"],
            "qrank": quote_priority.index(q) if q in quote_priority else 999,
            "brank": rank.get(b, 10_000)
        })
    cand.sort(key=lambda r: (r["brank"], r["qrank"], r["symbol"]))
    return [{k: r[k] for k in ("symbol","name","market","tf_supported")} for r in cand[:limit]]

def get_universe() -> List[Dict[str, Any]]:
    now = time.time()
    if STATE["universe"] and now - STATE["universe_fetched_at"] < CACHE_TTL_SEC:
        return STATE["universe"]

    dbg = STATE["universe_debug"]
    dbg.update({"path": None, "supported_count": 0, "last_error": None,
                "exchange": EXCHANGE, "quote": QUOTE, "top_n": TOP_N})

    try:
        ex = get_exchange()
        markets = ex.load_markets()

        curated = curated_universe(ex, markets, CURATED_BASES, QUOTE_FALLBACKS)
        if curated:
            dbg["path"] = "curated"
            dbg["supported_count"] = len(curated)
            STATE["universe"] = curated
            STATE["universe_fetched_at"] = now
            return curated

        tier_b = tier_b_universe(ex, markets, TOP_N, QUOTE_FALLBACKS)
        if tier_b:
            dbg["path"] = "tierB_exchange"
            dbg["supported_count"] = len(tier_b)
            STATE["universe"] = tier_b
            STATE["universe_fetched_at"] = now
            return tier_b

        dbg["path"] = "fallback_trio"
        dbg["supported_count"] = 3
        fallback = [
            {"symbol": f"BTC/{QUOTE}", "name": "Bitcoin",  "market": EXCHANGE, "tf_supported": ["1h","1d"]},
            {"symbol": f"ETH/{QUOTE}", "name": "Ethereum", "market": EXCHANGE, "tf_supported": ["1h","1d"]},
            {"symbol": f"SOL/{QUOTE}", "name": "Solana",   "market": EXCHANGE, "tf_supported": ["1h","1d"]},
        ]
        STATE["universe"] = fallback
        STATE["universe_fetched_at"] = now
        return fallback

    except Exception as e:
        dbg["path"] = dbg.get("path") or "exception"
        dbg["last_error"] = str(e)
        if STATE["universe"]:
            return STATE["universe"]
        fallback = [
            {"symbol": f"BTC/{QUOTE}", "name": "Bitcoin",  "market": EXCHANGE, "tf_supported": ["1h","1d"]},
            {"symbol": f"ETH/{QUOTE}", "name": "Ethereum", "market": EXCHANGE, "tf_supported": ["1h","1d"]},
            {"symbol": f"SOL/{QUOTE}", "name": "Solana",   "market": EXCHANGE, "tf_supported": ["1h","1d"]},
        ]
        STATE["universe"] = fallback
        STATE["universe_fetched_at"] = now
        return fallback

# ---------------- Data fetch ----------------
def fetch_ohlcv(symbol: str, timeframe: str) -> pd.DataFrame:
    key = (symbol, timeframe)
    now = time.time()
    cached = STATE["ohlcv_cache"].get(key)
    if cached and now - cached["at"] < 900:   # 15m cache
        return cached["df"]

    ex = get_exchange()
    limit = 1000
    ms_per = {"1h": 3600_000, "1d": 86_400_000}.get(timeframe)
    if not ms_per:
        raise HTTPException(status_code=400, detail="Unsupported timeframe")
    since = int(pd.Timestamp.utcnow().timestamp()*1000 - ms_per * 24 * 365 * 2)  # ~2y
    rows = []
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not batch: break
        rows += batch
        since = batch[-1][0] + 1
        if len(batch) < limit: break
        delay_ms = getattr(ex, "rateLimit", 1000) or 1000
        time.sleep(delay_ms/1000)
        if len(rows) > 100_000: break

    if not rows:
        raise HTTPException(status_code=502, detail=f"No OHLCV for {symbol} {timeframe}")

    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df[(df["close"] > 0) & (df["high"] > 0) & (df["low"] > 0)].copy()
    STATE["ohlcv_cache"][key] = {"at": now, "df": df}
    return df

# ---------------- Models ----------------
class Instrument(BaseModel):
    symbol: str
    name: str
    market: str
    tf_supported: List[str]

class Trade(BaseModel):
    direction: str
    entry: float
    stop: float
    targets: List[float]
    rr: List[float]
    volatility: Dict[str, float]
    risk_model: Dict[str, Any]
    position_size: Optional[Dict[str, float]] = None
    risk_suggestions: Optional[Dict[str, Any]] = None

class SignalOut(BaseModel):
    symbol: str
    timeframe: str
    signal: str
    confidence: float
    updated: str
    features: Dict[str, Optional[float]] = {}  # allow None safely
    trade: Optional[Trade] = None
    filters: Dict[str, Any] = {}
    advice: str = Field(default="Consider")  # Consider / Skip

# ---------------- Routes ----------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "ts": time.time(),
        "exchange": EXCHANGE,
        "quote": QUOTE,
        "top_n": TOP_N,
        "filters": {
            "MIN_CONFIDENCE": MIN_CONFIDENCE,
            "VOL_CAP_ATR_PCT": VOL_CAP_ATR_PCT,
            "VOL_MIN_ATR_PCT": VOL_MIN_ATR_PCT,
        },
        "universe_debug": STATE["universe_debug"],
    }

@app.get("/debug/universe")
def debug_universe(_: None = Depends(require_key)):
    return {
        "cached": bool(STATE["universe"]),
        "cached_age_sec": time.time() - STATE["universe_fetched_at"] if STATE["universe_fetched_at"] else None,
        "universe_len": len(STATE["universe"]),
        "universe": STATE["universe"],
        "debug": STATE["universe_debug"],
    }

@app.get("/instruments", response_model=List[Instrument])
def instruments(_: None = Depends(require_key)):
    return get_universe()

def _safe_float(series, key):
    try:
        v = series.get(key, None)
    except Exception:
        v = None
    return float(v) if v is not None and np.isfinite(v) else None

@app.get("/signals", response_model=SignalOut)
def signals(
    symbol: str,
    tf: str = "1h",
    # sizing
    risk_pct: float = Query(1.0, ge=0.1, le=5.0),
    equity: Optional[float] = Query(None, ge=0.0),
    leverage: float = Query(1.0, ge=1.0, le=10.0),
    # NEW: filter overrides from the UI (all optional)
    min_confidence: Optional[float] = Query(None),   # e.g. 0.12
    vol_cap: Optional[float] = Query(None),          # e.g. 0.10  (10% ATR)
    vol_min: Optional[float] = Query(None),          # e.g. 0.002 (0.2% ATR)
    allow_neutral: int = Query(0, ge=0, le=1),       # 1 => nudge neutral to long/short by SMA20 slope
    ignore_trend: int = Query(0, ge=0, le=1),        # 1 => skip trend filter
    ignore_vol: int = Query(0, ge=0, le=1),          # 1 => skip volatility filter
    _: None = Depends(require_key)
):
    key = (symbol, tf)
    now = time.time()

    # use request overrides if provided, else env defaults
    MINC = float(min_confidence) if min_confidence is not None else MIN_CONFIDENCE
    VCAP = float(vol_cap) if vol_cap is not None else VOL_CAP_ATR_PCT
    VMIN = float(vol_min) if vol_min is not None else VOL_MIN_ATR_PCT
    IGN_TREND = bool(ignore_trend)
    IGN_VOL   = bool(ignore_vol)
    ALLOW_NEU = bool(allow_neutral)

    cached = STATE["signal_cache"].get(key)
    if cached and now - cached["at"] < 300:
        payload = copy.deepcopy(cached["payload"])
        # Re-apply sizing with new equity/risk/leverage if provided
        if payload.get("trade") and equity is not None:
            tr = payload["trade"]
            entry, stop = tr["entry"], tr["stop"]
            R = abs(entry - stop)
            risk_amt = float(equity) * (risk_pct/100.0)
            qty = max(risk_amt / (R + 1e-12), 0.0)
            notional = qty * entry / max(leverage, 1.0)
            tr["risk_model"].update({"suggested_risk_pct": risk_pct, "leverage": leverage})
            tr["position_size"] = {"equity": float(equity), "risk_amount": risk_amt, "qty": float(qty), "notional": float(notional)}
        return payload

    df = fetch_ohlcv(symbol, tf)
    df = compute_features(df).dropna()
    if "atr14" not in df.columns or "atr14_pct" not in df.columns or len(df) < 200:
        raise HTTPException(status_code=502, detail="Not enough features/history")

    pred = rule_inference(df)
    feats = df.iloc[-1]

    # possibly nudge neutrals by SMA20 slope when allowed
    if pred["signal"] == "Neutral" and ALLOW_NEU:
        slope = (df["sma20"].iloc[-1] - df["sma20"].iloc[-5]) if df["sma20"].notna().tail(5).all() else 0.0
        direction = "Long" if slope >= 0 else "Short"
    else:
        direction = "Long" if pred["signal"] == "Bullish" else "Short" if pred["signal"] == "Bearish" else "Neutral"

    # filters
    filters = {"trend_ok": True, "vol_ok": True, "confidence_ok": True, "reasons": []}

    if not IGN_TREND:
        ok, msg = trend_ok(direction, df)
        filters["trend_ok"] = ok if direction != "Neutral" else False
        if direction == "Neutral":
            filters["reasons"].append("Neutral signal")
        elif not ok:
            filters["reasons"].append(msg)

    atr_pct = float(feats.get("atr14_pct", np.nan))
    if not IGN_VOL:
        if np.isfinite(atr_pct):
            if atr_pct > VCAP:
                filters["vol_ok"] = False
                filters["reasons"].append(f"ATR% {atr_pct:.1%} > cap {VCAP:.0%}")
            if atr_pct < VMIN:
                filters["vol_ok"] = False
                filters["reasons"].append(f"ATR% {atr_pct:.2%} < min {VMIN:.2%}")
        else:
            filters["vol_ok"] = False
            filters["reasons"].append("ATR% unavailable")

    if abs(pred["confidence"]) < MINC:
        filters["confidence_ok"] = False
        filters["reasons"].append(f"Confidence {pred['confidence']:.2f} < {MINC:.2f}")

    trade = None
    advice = "Consider"
    if direction == "Neutral" or not (filters["trend_ok"] or IGN_TREND) or not (filters["vol_ok"] or IGN_VOL) or not filters["confidence_ok"]:
        advice = "Skip"
    else:
        trade = build_trade(df, direction, risk_pct=risk_pct, equity=equity, leverage=leverage)
        if trade is None:
            advice = "Skip"

    payload = {
        "symbol": symbol,
        "timeframe": tf,
        "signal": pred["signal"],
        "confidence": float(pred["confidence"]),
        "updated": pd.Timestamp.utcnow().isoformat(),
        "features": {
            "rsi14": _safe_float(feats, "rsi14"),
            "sma20_50_diff": _safe_float(feats, "sma20_50_diff"),
            "bb_pos": _safe_float(feats, "bb_pos"),
            "ret5": _safe_float(feats, "ret5"),
            "atr14": _safe_float(feats, "atr14"),
            "atr14_pct": _safe_float(feats, "atr14_pct"),
            "sma200": _safe_float(feats, "sma200"),
        },
        "trade": trade,
        "filters": filters,
        "advice": advice
    }
    STATE["signal_cache"][key] = {"at": now, "payload": payload}
    return payload

@app.get("/summary")
def summary(_: None = Depends(require_key)):
    uni = get_universe()
    board = []
    for inst in uni[: min(10, len(uni))]:
        try:
            s = signals(inst["symbol"], "1h")
            board.append({"symbol": inst["symbol"], "signal": s["signal"], "confidence": s["confidence"]})
        except Exception:
            board.append({"symbol": inst["symbol"], "signal": "Neutral", "confidence": 0.0})
    return {"universe": uni, "leaderboard": board}

@app.post("/admin/refresh")
def refresh(_: None = Depends(require_key)):
    STATE["universe"] = []
    STATE["universe_fetched_at"] = 0
    STATE["ohlcv_cache"].clear()
    STATE["signal_cache"].clear()
    STATE["universe_debug"].update({"path": "manual_refresh", "supported_count": 0, "last_error": None})
    uni = get_universe()
    return {"ok": True, "universe": len(uni)}

@app.get("/chart")
def chart(
    symbol: str,
    tf: str = "1h",
    n: int = Query(120, ge=20, le=500),
    _: None = Depends(require_key)
):
    df = fetch_ohlcv(symbol, tf)
    if df.empty:
        raise HTTPException(status_code=502, detail="No OHLCV")
    tail = df.tail(n).copy()
    closes = tail["close"].astype(float).tolist()
    ts = tail["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
    lo, hi = float(min(closes)), float(max(closes))
    if hi - lo < 1e-12:
        hi, lo = hi + 1e-6, lo - 1e-6
    chg = (closes[-1] / closes[0] - 1.0) if closes[0] else 0.0
    return {
        "symbol": symbol,
        "timeframe": tf,
        "n": len(closes),
        "timestamps": ts,
        "closes": closes,
        "min": lo,
        "max": hi,
        "change": chg,
    }
