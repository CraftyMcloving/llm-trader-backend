# app/main.py  — v1.5.0 “Pro signals + Learning”
from fastapi import FastAPI, HTTPException, Depends, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple
import os, time, copy, json, math
import pandas as pd
import numpy as np

# =========================
# Config (sane, trade-finding defaults)
# =========================
API_KEY  = os.getenv("API_KEY", "change-me")
EXCHANGE = os.getenv("EXCHANGE", "kraken").lower()
QUOTE    = os.getenv("QUOTE", "USD").upper()
TOP_N    = int(os.getenv("TOP_N", "20"))
CACHE_TTL_SEC = int(os.getenv("CACHE_TTL_SEC", "1800"))

# Curated names first; we fall back if missing
CURATED_BASES = [b.strip().upper() for b in os.getenv(
    "CURATED_BASES",
    "BTC,XRP,ETH,SOL,ADA,HYPE,USDT,BNB,USDC,DOGE,STETH,TRX,LINK,WBTC,SUI,AVAX,XLM,BCH,LTC,CRO,TON,USDS,SHIB,DOT,XMR,MNT,UNI"
).split(",") if b.strip()]
QUOTE_FALLBACKS = [q.strip().upper() for q in os.getenv(
    "QUOTE_FALLBACKS",
    f"{QUOTE},USD,USDT,USDC,EUR"
).split(",") if q.strip()]

# Filters / risk defaults — made more permissive to find trades
MIN_CONFIDENCE   = float(os.getenv("MIN_CONFIDENCE", "0.10"))   # was 0.18
VOL_CAP_ATR_PCT  = float(os.getenv("VOL_CAP_ATR_PCT", "0.25"))  # allow up to 25% ATR
VOL_MIN_ATR_PCT  = float(os.getenv("VOL_MIN_ATR_PCT", "0.001")) # >= 0.10% ATR
ALLOW_NEUTRAL_DEFAULT = int(os.getenv("ALLOW_NEUTRAL_DEFAULT", "1"))  # nudge neutrals on by default

# Learning store
FEEDBACK_PATH = os.getenv("FEEDBACK_PATH", "/tmp/ai_trade_feedback.json")  # ephemeral on Render, but fine to start
LEARN_RATE = float(os.getenv("LEARN_RATE", "0.05"))  # tiny online weight step

# =========================
# Lazy deps
# =========================
_ccxt = None
def ccxt():
    global _ccxt
    if _ccxt is None:
        import ccxt as c
        _ccxt = c
    return _ccxt

# =========================
# Security
# =========================
def require_key(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    token = authorization.split(" ", 1)[1].strip()
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid token")

# =========================
# App
# =========================
app = FastAPI(title="AI Trade Advisor", version="1.5.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE: Dict[str, Any] = {
    "universe": [],
    "universe_fetched_at": 0,
    "ohlcv_cache": {},     # (symbol, tf) -> {at, df}
    "signal_cache": {},    # (symbol, tf) -> {at, payload}
    "edge_cache": {},      # (symbol, tf) -> {at, edge}
    "universe_debug": {"path": None, "supported_count": 0, "last_error": None, "exchange": EXCHANGE, "quote": QUOTE, "top_n": TOP_N},
    "weights": {"rule": 0.5, "edge": 0.35, "trend": 0.15, "bias": 0.0},  # learned tweaks
}

# =========================
# TA helpers
# =========================
def ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False).mean()

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = pd.Series(np.where(delta > 0, delta, 0.0), index=series.index)
    dn = pd.Series(np.where(delta < 0, -delta, 0.0), index=series.index)
    rs = up.rolling(n, min_periods=n).mean() / (dn.rolling(n, min_periods=n).mean() + 1e-12)
    return 100 - (100/(1+rs))

def bollinger(series: pd.Series, n: int = 20, k: float = 2.0) -> Tuple[pd.Series, pd.Series]:
    ma = series.rolling(n, min_periods=n).mean()
    sd = series.rolling(n, min_periods=n).std()
    return ma + k*sd, ma - k*sd

def macd_hist(close: pd.Series, fast=12, slow=26, sig=9) -> pd.Series:
    macd = ema(close, fast) - ema(close, slow)
    signal = ema(macd, sig)
    return macd - signal

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def donch_pos(df: pd.DataFrame, n: int = 20) -> pd.Series:
    hh = df["high"].rolling(n, min_periods=n).max()
    ll = df["low"].rolling(n, min_periods=n).min()
    return (df["close"] - ll) / ((hh - ll) + 1e-12)  # 0..1 channel position

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
    df["macd_hist"] = macd_hist(df["close"])
    df["donch20"] = donch_pos(df, 20)  # 0..1
    return df

def rule_inference(df: pd.DataFrame) -> Dict[str, Any]:
    last = df.iloc[-1]
    slope_ok = df["sma20"].notna().tail(5).all()
    slope = (df["sma20"].iloc[-1] - df["sma20"].iloc[-5]) if slope_ok else 0.0
    score = 0.0
    # momentum extremes
    score += 0.50 if last["rsi14"] < 30 else 0.0
    score -= 0.50 if last["rsi14"] > 70 else 0.0
    # slope adds/subtracts softly
    base = abs(df["sma20"].iloc[-5]) + 1e-6 if slope_ok else 1.0
    score += np.tanh(slope / base) * 0.30
    # channel/Bollinger location
    score += np.clip((last["bb_pos"] - 0.5) * 0.30, -0.30, 0.30)
    # MACD hist sign
    if np.isfinite(last.get("macd_hist", np.nan)):
        score += np.clip(last["macd_hist"], -1, 1) * 0.15
    label = "Bullish" if score > 0.08 else "Bearish" if score < -0.08 else "Neutral"
    conf = float(np.clip(abs(score), 0, 1))
    return {"signal": label, "confidence": conf, "score": float(score)}

def trend_ok(direction: str, df: pd.DataFrame) -> tuple[bool, str]:
    if direction not in ("Long","Short"):
        return False, "Neutral signal"
    last = df.iloc[-1]
    px = float(last["close"])
    sma200 = float(last.get("sma200") or np.nan)
    sma50  = float(last.get("sma50") or np.nan)
    if not np.isfinite(sma200) or not np.isfinite(sma50):
        return True, "No long MA yet"
    if direction == "Long" and (px <= sma200 or sma50 <= sma200):
        return False, "Below SMA200 / weak trend"
    if direction == "Short" and (px >= sma200 or sma50 >= sma200):
        return False, "Above SMA200 / weak downtrend"
    return True, "Trend aligned"

# ========== Historical analogue edge ==========
def _zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std() + 1e-12)

def historical_edge(df: pd.DataFrame, tf: str) -> Dict[str, Any]:
    """
    k-NN analogue search on feature space to estimate forward return.
    Returns dict with winrate, exp_ret, horizon_bars, k, score in [-1,1].
    Cached per (symbol, tf) via STATE["edge_cache"] by caller.
    """
    feat = df.dropna().copy()
    if len(feat) < 300:  # need enough history
        return {"winrate": None, "exp_ret": None, "horizon": None, "k": 0, "score": 0.0}

    # feature vector (normalized)
    cols = ["rsi14","sma20_50_diff","macd_hist","bb_pos","atr14_pct","donch20"]
    X = pd.DataFrame({c: _zscore(feat[c]) for c in cols})
    current = X.iloc[-1].values

    # search pool excludes the most recent 100 bars to avoid leakage
    pool = X.iloc[:-100].copy()
    # cosine similarity
    num = (pool * current).sum(axis=1)
    den = (np.sqrt((pool**2).sum(axis=1)) * np.sqrt((current**2).sum()))
    sim = (num / (den + 1e-12)).clip(-1,1)

    k = 60
    idx = sim.nlargest(k).index
    horizon = 12 if tf == "1h" else 5   # ~12h or 5d median
    fwd_ret = (feat["close"].shift(-horizon) / feat["close"] - 1.0).reindex(idx)
    fwd_ret = fwd_ret.dropna()
    if fwd_ret.empty:
        return {"winrate": None, "exp_ret": None, "horizon": horizon, "k": 0, "score": 0.0}

    winrate = float((fwd_ret > 0).mean())
    exp_ret = float(fwd_ret.mean())  # fraction
    # Normalize expected return by typical volatility (ATR%) to form a [-1,1]-ish edge score
    atr_pct_now = float(feat["atr14_pct"].iloc[-1])
    vol_norm = max(atr_pct_now, 1e-3)
    edge_raw = exp_ret / (vol_norm * 2.0)  # 2×ATR% ~ 1R-ish
    edge_score = float(np.tanh(edge_raw))  # squash to [-1,1]
    return {"winrate": winrate, "exp_ret": exp_ret, "horizon": horizon, "k": int(len(fwd_ret)), "score": edge_score}

# ========== Composite & trade plan ==========
def composite_score(rule_score: float, edge_score: float, trend_dir_score: float, weights: Dict[str, float]) -> float:
    # Weighted sum then tanh to bound
    total = weights["rule"]*rule_score + weights["edge"]*edge_score + weights["trend"]*trend_dir_score + weights.get("bias",0.0)
    return float(np.tanh(total))

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

# =========================
# Exchange / universe
# =========================
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
            "symbol": sym, "name": b, "market": EXCHANGE, "tf_supported": ["1h","1d"],
            "qrank": quote_priority.index(q) if q in quote_priority else 999,
            "brank": rank.get(b, 10_000)
        })
    cand.sort(key=lambda r: (r["brank"], r["qrank"], r["symbol"]))
    return [{k: r[k] for k in ("symbol","name","market","tf_supported")} for r in cand[:limit]]

def get_universe() -> List[Dict[str, Any]]:
    now = time.time()
    if STATE["universe"] and now - STATE["universe_fetched_at"] < CACHE_TTL_SEC:
        return STATE["universe"]
    dbg = STATE["universe_debug"]; dbg.update({"path": None, "supported_count": 0, "last_error": None,
                                               "exchange": EXCHANGE, "quote": QUOTE, "top_n": TOP_N})
    try:
        ex = get_exchange()
        markets = ex.load_markets()
        curated = curated_universe(ex, markets, CURATED_BASES, QUOTE_FALLBACKS)
        if curated:
            dbg["path"] = "curated"; dbg["supported_count"] = len(curated)
            STATE["universe"], STATE["universe_fetched_at"] = curated, now
            return curated
        tier_b = tier_b_universe(ex, markets, TOP_N, QUOTE_FALLBACKS)
        if tier_b:
            dbg["path"] = "tierB_exchange"; dbg["supported_count"] = len(tier_b)
            STATE["universe"], STATE["universe_fetched_at"] = tier_b, now
            return tier_b
        fallback = [
            {"symbol": f"BTC/{QUOTE}", "name": "Bitcoin",  "market": EXCHANGE, "tf_supported": ["1h","1d"]},
            {"symbol": f"ETH/{QUOTE}", "name": "Ethereum", "market": EXCHANGE, "tf_supported": ["1h","1d"]},
            {"symbol": f"SOL/{QUOTE}", "name": "Solana",   "market": EXCHANGE, "tf_supported": ["1h","1d"]},
        ]
        dbg["path"] = "fallback_trio"; dbg["supported_count"] = 3
        STATE["universe"], STATE["universe_fetched_at"] = fallback, now
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
        STATE["universe"], STATE["universe_fetched_at"] = fallback, now
        return fallback

# =========================
# Data fetch
# =========================
def fetch_ohlcv(symbol: str, timeframe: str) -> pd.DataFrame:
    key = (symbol, timeframe)
    now = time.time()
    cached = STATE["ohlcv_cache"].get(key)
    if cached and now - cached["at"] < 900:
        return cached["df"]
    ex = get_exchange()
    ms_per = {"1h": 3600_000, "1d": 86_400_000}.get(timeframe)
    if not ms_per:
        raise HTTPException(status_code=400, detail="Unsupported timeframe")
    since = int(pd.Timestamp.utcnow().timestamp()*1000 - ms_per * 24 * 365 * 2)
    limit = 1000
    rows = []
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not batch: break
        rows += batch
        since = batch[-1][0] + 1
        if len(batch) < limit: break
        delay_ms = getattr(ex, "rateLimit", 1000) or 1000
        time.sleep(delay_ms/1000)
        if len(rows) > 120_000: break
    if not rows:
        raise HTTPException(status_code=502, detail=f"No OHLCV for {symbol} {timeframe}")
    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df[(df["close"] > 0) & (df["high"] > 0) & (df["low"] > 0)].copy()
    STATE["ohlcv_cache"][key] = {"at": now, "df": df}
    return df

# =========================
# Models
# =========================
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
    features: Dict[str, Optional[float]] = {}
    trade: Optional[Trade] = None
    filters: Dict[str, Any] = {}
    advice: str = Field(default="Consider")
    edge: Dict[str, Any] = {}
    composite: Dict[str, Any] = {}

# =========================
# Learning store
# =========================
def _load_feedback():
    try:
        with open(FEEDBACK_PATH, "r") as f:
            obj = json.load(f)
            STATE["weights"] = obj.get("weights", STATE["weights"])
            return obj
    except Exception:
        return {"weights": STATE["weights"], "events": []}

def _save_feedback(obj):
    obj["weights"] = STATE["weights"]
    try:
        with open(FEEDBACK_PATH, "w") as f:
            json.dump(obj, f)
    except Exception:
        pass

FEEDBACK = _load_feedback()

def _update_weights(components: Dict[str, float], outcome: int):
    """
    outcome: 1 for success (trade idea good), 0 for fail.
    components: {'rule': score, 'edge': score, 'trend': score}
    Simple online update nudging weights toward components that helped.
    """
    w = STATE["weights"]
    # predicted "probability" proxy from current weights
    pred = 0.5 + 0.5 * np.tanh(w["rule"]*components.get("rule",0)
                                + w["edge"]*components.get("edge",0)
                                + w["trend"]*components.get("trend",0)
                                + w.get("bias",0))
    error = outcome - float(pred)  # positive → increase
    for k in ("rule","edge","trend"):
        w[k] = float(np.clip(w[k] + LEARN_RATE * error * components.get(k,0), 0.05, 1.0))
    w["bias"] = float(np.clip(w.get("bias",0) + LEARN_RATE * error * 0.5, -0.5, 0.5))

# =========================
# Routes
# =========================
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
            "ALLOW_NEUTRAL_DEFAULT": ALLOW_NEUTRAL_DEFAULT
        },
        "weights": STATE["weights"],
        "universe_debug": STATE["universe_debug"],
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
    # overrides
    min_confidence: Optional[float] = Query(None),
    vol_cap: Optional[float] = Query(None),
    vol_min: Optional[float] = Query(None),
    allow_neutral: int = Query(ALLOW_NEUTRAL_DEFAULT, ge=0, le=1),
    ignore_trend: int = Query(0, ge=0, le=1),
    ignore_vol: int = Query(0, ge=0, le=1),
    # force plan anyway
    force: int = Query(0, ge=0, le=1),
    _: None = Depends(require_key)
):
    # cache shallow (not for overrides), we still re-size on the fly
    key = (symbol, tf)
    now = time.time()
    cached = STATE["signal_cache"].get(key)
    if cached and now - cached["at"] < 180:
        payload = copy.deepcopy(cached["payload"])
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

    df_raw = fetch_ohlcv(symbol, tf)
    df = compute_features(df_raw).dropna()
    if "atr14" not in df.columns or "atr14_pct" not in df.columns or len(df) < 240:
        raise HTTPException(status_code=502, detail="Not enough features/history")

    # Rule-based signal
    pred = rule_inference(df)
    feats = df.iloc[-1]
    # Trend direction score: +0.25 if aligned, -0.25 if opposite, 0 if neutral
    trend_dir_score = 0.0
    if pred["signal"] != "Neutral":
        ok, _ = trend_ok("Long" if pred["signal"]=="Bullish" else "Short", df)
        trend_dir_score = 0.25 if ok else -0.25

    # Historical edge (cache for 5m)
    ekey = (symbol, tf)
    ec = STATE["edge_cache"].get(ekey)
    if not ec or now - ec["at"] > 300:
        edge = historical_edge(df, tf)
        STATE["edge_cache"][ekey] = {"at": now, "edge": edge}
    else:
        edge = ec["edge"]

    # Composite score decides direction & confidence
    total_score = composite_score(pred["score"], edge.get("score",0.0), trend_dir_score, STATE["weights"])
    comp_dir = "Long" if total_score > 0.03 else "Short" if total_score < -0.03 else "Neutral"
    comp_conf = float(min(0.99, abs(total_score)))  # 0..~1

    # Final direction with optional nudge for neutral
    ALLOW_NEU = bool(allow_neutral)
    if comp_dir == "Neutral" and ALLOW_NEU:
        # nudge using SMA20 slope
        slope_ok = df["sma20"].notna().tail(5).all()
        slope = (df["sma20"].iloc[-1] - df["sma20"].iloc[-5]) if slope_ok else 0.0
        direction = "Long" if slope >= 0 else "Short"
    else:
        direction = comp_dir if comp_dir != "Neutral" else ("Long" if pred["signal"]=="Bullish" else "Short" if pred["signal"]=="Bearish" else "Neutral")

    # Effective thresholds
    MINC = float(min_confidence) if min_confidence is not None else MIN_CONFIDENCE
    VCAP = float(vol_cap) if vol_cap is not None else VOL_CAP_ATR_PCT
    VMIN = float(vol_min) if vol_min is not None else VOL_MIN_ATR_PCT
    IGN_TREND = bool(ignore_trend)
    IGN_VOL   = bool(ignore_vol)
    FORCE     = bool(force)

    # Filters
    filters = {"trend_ok": True, "vol_ok": True, "confidence_ok": True, "reasons": []}

    if not IGN_TREND:
        ok, msg = trend_ok(direction, df)
        filters["trend_ok"] = ok if direction != "Neutral" else False
        if direction == "Neutral":
            filters["reasons"].append("Neutral signal")
        elif not ok:
            filters["reasons"].append(msg)

    atr_pct = float(_safe_float(feats, "atr14_pct") or 0.0)
    if not IGN_VOL:
        if atr_pct > VCAP:
            filters["vol_ok"] = False; filters["reasons"].append(f"ATR% {atr_pct:.1%} > cap {VCAP:.0%}")
        if atr_pct < VMIN:
            filters["vol_ok"] = False; filters["reasons"].append(f"ATR% {atr_pct:.2%} < min {VMIN:.2%}")

    # Confidence filter uses composite confidence (more informative)
    if comp_conf < MINC:
        filters["confidence_ok"] = False
        filters["reasons"].append(f"Composite conf {comp_conf:.2f} < {MINC:.2f}")

    pass_ok = (direction != "Neutral") and filters["confidence_ok"] and ((filters["trend_ok"] or IGN_TREND) and (filters["vol_ok"] or IGN_VOL))

    trade = None
    if pass_ok or FORCE:
        if direction == "Neutral":
            # last resort nudge
            slope_ok = df["sma20"].notna().tail(5).all()
            slope = (df["sma20"].iloc[-1] - df["sma20"].iloc[-5]) if slope_ok else 0.0
            direction = "Long" if slope >= 0 else "Short"
        trade = build_trade(df, direction, risk_pct=risk_pct, equity=equity, leverage=leverage)

    advice = "Consider" if pass_ok and trade else ("Draft" if FORCE and trade else "Skip")

    payload = {
        "symbol": symbol,
        "timeframe": tf,
        "signal": pred["signal"],            # original label
        "confidence": comp_conf,             # composite confidence drives UI bar
        "updated": pd.Timestamp.utcnow().isoformat(),
        "features": {
            "rsi14": _safe_float(feats, "rsi14"),
            "sma20_50_diff": _safe_float(feats, "sma20_50_diff"),
            "bb_pos": _safe_float(feats, "bb_pos"),
            "ret5": _safe_float(feats, "ret5"),
            "atr14": _safe_float(feats, "atr14"),
            "atr14_pct": atr_pct,
            "sma200": _safe_float(feats, "sma200"),
            "macd_hist": _safe_float(feats, "macd_hist"),
            "donch20": _safe_float(feats, "donch20"),
        },
        "trade": trade,
        "filters": filters,
        "advice": advice,
        "edge": edge,
        "composite": {"score": total_score, "direction": direction}
    }
    STATE["signal_cache"][key] = {"at": now, "payload": payload}
    return payload

@app.post("/feedback")
def feedback(evt: Dict[str, Any], _: None = Depends(require_key)):
    """
    Body example:
    {
      "symbol":"BTC/USD","tf":"1h",
      "outcome":"win|loss|be",
      "components":{"rule":0.12,"edge":0.35,"trend":0.25},
      "meta":{"direction":"Long","entry":..., "stop":..., "tp_hit":1}
    }
    """
    req = {**evt}
    outcome = (evt.get("outcome") or "").lower()
    y = 1 if outcome == "win" else (0 if outcome == "loss" else None)
    FEEDBACK.setdefault("events", []).append({"ts": time.time(), **req})
    if y is not None and isinstance(evt.get("components"), dict):
        _update_weights(evt["components"], y)
    _save_feedback(FEEDBACK)
    return {"ok": True, "weights": STATE["weights"], "stored": True}

@app.get("/feedback/stats")
def feedback_stats(_: None = Depends(require_key)):
    ev = FEEDBACK.get("events", [])
    wins = sum(1 for e in ev if (e.get("outcome") or "").lower() == "win")
    losses = sum(1 for e in ev if (e.get("outcome") or "").lower() == "loss")
    return {"count": len(ev), "wins": wins, "losses": losses, "weights": STATE["weights"]}

@app.post("/admin/refresh")
def refresh(_: None = Depends(require_key)):
    STATE["universe"] = []; STATE["universe_fetched_at"] = 0
    STATE["ohlcv_cache"].clear(); STATE["signal_cache"].clear(); STATE["edge_cache"].clear()
    STATE["universe_debug"].update({"path": "manual_refresh", "supported_count": 0, "last_error": None})
    uni = get_universe()
    return {"ok": True, "universe": len(uni)}

@app.get("/chart")
def chart(symbol: str, tf: str = "1h", n: int = Query(120, ge=20, le=500), _: None = Depends(require_key)):
    df = fetch_ohlcv(symbol, tf)
    if df.empty: raise HTTPException(status_code=502, detail="No OHLCV")
    tail = df.tail(n).copy()
    closes = tail["close"].astype(float).tolist()
    ts = tail["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
    lo, hi = float(min(closes)), float(max(closes))
    if hi - lo < 1e-12: hi, lo = hi + 1e-6, lo - 1e-6
    chg = (closes[-1] / closes[0] - 1.0) if closes[0] else 0.0
    return {"symbol": symbol, "timeframe": tf, "n": len(closes), "timestamps": ts, "closes": closes, "min": lo, "max": hi, "change": chg}
