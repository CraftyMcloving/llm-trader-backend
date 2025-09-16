# app/main.py
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
import os, time, math, json, random
import pandas as pd
import numpy as np

# ---- Config ----
API_KEY = os.getenv("API_KEY", "change-me")               # set on Render
EXCHANGE = os.getenv("EXCHANGE", "binance")               # binance/kraken/coinbase
QUOTE = os.getenv("QUOTE", "USDT")                        # default quote
TOP_N = int(os.getenv("TOP_N", "25"))                     # top-25 by mcap
CACHE_TTL_SEC = int(os.getenv("CACHE_TTL_SEC", "1800"))   # 30m instruments cache

# ---- External (lazy import so app boots even if libs missing) ----
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
app = FastAPI(title="AI Trade Advisor", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # optionally lock to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- In-memory caches (you can swap for Redis/Postgres later) ----
STATE = {
    "universe": [],          # list[Instrument]
    "universe_fetched_at": 0,
    "ohlcv_cache": {},       # key: (symbol, tf) -> {"at": ts, "df": DataFrame}
    "signal_cache": {},      # key: (symbol, tf) -> {"at": ts, "payload": dict}
}

# ---------------- Feature engineering & simple inference ----------------
def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = pd.Series(np.where(delta > 0, delta, 0.0), index=series.index)
    dn = pd.Series(np.where(delta < 0, -delta, 0.0), index=series.index)
    rs = up.rolling(n).mean() / (dn.rolling(n).mean() + 1e-12)
    return 100 - (100/(1+rs))

def bollinger(series: pd.Series, n: int = 20, k: float = 2.0):
    ma = series.rolling(n).mean()
    sd = series.rolling(n).std()
    return ma + k*sd, ma - k*sd

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret1"] = df["close"].pct_change()
    df["ret5"] = df["close"].pct_change(5)
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()
    df["sma20_50_diff"] = (df["sma20"] - df["sma50"]) / (df["sma50"] + 1e-12)
    df["rsi14"] = rsi(df["close"], 14)
    up, dn = bollinger(df["close"], 20, 2)
    df["bb_up"], df["bb_dn"] = up, dn
    df["bb_pos"] = (df["close"] - df["bb_dn"]) / ((df["bb_up"] - df["bb_dn"]) + 1e-12)
    return df

def rule_inference(df: pd.DataFrame) -> Dict[str, Any]:
    """Transparent baseline. Swap with ML later."""
    last = df.iloc[-1]
    slope = (df["sma20"].iloc[-1] - df["sma20"].iloc[-5]) if df["sma20"].notna().tail(5).all() else 0.0
    score = 0.0
    # RSI zones
    if last["rsi14"] < 30: score += 0.45
    if last["rsi14"] > 70: score -= 0.45
    # Trend tilt
    base = abs(df["sma20"].iloc[-5]) + 1e-6 if df["sma20"].notna().iloc[-5] else 1.0
    score += np.tanh(slope / base) * 0.3
    # Bollinger position (mean-revert bias)
    score += np.clip(last["bb_pos"] - 0.5, -0.5, 0.5) * 0.25
    label = "Bullish" if score > 0.15 else "Bearish" if score < -0.15 else "Neutral"
    conf = float(np.clip(abs(score), 0, 1))
    return {"signal": label, "confidence": conf, "score": float(score)}

# ---------------- Data sources ----------------
def coingecko_top_n(n: int) -> List[Dict[str, Any]]:
    sess = requests().Session()
    sess.headers.update({"User-Agent": "AI-Trade-Advisor/1.0 (contact: site-owner)"})
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = dict(vs_currency="usd", order="market_cap_desc", per_page=n, page=1)
    last_err = None
    for _ in range(3):  # up to 3 retries
        try:
            r = sess.get(url, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(2)  # gentle backoff
                continue
            r.raise_for_status()
            coins = r.json()
            out = []
            for c in coins:
                sym = (c.get("symbol") or "").upper()
                name = c.get("name") or sym
                out.append({"base": sym, "name": name})
            return out
        except Exception as e:
            last_err = e
            time.sleep(1)
    # if we’re here, all retries failed
    raise HTTPException(status_code=502, detail=f"CoinGecko error: {last_err}")

def get_universe() -> List[Dict[str, Any]]:
    now = time.time()
    if STATE["universe"] and now - STATE["universe_fetched_at"] < CACHE_TTL_SEC:
        return STATE["universe"]

    try:
        # 1) fetch top list from CoinGecko
        top = coingecko_top_n(TOP_N)  # list of {"base","name"}
        # 2) load exchange markets and keep only supported pairs
        ex = get_exchange()
        markets = ex.load_markets()
        wanted = set(f"{t['base']}/{QUOTE}" for t in top)
        supported = []
        for sym, m in markets.items():
            if sym in wanted and m.get("spot", True) and not m.get("contract"):
                base = m.get("base") or sym.split("/")[0]
                name = next((t["name"] for t in top if t["base"] == base), base)
                supported.append({"symbol": sym, "name": name, "market": EXCHANGE, "tf_supported": ["1h","1d"]})
        # If some are missing, we still return whatever is supported (could be < TOP_N)
        # If *none* matched (e.g., QUOTE not offered), fallback to the safe trio
        if not supported:
            supported = [
                {"symbol": f"BTC/{QUOTE}", "name": "Bitcoin", "market": EXCHANGE, "tf_supported": ["1h","1d"]},
                {"symbol": f"ETH/{QUOTE}", "name": "Ethereum", "market": EXCHANGE, "tf_supported": ["1h","1d"]},
                {"symbol": f"SOL/{QUOTE}", "name": "Solana", "market": EXCHANGE, "tf_supported": ["1h","1d"]},
            ]
        STATE["universe"] = supported
        STATE["universe_fetched_at"] = now
        return supported
    except HTTPException:
        # propagate API error; if we have stale, serve it
        if STATE["universe"]:
            return STATE["universe"]
        # final safety fallback
        return [
            {"symbol": f"BTC/{QUOTE}", "name": "Bitcoin", "market": EXCHANGE, "tf_supported": ["1h","1d"]},
            {"symbol": f"ETH/{QUOTE}", "name": "Ethereum","market": EXCHANGE, "tf_supported": ["1h","1d"]},
            {"symbol": f"SOL/{QUOTE}", "name": "Solana",  "market": EXCHANGE, "tf_supported": ["1h","1d"]},
        ]

def get_exchange():
    # build CCXT exchange with sane defaults
    klass = getattr(ccxt(), EXCHANGE)
    ex = klass({
        "enableRateLimit": True,
        "timeout": 20000,
        # add keys here if needed, most public OHLCV doesn’t require
    })
    return ex

def fetch_ohlcv(symbol: str, timeframe: str) -> pd.DataFrame:
    # cache 15 minutes per (symbol, tf) to keep Render costs down
    key = (symbol, timeframe)
    now = time.time()
    cached = STATE["ohlcv_cache"].get(key)
    if cached and now - cached["at"] < 900:
        return cached["df"]

    ex = get_exchange()
    limit = 1000
    ms_per = {"1h": 3600_000, "1d": 86_400_000}.get(timeframe)
    if not ms_per:
        raise HTTPException(status_code=400, detail="Unsupported timeframe")
    since = int(pd.Timestamp.utcnow().timestamp()*1000 - ms_per * 24 * 365 * 2)  # ~2 years
    rows = []
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not batch: break
        rows += batch
        since = batch[-1][0] + 1
        # exit if exchange truncated
        if len(batch) < limit: break
        time.sleep(ex.rateLimit/1000)
        # safety: stop after ~100k rows
        if len(rows) > 100_000: break

    if not rows:
        raise HTTPException(status_code=502, detail=f"No OHLCV for {symbol} {timeframe}")

    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    # sanity filter (drop zeros/negatives)
    df = df[(df["close"] > 0) & (df["high"] > 0) & (df["low"] > 0)].copy()

    STATE["ohlcv_cache"][key] = {"at": now, "df": df}
    return df

# ---------------- Schemas ----------------
class Instrument(BaseModel):
    symbol: str
    name: str
    market: str
    tf_supported: List[str]

class Signal(BaseModel):
    symbol: str
    timeframe: str
    signal: str
    confidence: float
    updated: str
    features: Dict[str, float] = {}

# ---------------- Routes ----------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "ts": time.time(),
        "exchange": EXCHANGE,
        "quote": QUOTE,
        "top_n": TOP_N
    }

@app.get("/instruments", response_model=List[Instrument])
def instruments(_: None = Depends(require_key)):
    return get_universe()

@app.get("/signals", response_model=Signal)
def signals(symbol: str, tf: str = "1h", _: None = Depends(require_key)):
    key = (symbol, tf)
    now = time.time()
    cached = STATE["signal_cache"].get(key)
    if cached and now - cached["at"] < 300:  # 5m cache
        return cached["payload"]

    # Fetch & featurize
    df = fetch_ohlcv(symbol, tf)
    df = compute_features(df).dropna()
    if len(df) < 60:
        raise HTTPException(status_code=502, detail="Not enough data to compute features")

    pred = rule_inference(df)
    feats = df.iloc[-1]
    payload = {
        "symbol": symbol,
        "timeframe": tf,
        "signal": pred["signal"],
        "confidence": float(pred["confidence"]),
        "updated": pd.Timestamp.utcnow().isoformat(),
        "features": {
            "rsi14": float(feats["rsi14"]),
            "sma20_50_diff": float(feats["sma20_50_diff"]),
            "bb_pos": float(feats["bb_pos"]),
            "ret5": float(feats["ret5"]),
        }
    }
    STATE["signal_cache"][key] = {"at": now, "payload": payload}
    return payload

@app.get("/summary")
def summary(_: None = Depends(require_key)):
    uni = get_universe()
    board = []
    for inst in uni[: min(10, len(uni))]:  # keep this light
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
    # trigger fetch once so the next call is fast
    uni = get_universe()
    return {"ok": True, "universe": len(uni)}