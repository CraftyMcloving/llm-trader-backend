# app/main.py
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os, time
import pandas as pd
import numpy as np

# ---- Config (normalize) ----
API_KEY  = os.getenv("API_KEY", "change-me")
EXCHANGE = os.getenv("EXCHANGE", "kraken").lower()   # kraken/binance/coinbase/...
QUOTE    = os.getenv("QUOTE", "USD").upper()
TOP_N    = int(os.getenv("TOP_N", "20"))
CACHE_TTL_SEC = int(os.getenv("CACHE_TTL_SEC", "1800"))

# Curated list (ordered) – will be tried first, in this exact order:
CURATED_BASES = [b.strip().upper() for b in os.getenv(
    "CURATED_BASES",
    "BTC,XRP,ETH,SOL,ADA,HYPE,USDT,BNB,USDC,DOGE,STETH,TRX,LINK,XLM,WBTC,SUI,AVAX,BCH,LTC,CRO,TON,USDS,DOT,XMR,MNT,UNI"
).split(",") if b.strip()]

# When a base doesn’t have QUOTE on this exchange, we’ll try these quotes in order:
QUOTE_FALLBACKS = [q.strip().upper() for q in os.getenv(
    "QUOTE_FALLBACKS",
    f"{QUOTE},USD,USDT,USDC,EUR,USDE,HBAR,WETH"
).split(",") if q.strip()]


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
    
# ---- Helpers ----
def get_exchange():
    klass = getattr(ccxt(), EXCHANGE)
    return klass({"enableRateLimit": True, "timeout": 20000})

def first_supported_symbol(markets: dict, base: str, quote_priority: list[str]) -> Optional[str]:
    """
    Return first market symbol like 'BTC/USD' that exists in markets for the given base,
    walking the quote_priority list. Uses CCXT unified symbols.
    """
    for q in quote_priority:
        sym = f"{base}/{q}"
        if sym in markets and markets[sym].get("spot", True) and not markets[sym].get("contract"):
            return sym
    return None
    
def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def recent_extrema(series: pd.Series, lookback: int = 10) -> tuple[float, float]:
    # exclude the current bar for structure
    window = series.iloc[-(lookback+1):-1]
    return float(window.min()), float(window.max())

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
    df["atr14"] = atr(df, 14)
    df["atr14_pct"] = df["atr14"] / (df["close"].abs() + 1e-12)
    return df


def curated_universe(ex, markets, bases: list[str], quote_priority: list[str]) -> list[dict]:
    out = []
    for b in bases:
        sym = first_supported_symbol(markets, b, quote_priority)
        if not sym:
            continue  # skip coins Kraken doesn't list in the preferred quotes
        m = markets[sym]
        out.append({
            "symbol": sym,
            "name": m.get("base") or b,
            "market": EXCHANGE,
            "tf_supported": ["1h", "1d"]
        })
    return out[:TOP_N]

def tier_b_universe(ex, markets, limit: int, quote_priority: list[str]) -> list[dict]:
    """
    Fill from exchange catalogue using preferred quotes.
    We prefer symbols whose quote is earlier in quote_priority and sort by our curated bases rank.
    """
    # Rank bases: curated bases first (keep their relative order), others after alphabetically
    rank = {b: i for i, b in enumerate(CURATED_BASES)}
    cand = []
    for sym, m in markets.items():
        if m.get("contract") or not m.get("spot", True):
            continue
        q = m.get("quote")
        b = m.get("base")
        if not b or not q: 
            continue
        if q.upper() not in {x.upper() for x in quote_priority}:
            continue
        cand.append({
            "symbol": sym,
            "base": b.upper(),
            "quote": q.upper(),
            "name": b,
            "market": EXCHANGE,
            "tf_supported": ["1h","1d"],
            "qrank": quote_priority.index(q.upper()) if q.upper() in quote_priority else 999,
            "brank": rank.get(b.upper(), 10_000)  # curated first
        })
    # Sort: curated-order first (brank), then quote priority (qrank), then symbol for stability
    cand.sort(key=lambda r: (r["brank"], r["qrank"], r["symbol"]))
    return [{k: r[k] for k in ("symbol","name","market","tf_supported")} for r in cand[:limit]]


# ---- Security ----
def require_key(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    token = authorization.split(" ", 1)[1].strip()
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid token")
        
# ---- Interface ----
def build_trade(df: pd.DataFrame, direction: str, risk_pct: float = 1.0, equity: Optional[float] = None, leverage: float = 1.0) -> Optional[Dict[str, Any]]:
    if direction not in ("Long", "Short"):
        return None
    last = df.iloc[-1]
    px   = float(last["close"])
    a    = float(last["atr14"])
    if not np.isfinite(a) or a <= 0:  # safety
        return None

    # structure levels
    recent_low, recent_high = recent_extrema(df["low"], 10)[0], recent_extrema(df["high"], 10)[1]
    if direction == "Long":
        base_sl   = px - 1.5*a
        struct_sl = recent_low - 0.2*a
        stop      = min(base_sl, struct_sl)
        R         = px - stop
        tps       = [px + k*R for k in (1, 2, 3)]
    else:
        base_sl   = px + 1.5*a
        struct_sl = recent_high + 0.2*a
        stop      = max(base_sl, struct_sl)
        R         = stop - px
        tps       = [px - k*R for k in (1, 2, 3)]

    if R <= 0 or not np.isfinite(R):
        return None

    trade = {
        "direction": direction,
        "entry": px,
        "stop": float(stop),
        "targets": [float(t) for t in tps],
        "rr": [1.0, 2.0, 3.0],               # TP multiples
        "volatility": {"atr": a, "atr_pct": float(last["atr14_pct"])},
        "risk_model": {"suggested_risk_pct": risk_pct, "leverage": leverage},
    }

    # Optional position sizing if equity provided (risk-based sizing)
    if equity and equity > 0:
        risk_amt = float(equity) * (risk_pct/100.0)
        risk_per_unit = R  # quoted in quote currency per 1 base
        qty = max(risk_amt / (risk_per_unit + 1e-12), 0.0)
        notional = qty * px / max(leverage, 1.0)
        trade["position_size"] = {
            "equity": float(equity),
            "risk_amount": risk_amt,
            "qty": float(qty),            # base units
            "notional": float(notional),  # quote currency
        }
    return trade


# ---- App ----
app = FastAPI(title="AI Trade Advisor", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
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
    # debug
    "universe_debug": {
        "path": None,            # "tierA" | "tierB" | "fallback"
        "supported_count": 0,
        "last_error": None,
        "exchange": EXCHANGE,
        "quote": QUOTE,
        "top_n": TOP_N,
    }
}

# ---------------- Features & simple rule ----------------
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

# ---------------- Data sources ----------------
def get_exchange():
    klass = getattr(ccxt(), EXCHANGE)
    return klass({"enableRateLimit": True, "timeout": 20000})

def coingecko_top_bases(n: int) -> List[Dict[str, str]]:
    """Return [{'base': 'BTC', 'name': 'Bitcoin'}, ...]"""
    sess = requests().Session()
    sess.headers.update({"User-Agent": "AI-Trade-Advisor/1.1 (contact: admin@yourdomain)"})
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = dict(vs_currency="usd", order="market_cap_desc", per_page=n, page=1)
    last_err = None
    for _ in range(3):
        try:
            r = sess.get(url, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(2)
                continue
            r.raise_for_status()
            coins = r.json()
            return [{"base": (c.get("symbol") or "").upper(), "name": c.get("name") or ""} for c in coins]
        except Exception as e:
            last_err = str(e)
            time.sleep(1)
    raise HTTPException(status_code=502, detail=f"CoinGecko error: {last_err}")

def tier_a_universe(ex, markets, top_bases: List[Dict[str,str]]) -> List[Dict[str, Any]]:
    wanted = [f"{t['base']}/{QUOTE}" for t in top_bases]
    out = []
    for pair in wanted:
        m = markets.get(pair)
        if not m: 
            continue
        if m.get("contract") or not m.get("spot", True):
            continue
        base = m.get("base") or pair.split("/")[0]
        name = next((t["name"] for t in top_bases if t["base"] == base), base)
        out.append({"symbol": pair, "name": name, "market": EXCHANGE, "tf_supported": ["1h","1d"]})
    return out

def tier_b_universe(ex, markets, limit: int) -> List[Dict[str, Any]]:
    """No CoinGecko: take the first N USDT spot symbols from the exchange catalogue."""
    candidates = []
    for sym, m in markets.items():
        if m.get("contract") or not m.get("spot", True):
            continue
        if m.get("quote") == QUOTE:
            name = m.get("base") or sym.split("/")[0]
            candidates.append({"symbol": sym, "name": name, "market": EXCHANGE, "tf_supported": ["1h","1d"]})
    # Keep active first; then sort by symbol for determinism
    candidates.sort(key=lambda d: (markets[d["symbol"]].get("active") is not False, d["symbol"]), reverse=True)
    return candidates[:limit]

def get_universe() -> List[Dict[str, Any]]:
    now = time.time()
    if STATE["universe"] and now - STATE["universe_fetched_at"] < CACHE_TTL_SEC:
        return STATE["universe"]

    dbg = STATE.get("universe_debug", {})
    dbg.update({"path": None, "supported_count": 0, "last_error": None, "exchange": EXCHANGE, "quote": QUOTE, "top_n": TOP_N})
    STATE["universe_debug"] = dbg

    try:
        ex = get_exchange()
        markets = ex.load_markets()

        # ---------- Tier A: CURATED (exact order you supplied) ----------
        curated = curated_universe(ex, markets, CURATED_BASES, QUOTE_FALLBACKS)
        if curated:
            dbg["path"] = "curated"
            dbg["supported_count"] = len(curated)
            STATE["universe"] = curated
            STATE["universe_fetched_at"] = now
            return curated

        # ---------- Tier B: Fill from exchange catalogue using preferred quotes ----------
        tier_b = tier_b_universe(ex, markets, TOP_N, QUOTE_FALLBACKS)
        if tier_b:
            dbg["path"] = "tierB_exchange"
            dbg["supported_count"] = len(tier_b)
            STATE["universe"] = tier_b
            STATE["universe_fetched_at"] = now
            return tier_b

        # ---------- Fallback trio ----------
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
        time.sleep(ex.rateLimit/1000)
        if len(rows) > 100_000: break

    if not rows:
        raise HTTPException(status_code=502, detail=f"No OHLCV for {symbol} {timeframe}")

    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
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
        "top_n": TOP_N,
        "universe_debug": STATE["universe_debug"],  # <- see path/size/last_error
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

from fastapi import Query

@app.get("/signals", response_model=Signal)
def signals(
    symbol: str,
    tf: str = "1h",
    risk_pct: float = Query(1.0, ge=0.1, le=5.0),
    equity: Optional[float] = Query(None, ge=0.0),
    leverage: float = Query(1.0, ge=1.0, le=10.0),
    _: None = Depends(require_key)
):
    key = (symbol, tf)
    now = time.time()
    cached = STATE["signal_cache"].get(key)
    if cached and now - cached["at"] < 300:  # 5m cache
        payload = cached["payload"]
        # enrich on the fly with requested risk params (doesn't break cache)
        if "trade" not in payload or payload["trade"] is None:
            pass  # rebuild below
        else:
            # if user passed equity/risk, recompute position sizing
            if equity is not None:
                tr = payload["trade"]
                # rebuild position size only, preserving levels
                entry, stop = tr["entry"], tr["stop"]
                R = abs(entry - stop)
                risk_amt = float(equity) * (risk_pct/100.0)
                qty = max(risk_amt / (R + 1e-12), 0.0)
                notional = qty * entry / max(leverage, 1.0)
                tr["risk_model"]["suggested_risk_pct"] = risk_pct
                tr["risk_model"]["leverage"] = leverage
                tr["position_size"] = {"equity": float(equity), "risk_amount": risk_amt, "qty": float(qty), "notional": float(notional)}
            return payload

    df = fetch_ohlcv(symbol, tf)
    df = compute_features(df).dropna()
    if len(df) < 60:
        raise HTTPException(status_code=502, detail="Not enough data to compute features")

    pred = rule_inference(df)
    feats = df.iloc[-1]

    # direction from score/label
    direction = "Long" if pred["signal"] == "Bullish" else "Short" if pred["signal"] == "Bearish" else "Neutral"
    trade = build_trade(df, direction, risk_pct=risk_pct, equity=equity, leverage=leverage)

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
            "atr14": float(feats["atr14"]),
            "atr14_pct": float(feats["atr14_pct"]),
        },
        "trade": trade
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

from fastapi import Query

@app.get("/chart")
def chart(
    symbol: str,
    tf: str = "1h",
    n: int = Query(120, ge=20, le=500),
    _: None = Depends(require_key)
):
    """
    Compact sparkline data for a symbol/timeframe.
    Returns last `n` closes + timestamps and basic stats.
    """
    df = fetch_ohlcv(symbol, tf)
    if df.empty:
        raise HTTPException(status_code=502, detail="No OHLCV")
    tail = df.tail(n).copy()
    closes = tail["close"].astype(float).tolist()
    ts = tail["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
    lo, hi = float(min(closes)), float(max(closes))
    # avoid flatline zero range
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
