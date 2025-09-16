from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import time, os, math
import pandas as pd
import pandas_ta as ta
import ccxt
import requests
from typing import List, Optional, Dict, Tuple

# ---------------------------
# Config / constants
# ---------------------------
SECRET = os.getenv("AI_TRADE_SECRET", "")
DEFAULT_TF = "4h"

# FX providers
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")  # primary
TWELVE_DATA_KEY   = os.getenv("TWELVE_DATA_KEY")    # optional fallback

# Caching (in-memory)
FX_CACHE_TTL_SEC = int(os.getenv("FX_CACHE_TTL_SEC", "300"))  # 5 minutes
CRYPTO_CACHE_TTL_SEC = int(os.getenv("CRYPTO_CACHE_TTL_SEC", "60"))  # 1 minute

app = FastAPI(title="AI Trade Advisor Backend")

# ---------------------------
# Models
# ---------------------------
class AnalyzeRequest(BaseModel):
    symbol: str
    asset_type: str  # "crypto" or "fx"
    timeframe: str = DEFAULT_TF
    equity: float = 10000.0
    risk_pct: float = 0.8

class AnalyzeBatchRequest(BaseModel):
    items: List[AnalyzeRequest]

# ---------------------------
# Auth
# ---------------------------
def require_auth(auth: Optional[str]):
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing auth")
    token = auth.split(" ", 1)[1]
    if token != SECRET:
        raise HTTPException(status_code=403, detail="Invalid token")

# ---------------------------
# Utilities
# ---------------------------
def _first_col_startswith(df: pd.DataFrame, prefix: str) -> Optional[str]:
    if isinstance(df, pd.DataFrame):
        for c in df.columns:
            if str(c).startswith(prefix):
                return c
    return None

def _bollinger_mid(close: pd.Series, length: int = 20, std: int = 2) -> pd.Series:
    bb = ta.bbands(close, length=length, std=std)
    if isinstance(bb, pd.DataFrame) and not bb.empty:
        mid = _first_col_startswith(bb, "BBM_") or ("BBM" if "BBM" in bb.columns else None)
        if mid and mid in bb:
            return bb[mid]
        up = _first_col_startswith(bb, "BBU_") or ("BBU" if "BBU" in bb.columns else None)
        lo = _first_col_startswith(bb, "BBL_") or ("BBL" if "BBL" in bb.columns else None)
        if (up in bb) and (lo in bb):
            return (bb[up] + bb[lo]) / 2.0
    return ta.sma(close, length=length)

def _macd_hist(close: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
    macd = ta.macd(close, fast=fast, slow=slow, signal=signal)
    if isinstance(macd, pd.DataFrame) and not macd.empty:
        hist_col = _first_col_startswith(macd, "MACDh_") or _first_col_startswith(macd, "MACD_Hist")
        if hist_col and hist_col in macd:
            return macd[hist_col]
        macd_line = _first_col_startswith(macd, "MACD_")
        sig_line  = _first_col_startswith(macd, "MACDs_") or _first_col_startswith(macd, "MACD_Signal")
        if (macd_line in macd) and (sig_line in macd):
            return macd[macd_line] - macd[sig_line]
    return pd.Series(index=close.index, dtype=float)

def _resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    if tf.lower() in ("4h", "4hr", "240m"):
        rule = "4H"
    elif tf.lower() in ("1h", "1hr", "60m"):
        rule = "1H"
    elif tf.lower() in ("1d", "d", "day"):
        rule = "1D"
    else:
        return df  # return raw if unknown; callers supply correct interval upstream
    return (
        df.resample(rule)
          .agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
          .dropna()
    )

def _now() -> float:
    return time.time()

def _http_get_json(url: str, timeout: int = 30, retries: int = 2, backoff_sec: float = 0.8) -> dict:
    last_err = None
    for i in range(retries + 1):
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent":"ai-trade-advisor/1.0"})
            if r.status_code >= 500:
                last_err = f"HTTP {r.status_code}"
            else:
                return r.json()
        except Exception as e:
            last_err = str(e)
        if i < retries:
            time.sleep(backoff_sec * (2 ** i))
    raise RuntimeError(f"GET failed: {last_err or 'unknown error'}")

# ---------------------------
# Health / instruments
# ---------------------------
@app.get("/health")
def health():
    return {"ok": True, "ts": int(time.time())}

@app.get("/instruments")
def instruments():
    return {
        "crypto": [
            "BTC/USDT","ETH/USDT","XRP/USDT","SOL/USDT","BNB/USDT",
            "ADA/USDT","DOGE/USDT","AVAX/USDT","DOT/USDT","MATIC/USDT",
            "LTC/USDT","TRX/USDT","LINK/USDT","ATOM/USDT","XLM/USDT",
            "NEAR/USDT","FIL/USDT","APT/USDT","AAVE/USDT","ETC/USDT",
            "ICP/USDT","HBAR/USDT","ARB/USDT","OP/USDT","SUI/USDT"
        ],
        "fx": [
            "EUR/USD","GBP/USD","USD/JPY","USD/CHF","AUD/USD",
            "USD/CAD","NZD/USD","EUR/GBP","EUR/JPY","GBP/JPY",
            "EUR/AUD","AUD/JPY","CHF/JPY","EUR/CHF","GBP/CHF",
            "AUD/NZD","CAD/JPY","EUR/CAD","GBP/CAD","NZD/JPY",
            "USD/SEK","USD/NOK","USD/MXN","USD/TRY","USD/ZAR"
        ]
    }

# ---------------------------
# Caches
# ---------------------------
_CRYPTO_CACHE: Dict[Tuple[str,str], Tuple[float, pd.DataFrame]] = {}
_FX_CACHE: Dict[Tuple[str,str,str], Tuple[float, pd.DataFrame]] = {}
# keys: (provider, pair, timeframe)

def _get_cache(cache: dict, key: tuple, ttl: int) -> Optional[pd.DataFrame]:
    hit = cache.get(key)
    if not hit: return None
    ts, df = hit
    if _now() - ts <= ttl:
        return df.copy()
    return None

def _set_cache(cache: dict, key: tuple, df: pd.DataFrame):
    cache[key] = (_now(), df.copy())

# ---------------------------
# Data fetchers
# ---------------------------
def fetch_ohlcv_crypto(symbol: str, timeframe: str) -> pd.DataFrame:
    cache_key = (symbol, timeframe)
    cached = _get_cache(_CRYPTO_CACHE, cache_key, CRYPTO_CACHE_TTL_SEC)
    if cached is not None:
        return cached

    ex = ccxt.binance({"enableRateLimit": True})
    # Map common tf strings
    tf_map = {"4h":"4h","1h":"1h","1d":"1d","d":"1d"}
    tf = tf_map.get(timeframe.lower(), timeframe)
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=tf, limit=400)
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"]).set_index("ts")
    df.index = pd.to_datetime(df.index, unit="ms", utc=True)
    _set_cache(_CRYPTO_CACHE, cache_key, df)
    return df

def _alpha_vantage_fx(pair: str, timeframe: str) -> pd.DataFrame:
    if not ALPHA_VANTAGE_KEY:
        raise HTTPException(500, "ALPHA_VANTAGE_KEY not set")
    base, quote = pair.split("/")
    if timeframe.lower() in ("1h","4h"):
        interval = "60min"  # request 60min and resample for 4h
        url = (
            "https://www.alphavantage.co/query"
            f"?function=FX_INTRADAY&from_symbol={base}&to_symbol={quote}"
            f"&interval={interval}&apikey={ALPHA_VANTAGE_KEY}&outputsize=compact"
        )
        data = _http_get_json(url)
        if "Note" in data:
            # Rate limited
            raise HTTPException(429, data["Note"])
        if "Error Message" in data:
            raise HTTPException(502, data["Error Message"])
        key = f"Time Series FX ({interval})"
    else:
        url = (
            "https://www.alphavantage.co/query"
            f"?function=FX_DAILY&from_symbol={base}&to_symbol={quote}"
            f"&apikey={ALPHA_VANTAGE_KEY}&outputsize=compact"
        )
        data = _http_get_json(url)
        if "Note" in data:
            raise HTTPException(429, data["Note"])
        if "Error Message" in data:
            raise HTTPException(502, data["Error Message"])
        key = "Time Series FX (Daily)"

    ts = data.get(key, {})
    if not ts:
        raise HTTPException(502, f"Alpha Vantage returned no '{key}' data for {pair}")

    df = (
        pd.DataFrame.from_dict(ts, orient="index")
          .rename(columns={"1. open":"open","2. high":"high","3. low":"low","4. close":"close"})
          .astype(float)
    )
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    df["volume"] = 0.0
    if timeframe.lower() == "4h":
        df = _resample(df, "4h")
    return df.tail(1000)

def _twelve_data_fx(pair: str, timeframe: str) -> pd.DataFrame:
    if not TWELVE_DATA_KEY:
        raise HTTPException(502, "Twelve Data fallback not configured")
    # Twelve Data symbols look like "EUR/USD"
    interval = "60min" if timeframe.lower() in ("1h","4h") else "1day"
    # Big outputsize for intraday; TD caps free tier but still ok
    url = (
        "https://api.twelvedata.com/time_series"
        f"?symbol={pair}&interval={interval}&outputsize=5000&apikey={TWELVE_DATA_KEY}&format=JSON"
    )
    data = _http_get_json(url)
    if "status" in data and data["status"] == "error":
        raise HTTPException(502, f"Twelve Data error: {data.get('message','unknown')}")
    values = data.get("values")
    if not values:
        raise HTTPException(502, "Twelve Data returned no values")

    # values is list of dicts newest-first; normalize to OHLC
    df = pd.DataFrame(values)
    # Expected keys: datetime, open, high, low, close, volume (volume may be None for FX)
    for col in ["open","high","low","close"]:
        df[col] = df[col].astype(float)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()
    if "volume" not in df.columns:
        df["volume"] = 0.0
    else:
        # Some FX volumes are empty strings
        with pd.option_context("mode.use_inf_as_na", True):
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)

    if timeframe.lower() == "4h" and interval == "60min":
        df = _resample(df, "4h")
    return df.tail(1000)

def fetch_ohlcv_fx(pair: str, timeframe: str) -> pd.DataFrame:
    """
    FX fetcher with cache + fallback:
    1) Alpha Vantage (if configured)
    2) Twelve Data (if configured)
    Serve cached data when provider is rate-limited.
    """
    # Cache key includes provider so we can hold both
    cache_key_av = ("alpha_vantage", pair, timeframe)
    cache_key_td = ("twelve_data", pair, timeframe)

    # Serve fresh cache first (any provider)
    cached = _get_cache(_FX_CACHE, cache_key_av, FX_CACHE_TTL_SEC) or _get_cache(_FX_CACHE, cache_key_td, FX_CACHE_TTL_SEC)
    if cached is not None:
        return cached

    # Try Alpha Vantage
    if ALPHA_VANTAGE_KEY:
        try:
            df = _alpha_vantage_fx(pair, timeframe)
            _set_cache(_FX_CACHE, cache_key_av, df)
            return df
        except HTTPException as e:
            # If rate-limited, try fallback (if available) or serve stale-if-present
            if e.status_code == 429 and TWELVE_DATA_KEY:
                try:
                    df = _twelve_data_fx(pair, timeframe)
                    _set_cache(_FX_CACHE, cache_key_td, df)
                    return df
                except HTTPException:
                    pass
            # Stale-if-error: serve any stale cache if present
            stale_av = _FX_CACHE.get(cache_key_av)
            if stale_av:
                return stale_av[1].copy()
            stale_td = _FX_CACHE.get(cache_key_td)
            if stale_td:
                return stale_td[1].copy()
            raise  # no cache to fall back to: bubble up

    # No AV or it failed, try Twelve Data if configured
    if TWELVE_DATA_KEY:
        try:
            df = _twelve_data_fx(pair, timeframe)
            _set_cache(_FX_CACHE, cache_key_td, df)
            return df
        except HTTPException:
            stale_td = _FX_CACHE.get(cache_key_td)
            if stale_td:
                return stale_td[1].copy()
            raise

    # No provider available
    raise HTTPException(502, "No FX provider configured or providers unavailable")

# ---------------------------
# Signal engine
# ---------------------------
def compute_signals(df: pd.DataFrame) -> Dict:
    df = df.copy()
    # Core indicators
    df["ema20"]  = ta.ema(df["close"], length=20)
    df["ema50"]  = ta.ema(df["close"], length=50)
    df["ema200"] = ta.ema(df["close"], length=200)
    df["macd_hist"] = _macd_hist(df["close"], fast=12, slow=26, signal=9)
    df["rsi"]    = ta.rsi(df["close"], length=14)
    df["bb_mid"] = _bollinger_mid(df["close"], length=20, std=2)
    df["atr"]    = ta.atr(df["high"], df["low"], df["close"], length=14)

    ready_cols = ["ema20","ema50","ema200","macd_hist","rsi","bb_mid","atr","close"]
    df_ready = df.dropna(subset=ready_cols)
    if df_ready.empty:
        raise HTTPException(503, "Indicators not ready; need more history")

    latest = df_ready.iloc[-1]

    # Trend filter
    trend_long  = latest.close > latest.ema200 and latest.ema50 > latest.ema200
    trend_short = latest.close < latest.ema200 and latest.ema50 < latest.ema200

    # Momentum
    mom_long  = (latest.rsi > 55) and (latest.macd_hist > 0) and (latest.close > latest.bb_mid)
    mom_short = (latest.rsi < 45) and (latest.macd_hist < 0) and (latest.close < latest.bb_mid)

    # Pullback context
    pullback_long  = latest.close > latest.ema20 > latest.ema50 if trend_long else False
    pullback_short = latest.close < latest.ema20 < latest.ema50 if trend_short else False

    # Direction
    direction = "flat"
    if trend_long and mom_long:
        direction = "long"
    elif trend_short and mom_short:
        direction = "short"

    # Levels
    atr   = float(latest.atr)
    entry = float(latest.close)
    if direction == "long":
        stop = entry - max(1.5*atr, entry*0.004)
        tp1  = entry + (entry - stop)
        tp2  = entry + 2*(entry - stop)
    elif direction == "short":
        stop = entry + max(1.5*atr, entry*0.004)
        tp1  = entry - (stop - entry)
        tp2  = entry - 2*(stop - entry)
    else:
        stop = tp1 = tp2 = entry

    # Confidence
    conf = 0
    if trend_long or trend_short: conf += 20
    if mom_long or mom_short:     conf += 20
    if pullback_long or pullback_short: conf += 15

    rr = 0.0
    if direction != "flat" and entry != stop:
        rr = abs((tp2 - entry) / (entry - stop))
        if rr >= 2.0:
            conf += 15

    atr_pct = (atr / entry * 100) if entry else 0
    if 0.5 <= atr_pct <= 5:
        conf += 10

    # Rationale
    rationale = []
    if direction == "long":
        rationale += [
            "Price above rising 200 EMA" if trend_long else "200 EMA trend not confirmed",
            "RSI>55 & MACD>0" if mom_long else "Momentum mixed",
            "Pullback structure intact" if pullback_long else "No ideal pullback",
        ]
    elif direction == "short":
        rationale += [
            "Price below falling 200 EMA" if trend_short else "200 EMA trend not confirmed",
            "RSI<45 & MACD<0" if mom_short else "Momentum mixed",
            "Pullback structure intact" if pullback_short else "No ideal pullback",
        ]
    else:
        rationale.append("No aligned signals; staying flat")

    return {
        "direction": direction,
        "entry": round(entry, 5),
        "stop": round(stop, 5),
        "take_profits": [round(tp1,5), round(tp2,5)],
        "risk_reward": round(rr, 2),
        "confidence": conf,
        "rationale": rationale,
        "indicators": {
            "ema200": round(float(latest.ema200), 5),
            "rsi": round(float(latest.rsi), 2),
            "macd_hist": round(float(latest.macd_hist), 4),
            "atr": round(float(atr), 5),
        },
    }

# ---------------------------
# Risk / sizing
# ---------------------------
def position_size(entry: float, stop: float, equity: float, risk_pct: float) -> float:
    risk_amt = equity * (risk_pct/100.0)
    risk_per_unit = abs(entry - stop)
    if risk_per_unit <= 0:
        return 0.0
    return round(risk_amt / risk_per_unit, 6)

# ---------------------------
# API routes
# ---------------------------
@app.get("/analyze")
def analyze(
    symbol: str,
    asset_type: str,
    tf: str = DEFAULT_TF,
    equity: float = 10000.0,
    risk_pct: float = 0.8,
    Authorization: Optional[str] = Header(None)
):
    require_auth(Authorization)

    if asset_type == "crypto":
        df = fetch_ohlcv_crypto(symbol, tf)
    elif asset_type == "fx":
        df = fetch_ohlcv_fx(symbol.replace(" ", "/"), tf)
    else:
        raise HTTPException(400, "asset_type must be crypto or fx")

    # Need enough history for EMA200 etc.
    if len(df.dropna()) < 150:
        raise HTTPException(503, "Not enough data for indicators")

    sig = compute_signals(df)
    size_units = position_size(sig["entry"], sig["stop"], equity, risk_pct)
    return {
        "symbol": symbol,
        "asset_type": asset_type,
        "timeframe": tf,
        "updated_at": pd.Timestamp.utcnow().isoformat(),
        **sig,
        "suggested_risk_pct": risk_pct,
        "position_size_units": size_units,
        "risk": {"account_equity": equity, "risk_amount": equity*(risk_pct/100.0)}
    }

@app.post("/batch")
def batch(req: AnalyzeBatchRequest, Authorization: Optional[str] = Header(None)):
    require_auth(Authorization)
    results = []
    for item in req.items:
        try:
            r = analyze(
                item.symbol,
                item.asset_type,
                item.timeframe,
                item.equity,
                item.risk_pct,
                Authorization=f"Bearer {SECRET}"
            )
            results.append(r)
        except HTTPException as e:
            results.append({"symbol": item.symbol, "status": e.status_code, "error": e.detail})
        except Exception as e:
            results.append({"symbol": item.symbol, "status": 500, "error": str(e)})
    return {"results": results}
