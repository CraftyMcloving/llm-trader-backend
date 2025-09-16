from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import time, os
import pandas as pd
import pandas_ta as ta
import ccxt
import requests
from typing import List, Optional, Dict, Tuple

# =========================
# Config / constants
# =========================
SECRET = os.getenv("AI_TRADE_SECRET", "XxUjb7DilVuqcnmeLXmUCURndUzC4Vmf")
DEFAULT_TF = "4h"

# FX providers (set any subset; priority is OANDA -> AlphaVantage -> TwelveData)
OANDA_API_KEY     = os.getenv("OANDA_API_KEY")          # required for OANDA
OANDA_ENV         = (os.getenv("OANDA_ENV") or "practice").lower()  # "practice" | "live"
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")      # optional fallback
TWELVE_DATA_KEY   = os.getenv("TWELVE_DATA_KEY")        # optional fallback

# Caching (in-memory)
FX_CACHE_TTL_SEC     = int(os.getenv("FX_CACHE_TTL_SEC", "300"))   # 5 min
CRYPTO_CACHE_TTL_SEC = int(os.getenv("CRYPTO_CACHE_TTL_SEC", "60"))

# OANDA endpoints
OANDA_BASE = "https://api-fxtrade.oanda.com" if OANDA_ENV == "live" else "https://api-fxpractice.oanda.com"

app = FastAPI(title="AI Trade Advisor Backend")

# =========================
# Models
# =========================
class AnalyzeRequest(BaseModel):
    symbol: str
    asset_type: str  # "crypto" or "fx"
    timeframe: str = DEFAULT_TF
    equity: float = 10000.0
    risk_pct: float = 0.8

class AnalyzeBatchRequest(BaseModel):
    items: List[AnalyzeRequest]

# =========================
# Auth
# =========================
def require_auth(auth: Optional[str]):
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing auth")
    token = auth.split(" ", 1)[1]
    if token != SECRET:
        raise HTTPException(status_code=403, detail="Invalid token")

# =========================
# Utils
# =========================
def _now() -> float:
    return time.time()

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
    tfl = tf.lower()
    if tfl in ("4h", "4hr", "240m"):
        rule = "4H"
    elif tfl in ("1h", "1hr", "60m"):
        rule = "1H"
    elif tfl in ("1d", "d", "day"):
        rule = "1D"
    else:
        return df
    return (
        df.resample(rule)
          .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
          .dropna()
    )

def _http_get_json(url: str, timeout: int = 30, retries: int = 2, backoff_sec: float = 0.8, headers: Optional[Dict[str, str]] = None) -> dict:
    last_err = None
    for i in range(retries + 1):
        try:
            r = requests.get(url, timeout=timeout, headers=headers or {"User-Agent": "ai-trade-advisor/1.0"})
            # Prefer returning JSON even for 4xx to capture provider messages
            data = {}
            try:
                data = r.json()
            except Exception:
                pass
            if r.status_code < 400:
                return data
            # Bubble known rate limits
            if r.status_code == 429:
                msg = data.get("errorMessage") or data.get("message") or "Rate limited"
                raise HTTPException(429, msg)
            last_err = f"HTTP {r.status_code}: {data or r.text}"
        except HTTPException:
            raise
        except Exception as e:
            last_err = str(e)
        if i < retries:
            time.sleep(backoff_sec * (2 ** i))
    raise RuntimeError(f"GET failed: {last_err or 'unknown error'}")

# =========================
# Health & instruments
# =========================
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

# =========================
# Caches
# =========================
_CRYPTO_CACHE: Dict[Tuple[str, str], Tuple[float, pd.DataFrame]] = {}
_FX_CACHE: Dict[Tuple[str, str, str], Tuple[float, pd.DataFrame]] = {}  # (provider, pair, tf)

def _get_cache(cache: dict, key: tuple, ttl: int) -> Optional[pd.DataFrame]:
    hit = cache.get(key)
    if not hit:
        return None
    ts, df = hit
    if _now() - ts <= ttl:
        return df.copy()
    return None

def _set_cache(cache: dict, key: tuple, df: pd.DataFrame):
    cache[key] = (_now(), df.copy())

# =========================
# Data fetchers
# =========================
def fetch_ohlcv_crypto(symbol: str, timeframe: str) -> pd.DataFrame:
    ck = (symbol, timeframe)
    cached = _get_cache(_CRYPTO_CACHE, ck, CRYPTO_CACHE_TTL_SEC)
    if cached is not None:
        return cached
    ex = ccxt.binance({"enableRateLimit": True})
    tf_map = {"4h": "4h", "1h": "1h", "1d": "1d", "d": "1d"}
    tf = tf_map.get(timeframe.lower(), timeframe)
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=tf, limit=400)
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"]).set_index("ts")
    df.index = pd.to_datetime(df.index, unit="ms", utc=True)
    _set_cache(_CRYPTO_CACHE, ck, df)
    return df

def _fx_pair_to_oanda(pair: str) -> str:
    # "EUR/USD" -> "EUR_USD"
    return pair.replace("/", "_").upper()

def _oanda_fx(pair: str, timeframe: str) -> pd.DataFrame:
    if not OANDA_API_KEY:
        raise HTTPException(500, "OANDA_API_KEY not set")
    instrument = _fx_pair_to_oanda(pair)
    tf = timeframe.lower()
    gran_map = {"1h": "H1", "4h": "H4", "1d": "D", "d": "D"}
    gran = gran_map.get(tf, "H4")  # default H4
    url = (
        f"{OANDA_BASE}/v3/instruments/{instrument}/candles"
        f"?granularity={gran}&count=2000&price=M"  # mid prices
    )
    headers = {
        "Authorization": f"Bearer {OANDA_API_KEY}",
        "Accept": "application/json",
        "User-Agent": "ai-trade-advisor/1.0"
    }
    data = _http_get_json(url, headers=headers, timeout=30, retries=2)
    if "candles" not in data:
        msg = data.get("errorMessage") or data.get("message") or "OANDA returned no candles"
        raise HTTPException(502, msg)

    rows = []
    for c in data["candles"]:
        # skip incomplete last candle
        if not c.get("complete", False):
            continue
        t = pd.to_datetime(c["time"], utc=True)
        mid = c.get("mid") or {}
        try:
            o = float(mid["o"]); h = float(mid["h"]); l = float(mid["l"]); cl = float(mid["c"])
        except Exception:
            # fall back if price=B/A someday
            bid = c.get("bid") or {}
            ask = c.get("ask") or {}
            o = (float(bid.get("o", 0)) + float(ask.get("o", 0))) / 2.0
            h = (float(bid.get("h", 0)) + float(ask.get("h", 0))) / 2.0
            l = (float(bid.get("l", 0)) + float(ask.get("l", 0))) / 2.0
            cl = (float(bid.get("c", 0)) + float(ask.get("c", 0))) / 2.0
        vol = float(c.get("volume", 0))
        rows.append((t, o, h, l, cl, vol))
    if not rows:
        raise HTTPException(502, "OANDA returned empty candles (after filtering incomplete)")
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]).set_index("ts")
    df = df.sort_index()
    return df.tail(1000)

def _alpha_vantage_fx(pair: str, timeframe: str) -> pd.DataFrame:
    if not ALPHA_VANTAGE_KEY:
        raise HTTPException(500, "ALPHA_VANTAGE_KEY not set")
    base, quote = pair.split("/")
    if timeframe.lower() in ("1h", "4h"):
        interval = "60min"  # request 60m then resample to 4h if needed
        url = (
            "https://www.alphavantage.co/query"
            f"?function=FX_INTRADAY&from_symbol={base}&to_symbol={quote}"
            f"&interval={interval}&apikey={ALPHA_VANTAGE_KEY}&outputsize=compact"
        )
        data = _http_get_json(url, timeout=30, retries=2)
        if "Note" in data:
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
        data = _http_get_json(url, timeout=30, retries=2)
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
          .rename(columns={"1. open": "open", "2. high": "high", "3. low": "low", "4. close": "close"})
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
        raise HTTPException(500, "TWELVE_DATA_KEY not set")
    interval = "60min" if timeframe.lower() in ("1h", "4h") else "1day"
    url = (
        "https://api.twelvedata.com/time_series"
        f"?symbol={pair}&interval={interval}&outputsize=5000&apikey={TWELVE_DATA_KEY}&format=JSON"
    )
    data = _http_get_json(url, timeout=30, retries=2)
    if data.get("status") == "error":
        raise HTTPException(502, data.get("message", "Twelve Data error"))
    values = data.get("values")
    if not values:
        raise HTTPException(502, "Twelve Data returned no values")
    df = pd.DataFrame(values)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.set_index("datetime").sort_index()
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    else:
        df["volume"] = 0.0
    if timeframe.lower() == "4h" and interval == "60min":
        df = _resample(df, "4h")
    return df.tail(1000)

def fetch_ohlcv_fx(pair: str, timeframe: str) -> pd.DataFrame:
    """
    FX data with provider priority + cache + stale-if-error:
    1) OANDA (if configured)
    2) Alpha Vantage (if configured)
    3) Twelve Data (if configured)
    """
    # Serve fresh cache first (any provider)
    ck_oanda = ("oanda", pair, timeframe)
    ck_av    = ("alpha_vantage", pair, timeframe)
    ck_td    = ("twelve_data", pair, timeframe)
    cached = (
        _get_cache(_FX_CACHE, ck_oanda, FX_CACHE_TTL_SEC)
        or _get_cache(_FX_CACHE, ck_av, FX_CACHE_TTL_SEC)
        or _get_cache(_FX_CACHE, ck_td, FX_CACHE_TTL_SEC)
    )
    if cached is not None:
        return cached

    # OANDA first
    if OANDA_API_KEY:
        try:
            df = _oanda_fx(pair, timeframe)
            _set_cache(_FX_CACHE, ck_oanda, df)
            return df
        except HTTPException as e:
            # keep going to fallbacks; serve stale if any
            stale = _FX_CACHE.get(ck_oanda)
            if stale:
                return stale[1].copy()
            if e.status_code == 429:
                # hard rate limit; try next providers
                pass
            elif e.status_code >= 500:
                pass
            else:
                # For 4xx other than rate limit, still try fallbacks
                pass

    # Alpha Vantage fallback
    if ALPHA_VANTAGE_KEY:
        try:
            df = _alpha_vantage_fx(pair, timeframe)
            _set_cache(_FX_CACHE, ck_av, df)
            return df
        except HTTPException as e:
            stale = _FX_CACHE.get(ck_av)
            if stale:
                return stale[1].copy()
            if e.status_code != 429:
                pass  # proceed to next

    # Twelve Data fallback
    if TWELVE_DATA_KEY:
        try:
            df = _twelve_data_fx(pair, timeframe)
            _set_cache(_FX_CACHE, ck_td, df)
            return df
        except HTTPException as e:
            stale = _FX_CACHE.get(ck_td)
            if stale:
                return stale[1].copy()
            raise e

    raise HTTPException(502, "No FX provider configured or providers unavailable")

# =========================
# Signals
# =========================
def compute_signals(df: pd.DataFrame) -> Dict:
    df = df.copy()
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

    # Trend
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

# =========================
# Risk / sizing
# =========================
def position_size(entry: float, stop: float, equity: float, risk_pct: float) -> float:
    risk_amt = equity * (risk_pct / 100.0)
    risk_per_unit = abs(entry - stop)
    if risk_per_unit <= 0:
        return 0.0
    return round(risk_amt / risk_per_unit, 6)

# =========================
# Routes
# =========================
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
