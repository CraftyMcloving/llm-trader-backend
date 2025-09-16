from fastapi import FastAPI, HTTPException, Header, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Tuple
import os, time, threading
import pandas as pd
import numpy as np
import ccxt
import requests

# ------------------------------- Config -------------------------------

SECRET = os.getenv("AI_TRADE_SECRET", "XxUjb7DilVuqcnmeLXmUCURndUzC4Vmf")
DEFAULT_TF = "4h"

ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")  # required for FX
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT_SECONDS", "12"))   # keep small
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "150"))  # ~2.5 minutes
BATCH_LIMIT_SUCCESS = int(os.getenv("BATCH_LIMIT_SUCCESS", "6"))
CRYPTO_LIMIT = int(os.getenv("CRYPTO_CANDLES_LIMIT", "300"))  # enough for EMA200 on 4h

# Force single worker (set this in Render start cmd env, too)
# uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1

# ------------------------------ FastAPI -------------------------------

app = FastAPI(title="AI Trade Advisor Backend (lean)")

class AnalyzeRequest(BaseModel):
    symbol: str
    asset_type: str  # "crypto" or "fx"
    timeframe: str = DEFAULT_TF
    equity: float = 10000.0
    risk_pct: float = 0.8

class AnalyzeBatchRequest(BaseModel):
    items: List[AnalyzeRequest]

# ----------------------------- Utilities ------------------------------

def _downcast_df(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast to float32 to save memory."""
    for c in ("open","high","low","close","volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce", downcast="float")
            df[c] = df[c].astype("float32")
    return df

class TTLCache:
    def __init__(self, ttl_seconds: int):
        self.ttl = ttl_seconds
        self._d: Dict[Tuple[str, str, str], Tuple[float, pd.DataFrame]] = {}
        self._lock = threading.Lock()

    def get(self, key: Tuple[str, str, str]) -> Optional[pd.DataFrame]:
        now = time.time()
        with self._lock:
            it = self._d.get(key)
            if not it:
                return None
            ts, df = it
            if now - ts <= self.ttl:
                return df.copy()
            self._d.pop(key, None)
        return None

    def set(self, key: Tuple[str, str, str], df: pd.DataFrame):
        with self._lock:
            self._d[key] = (time.time(), df.copy())

CACHE = TTLCache(CACHE_TTL_SECONDS)

# Reuse a single CCXT client to avoid overhead/memory churn
_EXCHANGE = None
def get_exchange():
    global _EXCHANGE
    if _EXCHANGE is None:
        _EXCHANGE = ccxt.binance({"enableRateLimit": True, "timeout": HTTP_TIMEOUT * 1000})
    return _EXCHANGE

# ------------------------- Auth / Health / List -----------------------

def require_auth(auth: Optional[str]):
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing auth")
    token = auth.split(" ",1)[1]
    if token != SECRET:
        raise HTTPException(status_code=403, detail="Invalid token")

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

# ----------------------------- Data fetchers --------------------------

def fetch_ohlcv_crypto(symbol: str, timeframe: str) -> pd.DataFrame:
    key = ("crypto", symbol, timeframe)
    cached = CACHE.get(key)
    if cached is not None:
        return cached

    ex = get_exchange()
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=CRYPTO_LIMIT)
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"]).set_index("ts")
    df.index = pd.to_datetime(df.index, unit="ms", utc=True)
    df = _downcast_df(df)
    CACHE.set(key, df)
    return df

def fetch_ohlcv_fx(pair: str, timeframe: str) -> pd.DataFrame:
    if not ALPHA_VANTAGE_KEY:
        raise HTTPException(500, "ALPHA_VANTAGE_KEY not set")
    key = ("fx", pair, timeframe)
    cached = CACHE.get(key)
    if cached is not None:
        return cached

    base, quote = pair.split("/")
    try:
        if timeframe in ("1h", "4h"):
            # 60-min then resample → 4H
            url = (
                "https://www.alphavantage.co/query"
                f"?function=FX_INTRADAY&from_symbol={base}&to_symbol={quote}"
                f"&interval=60min&outputsize=full&apikey={ALPHA_VANTAGE_KEY}"
            )
            data = requests.get(url, timeout=HTTP_TIMEOUT).json()
            key_ts = "Time Series FX (60min)"
        else:
            url = (
                "https://www.alphavantage.co/query"
                f"?function=FX_DAILY&from_symbol={base}&to_symbol={quote}"
                f"&outputsize=compact&apikey={ALPHA_VANTAGE_KEY}"
            )
            data = requests.get(url, timeout=HTTP_TIMEOUT).json()
            key_ts = "Time Series FX (Daily)"

        ts = data.get(key_ts, {})
        if not ts:
            detail = data.get("Note") or data.get("Error Message") or "FX data unavailable"
            raise HTTPException(502, f"{detail}")

        df = (
            pd.DataFrame.from_dict(ts, orient="index")
            .rename(columns={"1. open":"open","2. high":"high","3. low":"low","4. close":"close"})
        )
        df.index = pd.to_datetime(df.index, utc=True)
        for c in ["open","high","low","close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["volume"] = 0.0
        df = df.sort_index()

        if timeframe == "4h":
            df = (
                df.resample("4H")
                  .agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
                  .dropna()
            )

        df = df.tail(1000)
        df = _downcast_df(df)
        CACHE.set(key, df)
        return df
    except requests.Timeout:
        raise HTTPException(504, f"FX provider timeout for {pair}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"FX fetch error for {pair}: {e}")

# ---------------------------- Indicators (lean) -----------------------

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()

def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/length, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.astype("float32")

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line.astype("float32"), signal_line.astype("float32"), hist.astype("float32")

def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=length).mean().astype("float32")

def bb_mid(series: pd.Series, length: int = 20) -> pd.Series:
    return sma(series, length).astype("float32")

# --------------------------- Signal engine ----------------------------

def compute_signals(df: pd.DataFrame) -> Dict:
    df = df.copy()

    df["ema20"]  = ema(df["close"], 20)
    df["ema50"]  = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200)
    _, _, macd_hist = macd(df["close"], 12, 26, 9)
    df["macd_hist"] = macd_hist
    df["rsi"]    = rsi(df["close"], 14)
    df["bb_mid"] = bb_mid(df["close"], 20)
    df["atr"]    = atr(df["high"], df["low"], df["close"], 14)

    ready_cols = ["ema20","ema50","ema200","macd_hist","rsi","bb_mid","atr","close"]
    df_ready = df.dropna(subset=ready_cols)
    if df_ready.shape[0] < 5:
        raise HTTPException(503, "Indicators not ready; need more history")

    latest = df_ready.iloc[-1]

    # Trend filter
    trend_long  = bool(latest.close > latest.ema200 and latest.ema50 > latest.ema200)
    trend_short = bool(latest.close < latest.ema200 and latest.ema50 < latest.ema200)

    # Momentum
    mom_long  = bool(latest.rsi > 55 and latest.macd_hist > 0 and latest.close > latest.bb_mid)
    mom_short = bool(latest.rsi < 45 and latest.macd_hist < 0 and latest.close < latest.bb_mid)

    # Pullback context
    pullback_long  = bool(latest.close > latest.ema20 > latest.ema50) if trend_long else False
    pullback_short = bool(latest.close < latest.ema20 < latest.ema50) if trend_short else False

    # Direction
    direction = "flat"
    if trend_long and mom_long:
        direction = "long"
    elif trend_short and mom_short:
        direction = "short"

    # Levels
    atr_v = float(latest.atr)
    entry = float(latest.close)
    if direction == "long":
        stop = entry - max(1.5*atr_v, entry*0.004)
        tp1  = entry + (entry - stop)
        tp2  = entry + 2*(entry - stop)
    elif direction == "short":
        stop = entry + max(1.5*atr_v, entry*0.004)
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

    atr_pct = (atr_v / entry * 100) if entry else 0
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
            "atr": round(float(atr_v), 5),
        },
    }

# ------------------------ Position sizing / API -----------------------

def position_size(entry: float, stop: float, equity: float, risk_pct: float) -> float:
    risk_amt = equity * (risk_pct/100.0)
    risk_per_unit = abs(entry - stop)
    if risk_per_unit <= 0:
        return 0.0
    return round(risk_amt / risk_per_unit, 6)

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
        df = fetch_ohlcv_fx(symbol.replace(" ","/"), tf)
    else:
        raise HTTPException(400, "asset_type must be crypto or fx")

    if df.shape[0] < 220:
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
def batch(
    req: AnalyzeBatchRequest,
    limit_success: Optional[int] = Query(None, description="Early stop once N successes found"),
    Authorization: Optional[str] = Header(None)
):
    require_auth(Authorization)
    max_success = limit_success or BATCH_LIMIT_SUCCESS

    results = []
    success_count = 0
    for item in req.items:
        if success_count >= max_success:
            break
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
            success_count += 1
        except HTTPException as e:
            results.append({"symbol": item.symbol, "error": e.detail})
        except Exception as e:
            results.append({"symbol": item.symbol, "error": str(e)})
    return {"results": results}
