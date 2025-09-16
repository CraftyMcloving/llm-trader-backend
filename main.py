from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import time, os
import pandas as pd
import pandas_ta as ta
import ccxt
import requests
from typing import List, Optional, Dict
# --- helpers for resilient indicator access ---
from typing import Optional

def _first_col_startswith(df: pd.DataFrame, prefix: str) -> Optional[str]:
    if isinstance(df, pd.DataFrame):
        for c in df.columns:
            if str(c).startswith(prefix):
                return c
    return None

def _bollinger_mid(close: pd.Series, length: int = 20, std: int = 2) -> pd.Series:
    """Return BB midline robustly; fall back to SMA(length) if not present yet."""
    bb = ta.bbands(close, length=length, std=std)
    if isinstance(bb, pd.DataFrame) and not bb.empty:
        # pandas-ta usually names midline like 'BBM_20_2.0' or similar
        mid = _first_col_startswith(bb, "BBM_") or _first_col_startswith(bb, "BBM")
        if mid and mid in bb:
            return bb[mid]
    # Fallback if bb not fully populated or columns renamed
    return ta.sma(close, length=length)
    
class AnalyzeRequest(BaseModel):
    symbol: str
    asset_type: str  # "crypto" or "fx"
    timeframe: str = DEFAULT_TF
    equity: float = 10000.0
    risk_pct: float = 0.8

class AnalyzeBatchRequest(BaseModel):
    items: List[AnalyzeRequest]

SECRET = os.getenv("AI_TRADE_SECRET", "XxUjb7DilVuqcnmeLXmUCURndUzC4Vmf")
DEFAULT_TF = "4h"

app = FastAPI(title="AI Trade Advisor Backend")
    
def first_col_like(df: pd.DataFrame, startswith: str) -> Optional[str]:
    for c in df.columns:
        if c.startswith(startswith):
            return c
    return None

def bollinger_mid(close: pd.Series, length: int = 20, std: int = 2) -> pd.Series:
    bb = ta.bbands(close, length=length, std=std)
    if isinstance(bb, pd.DataFrame) and not bb.empty:
        mid = first_col_like(bb, "BBM_")
        if mid and mid in bb:
            return bb[mid]
    # Fallback if pandas-ta naming changed or BB not available yet
    return ta.sma(close, length=length)


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


def fetch_ohlcv_crypto(symbol: str, timeframe: str) -> pd.DataFrame:
    ex = ccxt.binance({"enableRateLimit": True})
    # Convert symbol like BTC/USDT
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=400)
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"]).set_index("ts")
    df.index = pd.to_datetime(df.index, unit="ms", utc=True)
    return df

# Placeholder for FX - replace with your provider
ALPHA_VANTAGE_KEY = os.getenv("4N0D0EY6X6W42UJV")

def fetch_ohlcv_fx(pair: str, timeframe: str) -> pd.DataFrame:
    base, quote = pair.split("/")
    if not ALPHA_VANTAGE_KEY:
        raise HTTPException(500, "ALPHA_VANTAGE_KEY not set")

    if timeframe in ("1h", "4h"):
        interval = "60min"
        url = (
            "https://www.alphavantage.co/query"
            f"?function=FX_INTRADAY&from_symbol={base}&to_symbol={quote}"
            f"&interval={interval}&apikey={ALPHA_VANTAGE_KEY}&outputsize=full"
        )
        data = requests.get(url, timeout=30).json()
        key = f"Time Series FX ({interval})"
    else:
        url = (
            "https://www.alphavantage.co/query"
            f"?function=FX_DAILY&from_symbol={base}&to_symbol={quote}"
            f"&apikey={ALPHA_VANTAGE_KEY}&outputsize=full"
        )
        data = requests.get(url, timeout=30).json()
        key = "Time Series FX (Daily)"

    ts = data.get(key, {})
    if not ts:
        raise HTTPException(502, f"FX data unavailable for {pair}")

    df = (
        pd.DataFrame.from_dict(ts, orient="index")
          .rename(columns={"1. open":"open","2. high":"high","3. low":"low","4. close":"close"})
          .astype(float)
    )
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    df["volume"] = 0.0

    # Resample if 4h requested
    if timeframe == "4h":
        df = (
            df.resample("4H")
              .agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
              .dropna()
        )

    return df.tail(1000)


def compute_signals(df: pd.DataFrame) -> Dict:
    df = df.copy()

    # Core indicators
    df["ema20"]  = ta.ema(df["close"], length=20)
    df["ema50"]  = ta.ema(df["close"], length=50)
    df["ema200"] = ta.ema(df["close"], length=200)

    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if isinstance(macd, pd.DataFrame) and not macd.empty:
        # Histogram can be 'MACDh_12_26_9' or 'MACD_Hist…' depending on version
        hist_col = _first_col_startswith(macd, "MACDh_") or _first_col_startswith(macd, "MACD_Hist")
        df["macd_hist"] = macd[hist_col] if hist_col else pd.Series(index=df.index, dtype=float)
    else:
        df["macd_hist"] = pd.Series(index=df.index, dtype=float)

    df["rsi"]    = ta.rsi(df["close"], length=14)
    df["bb_mid"] = _bollinger_mid(df["close"], length=20, std=2)
    df["atr"]    = ta.atr(df["high"], df["low"], df["close"], length=14)

    # Use only rows where everything is ready
    ready_cols = ["ema20","ema50","ema200","macd_hist","rsi","bb_mid","atr","close"]
    df_ready = df.dropna(subset=ready_cols)
    if df_ready.empty:
        # Not enough ready candles yet (e.g., newly listed symbol / short history)
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


def position_size(entry: float, stop: float, equity: float, risk_pct: float) -> float:
    risk_amt = equity * (risk_pct/100.0)
    risk_per_unit = abs(entry - stop)
    if risk_per_unit <= 0:
        return 0.0
    return round(risk_amt / risk_per_unit, 6)


@app.get("/analyze")
def analyze(symbol: str, asset_type: str, tf: str = DEFAULT_TF, equity: float = 10000.0, risk_pct: float = 0.8, Authorization: Optional[str] = Header(None)):
    require_auth(Authorization)
    if asset_type == "crypto":
        df = fetch_ohlcv_crypto(symbol, tf)
    elif asset_type == "fx":
        df = fetch_ohlcv_fx(symbol.replace(" ","/"), tf)
    else:
        raise HTTPException(400, "asset_type must be crypto or fx")

    # Need enough history for EMA200; resampled FX can be tighter in practice.
    # before computing signals in /analyze
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
            r = analyze(item.symbol, item.asset_type, item.timeframe, item.equity, item.risk_pct, Authorization=f"Bearer {SECRET}")
            results.append(r)
        except Exception as e:
            results.append({"symbol": item.symbol, "error": str(e)})
    return {"results": results}
