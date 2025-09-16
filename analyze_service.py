from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import time, os
import pandas as pd
import pandas_ta as ta
import ccxt
import requests
from typing import List, Optional, Dict

SECRET = os.getenv("AI_TRADE_SECRET", "XxUjb7DilVuqcnmeLXmUCURndUzC4Vmf")
DEFAULT_TF = "4h"

app = FastAPI(title="AI Trade Advisor Backend")

class AnalyzeRequest(BaseModel):
    symbol: str
    asset_type: str  # "crypto" or "fx"
    timeframe: str = DEFAULT_TF
    equity: float = 10000.0
    risk_pct: float = 0.8

class AnalyzeBatchRequest(BaseModel):
    items: List[AnalyzeRequest]


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
    # Minimal demo using Alpha Vantage FX_INTRADAY/FX_DAILY
    base, quote = pair.split("/")
    if timeframe in ("1h","4h"):
        interval = "60min" if timeframe=="1h" else "60min"
        url = f"https://www.alphavantage.co/query?function=FX_INTRADAY&from_symbol={base}&to_symbol={quote}&interval={interval}&apikey={ALPHA_VANTAGE_KEY}&outputsize=full"
        data = requests.get(url, timeout=30).json()
        key = f"Time Series FX ({interval})"
    else:
        url = f"https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={base}&to_symbol={quote}&apikey={ALPHA_VANTAGE_KEY}&outputsize=full"
        data = requests.get(url, timeout=30).json()
        key = "Time Series FX (Daily)"
    ts = data.get(key, {})
    df = pd.DataFrame.from_dict(ts, orient="index").rename(columns={
        "1. open":"open","2. high":"high","3. low":"low","4. close":"close"
    }).astype(float)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    df["volume"] = 0.0
    return df.tail(500)


def compute_signals(df: pd.DataFrame) -> Dict:
    out = {}
    df = df.copy()
    df["ema20"] = ta.ema(df["close"], length=20)
    df["ema50"] = ta.ema(df["close"], length=50)
    df["ema200"] = ta.ema(df["close"], length=200)
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["macd_hist"] = macd["MACDh_12_26_9"]
    df["rsi"] = ta.rsi(df["close"], length=14)
    bb = ta.bbands(df["close"], length=20, std=2)
    df["bb_mid"] = bb["BBM_20_2.0"]
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    latest = df.dropna().iloc[-1]

    # Trend filter
    trend_long = latest.close > latest.ema200 and latest.ema50 > latest.ema200
    trend_short = latest.close < latest.ema200 and latest.ema50 < latest.ema200

    # Momentum
    mom_long = latest.rsi > 55 and latest.macd_hist > 0 and latest.close > latest.bb_mid
    mom_short = latest.rsi < 45 and latest.macd_hist < 0 and latest.close < latest.bb_mid

    # Entry suggestion: pullback to ema20/ema50
    pullback_long = latest.close > latest.ema20 > latest.ema50 if trend_long else False
    pullback_short = latest.close < latest.ema20 < latest.ema50 if trend_short else False

    direction = "flat"
    if trend_long and mom_long:
        direction = "long"
    elif trend_short and mom_short:
        direction = "short"

    atr = float(latest.atr)
    entry = float(latest.close)
    if direction == "long":
        stop = entry - max(1.5*atr, entry*0.004)
        tp1 = entry + (entry - stop)
        tp2 = entry + 2*(entry - stop)
    elif direction == "short":
        stop = entry + max(1.5*atr, entry*0.004)
        tp1 = entry - (stop - entry)
        tp2 = entry - 2*(stop - entry)
    else:
        stop, tp1, tp2 = entry, entry, entry

    # Confidence score
    conf = 0
    if trend_long or trend_short: conf += 20
    if mom_long or mom_short: conf += 20
    if pullback_long or pullback_short: conf += 15
    rr = 0.0
    if direction != "flat":
        rr = abs((tp2 - entry) / (entry - stop)) if entry != stop else 0.0
        if rr >= 2.0: conf += 15
    # Volatility regime
    atr_pct = atr / entry * 100 if entry else 0
    if 0.5 <= atr_pct <= 5: conf += 10

    rationale = []
    if direction == "long":
        rationale += ["Price above rising 200 EMA" if trend_long else "200 EMA trend not confirmed",
                      "RSI>55 & MACD>0" if mom_long else "Momentum mixed",
                      "Pullback structure intact" if pullback_long else "No ideal pullback"]
    elif direction == "short":
        rationale += ["Price below falling 200 EMA" if trend_short else "200 EMA trend not confirmed",
                      "RSI<45 & MACD<0" if mom_short else "Momentum mixed",
                      "Pullback structure intact" if pullback_short else "No ideal pullback"]
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
            "ema200": round(float(latest.ema200),5),
            "rsi": round(float(latest.rsi),2),
            "macd_hist": round(float(latest.macd_hist),4),
            "atr": round(float(atr),5)
        }
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

    if len(df) < 200:
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
