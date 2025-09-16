from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os, time, math
import pandas as pd
import numpy as np

# --- Security ---
API_KEY = os.getenv("API_KEY", "change-me")

def require_key(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    token = authorization.split(" ", 1)[1].strip()
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid token")

# --- App ---
app = FastAPI(title="AI Trade Advisor", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- In-memory cache (replace with Postgres/Redis in prod) ---
STATE = {
    "universe": [],      # list of {symbol, name, market, tf_supported}
    "signals": {},       # {(symbol, tf): {signal, conf, updated, features...}}
    "backtests": {},     # {symbol: {metrics}}
    "last_refresh": 0,
}

# ---- Models / featurization demo (replace with your ML pipeline) ----
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    # df columns: ["ts","open","high","low","close","volume"]
    df = df.copy()
    df["ret1"] = df["close"].pct_change()
    df["ret5"] = df["close"].pct_change(5)
    df["vol_mean20"] = df["volume"].rolling(20).mean()
    df["rsi14"] = rsi(df["close"], 14)
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()
    df["sma20_50_diff"] = (df["sma20"] - df["sma50"]) / df["sma50"]
    df["bb_up"], df["bb_dn"] = bollinger(df["close"], 20, 2)
    df["bb_pos"] = (df["close"] - df["bb_dn"]) / (df["bb_up"] - df["bb_dn"])
    return df

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up).rolling(n).mean()
    roll_down = pd.Series(down).rolling(n).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def bollinger(series: pd.Series, n: int, k: float):
    ma = series.rolling(n).mean()
    sd = series.rolling(n).std()
    return ma + k*sd, ma - k*sd

def simple_rule_inference(df: pd.DataFrame) -> Dict[str, Any]:
    """Placeholder 'AI' signal: combine RSI and MA slope. Swap out with your ML model."""
    last = df.iloc[-1]
    slope = df["sma20"].iloc[-1] - df["sma20"].iloc[-5]
    score = 0.0
    # RSI contrarian banded logic + trend tilt
    if last["rsi14"] < 30: score += 0.4
    if last["rsi14"] > 70: score -= 0.4
    score += np.tanh(slope / (1e-6 + abs(df["sma20"].iloc[-5]))) * 0.3
    score += np.clip(last["bb_pos"] - 0.5, -0.5, 0.5) * 0.3

    label = "Bullish" if score > 0.15 else "Bearish" if score < -0.15 else "Neutral"
    conf = float(np.clip(abs(score), 0, 1))
    return {"signal": label, "confidence": conf, "score": float(score)}

# ---- Demo data: replace with CCXT/CoinGecko ingestion ----
def demo_prices(symbol: str, n: int = 300) -> pd.DataFrame:
    # Synthetic walk; swap with real OHLCV
    rng = pd.date_range(end=pd.Timestamp.utcnow(), periods=n, freq="H")
    price = 30000 + np.cumsum(np.random.randn(n)) * 50
    high = price + np.random.rand(n) * 30
    low = price - np.random.rand(n) * 30
    open_ = np.concatenate([[price[0]], price[:-1]])
    vol = 100 + np.random.rand(n) * 50
    return pd.DataFrame({
        "ts": rng, "open": open_, "high": high, "low": low, "close": price, "volume": vol
    })

def ensure_state():
    # populate demo universe once
    if not STATE["universe"]:
        STATE["universe"] = [
            {"symbol": "BTC/USDT", "name": "Bitcoin", "market": "Binance", "tf_supported": ["1h","1d"]},
            {"symbol": "ETH/USDT", "name": "Ethereum", "market": "Binance", "tf_supported": ["1h","1d"]},
            {"symbol": "SOL/USDT", "name": "Solana", "market": "Binance", "tf_supported": ["1h","1d"]},
        ]
    STATE["last_refresh"] = time.time()

# ---- Schemas ----
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

# ---- Routes ----
@app.get("/health")
def health():
    return {"ok": True, "ts": time.time()}

@app.get("/instruments", response_model=List[Instrument])
def instruments(_: None = Depends(require_key)):
    ensure_state()
    return STATE["universe"]

@app.get("/signals", response_model=Signal)
def signals(symbol: str, tf: str = "1h", _: None = Depends(require_key)):
    ensure_state()
    key = (symbol, tf)
    if key not in STATE["signals"]:
        df = demo_prices(symbol, 500 if tf=="1h" else 400)
        feats = compute_features(df).iloc[-1]
        pred = simple_rule_inference(compute_features(df))
        payload = {
            "symbol": symbol,
            "timeframe": tf,
            "signal": pred["signal"],
            "confidence": pred["confidence"],
            "updated": pd.Timestamp.utcnow().isoformat(),
            "features": {
                "rsi14": float(feats["rsi14"]),
                "sma20_50_diff": float(feats["sma20_50_diff"]),
                "bb_pos": float(feats["bb_pos"]),
                "ret5": float(feats["ret5"]),
            }
        }
        STATE["signals"][key] = payload
    return STATE["signals"][key]

@app.get("/summary")
def summary(_: None = Depends(require_key)):
    ensure_state()
    # dummy leaderboards
    board = []
    for inst in STATE["universe"]:
        s = STATE["signals"].get((inst["symbol"], "1h")) or {}
        board.append({
            "symbol": inst["symbol"],
            "signal": s.get("signal", "Neutral"),
            "confidence": s.get("confidence", 0.0),
        })
    return {"universe": STATE["universe"], "leaderboard": board}
