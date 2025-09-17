# main.py — AI Trade Advisor backend (Kraken/USD)
# FastAPI + ccxt + pandas (py3.12 recommended; see requirements.txt)
from fastapi import FastAPI, HTTPException, Depends, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
import os, time, math
import pandas as pd
import numpy as np
import ccxt

# ----- Security -----
API_KEY = os.getenv("API_KEY", "change-me")
def require_key(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    token = authorization.split(" ", 1)[1].strip()
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Bad token")
    return None

# ----- Exchange (Kraken) -----
EXCHANGE_ID = os.getenv("EXCHANGE", "kraken")
QUOTE = os.getenv("QUOTE", "USD")
TOP_N = int(os.getenv("TOP_N", "30"))

_ex = None
def get_exchange():
    global _ex
    if _ex is None:
        _ex = getattr(ccxt, EXCHANGE_ID)({
            "enableRateLimit": True,
            "timeout": 20000,
        })
    return _ex

def load_markets():
    ex = get_exchange()
    if not getattr(ex, "markets", None):
        ex.load_markets()
    return ex.markets

# ----- App -----
app = FastAPI(title="AI Trade Advisor API", version="2025.09")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ----- Cache -----
CACHE: Dict[str, Tuple[float, Any]] = {}
def cache_get(key, ttl):
    v = CACHE.get(key)
    if not v: return None
    ts, data = v
    return data if (time.time() - ts) <= ttl else None
def cache_set(key, data): CACHE[key] = (time.time(), data)

# ----- Universe -----
CURATED = [
    "BTC","ETH","XRP","SOL","ADA","DOGE","LINK","LTC","BCH","TRX",
    "DOT","ATOM","XLM","ETC","MATIC","UNI","APT","ARB","OP","AVAX",
    "NEAR","ALGO","FIL","SUI","SHIB","USDC","USDT","XMR","AAVE"
    ,"PAXG","ONDO","PEPE","SEI","IMX","FIL","TIA"
]
def get_universe(quote=QUOTE, limit=TOP_N) -> List[Dict[str, Any]]:
    key = f"uni:{EXCHANGE_ID}:{quote}:{limit}"
    u = cache_get(key, 1800)
    if u is not None: return u
    markets = load_markets()

    available: List[str] = []
    for base in CURATED:
        sym = f"{base}/{quote}"
        if sym in markets and markets[sym].get("active"):
            available.append(sym)

    if len(available) < limit:
        # Fallback: fill with other active {quote} tickers
        for m, info in markets.items():
            if info.get("quote") == quote and info.get("active"):
                if m not in available:
                    available.append(m)
                if len(available) >= limit:
                    break

    out = [{"symbol": s, "name": s.split('/')[0], "market": EXCHANGE_ID, "tf_supported": ["1h","1d"]} for s in available[:limit]]
    cache_set(key, out)
    return out

# ----- Market data -----
TF_MAP = {"1h": "1h", "1d": "1d"}

def fetch_ohlcv(symbol: str, tf: str, bars: int = 720) -> pd.DataFrame:
    ex = get_exchange()
    tf_ex = TF_MAP.get(tf)
    if tf_ex is None:
        raise HTTPException(400, detail=f"Unsupported timeframe: {tf}")
    key = f"ohlcv:{symbol}:{tf_ex}:{bars}"
    cached = cache_get(key, 900)
    if cached is not None:
        return cached.copy()
    try:
        data = ex.fetch_ohlcv(symbol, timeframe=tf_ex, limit=bars)
    except ccxt.BadSymbol:
        raise HTTPException(502, detail=f"Symbol not available on {EXCHANGE_ID}: {symbol}")
    except Exception as e:
        raise HTTPException(502, detail=f"fetch_ohlcv failed: {e}")
    if not data or len(data) < 50:
        raise HTTPException(502, detail="insufficient candles")
    df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    cache_set(key, df)
    return df.copy()

# ----- Indicators -----
def ema(s: pd.Series, n: int) -> pd.Series: return s.ewm(span=n, adjust=False).mean()
def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff(); up = d.clip(lower=0); dn = -d.clip(upper=0)
    rs = ema(up,n) / (ema(dn,n) + 1e-12)
    return 100 - (100/(1+rs))
def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h,l,c = df["high"], df["low"], df["close"]; pc = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return ema(tr,n)

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sma20"]   = out["close"].rolling(20).mean()
    out["sma50"]   = out["close"].rolling(50).mean()
    out["sma200"]  = out["close"].rolling(200).mean()
    out["rsi14"]   = rsi(out["close"], 14)
    out["atr14"]   = atr(out, 14)
    out["atr_pct"] = (out["atr14"]/out["close"]).clip(lower=0, upper=2.0)
    out["ret5"]    = out["close"].pct_change(5)
    out["slope20"] = out["sma20"] - out["sma20"].shift(5)
    out["bb_mid"]  = out["close"].rolling(20).mean()
    out["bb_std"]  = out["close"].rolling(20).std().replace(0, np.nan)
    out["bb_pos"]  = ((out["close"] - out["bb_mid"]) / (2*out["bb_std"])).clip(-1,1)
    return out

# ----- Signal logic -----
MIN_CONFIDENCE     = float(os.getenv("MIN_CONFIDENCE", "0.14"))
VOL_CAP_ATR_PCT    = float(os.getenv("VOL_CAP_ATR_PCT", "0.25"))
VOL_MIN_ATR_PCT    = float(os.getenv("VOL_MIN_ATR_PCT", "0.001"))

def build_trade(symbol: str, df: pd.DataFrame, direction: str,
                risk_pct: float = 1.0, equity: Optional[float] = None, leverage: float = 1.0) -> Dict[str, Any]:
    ex = get_exchange()
    markets = load_markets()
    mkt = markets.get(symbol, {})

    last  = df.iloc[-1]
    price = float(last["close"])
    a     = float(last["atr14"])
    if not math.isfinite(a) or a <= 0:
        raise ValueError("ATR not finite")

    mult = 2.2
    if direction == "Long":
        stop_raw = price - mult * a
        targets_raw = [price + k * a for k in (1.5, 2.5, 3.5)]
    else:
        stop_raw = price + mult * a
        targets_raw = [price - k * a for k in (1.5, 2.5, 3.5)]

    price_p = float(ex.price_to_precision(symbol, price))
    stop    = float(ex.price_to_precision(symbol, stop_raw))
    targets = [float(ex.price_to_precision(symbol, t)) for t in targets_raw]

    denom = (price_p - stop)
    rr = [round(abs((t - price_p) / denom), 2) if denom else 0.0 for t in targets]

    pos = None
    risk_amt = None
    if equity and risk_pct:
        risk_amt = float(equity) * (float(risk_pct) / 100.0)
        risk_per_unit = abs(denom) / max(float(leverage), 1.0)
        qty_raw = risk_amt / max(risk_per_unit, 1e-8)
        qty = float(ex.amount_to_precision(symbol, qty_raw))
        notional = qty * price_p
        pos = {"qty": qty, "notional": notional}

    return {
        "direction": direction,
        "entry": price_p,
        "stop": stop,
        "targets": targets,
        "rr": rr,
        "position_size": pos,
        "risk_amount": risk_amt,         # NEW: shows $ at risk
        "leverage": leverage,            # handy for UI
        "risk_suggestions": {
            "breakeven_after_tp": 1,
            "trail_after_tp": 2,
            "trail_method": "ATR",
            "trail_multiple": 1.0
        },
        "precision": mkt.get("precision", {})
    }
}

def infer_signal(feats: pd.DataFrame, min_conf: float) -> Tuple[str,float,Dict[str,bool],List[str]]:
    last = feats.iloc[-1]; reasons=[]
    trend_up = bool(last["sma20"]>last["sma50"] and feats["slope20"].iloc[-1]>0)
    trend_dn = bool(last["sma20"]<last["sma50"] and feats["slope20"].iloc[-1]<0)
    trend_ok = trend_up or trend_dn
    if not trend_ok: reasons.append("no clear trend")

    atr_pct = float(last["atr_pct"])
    vol_ok = bool((atr_pct>=VOL_MIN_ATR_PCT) and (atr_pct<=VOL_CAP_ATR_PCT))
    if not vol_ok: reasons.append("ATR% outside bounds")

    rsi14 = float(last["rsi14"])
    bias_up = rsi14>=52; bias_dn = rsi14<=48

    conf=0.0
    if trend_ok: conf += 0.45
    if vol_ok:   conf += 0.25
    if bias_up or bias_dn: conf += 0.15
    conf += min(abs(float(feats["ret5"].iloc[-1] or 0.0))*2.0, 0.15)
    conf = float(max(0.0, min(conf,1.0)))

    if trend_up and bias_up:   sig="Bullish"
    elif trend_dn and bias_dn: sig="Bearish"
    else:                      sig="Neutral"

    # explicit reason for threshold gate
    if conf < min_conf:
        reasons.append(f"Composite conf {conf:.2f} < {min_conf:.2f}")

    return sig, conf, {"trend_ok":trend_ok,"vol_ok":vol_ok,"confidence_ok":conf>=min_conf}, reasons

# ----- Pure helper used by both endpoints -----
def evaluate_signal(
    symbol: str,
    tf: str,
    risk_pct: float,
    equity: Optional[float],
    leverage: float,
    min_confidence: Optional[float],
    ignore_trend: bool = False,
    ignore_vol: bool = False,
    allow_neutral: bool = False,
) -> Dict[str, Any]:
    df = fetch_ohlcv(symbol, tf, bars=400)
    feats = compute_features(df).dropna().iloc[-200:]
    if len(feats) < 50:
        raise HTTPException(502, detail="insufficient features window")

    thresh = min_confidence if (min_confidence is not None) else MIN_CONFIDENCE
    sig, conf, filt, reasons = infer_signal(feats, thresh)

    # Apply client relax flags to filters/reasons
    if ignore_trend:
        filt["trend_ok"] = True
        reasons = [r for r in reasons if "trend" not in r.lower()]
    if ignore_vol:
        filt["vol_ok"] = True
        reasons = [r for r in reasons if "atr%" not in r.lower() and "vol" not in r.lower()]

    trade = None
    advice = "Skip"

    # Direction gate: allow neutral if requested (choose side from slope/RSI/bb_pos)
    directional_ok = (sig in ("Bullish", "Bearish")) or allow_neutral
    if conf >= thresh and filt["vol_ok"] and directional_ok:
        if sig == "Bullish":
            direction = "Long"
        elif sig == "Bearish":
            direction = "Short"
        else:
            # pick a side for neutral
            slope = float(feats["slope20"].iloc[-1])
            rsi14 = float(feats["rsi14"].iloc[-1])
            bbpos = float(feats["bb_pos"].iloc[-1])
            direction = "Long" if (slope >= 0 or rsi14 >= 50 or bbpos >= 0) else "Short"

        trade = build_trade(symbol, feats, direction, risk_pct, equity, leverage)
        advice = "Consider"

    return {
        "symbol": symbol,
        "timeframe": tf,
        "signal": sig,
        "confidence": conf,
        "updated": pd.Timestamp.utcnow().isoformat(),
        "trade": trade,
        "filters": {**filt, "reasons": reasons},
        "advice": advice,
    }

# ----- Schemas (light) -----
class Instrument(BaseModel):
    symbol: str; name: str; market: str; tf_supported: List[str]

# ----- Endpoints -----
@app.get("/health")
def health():
    try: name = get_exchange().id
    except Exception as e: name = f"error: {e}"
    return {"ok": True, "exchange": name, "quote": QUOTE}

@app.get("/instruments", response_model=List[Instrument])
def instruments(_: None = Depends(require_key)):
    return get_universe()

@app.get("/chart")
def chart(symbol: str, tf: str = "1h", n: int = 120, _: None = Depends(require_key)):
    df = fetch_ohlcv(symbol, tf, bars=max(200, n+20))
    closes = df["close"].tail(n).astype(float).tolist()
    return {"symbol": symbol, "tf": tf, "closes": closes}

@app.get("/signals")
def signals(
    symbol: str,
    tf: str = "1h",
    risk_pct: float = Query(1.0, ge=0.1, le=5.0),
    equity: Optional[float] = Query(None, ge=0),
    leverage: float = Query(1.0, ge=1.0, le=100.0),
    min_confidence: Optional[float] = Query(None),
    _: None = Depends(require_key)
):
    try:
        return evaluate_signal(symbol, tf, risk_pct, equity, leverage, min_confidence)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, detail=f"signals failed: {e}")

@app.get("/scan")
def scan(
    tf: str = "1h",
    limit: int = Query(TOP_N, ge=3, le=50),
    top: int = Query(6, ge=1, le=12),
    min_confidence: Optional[float] = Query(None),
    auto_relax: int = Query(1, ge=0, le=1),
    allow_neutral: int = Query(1, ge=0, le=1),
    ignore_trend: int = Query(0, ge=0, le=1),   # NEW
    ignore_vol: int = Query(0, ge=0, le=1),     # NEW
    risk_pct: float = Query(1.0, ge=0.1, le=5.0),
    equity: Optional[float] = Query(None, ge=0.0),
    leverage: float = Query(1.0, ge=1.0, le=100.0),   # raised cap
    include_chart: int = Query(1, ge=0, le=1),
    _: None = Depends(require_key)
):
    uni = get_universe(limit=limit)
    results, ok = [], []

    for it in uni:
        try:
            s = evaluate_signal(
                symbol=it["symbol"], tf=tf,
                risk_pct=risk_pct, equity=equity, leverage=leverage,
                min_confidence=min_confidence,
                ignore_trend=bool(ignore_trend),
                ignore_vol=bool(ignore_vol),
                allow_neutral=bool(allow_neutral),
            )
            s["market"] = it["market"]
            if s["advice"] == "Consider":
                ok.append(s)
            results.append(s)
        except HTTPException as he:
            results.append({
                "symbol": it["symbol"], "timeframe": tf, "signal": "Neutral",
                "confidence": 0.0, "updated": pd.Timestamp.utcnow().isoformat(),
                "trade": None,
                "filters": {"trend_ok": False, "vol_ok": False, "confidence_ok": False, "reasons": [he.detail]},
                "advice": "Skip", "market": it["market"]
            })
        except Exception as e:
            results.append({
                "symbol": it["symbol"], "timeframe": tf, "signal": "Neutral",
                "confidence": 0.0, "updated": pd.Timestamp.utcnow().isoformat(),
                "trade": None,
                "filters": {"trend_ok": False, "vol_ok": False, "confidence_ok": False, "reasons": [f"exception: {e}"]},
                "advice": "Skip", "market": it["market"]
            })
    # ... (rest of your /scan stays the same)

    # choose pool
    pool = ok if ok else results
    pool_sorted = sorted(pool, key=lambda s: abs(s.get("confidence") or 0.0), reverse=True)
    topK = pool_sorted[:top]

    # attach chart data if requested
    out: List[Dict[str, Any]] = []
    for s in topK:
        if include_chart:
            try:
                df = fetch_ohlcv(s["symbol"], tf, bars=160)
                s["chart"] = {"closes": df["close"].tail(120).astype(float).tolist()}
            except Exception:
                s["chart"] = None
        out.append(s)

    note = None
    if not ok and allow_neutral:
        note = "No high-confidence setups; returning best candidates by confidence."

    return {"universe": len(uni), "note": note, "results": out}
