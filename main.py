# main.py — AI Trade Advisor backend v3.0.4 (Kraken/USD)
# FastAPI + ccxt + pandas (py3.12 recommended; see requirements.txt)
from __future__ import annotations
from fastapi import FastAPI, HTTPException, Depends, Header, Query, Request
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple
import os, json, sqlite3, threading, time
import math
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

# put this near the top, before app = FastAPI(...)
origins_env = os.getenv("ALLOWED_ORIGINS", "*")
origins = ["*"] if origins_env == "*" else [o.strip() for o in origins_env.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)
app = FastAPI(title="AI Trade Advisor API", version="2025.09")

# ========= FEEDBACK STORAGE & ONLINE WEIGHTS =========
DB_PATH = os.getenv("FEEDBACK_DB", "/tmp/ai_trade_feedback.db")
_db_lock = threading.Lock()

def _db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_feedback_db():
    with _db_lock:
        conn = _db()
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS feedback(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts REAL NOT NULL,
          symbol TEXT, tf TEXT, market TEXT,
          direction TEXT, entry REAL, stop REAL, targets TEXT,
          confidence REAL, advice TEXT, outcome INTEGER,  -- +1 good, -1 bad
          features TEXT, edge TEXT, composite TEXT,
          equity REAL, risk_pct REAL, leverage REAL,
          ua TEXT, ip TEXT
        );
        CREATE TABLE IF NOT EXISTS weights(
          feature TEXT PRIMARY KEY,
          w REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS offers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_ts REAL NOT NULL,     -- epoch seconds when the card list was shown
        expires_ts REAL,              -- planned eval time (created_ts + hold_ms)
        symbol     TEXT NOT NULL,
        tf         TEXT NOT NULL,
        market     TEXT,
        direction  TEXT,
        entry      REAL,
        stop       REAL,
        targets    TEXT,              -- JSON array
        confidence REAL,
        advice     TEXT,
        features   TEXT,              -- JSON map
        equity     REAL,
        risk_pct   REAL,
        leverage   REAL,
        currency   TEXT,              -- "USD"/"GBP"/"EUR"
        universe   INTEGER,
        note       TEXT,              -- free-form note for scan settings
        resolved   INTEGER DEFAULT 0, -- Phase 2: 0=pending, 1=resolved
        result     TEXT,              -- Phase 2: 'TP','SL','PNL','NEITHER' etc
        pnl        REAL,              -- Phase 2: realized P/L at resolution
        resolved_ts REAL              -- Phase 2: epoch seconds
        );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_offers_symbol_tf ON offers(symbol, tf);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_offers_created ON offers(created_ts);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_offers_resolved ON offers(resolved, expires_ts);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_offers_exp ON offers(expires_ts);")
        conn.commit()
        conn.close()

def ensure_feedback_migrations(conn):
    cur = conn.cursor()
    cols = {r[1] for r in cur.execute("PRAGMA table_info(feedback)").fetchall()}
    if "uid" not in cols:
        cur.execute("ALTER TABLE feedback ADD COLUMN uid INTEGER DEFAULT 0")
    if "fingerprint" not in cols:
        cur.execute("ALTER TABLE feedback ADD COLUMN fingerprint TEXT")
    if "accepted" not in cols:
        cur.execute("ALTER TABLE feedback ADD COLUMN accepted INTEGER DEFAULT 1")

    # Pre-dedupe to allow unique index creation on legacy data
    cur.execute("""
        DELETE FROM feedback
        WHERE id IN (
          SELECT id FROM (
            SELECT id,
                   ROW_NUMBER() OVER(
                     PARTITION BY COALESCE(uid,0), COALESCE(fingerprint,'')
                     ORDER BY id DESC
                   ) AS rn
            FROM feedback
            WHERE fingerprint IS NOT NULL
          ) t
          WHERE t.rn > 1
        )
    """)

    # Safe to create indexes now that columns exist
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_feedback_uid_fp ON feedback(uid, fingerprint)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_fb_sym_tf_acc_ts ON feedback(symbol, tf, accepted, ts)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_fb_uid_ts ON feedback(uid, ts)")
    conn.commit()

def get_weights() -> Dict[str, float]:
    with _db_lock:
        conn = _db()
        cur = conn.execute("SELECT feature, w FROM weights")
        d = {k: float(v) for k, v in cur.fetchall()}
        conn.close()
        return d

def update_weights(features: Dict[str, Any], outcome: int, lr: float = 0.05):
    """
    Super-light online learning: w_i += lr * outcome * norm(feature_i)
    outcome: +1 good (TP), -1 bad (SL). Features loosely normalized.
    """
    if not isinstance(features, dict):
        return
    # normalize a few known numeric features; ignore non-numerics
    norm: Dict[str, float] = {}
    for k, v in features.items():
        if v is None:
            continue
        try:
            x = float(v)
        except Exception:
            continue
        # coarse normalization per feature name
        if "rsi" in k:          # RSI ~ [0..100]
            x = (x - 50.0) / 50.0
        elif "macd" in k or "diff" in k or "ret" in k or "bb_pos" in k or "donch" in k:
            # typically around [-1..+1] already-ish
            x = max(-2.0, min(2.0, x))
        elif "atr" in k:        # ATR or atr% → keep small
            x = max(-1.0, min(1.0, x))
        else:
            x = max(-3.0, min(3.0, x))
        norm[k] = x

    if not norm:
        return

    with _db_lock:
        conn = _db()
        for k, x in norm.items():
            cur = conn.execute("SELECT w FROM weights WHERE feature=?", (k,))
            row = cur.fetchone()
            w = float(row[0]) if row else 0.0
            w = w + lr * outcome * x
            conn.execute("INSERT INTO weights(feature, w) VALUES(?,?) ON CONFLICT(feature) DO UPDATE SET w=excluded.w",
                         (k, w))
        conn.commit()
        conn.close()

def feedback_bias(features: Dict[str, Any], alpha: float = 0.15) -> float:
    """
    Compute a bias adjustment from learned weights; clamp to [-1..1] then scale by alpha.
    """
    ws = get_weights()
    s = 0.0
    for k, v in (features or {}).items():
        try:
            x = float(v)
        except Exception:
            continue
        # re-use the same crude normalization as in update_weights
        if "rsi" in k:          x = (x - 50.0) / 50.0
        elif "macd" in k or "diff" in k or "ret" in k or "bb_pos" in k or "donch" in k:
            x = max(-2.0, min(2.0, x))
        elif "atr" in k:        x = max(-1.0, min(1.0, x))
        else:                   x = max(-3.0, min(3.0, x))
        w = ws.get(k, 0.0)
        s += w * x
    # squish
    s = max(-1.0, min(1.0, s))
    return alpha * s

@app.on_event("startup")
def _fb_startup():
    init_feedback_db()

BG_INTERVAL = int(os.getenv("LEARNER_BG_INTERVAL_SEC", "0"))  # e.g. 300 for 5min

def _bg_worker():
    while True:
        try:
            resolve_due_offers(limit=100)
        except Exception:
            pass
        time.sleep(max(60, BG_INTERVAL))

@app.on_event("startup")
def _start_bg():
    if BG_INTERVAL > 0:
        t = threading.Thread(target=_bg_worker, daemon=True)
        t.start()

@app.on_event("startup")
def _startup():
    with _db_lock:
        conn = _db()
        try:
            ensure_feedback_migrations(conn)
        finally:
            conn.close()

# =====================================================

def apply_feedback_bias(confidence: float, features: Dict[str, Any]) -> float:
    """Return confidence nudged by learned weights; always safe."""
    try:
        return max(-1.0, min(1.0, float(confidence) + feedback_bias(features)))
    except Exception:
        return float(confidence)

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
    ,"PAXG","ONDO","PEPE","SEI","IMX","TIA"
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

# ms per bar
def tf_ms(tf: str) -> int:
    if tf == "1h": return 60*60*1000
    if tf == "1d": return 24*60*60*1000
    raise HTTPException(400, detail=f"Unsupported tf for eval: {tf}")

def fetch_ohlcv_window(symbol: str, tf: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """
    Fetch enough OHLCV to cover [start_ms, end_ms]. We use ccxt 'since' + 'limit' with a small buffer.
    """
    ex = get_exchange()
    per = tf_ms(tf)
    need = max(2, int((end_ms - start_ms) / per) + 4)
    since = max(0, start_ms - 2*per)
    try:
        data = ex.fetch_ohlcv(symbol, timeframe=TF_MAP[tf], since=since, limit=min(need+20, 2000))
    except Exception as e:
        raise HTTPException(502, detail=f"eval window fetch failed: {e}")

    if not data:
        raise HTTPException(502, detail="no candles for eval window")

    df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
    # keep only overlap window (with one bar margin on each side)
    df = df[(df["ts"] >= (start_ms - per)) & (df["ts"] <= (end_ms + per))]
    if len(df) < 1:
        raise HTTPException(502, detail="empty eval slice")
    return df.reset_index(drop=True)

EVAL_STOP_FIRST = bool(int(os.getenv("EVAL_STOP_FIRST", "0")))  # if 1: SL wins ties within same bar

# --- Learning: default hold windows (env-overridable) ---
HOLD_1H_SECS = int(os.getenv("HOLD_1H_SECS", "28800"))   # 8h default
HOLD_1D_SECS = int(os.getenv("HOLD_1D_SECS", "604800"))  # 7d default

def hold_secs_for(tf: str) -> int:
    if tf == "1h": return HOLD_1H_SECS
    if tf == "1d": return HOLD_1D_SECS
    return HOLD_1H_SECS
# --------------------------------------------------------

def _tp_hit_first(entry, stop, tps, df, direction: str) -> Tuple[str, float]:
    """
    Walk forward per candle from offer creation to expiry, returning ("TP"/"SL"/"NONE", pnl_frac_at_event_or_expiry)
    pnl_frac is (price-entry)/entry for Long, (entry-price)/entry for Short.
    For ties in the same candle, TP wins unless EVAL_STOP_FIRST=1.
    """
    if not isinstance(tps, (list, tuple)) or not tps:
        tps = []

    # choose first target as success threshold
    tp = min(tps) if direction == "Long" else max(tps) if tps else None

    def pnl_frac(price: float) -> float:
        if direction == "Long":
            return (price - entry) / max(abs(entry), 1e-12)
        else:
            return (entry - price) / max(abs(entry), 1e-12)

    first_event = "NONE"
    event_pnl   = 0.0

    for _, bar in df.iterrows():
        high = float(bar["high"]); low = float(bar["low"]); close = float(bar["close"])
        tp_hit = False
        sl_hit = False

        if direction == "Long":
            if tp is not None and high >= tp: tp_hit = True
            if low <= stop: sl_hit = True
        else:  # Short
            if tp is not None and low <= tp: tp_hit = True
            if high >= stop: sl_hit = True

        if tp_hit and sl_hit:
            # tie-break rule in same candle
            if EVAL_STOP_FIRST:
                first_event = "SL"
                event_pnl   = pnl_frac(stop)
            else:
                first_event = "TP"
                event_pnl   = pnl_frac(tp if tp is not None else close)
            break
        elif tp_hit:
            first_event = "TP"
            event_pnl   = pnl_frac(tp if tp is not None else close)
            break
        elif sl_hit:
            first_event = "SL"
            event_pnl   = pnl_frac(stop)
            break

    if first_event == "NONE":
        # judge by expiry-close
        close = float(df.iloc[-1]["close"])
        return "PNL", pnl_frac(close)
    return first_event, event_pnl

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

# ---- Feature snapshot for learning/bias (add after compute_features) ----
def last_features_snapshot(feats: pd.DataFrame) -> Dict[str, float]:
    """
    Take the latest row of computed features and return a small, clean dict
    of numeric values the learner can use. Keep names stable.
    """
    row = feats.iloc[-1]
    keys = [
        "rsi14", "atr_pct", "ret5", "slope20", "bb_pos",
        # you can add more later; keep them numeric and reasonably bounded
    ]
    out: Dict[str, float] = {}
    for k in keys:
        v = row.get(k)
        try:
            x = float(v)
            if math.isfinite(x):
                out[k] = x
        except Exception:
            pass
    # Back-compat: your frontend used `atr14_pct`; duplicate the value
    if "atr_pct" in out and "atr14_pct" not in out:
        out["atr14_pct"] = out["atr_pct"]
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

        # risk-based size: leverage should NOT change risk_per_unit
        risk_per_unit = abs(denom)
        qty_raw = risk_amt / max(risk_per_unit, 1e-8)
        qty = float(ex.amount_to_precision(symbol, qty_raw))
        notional = qty * price_p

        # margin uses leverage; risk does not
        levf = max(float(leverage or 1.0), 1.0)
        margin = notional / levf

        # Optional safety: cap to available equity by margin (keeps margin <= equity)
        if equity and margin > float(equity):
            cap_notional = float(equity) * levf
            scale = cap_notional / max(notional, 1e-8)
            qty = float(ex.amount_to_precision(symbol, qty * scale))
            notional = qty * price_p
            margin = notional / levf

        pos = {"qty": qty, "notional": notional, "margin": margin, "leverage": levf}


    return {
        "direction": direction,
        "entry": price_p,
        "stop": stop,
        "targets": targets,
        "rr": rr,
        "position_size": pos,
        "risk_amount": risk_amt,         # NEW: shows $ at risk
        "leverage": float(leverage or 1.0),            # handy for UI
        "risk_suggestions": {
            "breakeven_after_tp": 1,
            "trail_after_tp": 2,
            "trail_method": "ATR",
            "trail_multiple": 1.0
        },
        "precision": mkt.get("precision", {})
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

def _vote_matches_market(fb: FeedbackIn) -> Tuple[bool, str]:
    # Need direction+entry for any meaningful check
    if not fb.direction or fb.entry is None:
        return False, "missing direction/entry for consistency check"

    # Prefer real-time last price; fallback to last close
    ex = get_exchange()
    px = None
    try:
        t = ex.fetch_ticker(fb.symbol)
        if t and t.get("last"):
            px = float(t["last"])
    except Exception:
        pass
    if px is None:
        df = fetch_ohlcv(fb.symbol, fb.tf, bars=2)
        px = float(df["close"].iloc[-1])

    is_long = fb.direction.lower().startswith("long")
    tp1 = None
    if fb.targets:
        tps = [float(x) for x in fb.targets if x is not None]
        if tps:
            tp1 = min(tps) if is_long else max(tps)

    if fb.outcome > 0:  # 👍 “good”
        ok = (tp1 is not None and ((px >= tp1) if is_long else (px <= tp1))) or \
             ((px > float(fb.entry)) if is_long else (px < float(fb.entry)))
        return ok, "upvote requires TP1 hit or profit at vote time"

    if fb.outcome < 0:  # 👎 “bad”
        if fb.stop is not None:
            ok = ((px <= float(fb.stop)) if is_long else (px >= float(fb.stop))) or \
                 ((px < float(fb.entry)) if is_long else (px > float(fb.entry)))
        else:
            # No SL provided; allow pure P&L check
            ok = (px < float(fb.entry)) if is_long else (px > float(fb.entry))
        return ok, "downvote requires SL hit or loss at vote time"

    return False, "invalid outcome"

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

    # stable feature snapshot for learning/bias
    feat_map = last_features_snapshot(feats)

    return {
        "symbol": symbol,
        "timeframe": tf,
        "signal": sig,
        "confidence": conf,
        "updated": pd.Timestamp.utcnow().isoformat(),
        "trade": trade,
        "filters": {**filt, "reasons": reasons},
        "advice": advice,
        "features": feat_map,
    }

def resolve_offer_row(row: sqlite3.Row) -> dict:
    symbol     = row["symbol"]
    tf         = row["tf"]
    direction  = row["direction"]
    entry      = row["entry"]
    stop       = row["stop"]

    # targets: safe parse
    try:
        targets = [float(x) for x in json.loads(row["targets"] or "[]") if x is not None]
    except Exception:
        targets = []

    # these must be set unconditionally
    created_ts = int(row["created_ts"])
    expires_ts = int(row["expires_ts"])

    fallback_only = not (
        direction and isinstance(entry, (int, float)) and isinstance(stop, (int, float))
    )

    df = fetch_ohlcv_window(symbol, tf, created_ts * 1000, expires_ts * 1000)

    if fallback_only:
        created_close = float(df.iloc[0]["close"])
        close = float(df.iloc[-1]["close"])
        if not direction:
            direction = "Long" if close >= created_close else "Short"
        entry_eff = float(entry) if isinstance(entry, (int, float)) and math.isfinite(float(entry)) else created_close
        denom = max(abs(entry_eff), 1e-12)
        pnl_frac = (close - entry_eff) / denom if direction == "Long" else (entry_eff - close) / denom
        result = "PNL"
        outcome = 1 if pnl_frac > 0 else (-1 if pnl_frac < 0 else 0)
    else:
        result, pnl_frac = _tp_hit_first(float(entry), float(stop), [float(x) for x in targets], df, direction)
        outcome = 1 if (result == "TP" or (result == "PNL" and pnl_frac > 0)) else (
            -1 if (result == "SL" or (result == "PNL" and pnl_frac < 0)) else 0
        )

    try:
        feats = json.loads(row["features"] or "{}")
        if outcome != 0:
            update_weights(feats, outcome)
    except Exception:
        pass

    return {"result": result, "pnl": float(pnl_frac), "outcome": int(outcome), "resolved_ts": time.time()}

def resolve_due_offers(limit: int = 50) -> dict:
    """
    Resolve all due offers (expires_ts <= now) that are not yet resolved.
    """
    now = time.time()
    done = 0
    good = 0
    bad  = 0

    with _db_lock:
        conn = _db()
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM offers WHERE resolved=0 AND expires_ts<=? ORDER BY expires_ts ASC LIMIT ?",
            (now, int(limit)),
        ).fetchall()

        for row in rows:
            try:
                upd = resolve_offer_row(row)
                conn.execute(
                    "UPDATE offers SET resolved=1, result=?, pnl=?, resolved_ts=? WHERE id=?",
                    (upd["result"], upd["pnl"], upd["resolved_ts"], row["id"])
                )
                # NEW: log a synthetic feedback row (only if outcome is non-neutral)
                try:
                    if int(upd.get("outcome", 0)) != 0:
                        row_map = {k: row[k] for k in row.keys()}  # sqlite3.Row -> dict
                        _record_feedback_from_offer(row_map, int(upd["outcome"]), float(upd["resolved_ts"]), pnl=float(upd["pnl"]))
                except Exception:
                    pass
                done += 1
                if upd["outcome"] > 0: good += 1
                elif upd["outcome"] < 0: bad += 1
            except Exception as e:
                # mark resolved with NEITHER on hard failure to avoid blocking forever
                conn.execute(
                    "UPDATE offers SET resolved=1, result=?, pnl=?, resolved_ts=? WHERE id=?",
                    ("NEITHER", 0.0, time.time(), row["id"])
                )
                done += 1
        conn.commit()
        conn.close()
    return {"resolved": done, "good": good, "bad": bad}

# ----- Schemas (light) -----
class Instrument(BaseModel):
    symbol: str; name: str; market: str; tf_supported: List[str]

# ----- Endpoints -----

@app.get("/healthz", include_in_schema=False)
def healthz():
    return PlainTextResponse("ok")

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
    _: None = Depends(require_key),
):
    try:
        # 1) get your base result (your existing function)
        res = evaluate_signal(
            symbol=symbol,
            tf=tf,
            risk_pct=risk_pct,
            equity=equity,
            leverage=leverage,
            min_confidence=min_confidence,
        )

        # Nudge confidence by learned weights
        base_conf = float(res.get("confidence", 0.0))
        feats: Dict[str, Any] = res.get("features") or {}
        res["confidence"] = apply_feedback_bias(base_conf, feats)

        # Keep filter in sync with the threshold if provided
        if isinstance(res.get("filters"), dict) and min_confidence is not None:
            res["filters"]["confidence_ok"] = (res["confidence"] >= float(min_confidence))

        return res

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"signals failed: {e}")

@app.get("/scan")
def scan(
    tf: str = "1h",
    limit: int = Query(TOP_N, ge=3, le=50),
    top: int = Query(6, ge=1, le=12),
    min_confidence: Optional[float] = Query(None),
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

            # Apply learned bias so ranking prefers what historically worked
            base_conf = float(s.get("confidence", 0.0))
            feats_map = s.get("features") or {}
            s["confidence"] = apply_feedback_bias(base_conf, feats_map)

            # keep the filter gate consistent if threshold provided
            if isinstance(s.get("filters"), dict) and (min_confidence is not None):
                s["filters"]["confidence_ok"] = (s["confidence"] >= float(min_confidence))

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

# ========= FEEDBACK API =========
class FeedbackIn(BaseModel):
    symbol: str
    tf: str
    market: Optional[str] = None
    direction: Optional[str] = None
    entry: Optional[float] = None
    stop: Optional[float] = None
    targets: Optional[List[float]] = None
    confidence: Optional[float] = None
    advice: Optional[str] = None
    features: Optional[Dict[str, Any]] = None
    edge: Optional[Dict[str, Any]] = None
    composite: Optional[Dict[str, Any]] = None
    equity: Optional[float] = None
    risk_pct: Optional[float] = None
    leverage: Optional[float] = None
    currency: Optional[str] = None
    outcome: int = Field(..., description="+1 good (TP/favorable), -1 bad (SL hit)")
    uid: Optional[int] = None
    fingerprint: Optional[str] = None

class FeedbackAck(BaseModel):
    ok: bool
    stored_id: Optional[int] = None
    accepted: Optional[bool] = None
    detail: Optional[str] = None

class OfferIn(BaseModel):
    symbol: str
    tf: str
    market: Optional[str] = None
    direction: Optional[str] = None
    entry: Optional[float] = None
    stop: Optional[float] = None
    targets: Optional[List[float]] = None
    confidence: Optional[float] = None
    advice: Optional[str] = None
    features: Optional[Dict[str, Any]] = None
    equity: Optional[float] = None
    risk_pct: Optional[float] = None
    leverage: Optional[float] = None
    currency: Optional[str] = None

    created_ts: float
    expires_ts: float

class OffersBatch(BaseModel):
    items: List[OfferIn]
    universe: Optional[int] = None
    note: Optional[str] = None

@app.post("/feedback", response_model=FeedbackAck, dependencies=[Depends(require_key)])
def post_feedback(fb: FeedbackIn, request: Request, x_user_id: Optional[int] = Header(None)):
    ua = request.headers.get("user-agent", "")[:300]
    ip = request.client.host if request.client else ""
    now = time.time()

    uid = x_user_id
    if not uid or uid <= 0:
        raise HTTPException(status_code=401, detail="login required")
    fb.uid = int(uid)

    # Must be logged in (WP proxy stamps this) – keep, but it’s redundant now
    if not fb.uid or fb.uid <= 0:
        raise HTTPException(status_code=401, detail="login required")

    if not fb.fingerprint:
        d = (fb.direction or "").upper()
        e = f"{fb.entry:.6f}" if fb.entry is not None else "0"
        s = f"{fb.stop:.6f}" if fb.stop is not None else "0"
        t = f"{(fb.targets or [None])[0]:.6f}" if (fb.targets and fb.targets[0] is not None) else "0"
        fb.fingerprint = "|".join([fb.symbol, fb.tf, d, e, s, t])

    ok, why = _vote_matches_market(fb)
    if not ok:
        raise HTTPException(status_code=422, detail=f"Vote rejected: {why}")

    with _db_lock:
        conn = _db()
        try:
            # per-user cooldown
            row_u = conn.execute(
                "SELECT ts FROM feedback WHERE uid=? ORDER BY ts DESC LIMIT 1",
                (int(fb.uid or 0),)
            ).fetchone()
            if row_u is not None and isinstance(row_u[0], (float, int)) and (now - row_u[0]) < 30:
                raise HTTPException(status_code=429, detail="Too fast. Please wait a moment and try again.")

            # IP-based cooldown
            row = conn.execute(
                "SELECT ts FROM feedback WHERE ip=? ORDER BY ts DESC LIMIT 1", (ip,)
            ).fetchone()
            if row is not None:
                last_ts = row[0]
                if isinstance(last_ts, (float, int)) and (now - last_ts) < 30:
                    raise HTTPException(status_code=429, detail="Feedback rate limit exceeded. Only one vote is allowed per trade.")

            # unique (uid,fingerprint)
            try:
                cur = conn.execute(
                    """INSERT INTO feedback
                       (ts, symbol, tf, market, direction, entry, stop, targets, confidence, advice,
                        outcome, features, edge, composite, equity, risk_pct, leverage, ua, ip,
                        uid, fingerprint, accepted)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)""",
                    (
                        now, fb.symbol, fb.tf, fb.market, fb.direction, fb.entry, fb.stop,
                        json.dumps(fb.targets or []), fb.confidence, fb.advice, fb.outcome,
                        json.dumps(fb.features or {}), json.dumps(fb.edge or {}),
                        json.dumps(fb.composite or {}), fb.equity, fb.risk_pct, fb.leverage,
                        ua, ip, int(fb.uid), fb.fingerprint
                    )
                )
            except sqlite3.IntegrityError:
                raise HTTPException(status_code=409, detail="Already voted for this trade")

            rid = cur.lastrowid
            conn.commit()
        finally:
            conn.close()

    try:
        update_weights(fb.features or {}, fb.outcome)
    except Exception:
        pass

    return FeedbackAck(ok=True, stored_id=rid, accepted=True)

@app.get("/feedback/summary")
def feedback_summary(symbol: str, tf: str, direction: Optional[str] = None, window: int = Query(86400, ge=1)):
    """Summarize ONLY accepted, human votes in the last `window` seconds."""
    cutoff = time.time() - float(window)
    with _db_lock:
        conn = _db()
        params = [symbol, tf, cutoff]
        where = "symbol=? AND tf=? AND accepted=1 AND uid>0 AND ts>=?"
        if direction:
            where += " AND (direction=? OR (direction IS NULL AND ?=''))"
            params.extend([direction, direction])
        rows = conn.execute(f"SELECT outcome FROM feedback WHERE {where}", params).fetchall()
        conn.close()
    up = sum(1 for r in rows if (r[0] or 0) > 0)
    down = sum(1 for r in rows if (r[0] or 0) < 0)
    total = len(rows)
    return {"up": up, "down": down, "total": total}

@app.get("/feedback/stats")
def feedback_stats():
    with _db_lock:
        conn = _db()
        total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        good  = conn.execute("SELECT COUNT(*) FROM feedback WHERE outcome=1").fetchone()[0]
        bad   = conn.execute("SELECT COUNT(*) FROM feedback WHERE outcome=-1").fetchone()[0]
        W = {k: v for k, v in conn.execute("SELECT feature, w FROM weights ORDER BY ABS(w) DESC LIMIT 24")}
        conn.close()
    return {"total": total, "good": good, "bad": bad, "weights": W}

@app.post("/learning/offered")
def learning_offered(batch: OffersBatch, request: Request, _: None = Depends(require_key)):
    if not batch.items:
        return {"ok": True, "stored": 0}

    ua = request.headers.get("user-agent", "")[:300]
    ip = request.client.host if request.client else ""

    with _db_lock:
        conn = _db()
        try:
            for it in batch.items:
                conn.execute(
                    """INSERT INTO offers
                       (created_ts, expires_ts, symbol, tf, market, direction,
                        entry, stop, targets, confidence, advice, features,
                        equity, risk_pct, leverage, currency, universe, note)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        float(it.created_ts),
                        float(it.expires_ts),
                        it.symbol, it.tf, it.market, it.direction,
                        it.entry, it.stop, json.dumps(it.targets or []),
                        it.confidence, it.advice, json.dumps(it.features or {}),
                        it.equity, it.risk_pct, it.leverage, it.currency,
                        batch.universe, batch.note,
                    )
                )
            conn.commit()
        finally:
            conn.close()

    return {"ok": True, "stored": len(batch.items)}

class ResolveOneIn(BaseModel):
    id: int

@app.post("/learning/resolve-one")
def learning_resolve_one(body: ResolveOneIn, _: None = Depends(require_key)):
    with _db_lock:
        conn = _db()
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM offers WHERE id=?", (body.id,)).fetchone()
        if not row:
            conn.close()
            raise HTTPException(404, detail="offer not found")
        upd = resolve_offer_row(row)
        conn.execute(
            "UPDATE offers SET resolved=1, result=?, pnl=?, resolved_ts=? WHERE id=?",
            (upd["result"], upd["pnl"], upd["resolved_ts"], row["id"])
        )
        # NEW: log a synthetic feedback row (only if outcome is non-neutral)
        try:
            if int(upd.get("outcome", 0)) != 0:
                row_map = {k: row[k] for k in row.keys()}  # sqlite3.Row -> dict
                _record_feedback_from_offer(row_map, int(upd["outcome"]), float(upd["resolved_ts"]), pnl=float(upd["pnl"]))
        except Exception:
            pass  # never block resolution on logging
        conn.commit()
        conn.close()
    return {"ok": True, "id": body.id, **upd}

@app.post("/learning/resolve-due")
def learning_resolve_due(limit: int = Query(50, ge=1, le=500), _: None = Depends(require_key)):
    return resolve_due_offers(limit=limit)

@app.get("/learning/stats")
def learning_stats(_: None = Depends(require_key)):
    with _db_lock:
        conn = _db()
        pending = conn.execute("SELECT COUNT(*) FROM offers WHERE resolved=0").fetchone()[0]
        done    = conn.execute("SELECT COUNT(*) FROM offers WHERE resolved=1").fetchone()[0]
        tp      = conn.execute("SELECT COUNT(*) FROM offers WHERE resolved=1 AND result='TP'").fetchone()[0]
        sl      = conn.execute("SELECT COUNT(*) FROM offers WHERE resolved=1 AND result='SL'").fetchone()[0]
        pnlp    = conn.execute("SELECT COUNT(*) FROM offers WHERE resolved=1 AND result='PNL' AND pnl>0").fetchone()[0]
        pnln    = conn.execute("SELECT COUNT(*) FROM offers WHERE resolved=1 AND result='PNL' AND pnl<=0").fetchone()[0]
        conn.close()
    return {"offers": {"pending": pending, "resolved": done, "tp": tp, "sl": sl, "pnl_pos": pnlp, "pnl_neg": pnln}}

@app.get("/learning/config")
def learning_config(_: None = Depends(require_key)):
    """Small helper so the frontend knows how long to ‘hold’ before resolution."""
    return {
        "hold_secs": {"1h": HOLD_1H_SECS, "1d": HOLD_1D_SECS},
        "bg_interval": BG_INTERVAL,
        "eval_stop_first": bool(EVAL_STOP_FIRST),
    }

# ---- Learning: record a feedback row from an offer resolution ----
def _record_feedback_from_offer(offer: Dict[str, Any], outcome: int, resolved_ts: float, pnl: Optional[float] = None):
    """Insert a synthetic feedback row so /feedback/stats includes auto-resolutions."""
    try:
        feats = json.loads(offer.get("features") or "{}")
    except Exception:
        feats = {}

    try:
        targets = json.loads(offer.get("targets") or "[]")
    except Exception:
        targets = []

    ua = "learning/auto-resolve"
    ip = "127.0.0.1"
    with _db_lock:
        conn = _db()
        conn.execute(
            """INSERT INTO feedback
               (ts, symbol, tf, market, direction, entry, stop, targets,
                confidence, advice, outcome, features, edge, composite, equity, risk_pct, leverage, ua, ip)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                resolved_ts,
                offer.get("symbol"),
                offer.get("tf"),
                offer.get("market"),
                offer.get("direction"),
                offer.get("entry"),
                offer.get("stop"),
                json.dumps(targets),
                offer.get("confidence"),
                offer.get("advice"),
                outcome,
                json.dumps(feats),
                "{}", "{}",  # edge, composite (not used here)
                offer.get("equity"),
                offer.get("risk_pct"),
                offer.get("leverage"),
                ua, ip
            )
        )
        conn.commit()
        conn.close()
# =================================