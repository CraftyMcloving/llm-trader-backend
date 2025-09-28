# main.py — AI Trade Advisor v3.1.0
from __future__ import annotations

# ✅ keep imports first
import os
import json
import sqlite3
import threading
import time
import math

from typing import List, Optional, Dict, Any, Tuple
from fastapi import FastAPI, HTTPException, Depends, Header, Query, Request
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from adapters import CryptoCCXT

import pandas as pd
import numpy as np
import ccxt

# ✅ only now is it safe to read env vars
MARKET   = os.getenv("MARKET", "crypto")
EXCHANGE = os.getenv("EXCHANGE", "kraken")
QUOTE    = os.getenv("QUOTE", "USD")

app = FastAPI(title="AI Trade Advisor API", version="2025.09")

origins_env = os.getenv("ALLOWED_ORIGINS", "*")
origins = ["*"] if origins_env == "*" else [o.strip() for o in origins_env.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ----- Security -----
API_KEY = os.getenv("API_KEY", "change-me")
def require_key(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    api_key_q: Optional[str] = Query(None),
):
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
    elif x_api_key:       # fallback if proxy strips Authorization
        token = x_api_key.strip()
    elif api_key_q:       # last-resort fallback (avoid using in URLs routinely)
        token = api_key_q.strip()

    if not token:
        raise HTTPException(status_code=401, detail="Missing token")
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


# ========= FEEDBACK STORAGE & ONLINE WEIGHTS =========
DB_PATH = os.getenv("FEEDBACK_DB", "/tmp/ai_trade_feedback.db")

# Ensure parent dir exists (important when FEEDBACK_DB=/var/data/... on Render Disk)
try:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
except Exception:
    pass

_db_lock = threading.Lock()

def _db():
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    except Exception:
        pass
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
        CREATE TABLE IF NOT EXISTS meta(
        k TEXT PRIMARY KEY,
        v TEXT
        );
        CREATE TABLE IF NOT EXISTS tracked(
          uid INTEGER NOT NULL,
          id  TEXT PRIMARY KEY,
          item_json  TEXT NOT NULL,
          created_at_ms INTEGER NOT NULL,
          expires_at_ms INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_tracked_uid ON tracked(uid);
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_offers_symbol_tf ON offers(symbol, tf);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_offers_created ON offers(created_ts);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_offers_resolved ON offers(resolved, expires_ts);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_offers_exp ON offers(expires_ts);")
        conn.commit()
        conn.close()

def meta_get(key:str) -> Optional[str]:
    with _db_lock:
        conn = _db()
        row = conn.execute("SELECT v FROM meta WHERE k=?", (key,)).fetchone()
        conn.close()
    return row[0] if row else None

def meta_set(key:str, val:str):
    with _db_lock:
        conn = _db()
        conn.execute("INSERT OR REPLACE INTO meta(k,v) VALUES(?,?)", (key, val))
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
    
def ensure_weights2(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS weights2(
          scope   TEXT NOT NULL,
          feature TEXT NOT NULL,
          w       REAL NOT NULL,
          PRIMARY KEY(scope, feature)
        )
    """)
    # One-time migration from old global weights → 'global'
    have_old = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='weights'").fetchone()
    if have_old:
        for (feat, w) in conn.execute("SELECT feature, w FROM weights"):
            conn.execute(
                "INSERT OR IGNORE INTO weights2(scope, feature, w) VALUES(?,?,?)",
                ("global", feat, float(w))
            )
    conn.commit()

def get_weights() -> Dict[str, float]:
    with _db_lock:
        conn = _db()
        cur = conn.execute("SELECT feature, w FROM weights")
        d = {k: float(v) for k, v in cur.fetchall()}
        conn.close()
        return d

def update_weights(
    features: Dict[str, Any],
    outcome: int,
    tf: Optional[str] = None,
    symbol: Optional[str] = None,
    lr: float = 0.05,
    l2: float = 1e-5
):
    """
    Online logistic regression on the most specific scope.
    outcome: +1 good → y=1, -1 bad → y=0
    """
    if not isinstance(features, dict):
        return
    y = 1.0 if (int(outcome) > 0) else 0.0
    x = _norm_feat_map(features)
    scopes = _scopes_for(tf, symbol)
    scope = scopes[0]  # most specific first
    W = _read_weights(scopes)

    # forward
    z = sum(float(W.get(k, 0.0)) * float(v) for k, v in x.items())
    p = _sigmoid(z)
    g = (y - p)  # gradient for log-loss

    # SGD step with tiny L2 regularization
    for k, xv in x.items():
        w = float(W.get(k, 0.0))
        w_new = w + lr * (g * float(xv) - l2 * w)
        _upsert_weight(scope, k, w_new)

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
    
def _sigmoid(z: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-float(z)))
    except Exception:
        return 0.5

def _norm_feat_map(features: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {"__bias__": 1.0}
    for k, v in (features or {}).items():
        if v is None:
            continue
        try:
            x = float(v)
        except Exception:
            continue
        kk = str(k)
        if "rsi" in kk:          x = (x - 50.0) / 50.0
        elif ("macd" in kk) or ("diff" in kk) or ("ret" in kk) or ("bb_pos" in kk) or ("donch" in kk):
            x = max(-2.0, min(2.0, x))
        elif "atr" in kk:
            x = max(-1.0, min(1.0, x))
        else:
            x = max(-3.0, min(3.0, x))
        out[kk] = x
    return out

def _scopes_for(tf: Optional[str] = None, symbol: Optional[str] = None) -> list[str]:
    scopes = ["global"]
    if tf:     scopes.insert(0, f"tf:{tf}")
    if symbol: scopes.insert(0, f"sym:{symbol}")
    return scopes

def _read_weights(scopes: list[str]) -> Dict[str, float]:
    W: Dict[str, float] = {}
    with _db_lock:
        conn = _db()
        try:
            # less specific last → most specific overwrite
            for sc in reversed(scopes):
                for (f, w) in conn.execute("SELECT feature, w FROM weights2 WHERE scope=?", (sc,)):
                    W[f] = float(w)
        finally:
            conn.close()
    return W

def _upsert_weight(scope: str, feature: str, w_new: float):
    with _db_lock:
        conn = _db()
        try:
            conn.execute(
                "INSERT INTO weights2(scope, feature, w) VALUES(?,?,?) "
                "ON CONFLICT(scope, feature) DO UPDATE SET w=excluded.w",
                (scope, feature, float(w_new))
            )
            conn.commit()
        finally:
            conn.close()

def predict_prob(features: Dict[str, Any], tf: Optional[str] = None, symbol: Optional[str] = None) -> float:
    x = _norm_feat_map(features)
    W = _read_weights(_scopes_for(tf, symbol))
    z = 0.0
    for k, xv in x.items():
        z += float(W.get(k, 0.0)) * float(xv)
    return _sigmoid(z)

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
            ensure_weights2(conn)
        finally:
            conn.close()

# =====================================================

def apply_feedback_bias(
    confidence: float,
    features: Dict[str, Any],
    tf: Optional[str] = None,
    symbol: Optional[str] = None,
    beta: float = 0.35
) -> float:
    """
    Blend base model confidence with learned probability p.
    beta controls influence (0..1).
    """
    try:
        base = float(confidence)
        p = predict_prob(features or {}, tf=tf, symbol=symbol)
        return max(0.0, min(1.0, (1.0 - beta) * base + beta * float(p)))
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

CURATED = [
    "BTC","ETH","XRP","SOL","ADA","DOGE","LINK","LTC","BCH","TRX",
    "DOT","ATOM","XLM","ETC","MATIC","UNI","APT","ARB","OP","AVAX",
    "NEAR","ALGO","FIL","SUI","SHIB","USDC","USDT","XMR","AAVE"
    ,"PAXG","ONDO","PEPE","SEI","IMX","TIA"
]

# ----- Universe -----
def get_universe(quote=QUOTE, limit=TOP_N) -> List[Dict[str, Any]]:
    # Crypto adapter (current behavior)
    curated = [
        "BTC","ETH","XRP","SOL","ADA","DOGE","LINK","LTC","BCH","TRX",
        "DOT","ATOM","XLM","ETC","MATIC","UNI","APT","ARB","OP","AVAX",
        "NEAR","ALGO","FIL","SUI","SHIB","USDC","USDT","XMR","AAVE","PAXG","ONDO","PEPE","SEI","IMX","TIA"
    ]
    adapter = CryptoCCXT(exchange_id=os.getenv("EXCHANGE","kraken"), quote=quote, curated=curated)
    key = f"uni:{adapter.name()}:{limit}"
    u = cache_get(key, 1800)
    if u is not None: return u
    out = adapter.list_universe(limit)
    cache_set(key, out)
    return out

# ----- Market data -----
TF_MAP = {"5m":"5m","15m":"15m","1h":"1h","1d":"1d"}

def fetch_ohlcv(symbol: str, tf: str, bars: int = 720) -> pd.DataFrame:
    adapter = CryptoCCXT(exchange_id=os.getenv("EXCHANGE","kraken"), quote=QUOTE)
    tf_ex = TF_MAP.get(tf)
    if tf_ex is None:
        raise HTTPException(400, detail=f"Unsupported timeframe: {tf}")
    key = f"ohlcv:{adapter.name()}:{symbol}:{tf_ex}:{bars}"
    cached = cache_get(key, 900)
    if cached is not None:
        return cached.copy()
    df = adapter.fetch_ohlcv(symbol, tf_ex, bars=bars)
    if df.empty:
        raise HTTPException(502, detail="no candles")
    cache_set(key, df)
    return df.copy()

# ms per bar
def tf_ms(tf: str) -> int:
    if tf == "5m":  return 5*60*1000
    if tf == "15m": return 15*60*1000
    if tf == "1h":  return 60*60*1000
    if tf == "1d":  return 24*60*60*1000
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
    
def _mtf_gate(symbol: str, primary_direction: str) -> Dict[str, Any]:
    """
    Multi-timeframe gate:
    - Confirms 1h signal with 15m alignment.
    - Produces 5m management hint (tighten/abort/none) without blocking.
    Simple heuristic, fast and stable.
    """
    try:
        df15 = compute_features(fetch_ohlcv(symbol, "15m", bars=400)).dropna().iloc[-120:]
        df5  = compute_features(fetch_ohlcv(symbol, "5m",  bars=400)).dropna().iloc[-120:]
    except HTTPException:
        # If lower TFs unavailable, be permissive but mark it
        return {"has": False, "confirm15m": True, "reason15m": "lower TF unavailable", "manage5m": "none"}

    def _side_ok(feats, direction):
        slope = float(feats["slope20"].iloc[-1])
        rsi14 = float(feats["rsi14"].iloc[-1])
        bbpos = float(feats["bb_pos"].iloc[-1])
        if direction == "Long":
            return (slope >= 0) and (rsi14 >= 45) and (bbpos >= -0.15)
        else:
            return (slope <= 0) and (rsi14 <= 55) and (bbpos <= 0.15)

    confirm = _side_ok(df15, primary_direction)

    # 5m hint
    slope5 = float(df5["slope20"].iloc[-1])
    rsi5   = float(df5["rsi14"].iloc[-1])
    if primary_direction == "Long":
        if (slope5 < 0 and rsi5 < 40):   hint = "abort"
        elif (slope5 < 0 and rsi5 < 47): hint = "tighten"
        else:                            hint = "none"
    else:
        if (slope5 > 0 and rsi5 > 60):   hint = "abort"
        elif (slope5 > 0 and rsi5 > 53): hint = "tighten"
        else:                            hint = "none"

    return {
        "has": True,
        "confirm15m": bool(confirm),
        "reason15m": "15m aligned" if confirm else "15m contradicts",
        "manage5m": hint,
        "checks": {
            "m15": {"slope20": float(df15["slope20"].iloc[-1]), "rsi14": float(df15["rsi14"].iloc[-1]), "bb_pos": float(df15["bb_pos"].iloc[-1])},
            "m5":  {"slope20": float(df5["slope20"].iloc[-1]),  "rsi14": float(df5["rsi14"].iloc[-1])}
        }
    }

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
    mtf = {"has": False}  # <-- ensure defined in all paths

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

        # --- MTF gate (1h -> confirm on 15m; 5m manage hint) ---
        if tf == "1h" and direction:
            mtf = _mtf_gate(symbol, direction)
            if mtf.get("has") and not mtf.get("confirm15m", True):
                # veto entry; degrade to "Wait"
                trade = None
                advice = "Wait"
                if "reasons" in filt:
                    filt["reasons"] = (filt.get("reasons") or []) + ["15m contradicts"]

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
        "mtf": mtf,
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
            update_weights(feats, outcome, tf=tf, symbol=symbol)
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

@app.middleware("http")
async def _log_requests(request, call_next):
    if request.url.path in ("/healthz", "/health"):
        print(f"HEALTH HIT: {request.method} {request.url.path}")
    return await call_next(request)

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
        # 1) base evaluation
        res = evaluate_signal(
            symbol=symbol,
            tf=tf,
            risk_pct=risk_pct,
            equity=equity,
            leverage=leverage,
            min_confidence=min_confidence,
        )

        # 2) learned bias nudge
        base_conf = float(res.get("confidence", 0.0))
        feats: Dict[str, Any] = res.get("features") or {}
        res["confidence"] = apply_feedback_bias(base_conf, feats, tf=tf, symbol=symbol)

        # 3) optional calibration (piecewise linear through saved bins)
        try:
            cal = meta_get("calibration_map")
            if cal:
                import bisect, json as _json
                bins = _json.loads(cal)
                xs = [float(a) for a, _ in bins]
                ys = [float(b) for _, b in bins]
                x = float(res["confidence"])
                if x <= xs[0]:
                    res["confidence"] = ys[0]
                elif x >= xs[-1]:
                    res["confidence"] = ys[-1]
                else:
                    i = bisect.bisect_left(xs, x)
                    x0, x1 = xs[i-1], xs[i]
                    y0, y1 = ys[i-1], ys[i]
                    t = (x - x0) / max(1e-9, (x1 - x0))
                    res["confidence"] = (1 - t) * y0 + t * y1
        except Exception:
            pass

        # 4) keep filter gate consistent if client passed a threshold
        if isinstance(res.get("filters"), dict) and (min_confidence is not None):
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
            s["confidence"] = apply_feedback_bias(base_conf, feats_map, tf=tf, symbol=s["symbol"])

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

    uid = int(x_user_id or 0)
    if uid <= 0 and fb.uid:          # accept body fallback if header got dropped
        uid = int(fb.uid or 0)

    # Must be logged in (WP proxy stamps this) – keep, but it’s redundant now
    if uid <= 0:
        raise HTTPException(status_code=401, detail="login required")

    fb.uid = uid  # canonicalize

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
        update_weights(fb.features or {}, fb.outcome, tf=fb.tf, symbol=fb.symbol)
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
        try:
            total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
            good  = conn.execute("SELECT COUNT(*) FROM feedback WHERE outcome=1").fetchone()[0]
            bad   = conn.execute("SELECT COUNT(*) FROM feedback WHERE outcome=-1").fetchone()[0]
            # Prefer weights2 aggregation (magnitude of influence); fall back to legacy weights
            try:
                W = {k: v for k, v in conn.execute(
                    "SELECT feature, SUM(ABS(w)) AS mag FROM weights2 "
                    "GROUP BY feature ORDER BY mag DESC LIMIT 24"
                )}
            except Exception:
                W = {k: v for k, v in conn.execute(
                    "SELECT feature, w FROM weights ORDER BY ABS(w) DESC LIMIT 24"
                )}
        finally:
            conn.close()
    return {"total": total, "good": good, "bad": bad, "weights": W}

@app.get("/feedback/stats_series")
def feedback_stats_series(window: str = Query("30d"), bucket: str = Query("day")):
    """Rolling stats series for sparkline etc."""
    secs = _parse_window_to_seconds(window)
    cutoff = time.time() - secs
    with _db_lock:
        conn = _db()
        rows = conn.execute(
            "SELECT ts, outcome FROM feedback WHERE accepted=1 AND ts>=? ORDER BY ts ASC",
            (cutoff,)
        ).fetchall()
        conn.close()
    if not rows:
        return {"by_day": []}
    # bucket by day UTC
    by = {}
    for ts, outcome in rows:
        d = time.strftime("%Y-%m-%d", time.gmtime(float(ts)))
        x = by.setdefault(d, {"good":0,"bad":0,"n":0})
        if (outcome or 0) > 0: x["good"] += 1
        if (outcome or 0) < 0: x["bad"]  += 1
        x["n"] += 1
    series = []
    for d in sorted(by.keys()):
        g = by[d]["good"]; b = by[d]["bad"]; n = by[d]["n"]
        wr = (g / n) if n else 0.0
        series.append({"date": d, "good": g, "bad": b, "n": n, "win_rate": wr})
    return {"by_day": series}

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

# ========= TRACKED TRADES (for mobile app) =========
from fastapi import Body

def _uid_from_header(x_user_id: Optional[int]) -> int:
    uid = int(x_user_id or 0)
    if uid <= 0:
        raise HTTPException(401, detail="login required (X-User-Id)")
    return uid

def _tracked_list_for(uid: int):
    with _db_lock:
        conn = _db()
        now_ms = int(time.time() * 1000)
        # prune expired
        conn.execute("DELETE FROM tracked WHERE uid=? AND expires_at_ms<=?", (uid, now_ms))
        rows = conn.execute(
            "SELECT id, item_json, created_at_ms, expires_at_ms "
            "FROM tracked WHERE uid=? ORDER BY created_at_ms DESC",
            (uid,)
        ).fetchall()
        conn.close()

    out = []
    for (id_, js, ca, ea) in rows:
        try:
            item = json.loads(js)
        except Exception:
            item = None
        out.append({
            "id": id_, "item": item,
            "createdAt": ca, "expiresAt": ea
        })
    return out

@app.get("/tracked")
def tracked_get(x_user_id: Optional[int] = Header(None), _: None = Depends(require_key)):
    uid = _uid_from_header(x_user_id)
    return _tracked_list_for(uid)

class TrackIn(BaseModel):
    item: Dict[str, Any]
    expiresInMs: Optional[int] = None

@app.post("/tracked")
def tracked_post(body: TrackIn, x_user_id: Optional[int] = Header(None), _: None = Depends(require_key)):
    uid = _uid_from_header(x_user_id)
    now_ms = int(time.time() * 1000)
    expires_ms = now_ms + int(body.expiresInMs or 0)

    symbol = (body.item or {}).get("symbol", "NA")
    tf     = (body.item or {}).get("timeframe") or (body.item or {}).get("tf") or "1h"
    new_id = f"{symbol}|{tf}|{now_ms}"

    with _db_lock:
        conn = _db()
        # prune first
        conn.execute("DELETE FROM tracked WHERE uid=? AND expires_at_ms<=?", (uid, now_ms))
        # insert newest on top
        conn.execute(
            "INSERT OR REPLACE INTO tracked(uid, id, item_json, created_at_ms, expires_at_ms) "
            "VALUES(?,?,?,?,?)",
            (uid, new_id, json.dumps(body.item or {}), now_ms, expires_ms)
        )
        # enforce max 3: keep newest 3
        rows = conn.execute(
            "SELECT id FROM tracked WHERE uid=? ORDER BY created_at_ms DESC",
            (uid,)
        ).fetchall()
        if len(rows) > 3:
            for r in rows[3:]:
                conn.execute("DELETE FROM tracked WHERE uid=? AND id=?", (uid, r[0]))
        conn.commit()
        conn.close()

    return _tracked_list_for(uid)

@app.delete("/tracked")
def tracked_delete(id: Optional[str] = None, x_user_id: Optional[int] = Header(None), req: Request = None, _: None = Depends(require_key)):
    uid = _uid_from_header(x_user_id)
    # accept id in query or JSON body
    if not id:
        try:
            data = req.json() if hasattr(req, "json") else None
        except Exception:
            data = None
        if data and isinstance(data, dict):
            id = data.get("id")
    if not id:
        raise HTTPException(400, detail="id required")
    with _db_lock:
        conn = _db()
        conn.execute("DELETE FROM tracked WHERE uid=? AND id=?", (uid, id))
        conn.commit()
        conn.close()
    return _tracked_list_for(uid)

@app.delete("/tracked/{id}")
def tracked_delete_path(id: str, x_user_id: Optional[int] = Header(None), _: None = Depends(require_key)):
    uid = _uid_from_header(x_user_id)
    with _db_lock:
        conn = _db()
        conn.execute("DELETE FROM tracked WHERE uid=? AND id=?", (uid, id))
        conn.commit()
        conn.close()
    return _tracked_list_for(uid)
    
@app.post("/calibration/rebuild")
def calibration_rebuild(min_samples:int=200, bins:int=10, _: None = Depends(require_key)):
    """Compute a simple isotonic-like reliability mapping."""
    with _db_lock:
        conn = _db()
        rows = conn.execute("""
          SELECT confidence, outcome
          FROM feedback
          WHERE accepted=1 AND confidence IS NOT NULL
          ORDER BY ts DESC
          LIMIT 5000
        """).fetchall()
        conn.close()
    if not rows or len(rows) < min_samples:
        raise HTTPException(400, detail="not enough samples")

    pairs = [(float(c), 1 if (o or 0)>0 else 0) for c,o in rows if c is not None]
    if len(pairs) < min_samples:
        raise HTTPException(400, detail="not enough samples")

    # bin by predicted conf
    pairs.sort(key=lambda x: x[0])
    n = max(3, bins)
    out = []
    for i in range(n):
        lo = int(i*len(pairs)/n); hi = int((i+1)*len(pairs)/n)
        chunk = pairs[lo:hi] or []
        if not chunk: continue
        p_hat = sum(1 for _,y in chunk if y>0) / len(chunk)
        c_mid = sum(c for c,_ in chunk)/len(chunk)
        out.append([c_mid, p_hat])

    # enforce monotonicity (isotonic-like via running max)
    monot = []
    m = 0.0
    for c,ph in out:
        m = max(m, ph)
        monot.append([c, m])

    meta_set("calibration_map", json.dumps(monot))
    return {"bins": monot, "n": len(pairs)}

# ---- Learning: record a feedback row from an offer resolution ----
def _record_feedback_from_offer(offer: Dict[str, Any], outcome: int, resolved_ts: float, pnl: Optional[float] = None):
    """Insert a synthetic feedback row so /feedback/stats includes auto-resolutions, then learn."""
    # Parse fields safely
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

    # Write the feedback row (inside lock)
    with _db_lock:
        conn = _db()
        try:
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
                    int(outcome),
                    json.dumps(feats),
                    "{}", "{}",  # edge, composite (unused here)
                    offer.get("equity"),
                    offer.get("risk_pct"),
                    offer.get("leverage"),
                    ua, ip,
                )
            )
            conn.commit()
        finally:
            conn.close()

    # Learn from the outcome (outside lock to avoid deadlocks)
    try:
        update_weights(feats, int(outcome), tf=offer.get("tf"), symbol=offer.get("symbol"))
    except Exception:
        pass
# =================================