# main.py — AI Trade Advisor backend v3.1.0 (Kraken/USD)
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

# NEW: external adapters
from adapters import get_adapter, BaseAdapter, UniverseItem, AdapterError

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
TOP_N = int(os.getenv("TOP_N", "18"))

_ex = None
def get_exchange():
    global _ex
    if _ex is None:
        try:
            import ccxt as _ccxt  # import here to avoid ImportError at module load time
        except Exception:
            # Surface a clear error so callers know why the exchange features are unavailable
            raise HTTPException(status_code=500, detail="ccxt is not installed. Install ccxt to use the exchange features.")
        # Look up the exchange class dynamically; fail early if unsupported
        exchange_cls = getattr(_ccxt, EXCHANGE_ID, None)
        if not exchange_cls:
            raise HTTPException(status_code=500, detail=f"Exchange '{EXCHANGE_ID}' not supported by ccxt.")
        _ex = exchange_cls({
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
DB_PATH = os.getenv("FEEDBACK_DB", "ai_trade_feedback.db")
_db_lock = threading.RLock()

def _db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA mmap_size=300000000;")   # ~300MB if available (no-op if not)
    conn.execute("PRAGMA cache_size=-20000;")     # ~20MB page cache
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
        # ---- model tables ----
    cur.execute("""
      CREATE TABLE IF NOT EXISTS calibrations(
        tf TEXT PRIMARY KEY,
        a REAL NOT NULL,
        b REAL NOT NULL,
        updated_ts REAL NOT NULL
      )
    """)
    cur.execute("""
      CREATE TABLE IF NOT EXISTS weights_ns(
        namespace TEXT NOT NULL,
        feature   TEXT NOT NULL,
        w         REAL NOT NULL,
        PRIMARY KEY(namespace, feature)
      )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_weights_ns_ns ON weights_ns(namespace)")
    conn.commit()
        # ---- per-namespace (symbol|tf) performance counters ----
    cur.execute("""
      CREATE TABLE IF NOT EXISTS ns_stats(
        ns TEXT PRIMARY KEY,
        wins INTEGER NOT NULL DEFAULT 0,
        total INTEGER NOT NULL DEFAULT 0,
        updated_ts REAL NOT NULL
      )
    """)
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

def feedback_bias(features: Dict[str, Any], symbol: str = "", tf: str = "", alpha: float = 0.15) -> float:
    ws = get_weights_ns(symbol, tf) if (symbol and tf) else get_weights()
    s = 0.0
    for k, v in (features or {}).items():
        try: x = float(v)
        except: continue
        if "rsi" in k:          x = (x - 50.0) / 50.0
        elif any(t in k for t in ("macd","diff","ret","bb_pos","donch")): x = max(-2.0, min(2.0, x))
        elif "atr" in k:        x = max(-1.0, min(1.0, x))
        else:                   x = max(-3.0, min(3.0, x))
        w = ws.get(k, 0.0)
        s += w * x
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

def apply_feedback_bias(confidence: float, features: Dict[str, Any], symbol: str = "", tf: str = "") -> float:
    try:
        return max(-1.0, min(1.0, float(confidence) + feedback_bias(features, symbol, tf)))
    except Exception:
        return float(confidence)

# ====== Platt calibration (per timeframe) ======
CALIBRATION_ENABLED = bool(int(os.getenv("CALIBRATION_ENABLED", "1")))
CALIBRATION_MIN_SAMPLES = int(os.getenv("CALIBRATION_MIN_SAMPLES", "60"))
CALIBRATION_STALENESS_SEC = int(os.getenv("CALIBRATION_STALENESS_SEC", "21600"))  # 6h

def _sigmoid(x): return 1.0/(1.0+math.exp(-x))
def _logit(p):
    p = min(max(p, 1e-6), 1-1e-6)
    return math.log(p/(1.0-p))

def _fit_platt(z, y, iters=400, lr=0.05):
    # minimize log-loss of sigmoid(a*z + b) vs labels y in {0,1}
    a, b = 1.0, 0.0
    for _ in range(iters):
        s = a*z + b
        p = 1.0/(1.0+np.exp(-s))
        grad_a = float(((p - y) * z).mean())
        grad_b = float((p - y).mean())
        a -= lr * grad_a
        b -= lr * grad_b
    return float(a), float(b)

def _load_calib(tf: str):
    with _db_lock:
        conn = _db()
        row = conn.execute("SELECT a,b,updated_ts FROM calibrations WHERE tf=?", (tf,)).fetchone()
        conn.close()
    return (float(row[0]), float(row[1]), float(row[2])) if row else (None,None,None)

def _save_calib(tf: str, a: float, b: float):
    with _db_lock:
        conn = _db()
        conn.execute("INSERT INTO calibrations(tf,a,b,updated_ts) VALUES(?,?,?,?) "
                     "ON CONFLICT(tf) DO UPDATE SET a=excluded.a, b=excluded.b, updated_ts=excluded.updated_ts",
                     (tf, float(a), float(b), time.time()))
        conn.commit()
        conn.close()

def _maybe_fit_calibration(tf: str):
    a,b,ts = _load_calib(tf)
    if a is not None and (time.time() - ts) < CALIBRATION_STALENESS_SEC:
        return a,b
    # gather recent accepted votes w/ confidence
    with _db_lock:
        conn = _db()
        rows = conn.execute(
            "SELECT confidence, outcome FROM feedback "
            "WHERE tf=? AND accepted=1 AND confidence IS NOT NULL AND outcome IN (1,-1) "
            "AND ts >= ? ORDER BY ts DESC LIMIT 1000",
            (tf, time.time() - 30*24*3600)
        ).fetchall()
        conn.close()
    if not rows or len(rows) < CALIBRATION_MIN_SAMPLES:
        # sensible default (identity)
        _save_calib(tf, 1.0, 0.0)
        return 1.0, 0.0
    p = np.array([min(max(float(r[0]), 1e-6), 1-1e-6) for r in rows], dtype=float)
    y = np.array([1 if int(r[1])>0 else 0 for r in rows], dtype=float)
    z = np.array([_logit(x) for x in p], dtype=float)
    # standardize z for stable fit
    zm = z.mean(); zs = z.std() or 1.0
    z_std = (z - zm)/zs
    a, b = _fit_platt(z_std, y)
    # store parameters to be used with standardized z
    _save_calib(tf, a/ max(zs,1e-9), b - a*zm/ max(zs,1e-9))
    return _load_calib(tf)[:2]

def apply_calibration(tf: str, p: float) -> float:
    if not CALIBRATION_ENABLED:
        return float(p)
    try:
        a,b = _maybe_fit_calibration(tf)
        z = _logit(float(p))
        return float(max(0.0, min(1.0, _sigmoid(a*z + b))))
    except Exception:
        return float(p)

# ====== Symbol-aware weights ======
def _ns(symbol: str, tf: str) -> str:
    return f"{symbol}|{tf}"

    # Build weights in order of increasing specificity (global → timeframe-level → symbol-level),
    # allowing later assignments to override earlier ones.  We always include global weights
    # so that features missing at the namespace level fall back to a sensible base.
    with _db_lock:
        conn = _db()
        # Global base weights
        for k, v in conn.execute("SELECT feature, w FROM weights"):
            W[k] = float(v)
        # Timeframe-level overrides (e.g. "*|1h")
        for k, v in conn.execute("SELECT feature, w FROM weights_ns WHERE namespace=?", (ns_tf,)):
            W[k] = float(v)
        # Symbol-specific overrides (e.g. "BTC/USD|1h")
        for k, v in conn.execute("SELECT feature, w FROM weights_ns WHERE namespace=?", (ns_sym,)):
            W[k] = float(v)
        conn.close()
    return W

def update_weights_ns(features: Dict[str, Any], outcome: int, symbol: str, tf: str, lr: float = 0.05):
    if not isinstance(features, dict): return
    norm: Dict[str, float] = {}
    for k, v in features.items():
        if v is None: continue
        try: x = float(v)
        except: continue
        if "rsi" in k: x = (x - 50.0) / 50.0
        elif any(t in k for t in ("macd","diff","ret","bb_pos","donch")): x = max(-2.0,min(2.0,x))
        elif "atr" in k: x = max(-1.0, min(1.0, x))
        else: x = max(-3.0, min(3.0, x))
        norm[k] = x
    if not norm: return
    with _db_lock:
        conn = _db()
        for ns in (_ns(symbol, tf), f"*|{tf}"):
            for k, x in norm.items():
                row = conn.execute("SELECT w FROM weights_ns WHERE namespace=? AND feature=?", (ns,k)).fetchone()
                w = float(row[0]) if row else 0.0
                w = w + lr * outcome * x
                conn.execute("INSERT INTO weights_ns(namespace, feature, w) VALUES(?,?,?) "
                             "ON CONFLICT(namespace,feature) DO UPDATE SET w=excluded.w",
                             (ns, k, w))
        conn.commit()
        conn.close()

def ns_stats_get(symbol: str, tf: str) -> Tuple[int,int]:
    with _db_lock:
        conn = _db()
        row = conn.execute("SELECT wins,total FROM ns_stats WHERE ns=?", (_ns(symbol, tf),)).fetchone()
        conn.close()
    if not row: return (0,0)
    return (int(row[0]), int(row[1]))

def ns_stats_update(symbol: str, tf: str, outcome: int):
    """
    Incrementally update per-namespace win/total counters.
    This uses a single SQL statement with arithmetic increments to avoid
    reading the existing row under the same lock (which can deadlock
    with non-reentrant locks).  The update inserts a row if missing,
    otherwise adds to the existing counts.
    """
    if outcome == 0:
        return
    inc_win = 1 if outcome > 0 else 0
    inc_total = 1
    now_ts = time.time()
    with _db_lock:
        conn = _db()
        conn.execute(
            """
            INSERT INTO ns_stats(ns, wins, total, updated_ts) VALUES(?, ?, ?, ?)
            ON CONFLICT(ns) DO UPDATE SET
              wins = wins + ?,
              total = total + ?,
              updated_ts = excluded.updated_ts
            """,
            (_ns(symbol, tf), inc_win, inc_total, now_ts, inc_win, inc_total)
        )
        conn.commit()
        conn.close()

def wilson_lb(wins: int, total: int, z: float = 1.96) -> float:
    if total <= 0: return 0.0
    p = wins / total
    denom = 1 + z*z/total
    centre = p + z*z/(2*total)
    adj = z * math.sqrt((p*(1-p) + z*z/(4*total))/total)
    lb = (centre - adj) / denom
    return max(0.0, min(1.0, lb))

def _auto_adapter(symbol: str, market_name: Optional[str]):
    # honor explicit market (normalized by adapters.get_adapter)
    if market_name:
        return get_adapter(market_name)

    # heuristic routing when market is omitted
    s = symbol.upper()
    if "/" in s:                # e.g. BTC/USD, ETH/USDT
        return get_adapter("crypto")
    if s.endswith("=X"):        # e.g. EURUSD=X, USDJPY=X
        return get_adapter("forex")
    if s.endswith("=F"):        # e.g. GC=F, CL=F
        return get_adapter("commodities")
    # default guess for single-word tickers: stocks
    return get_adapter("stocks")

# ----- Cache -----
CACHE: Dict[str, Tuple[float, Any]] = {}
def cache_get(key, ttl):
    v = CACHE.get(key)
    if not v: return None
    ts, data = v
    return data if (time.time() - ts) <= ttl else None
def cache_set(key, data): CACHE[key] = (time.time(), data)

def get_universe(quote=QUOTE, limit=TOP_N, market_name: Optional[str] = None) -> List[Dict[str, Any]]:
    ad = get_adapter(market_name)

    # adapter key for cache
    ad_name_attr = getattr(ad, "name", None)
    if callable(ad_name_attr):
        adapter_key = ad_name_attr()
    elif isinstance(ad_name_attr, str) and ad_name_attr:
        adapter_key = ad_name_attr
    else:
        adapter_key = ad.__class__.__name__

    # cache key should reflect the effective (clamped) limit
    lim = min(max(6, limit), 50)
    key = f"uni:{adapter_key}:{lim}"
    u = cache_get(key, 1800)
    if u is not None:
        return u

    # tolerate different adapter signatures
    try:
        items = ad.list_universe(limit=lim)
    except TypeError:
        try:
            items = ad.list_universe(top=lim)
        except TypeError:
            items = ad.list_universe(lim)

    def _get(it, k, default=None):
        return it.get(k, default) if isinstance(it, dict) else getattr(it, k, default)

    out = []
    for it in items:
        sym = _get(it, "symbol")
        if not sym:
            continue
        out.append({
            "symbol": sym,
            "name": _get(it, "name", sym.split("/")[0]),
            "market": _get(it, "market", adapter_key),          # ← default to the adapter you used
            "tf_supported": _get(it, "tf_supported", ["5m","15m","1h","1d"]),
        })

    cache_set(key, out)
    return out

# ----- Market data -----
# Add lower timeframes (5m/15m) in addition to 1h/1d
TF_MAP = {"5m":"5m","15m":"15m","1h": "1h", "1d": "1d"}

def fetch_ohlcv(symbol: str, tf: str, bars: int = 720, market_name: Optional[str] = None) -> pd.DataFrame:
    ad = _auto_adapter(symbol, market_name)
    try:
        return ad.fetch_ohlcv(symbol, tf, bars)
    except AdapterError as e:
        # try second-chance heuristics if market was missing
        if not market_name:
            for alt in ("forex","commodities","stocks","crypto"):
                try:
                    if alt != ad.name.split(":",1)[0]:
                        return get_adapter(alt).fetch_ohlcv(symbol, tf, bars)
                except Exception:
                    pass
        raise HTTPException(502, detail=str(e))
    except Exception as e:
        raise HTTPException(502, detail=f"{ad.name} fetch_ohlcv error: {e}")

def fetch_ohlcv_window(symbol: str, tf: str, start_ms: int, end_ms: int, market_name: Optional[str] = None) -> pd.DataFrame:
    ad = _auto_adapter(symbol, market_name)
    k = f"w:{ad.name}:{symbol}:{tf}:{start_ms}:{end_ms}"
    c = _wcache_get(k)
    if c is not None:
        return c
    try:
        df = ad.fetch_window(symbol, tf, start_ms, end_ms)
        _wcache_set(k, df)
        return df
    except AdapterError as e:
        if not market_name:
            for alt in ("forex","commodities","stocks","crypto"):
                try:
                    if alt != ad.name.split(":",1)[0]:
                        df = get_adapter(alt).fetch_window(symbol, tf, start_ms, end_ms)
                        _wcache_set(k, df)
                        return df
                except Exception:
                    pass
        raise HTTPException(502, detail=str(e))
    except Exception as e:
        raise HTTPException(502, detail=f"{ad.name} fetch_window error: {e}")

# ms per bar
def tf_ms(tf: str) -> int:
    if tf == "5m":  return 5*60*1000
    if tf == "15m": return 15*60*1000
    if tf == "1h":  return 60*60*1000
    if tf == "1d":  return 24*60*60*1000
    raise HTTPException(400, detail=f"Unsupported tf for eval: {tf}")

from collections import OrderedDict
_WINDOW_CACHE = OrderedDict()
_WINDOW_TTL = 60  # seconds
_WINDOW_MAX = 64  # entries

def _wcache_get(key):
    now = time.time()
    v = _WINDOW_CACHE.get(key)
    if not v: return None
    ts, df = v
    if (now - ts) > _WINDOW_TTL:
        _WINDOW_CACHE.pop(key, None)
        return None
    # move to MRU
    _WINDOW_CACHE.move_to_end(key)
    return df.copy()

def _wcache_set(key, df):
    _WINDOW_CACHE[key] = (time.time(), df.copy())
    _WINDOW_CACHE.move_to_end(key)
    while len(_WINDOW_CACHE) > _WINDOW_MAX:
        _WINDOW_CACHE.popitem(last=False)

EVAL_STOP_FIRST = bool(int(os.getenv("EVAL_STOP_FIRST", "0")))  # if 1: SL wins ties within same bar

# --- Learning: default hold windows (env-overridable) ---
HOLD_5M_SECS  = int(os.getenv("HOLD_5M_SECS",  "5400"))     # ~90m
HOLD_15M_SECS = int(os.getenv("HOLD_15M_SECS", "10800"))    # ~3h
HOLD_1H_SECS  = int(os.getenv("HOLD_1H_SECS",  "28800"))    # 8h
HOLD_1D_SECS  = int(os.getenv("HOLD_1D_SECS",  "604800"))   # 7d


def hold_secs_for(tf: str) -> int:
    if tf == "5m":  return HOLD_5M_SECS
    if tf == "15m": return HOLD_15M_SECS
    if tf == "1h":  return HOLD_1H_SECS
    if tf == "1d":  return HOLD_1D_SECS
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
    # defensive normalization – make sure OHLCV are numeric Series (not DataFrames)
    out = df.copy()

    # Ensure expected columns exist
    need = ["open","high","low","close","volume"]
    # some sources may send capitalized keys; map them once
    lower_map = {str(c).lower(): c for c in out.columns}
    for k in need:
        if k not in out.columns and k in lower_map:
            out.rename(columns={lower_map[k]: k}, inplace=True)

    for k in need:
        if k in out.columns:
            col = out[k]
            if isinstance(col, pd.DataFrame):
                col = col.iloc[:, 0]
            out[k] = pd.to_numeric(col, errors="coerce")
        else:
            # if truly missing, create a safe default
            out[k] = np.nan if k != "volume" else 0.0

    # basic NA drop to stabilize rolling calcs (keep enough history)
    out = out.dropna(subset=["open","high","low","close"]).copy()

    # Core features
    out["sma20"]   = out["close"].rolling(20).mean()
    out["sma50"]   = out["close"].rolling(50).mean()
    out["sma200"]  = out["close"].rolling(200).mean()
    out["rsi14"]   = rsi(out["close"], 14)
    out["atr14"]   = atr(out, 14)

    # guard against div-by-zero and multi-col ops
    safe_close = out["close"].replace(0, np.nan)
    out["atr_pct"] = (out["atr14"] / safe_close).clip(lower=0, upper=2.0)

    out["ret5"]    = out["close"].pct_change(5)
    out["slope20"] = out["sma20"] - out["sma20"].shift(5)

    out["bb_mid"]  = out["close"].rolling(20).mean()
    out["bb_std"]  = out["close"].rolling(20).std().replace(0, np.nan)
    out["bb_pos"]  = ((out["close"] - out["bb_mid"]) / (2 * out["bb_std"])).clip(-1, 1)

    # Dollar turnover proxy (will be bypassed later for FX if volume is unreliable)
    out["turnover"] = (out["close"] * out["volume"]).replace([np.inf, -np.inf], np.nan)
    out["turnover_sma96"] = out["turnover"].rolling(96).mean()

    # === Additional indicators ===
    # Momentum: percentage change over 10 bars (longer horizon than ret5)
    out["momentum"] = out["close"].pct_change(10)
    # Bollinger band width: normalized width of the Bollinger Bands (upper – lower) / mid.
    out["bb_width"] = (4.0 * out["bb_std"] / out["bb_mid"]).replace([np.inf, -np.inf], np.nan)
    # MACD and related: compute standard MACD (12, 26) with a 9‑period signal line.
    ema12 = out["close"].ewm(span=12, adjust=False).mean()
    ema26 = out["close"].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    out["macd"] = macd_line
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_line - macd_signal

    return out

def last_features_snapshot(feats: pd.DataFrame) -> Dict[str, float]:
    """
    Take the latest row of computed features and return a small numeric dict
    the learner/bias can use. Keep names stable.
    """
    row = feats.iloc[-1]
    keys = [
        "rsi14", "atr_pct", "ret5", "slope20", "bb_pos",
        "turnover", "turnover_sma96",
        # additional indicators: momentum (10), Bollinger width, MACD line/histogram
        "momentum", "bb_width", "macd", "macd_hist",
    ]
    out: Dict[str, float] = {}
    for k in keys:
        if k in row.index:
            try:
                x = float(row[k])
                if math.isfinite(x):
                    out[k] = x
            except Exception:
                pass
    # Back-compat alias some front-ends use
    if "atr_pct" in out and "atr14_pct" not in out:
        out["atr14_pct"] = out["atr_pct"]
    return out

# ----- Signal logic -----
MIN_CONFIDENCE     = float(os.getenv("MIN_CONFIDENCE", "0.14"))
VOL_CAP_ATR_PCT    = float(os.getenv("VOL_CAP_ATR_PCT", "0.25"))
VOL_MIN_ATR_PCT    = float(os.getenv("VOL_MIN_ATR_PCT", "0.001"))
# ---- Precision Mode knobs (quality > quantity) ----
PRECISION_DEFAULT = int(os.getenv("PRECISION_DEFAULT", "1"))  # 1=on by default
TURNOVER_MIN_USD_PER_HR = float(os.getenv("TURNOVER_MIN_USD_PER_HR", "100000"))  # min $/hour
EV_MIN            = float(os.getenv("EV_MIN", "0.05"))  # min expected value in R units
WILSON_MIN_N      = int(os.getenv("WILSON_MIN_N", "50"))  # min samples before LB gate
WILSON_LB_MIN     = float(os.getenv("WILSON_LB_MIN", "0.55"))  # min lower bound win rate

def build_trade(symbol: str, df: pd.DataFrame, direction: str,
                risk_pct: float = 1.0, equity: Optional[float] = None, leverage: float = 1.0,
                market_name: Optional[str] = None) -> Dict[str, Any]:
    ad = _auto_adapter(symbol, market_name)
    mkt = {}  # only used for ccxt precision metadata in UI; safe empty for yfinance

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

    price_p = float(ad.price_to_precision(symbol, price))
    stop    = float(ad.price_to_precision(symbol, stop_raw))
    targets = [float(ad.price_to_precision(symbol, t)) for t in targets_raw]

    denom = (price_p - stop)
    rr = [round(abs((t - price_p) / denom), 2) if denom else 0.0 for t in targets]

    pos = None
    risk_amt = None
    if equity and risk_pct:
        risk_amt = float(equity) * (float(risk_pct) / 100.0)

        # risk-based size: leverage should NOT change risk_per_unit
        risk_per_unit = abs(denom)
        qty_raw = risk_amt / max(risk_per_unit, 1e-8)
        qty = float(ad.amount_to_precision(symbol, qty_raw))
        notional = qty * price_p

        # margin uses leverage; risk does not
        levf = max(float(leverage or 1.0), 1.0)
        margin = notional / levf

        # Optional safety: cap to available equity by margin (keeps margin <= equity)
        if equity and margin > float(equity):
            cap_notional = float(equity) * levf
            scale = cap_notional / max(notional, 1e-8)
            qty = float(ad.amount_to_precision(symbol, qty * scale))
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

def infer_regime(feats: pd.DataFrame) -> str:
    """
    Very light regime classifier:
    - 'trend' when SMA20 vs SMA50 aligned with sustained slope
    - 'chop' when bb_width is low (squeeze) and slopes near zero
    - else 'mixed'
    """
    w = feats.tail(100)
    sma20, sma50 = w["sma20"].iloc[-1], w["sma50"].iloc[-1]
    slope = float(w["slope20"].iloc[-1])
    bbw = (2*w["bb_std"]/w["bb_mid"]).replace([np.inf,-np.inf], np.nan).iloc[-20:].median()
    if pd.isna(bbw): bbw = 0.0
    if (sma20 > sma50 and slope > 0) or (sma20 < sma50 and slope < 0):
        if bbw > 0.01:
            return "trend"
    if abs(slope) < 0.001 and bbw < 0.004:
        return "chop"
    return "mixed"

def infer_signal(feats: pd.DataFrame, min_conf: float) -> Tuple[str,float,Dict[str,bool],List[str]]:
    last = feats.iloc[-1]; reasons=[]

    # Regime & trend gates
    regime = infer_regime(feats)
    trend_up = bool(last["sma20"]>last["sma50"] and feats["slope20"].iloc[-1]>0)
    trend_dn = bool(last["sma20"]<last["sma50"] and feats["slope20"].iloc[-1]<0)
    trend_ok = trend_up or trend_dn
    if not trend_ok: reasons.append("no clear trend")
    if regime == "chop":
        reasons.append("choppy regime")

    # Volatility guardrails (plus extreme tail)
    atr_pct = float(last["atr_pct"])
    vol_ok  = bool((atr_pct>=VOL_MIN_ATR_PCT) and (atr_pct<=VOL_CAP_ATR_PCT))
    try:
        q98 = float(feats["atr_pct"].tail(300).quantile(0.98))
        if math.isfinite(q98) and atr_pct >= q98 and atr_pct > 0:
            vol_ok = False
            reasons.append("ATR% extreme (top 2%)")
    except Exception:
        pass
    if not vol_ok and "ATR% extreme" not in reasons:
        reasons.append("ATR% outside bounds")

    # Direction bias by RSI bands
    rsi14 = float(last["rsi14"])
    bias_up = rsi14>=52; bias_dn = rsi14<=48

    # Base confidence (bounded)
    conf=0.0
    if trend_ok: conf += 0.40
    if vol_ok:   conf += 0.25
    if (bias_up or bias_dn): conf += 0.15
    try:
        conf += min(abs(float(feats["ret5"].iloc[-1] or 0.0))*2.0, 0.15)
    except Exception:
        pass
    conf = float(max(0.0, min(conf,1.0)))

    if trend_up and bias_up:   sig="Bullish"
    elif trend_dn and bias_dn: sig="Bearish"
    else:                      sig="Neutral"

    if conf < min_conf:
        reasons.append(f"Composite conf {conf:.2f} < {min_conf:.2f}")

    return sig, conf, {"trend_ok":trend_ok,"vol_ok":vol_ok,"confidence_ok":conf>=min_conf,"regime":regime}, reasons

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
        df = fetch_ohlcv(fb.symbol, fb.tf, bars=2, market_name=fb.market)
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
    market_name: Optional[str] = None,
) -> Dict[str, Any]:
    df = fetch_ohlcv(symbol, tf, bars=400, market_name=market_name)
    feats = compute_features(df).dropna().iloc[-200:]
    if len(feats) < 50:
        raise HTTPException(502, detail="insufficient features window")

    thresh = min_confidence if (min_confidence is not None) else MIN_CONFIDENCE
    sig, conf, filt, reasons = infer_signal(feats, thresh)

    # client relax flags
    if ignore_trend:
        filt["trend_ok"] = True
        reasons = [r for r in reasons if "trend" not in r.lower()]
    if ignore_vol:
        filt["vol_ok"] = True
        reasons = [r for r in reasons if "atr%" not in r.lower() and "vol" not in r.lower()]

    # ---- Liquidity gate (per timeframe) ----
    last = feats.iloc[-1]
    turnover = float(last.get("turnover_sma96") or last.get("turnover") or 0.0)
    # scale per-bar threshold to per-hour
    tf_minutes = (60 if tf=="1h" else 1440 if tf=="1d" else 5 if tf=="5m" else 15 if tf=="15m" else 60)
    min_turnover = TURNOVER_MIN_USD_PER_HR * (tf_minutes/60.0)
    liquid_ok = bool(turnover >= min_turnover)
    if not liquid_ok:
        reasons.append(f"turnover ${turnover:,.0f} < ${min_turnover:,.0f} min")
        
    # --- normalize mkey for volume reliability (handle 'top_traded' by inferring from symbol) ---
    mkey = (market_name or "").split(":", 1)[0]
    if mkey in ("", "top_traded", "mixed", "mixed_top"):
        su = symbol.upper()
        if su.endswith("=X"):      mkey = "forex"
        elif su.endswith("=F"):    mkey = "commodities"
        elif "/" not in su:        mkey = "stocks"
        else:                      mkey = "crypto"

    # For markets where yfinance volume is unreliable (e.g., forex),
    # if recent volume is missing/zero, bypass the turnover gate.
    try:
        if mkey in ("forex", "commodities", "stocks"):
            vol_sum = float(pd.to_numeric(feats["volume"].tail(200), errors="coerce").fillna(0).sum())
            if vol_sum <= 0:
                liquid_ok = True
                reasons = [r for r in reasons if "turnover" not in r.lower()]
    except Exception:
        # If we can't assess volume reliably, don't block the signal on liquidity.
        liquid_ok = True
        reasons = [r for r in reasons if "turnover" not in r.lower()]

    trade = None
    advice = "Skip"

    # Direction gate (allow neutral choose side)
    directional_ok = (sig in ("Bullish", "Bearish")) or allow_neutral
    if conf >= thresh and filt["vol_ok"] and directional_ok and liquid_ok:
        if sig == "Bullish":
            direction = "Long"
        elif sig == "Bearish":
            direction = "Short"
        else:
            slope = float(feats["slope20"].iloc[-1])
            rsi14 = float(feats["rsi14"].iloc[-1])
            bbpos = float(feats["bb_pos"].iloc[-1])
            direction = "Long" if (slope >= 0 or rsi14 >= 50 or bbpos >= 0) else "Short"

        trade = build_trade(symbol, feats, direction, risk_pct, equity, leverage, market_name=market_name)
        advice = "Consider"

        # ---- EV gate ----
        try:
            entry = float(trade["entry"]); stop = float(trade["stop"])
            tps = trade.get("targets") or []
            tp1 = float(tps[0]) if tps else None
            if direction == "Long":
                R = float((tp1 - entry) / max(1e-9, (entry - stop))) if tp1 else 1.2
            else:
                R = float((entry - tp1) / max(1e-9, (stop - entry))) if tp1 else 1.2
            p = float(conf)
            ev = p*R - (1-p)*1.0
            if ev < EV_MIN:
                advice = "Skip"
                reasons.append(f"EV {ev:.2f} < {EV_MIN:.2f}")
        except Exception:
            pass

        # ---- Reliability (Wilson LB) gate ----
        try:
            wins, total = ns_stats_get(symbol, tf)
            if total >= WILSON_MIN_N:
                lb = wilson_lb(wins, total, 1.96)
                if lb < WILSON_LB_MIN:
                    advice = "Skip"
                    reasons.append(f"LB {lb:.2f} < {WILSON_LB_MIN:.2f} (n={total})")
        except Exception:
            pass

    # Feature snapshot for learning/bias
    feat_map = last_features_snapshot(feats)

    return {
        "symbol": symbol,
        "timeframe": tf,
        "signal": sig,
        "confidence": conf,
        "updated": pd.Timestamp.utcnow().isoformat(),
        "trade": trade,
        "filters": {**filt, "reasons": reasons, "liquid_ok": liquid_ok},
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

    row_map = dict(row)
    df = fetch_ohlcv_window(symbol, tf, created_ts * 1000, expires_ts * 1000, market_name=row_map.get("market"))


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
        except Exception:
            feats = {}

        if outcome != 0:
            update_weights(feats, outcome)           # keep legacy global weights
            ns_stats_update(symbol, tf, outcome)     # NEW: reliability counters

    return {"result": result, "pnl": float(pnl_frac), "outcome": int(outcome), "resolved_ts": time.time()}

# --- REPLACE: resolve_due_offers with short, per-row lock usage ---
def resolve_due_offers(limit: int = 50) -> dict:
    """
    Resolve due offers with minimal lock time:
      1) read list of rows/ids under lock,
      2) compute outcome without lock,
      3) write each result under lock,
      4) (optionally) insert feedback outside existing locks.
    """
    now = time.time()
    done = good = bad = 0

    # 1) Read candidates under lock
    with _db_lock:
        conn = _db()
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, symbol, tf, market, direction, entry, stop, targets, confidence, advice, "
            "features, equity, risk_pct, leverage, created_ts, expires_ts "
            "FROM offers WHERE resolved=0 AND expires_ts<=? "
            "ORDER BY expires_ts ASC LIMIT ?",
            (now, int(limit)),
        ).fetchall()
        conn.close()

    # 2..4) For each row, compute (no lock) then write (lock) then log feedback (locks internally)
    for row in rows:
        try:
            upd = resolve_offer_row(row)  # network/CPU: NO DB LOCK HERE
        except Exception:
            upd = {"result": "NEITHER", "pnl": 0.0, "outcome": 0, "resolved_ts": time.time()}

        # write outcome under lock
        with _db_lock:
            conn = _db()
            conn.execute(
                "UPDATE offers SET resolved=1, result=?, pnl=?, resolved_ts=? WHERE id=?",
                (upd["result"], upd["pnl"], upd["resolved_ts"], row["id"])
            )
            conn.commit()
            conn.close()

        # insert synthetic feedback (helper has its own locking)
        try:
            if int(upd.get("outcome", 0)) != 0:
                _record_feedback_from_offer(
                    dict(row), int(upd["outcome"]), float(upd["resolved_ts"]), pnl=float(upd["pnl"])
                )
        except Exception:
            pass

        done += 1
        if upd["outcome"] > 0:   good += 1
        elif upd["outcome"] < 0: bad  += 1

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
def instruments(market: Optional[str] = None, _: None = Depends(require_key)):
    return get_universe(market_name=market)

@app.get("/chart")
def chart(symbol: str, tf: str = "1h", n: int = 120, market: Optional[str] = None, _: None = Depends(require_key)):
    df = fetch_ohlcv(symbol, tf, bars=max(200, n + 20), market_name=market)

    # Ensure we have a usable "close" series even if columns are duplicated or MI
    if "close" not in df.columns:
        # try case-insensitive fallback
        cols_lower = {str(c).lower(): c for c in df.columns}
        if "close" not in cols_lower:
            raise HTTPException(502, detail="no 'close' column in data")
        close_obj = df[cols_lower["close"]]
    else:
        close_obj = df["close"]

    # If it's a DataFrame (duplicate column names), squeeze to the first column
    if isinstance(close_obj, pd.DataFrame):
        if close_obj.shape[1] == 0:
            raise HTTPException(502, detail="'close' column is empty")
        close_obj = close_obj.iloc[:, 0]

    closes = pd.to_numeric(close_obj, errors="coerce").tail(n).dropna().astype(float).to_list()
    if not closes:
        raise HTTPException(502, detail="no numeric close data")

    return {"symbol": symbol, "tf": tf, "closes": closes}

@app.get("/signals")
def signals(
    symbol: str,
    tf: str = "1h",
    risk_pct: float = Query(1.0, ge=0.1, le=5.0),
    equity: Optional[float] = Query(None, ge=0),
    leverage: float = Query(1.0, ge=1.0, le=100.0),
    min_confidence: Optional[float] = Query(None),
    precision: int = Query(PRECISION_DEFAULT, ge=0, le=1),
    market: Optional[str] = Query(None),
    _: None = Depends(require_key),
):
    try:
        mc = (min_confidence if min_confidence is not None else MIN_CONFIDENCE)
        if precision:
            mc = min(1.0, mc + 0.05)  # stricter threshold in precision mode
        # 1) get your base result (your existing function)
        res = evaluate_signal(
            symbol=symbol,
            tf=tf,
            risk_pct=risk_pct,
            equity=equity,
            leverage=leverage,
            min_confidence=mc,
            market_name=market,
        )
        
        # include market for UI parity with /scan
        if market:
            res["market"] = market
        else:
            try:
                # infer from adapter used for routing
                res["market"] = _auto_adapter(symbol, market).name.split(":")[0]
            except Exception:
                pass

        # Nudge confidence by learned weights
        base_conf = float(res.get("confidence", 0.0))
        feats: Dict[str, Any] = res.get("features") or {}
        biased = apply_feedback_bias(base_conf, feats, symbol, tf)
        res["confidence"] = apply_calibration(tf, biased)


        # Keep filter in sync with the threshold if provided
        if isinstance(res.get("filters"), dict) and min_confidence is not None:
            res["filters"]["confidence_ok"] = (res["confidence"] >= float(min_confidence))

        # Include market for convenient round-trip to /chart
        resolved_market = market or _auto_adapter(symbol, market).name.split(":", 1)[0]
        res["market"] = resolved_market

        return res

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"signals failed: {e}")

@app.get("/scan")
def scan(
    tf: str = Query("1h"),
    limit: int = Query(18, ge=1, le=50),                # universe size to scan
    top: int = Query(6, ge=1, le=12),                  # how many results to return
    allow_neutral: int = Query(0, ge=0, le=1),
    ignore_trend: int = Query(0, ge=0, le=1),
    ignore_vol: int = Query(0, ge=0, le=1),
    risk_pct: float = Query(1.0, ge=0.1, le=5.0),
    equity: Optional[float] = Query(None, ge=0.0),
    leverage: float = Query(1.0, ge=1.0, le=100.0),
    include_chart: int = Query(1, ge=0, le=1),
    min_confidence: Optional[float] = Query(None),
    precision: int = Query(PRECISION_DEFAULT, ge=0, le=1),
    market: Optional[str] = Query(None),
    _: None = Depends(require_key)
):
    # Per-market scan caps to keep ccxt routes under the WP 65s proxy timeout
    MARKET_SCAN_CAP = {
        "crypto":   int(os.getenv("TOP_N_CRYPTO",  "18")),
        "futures":  int(os.getenv("TOP_N_FUTURES", "18")),   # binance_perps is treated as 'futures'
    }
    mkey = (market or "crypto").split(":", 1)[0]
    eff_limit = min(limit, MARKET_SCAN_CAP.get(mkey, limit))

    uni = get_universe(limit=eff_limit, market_name=market)
    results, ok = [], []

    for it in uni:
        try:
            mc = (min_confidence if min_confidence is not None else MIN_CONFIDENCE)
            if precision:
                mc = min(1.0, mc + 0.05)
            s = evaluate_signal(
                symbol=it["symbol"], tf=tf,
                risk_pct=risk_pct, equity=equity, leverage=leverage,
                min_confidence=mc,
                ignore_trend=bool(ignore_trend),
                ignore_vol=bool(ignore_vol),
                allow_neutral=bool(allow_neutral),
                market_name=market or it.get("market"),
            )

            base_conf = float(s.get("confidence", 0.0))
            feats_map = s.get("features") or {}
            biased = apply_feedback_bias(base_conf, feats_map, s["symbol"], s["timeframe"])
            s["confidence"] = apply_calibration(tf, biased)

            if isinstance(s.get("filters"), dict) and (min_confidence is not None):
                s["filters"]["confidence_ok"] = (s["confidence"] >= float(min_confidence))
                
            s["market"] = it["market"]
            # Attach a human-friendly name for UI purposes (fallback to symbol if missing)
            try:
                s["name"] = it.get("name", it["symbol"])
            except Exception:
                s["name"] = it["symbol"]
            if s["advice"] == "Consider":
                ok.append(s)
            results.append(s)
        except HTTPException as he:
            results.append({
                "symbol": it["symbol"], "timeframe": tf, "signal": "Neutral",
                "confidence": 0.0, "updated": pd.Timestamp.utcnow().isoformat(),
                "trade": None,
                "filters": {"trend_ok": False, "vol_ok": False, "confidence_ok": False, "reasons": [he.detail]},
                "advice": "Skip", "market": it["market"],
                "name": it.get("name", it["symbol"])
            })
        except Exception as e:
            results.append({
                "symbol": it["symbol"], "timeframe": tf, "signal": "Neutral",
                "confidence": 0.0, "updated": pd.Timestamp.utcnow().isoformat(),
                "trade": None,
                "filters": {"trend_ok": False, "vol_ok": False, "confidence_ok": False, "reasons": [f"exception: {e}"]},
                "advice": "Skip", "market": it["market"],
                "name": it.get("name", it["symbol"])
            })

    pool = ok if ok else results
    pool_sorted = sorted(pool, key=lambda s: abs(s.get("confidence") or 0.0), reverse=True)
    topK = pool_sorted[:top]   # <-- separate “top” from “limit”

    out: List[Dict[str, Any]] = []
    for s in topK:
        if include_chart:
            try:
                df = fetch_ohlcv(s["symbol"], tf, bars=160, market_name=market or s.get("market"))
                s["chart"] = {"closes": pd.to_numeric(df["close"], errors="coerce").tail(120).dropna().astype(float).to_list()}
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
        update_weights_ns(fb.features or {}, fb.outcome, fb.symbol, fb.tf)
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

# --- REPLACE: resolve-one to avoid nested locks and long critical sections ---
class ResolveOneIn(BaseModel):
    id: int

@app.post("/learning/resolve-one")
def learning_resolve_one(body: ResolveOneIn, _: None = Depends(require_key)):
    # 1) Read the row under lock, then release
    with _db_lock:
        conn = _db()
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM offers WHERE id=?", (body.id,)).fetchone()
        conn.close()
    if not row:
        raise HTTPException(404, detail="offer not found")

    # 2) Do network-heavy work OUTSIDE the lock
    upd = resolve_offer_row(row)

    # 3) Write update under lock (short transaction)
    with _db_lock:
        conn = _db()
        conn.execute(
            "UPDATE offers SET resolved=1, result=?, pnl=?, resolved_ts=? WHERE id=?",
            (upd["result"], upd["pnl"], upd["resolved_ts"], row["id"])
        )
        conn.commit()
        conn.close()

    # 4) Log synthetic feedback OUTSIDE existing lock (helper locks internally)
    try:
        if int(upd.get("outcome", 0)) != 0:
            row_map = dict(row)  # sqlite3.Row -> dict
            _record_feedback_from_offer(
                row_map, int(upd["outcome"]), float(upd["resolved_ts"]), pnl=float(upd["pnl"])
            )
    except Exception:
        pass  # never block resolution on logging

    return {"ok": True, "id": body.id, **upd}

@app.post("/learning/resolve-due")
def learning_resolve_due(limit: int = Query(25, ge=1, le=500), _: None = Depends(require_key)):
    return resolve_due_offers(limit=limit)

# Add this wrapper so GET works too (calls the same function)
@app.get("/learning/resolve-due")
def learning_resolve_due_get(limit: int = Query(25, ge=1, le=500), _: None = Depends(require_key)):
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
        "hold_secs": {
            "5m": HOLD_5M_SECS,
            "15m": HOLD_15M_SECS,
            "1h": HOLD_1H_SECS,
            "1d": HOLD_1D_SECS
        },
        "bg_interval": BG_INTERVAL,
        "eval_stop_first": bool(EVAL_STOP_FIRST),
    }

# === FX helper endpoint (USD/EUR/GBP) ===
from typing import Set

@app.get("/fx")
def fx(base: str = "USD", _: None = Depends(require_key)):
    """
    Return simple FX map for front-end display/sizing.
    Output format: {"base": "USD", "rates": {"EUR": <EUR per USD>, "GBP": <GBP per USD>}, "ts": <epoch_ms>}
    """
    base = (base or "USD").upper()
    want: Set[str] = {"EUR", "GBP"}

    # Use the existing Yahoo/forex adapter so we don't add new deps here.
    def _last_px(ticker: str) -> float:
        df = fetch_ohlcv(ticker, "1h", bars=2, market_name="forex")
        return float(df["close"].iloc[-1])

    rates = {}
    for iso in want:
        if iso == base:
            rates[iso] = 1.0
            continue
        rate = None
        # Prefer direct "BASEISO=X" (ISO per BASE) if Yahoo supports it, otherwise invert "ISOBASE=X"
        try:
            rate = _last_px(f"{base}{iso}=X")
        except Exception:
            try:
                inv = _last_px(f"{iso}{base}=X")  # USD per ISO (if base=USD)
                if inv and inv > 0:
                    rate = 1.0 / float(inv)
            except Exception:
                pass
        if rate is not None:
            rates[iso] = float(rate)

    return {"base": base, "rates": rates, "ts": int(time.time() * 1000)}

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
async def tracked_delete(
    id: Optional[str] = None,
    x_user_id: Optional[int] = Header(None),
    body: Optional[dict] = Body(None),
    _: None = Depends(require_key)
):
    uid = _uid_from_header(x_user_id)
    if not id and body and isinstance(body, dict):
        id = body.get("id")
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