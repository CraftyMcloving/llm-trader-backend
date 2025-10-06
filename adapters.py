# adapters.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import os, time
import pandas as pd
import numpy as np

# Optional deps
try:
    import ccxt  # for crypto/futures
except Exception:
    ccxt = None

try:
    import yfinance as yf  # for forex/commodities/stocks
except Exception:
    yf = None

# ---------- Shared helpers ----------
def _ensure(condition: bool, msg: str):
    if not condition:
        raise RuntimeError(msg)

# We keep TFs aligned with your app (5m/15m/1h/1d)
def tf_ms(tf: str) -> int:
    if tf == "5m":  return 5*60*1000
    if tf == "15m": return 15*60*1000
    if tf == "1h":  return 60*60*1000
    if tf == "1d":  return 24*60*60*1000
    raise ValueError(f"Unsupported tf: {tf}")

TF_MAP = {"5m": "5m", "15m": "15m", "1h": "1h", "1d": "1d"}

@dataclass
class UniverseItem:
    symbol: str
    name: str
    market: str
    tf_supported: List[str]

class AdapterError(Exception):
    pass

class BaseAdapter:
    """Minimal interface your app uses."""
    name: str = "base"
    def list_universe(self, limit:int) -> List[Dict[str,Any]]: raise NotImplementedError
    def fetch_ohlcv(self, symbol:str, tf:str, bars:int) -> pd.DataFrame: raise NotImplementedError

    def fetch_window(self, symbol:str, tf:str, start_ms:int, end_ms:int) -> pd.DataFrame:
        per = tf_ms(tf)
        need = max(2, int((end_ms - start_ms)/per) + 8)
        df = self.fetch_ohlcv(symbol, tf, bars=min(2000, max(need, 240)))
        # normalize 'ts' to ms (handle tz-aware safely)
        ts_col = df["ts"]
        if np.issubdtype(ts_col.dtype, np.datetime64):
            ts_col = pd.to_datetime(ts_col, utc=True)
            ms = (ts_col.view("int64") // 10**6)
        else:
            ms = pd.to_numeric(ts_col, errors="coerce").astype("int64")
        keep = (ms >= (start_ms - per)) & (ms <= (end_ms + per))
        return df.loc[keep].reset_index(drop=True)

    # Precision helpers used by build_trade
    def price_to_precision(self, symbol:str, price:float) -> float: return float(round(price, 2))
    def amount_to_precision(self, symbol:str, qty:float) -> float:  return float(qty)

    @staticmethod
    def _safe_float(v):
        try:
            return float(v)
        except Exception:
            return None

# ---------- Crypto (CCXT / Kraken default) ----------
class CryptoCCXT(BaseAdapter):
    def __init__(self, exchange_id:str="kraken", quote:str="USD",
                 curated:Optional[List[str]]=None, cache_ttl:int=900):
        _ensure(ccxt is not None, "ccxt not installed")
        self.name = f"crypto:{exchange_id}:{quote}"
        self.exchange_id = exchange_id
        self.quote = quote
        self.curated = curated or []
        self._ex = None
        self._markets = None
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Tuple[float, pd.DataFrame]] = {}
        
    def _tcache_get(self, key: str, ttl: Optional[int] = None):
        ttl = int(ttl or self.cache_ttl)
        ent = getattr(self, "_tcache", {}).get(key)
        if not ent:
            return None
        ts, val = ent
        return val if (time.time() - ts) < ttl else None

    def _tcache_set(self, key: str, val: dict):
        if not hasattr(self, "_tcache"):
            self._tcache = {}
        self._tcache[key] = (time.time(), val)

    def _top_traded(self, n: int, avail: Optional[List[str]] = None) -> List[str]:
        key = f"top:{self.exchange_id}:{self.quote}"
        cached = self._tcache_get(key)
        if cached and cached.get("syms"):
            return cached["syms"][:max(1, int(n or 0))]

        try:
            ex = self._exh()
            tickers = ex.fetch_tickers()
        except Exception:
            tickers = {}

        rows = []
        for sym, tk in (tickers or {}).items():
            if avail and sym not in avail:
                continue
            if ("/" not in sym) or (not sym.endswith(f"/{self.quote}")):
                continue

            qv = tk.get("quoteVolume")
            if qv is None:
                last = BaseAdapter._safe_float(tk.get("last")) or 0.0
                vol  = BaseAdapter._safe_float(tk.get("baseVolume")) or BaseAdapter._safe_float(tk.get("volume")) or 0.0
                qv = (last or 0.0) * (vol or 0.0)
            qv = BaseAdapter._safe_float(qv) or 0.0
            rows.append((sym, qv))

        rows.sort(key=lambda t: t[1], reverse=True)
        syms = [s for (s, _) in rows[:max(1, int(n or 0))]]
        self._tcache_set(key, {"syms": syms})
        return syms

    def _exh(self):
        if self._ex is None:
            self._ex = getattr(ccxt, self.exchange_id)({"enableRateLimit": True, "timeout": 12000})
            try:
                self._ex.load_markets()
            except Exception:
                # if this fails now, methods that need it will retry via _load_markets()
                pass
        return self._ex

    def _load_markets(self):
        if self._markets is None:
            self._markets = self._exh().load_markets()
        return self._markets

    def list_universe(self, limit: int | None = None, top: int | None = None) -> List[Dict[str, Any]]:
        # hard cap from env (0/neg => unlimited)
        raw_cap = int(os.getenv("TOP_N", "0"))
        hard_cap = None if raw_cap <= 0 else raw_cap

        try:
            m = self._load_markets()
        except Exception as e:
            raise AdapterError(f"load_markets failed: {e}")

        avail: List[str] = []
        for sym, info in (m or {}).items():
            if ("/" not in sym) or (not sym.endswith(f"/{self.quote}")):
                continue
            if (info or {}).get("active") is False:
                continue
            if info.get("type") and info["type"] != "spot":
                continue
            avail.append(sym)

        # curated to the front (keep order, no dupes)
        if self.curated:
            cur = [s for s in self.curated if s in avail]
            rest = [s for s in avail if s not in self.curated]
            avail = cur + rest

        # enforce request limit first, then hard cap from env
        if limit is not None:
            avail = avail[:limit]
        if hard_cap is not None:
            avail = avail[:hard_cap]

        # optional: top traded by 24h quote volume
        if top:
            syms = self._top_traded(top, avail=avail)
        else:
            syms = avail

        out = []
        for s in syms:
            out.append({
                "symbol": s,
                "name": s.split('/')[0],
                "market": self.name,
                "tf_supported": ["5m","15m","1h","1d"],
            })
        return out

    def _cache_get(self, key:str) -> Optional[pd.DataFrame]:
        v = self._cache.get(key)
        if not v: return None
        ts, df = v
        return df.copy() if (time.time()-ts) <= self.cache_ttl else None

    def _cache_set(self, key:str, df:pd.DataFrame):
        self._cache[key] = (time.time(), df.copy())

    def fetch_ohlcv(self, symbol:str, tf:str, bars:int) -> pd.DataFrame:
        ex = self._exh()
        tf_ex = TF_MAP.get(tf)
        if tf_ex is None:
            raise AdapterError(f"Unsupported tf: {tf}")
        key = f"ohlcv:{self.name}:{symbol}:{tf_ex}:{bars}"
        c = self._cache_get(key)
        if c is not None:
            return c
        try:
            data = ex.fetch_ohlcv(symbol, timeframe=tf_ex, limit=min(bars + 40, 2000))
        except Exception as e:
            raise AdapterError(f"{self.exchange_id} fetch_ohlcv({symbol},{tf_ex}) failed: {e}") from e
        if not data:
            raise AdapterError("no candles")
        df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        self._cache_set(key, df)
        return df

    def fetch_window(self, symbol:str, tf:str, start_ms:int, end_ms:int) -> pd.DataFrame:
        ex = self._exh()
        tf_ex = TF_MAP.get(tf)
        if tf_ex is None:
            raise AdapterError(f"Unsupported tf: {tf}")
        per = tf_ms(tf)
        need = max(2, int((end_ms - start_ms) / per) + 4)
        since = max(0, start_ms - 2 * per)
        try:
            data = ex.fetch_ohlcv(symbol, timeframe=tf_ex, since=since, limit=min(need + 20, 2000))
        except Exception as e:
            raise AdapterError(f"{self.exchange_id} fetch_window({symbol},{tf_ex}) failed: {e}") from e
        if not data:
            raise AdapterError("no candles for window")
        df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
        return df[(df["ts"] >= (start_ms - per)) & (df["ts"] <= (end_ms + per))].reset_index(drop=True)


    def price_to_precision(self, symbol:str, price:float) -> float:
        self._load_markets()  # ensure market metadata loaded
        return float(self._exh().price_to_precision(symbol, price))
    def amount_to_precision(self, symbol:str, qty:float) -> float:
        self._load_markets()
        return float(self._exh().amount_to_precision(symbol, qty))

# ---------- Binance Perps (USDT-M futures) via ccxt ----------
class BinancePerpsUSDT(BaseAdapter):
    def __init__(self, curated:Optional[List[str]]=None):
        _ensure(ccxt is not None, "ccxt not installed")
        self.name = "futures:binance:usdtm"
        self.curated = curated or ["BTC/USDT","ETH/USDT","SOL/USDT","XRP/USDT","DOGE/USDT","BNB/USDT","LTC/USDT","ADA/USDT","TRX/USDT","LINK/USDT","AVAX/USDT","DOT/USDT","MATIC/USDT","BCH/USDT","XLM/USDT","ETC/USDT","UNI/USDT","ATOM/USDT","FIL/USDT","NEAR/USDT"]
        self._ex = None

    def _exh(self):
        if self._ex is None:
            self._ex = ccxt.binance({
                "enableRateLimit": True,
                "timeout": 12000,
                "options": {"defaultType": "future"}  # USDT-M perpetuals
            })
            self._ex.load_markets()
        return self._ex

    def list_universe(self, limit:int) -> List[Dict[str,Any]]:
        ex = self._exh()
        out = []
        for s in self.curated[:max(1, limit)]:
            if s in ex.markets and ex.markets[s].get("active"):
                out.append({"symbol": s, "name": s.split('/')[0], "market": self.name,
                            "tf_supported": ["5m","15m","1h","1d"]})
        return out

    def fetch_ohlcv(self, symbol:str, tf:str, bars:int) -> pd.DataFrame:
        ex = self._exh()
        tf_ex = TF_MAP.get(tf)
        if tf_ex is None:
            raise AdapterError(f"Unsupported tf: {tf}")
        try:
            data = ex.fetch_ohlcv(symbol, timeframe=tf_ex, limit=min(bars+40, 1500))
        except Exception as e:
            raise AdapterError(f"binance fetch_ohlcv({symbol},{tf_ex}) failed: {e}") from e
        if not data:
            raise AdapterError("no candles")
        df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        return df

    def fetch_window(self, symbol:str, tf:str, start_ms:int, end_ms:int) -> pd.DataFrame:
        ex = self._exh()
        tf_ex = TF_MAP.get(tf)
        if tf_ex is None:
            raise AdapterError(f"Unsupported tf: {tf}")
        per = tf_ms(tf)
        need = max(2, int((end_ms - start_ms)/per) + 4)
        since = max(0, start_ms - 2*per)
        try:
            data = ex.fetch_ohlcv(symbol, timeframe=tf_ex, since=since, limit=min(need+20, 1500))
        except Exception as e:
            raise AdapterError(f"binance fetch_window({symbol},{tf_ex}) failed: {e}") from e
        if not data:
            raise AdapterError("no candles for window")
        df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
        return df[(df["ts"] >= (start_ms - per)) & (df["ts"] <= (end_ms + per))].reset_index(drop=True)

    def price_to_precision(self, symbol:str, price:float) -> float:
        return float(self._exh().price_to_precision(symbol, price))
    def amount_to_precision(self, symbol:str, qty:float) -> float:
        return float(self._exh().amount_to_precision(symbol, qty))

# ---------- yfinance generic + thin wrappers ----------
class YFAdapter(BaseAdapter):
    def __init__(self, name:str, tickers:List[str], price_decimals:int=4, cache_ttl:int=300):
        _ensure(yf is not None, "yfinance not installed; add yfinance>=0.2.44")
        self.name = name
        self.tickers = tickers
        self.price_decimals = price_decimals
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Tuple[float, pd.DataFrame]] = {}

    def _cache_get(self, key:str) -> Optional[pd.DataFrame]:
        v = self._cache.get(key)
        if not v: return None
        ts, df = v
        return df.copy() if (time.time() - ts) <= self.cache_ttl else None

    def _cache_set(self, key:str, df:pd.DataFrame):
        self._cache[key] = (time.time(), df.copy())

    def list_universe(self, limit:int) -> List[Dict[str,Any]]:
        syms = self.tickers[:max(1, limit)]
        # 5m is supported by Yahoo for ~60 days; you already pull 60d below.
        return [{"symbol": s, "name": s.replace("=F","").replace("^",""), "market": self.name,
                 "tf_supported": ["5m","15m","1h","1d"]} for s in syms]

    def _interval(self, tf:str) -> str:
        return {"5m":"5m", "15m":"15m", "1h":"60m", "1d":"1d"}.get(tf, "60m")

    def fetch_ohlcv(self, symbol:str, tf:str, bars:int) -> pd.DataFrame:
        key = f"yf:{self.name}:{symbol}:{tf}"
        cached = self._cache_get(key)
        if cached is not None:
            df = cached
        else:
            df_raw = yf.download(
                symbol,
                period="60d",
                interval=self._interval(tf),
                progress=False,
                prepost=False,
                threads=False,
                auto_adjust=False,
                actions=False,
            )
            if df_raw is None or df_raw.empty:
                raise AdapterError("no candles from yfinance")

            # --- flatten possible MultiIndex columns (common with some FX/indices) ---
            if isinstance(df_raw.columns, pd.MultiIndex):
                # if only one symbol level, drop it; else try to slice by our symbol
                try:
                    if df_raw.columns.nlevels >= 2:
                        lvl1 = df_raw.columns.get_level_values(-1)
                        if getattr(lvl1, "nunique", lambda: 1)() == 1:
                            df_raw.columns = df_raw.columns.droplevel(-1)
                        else:
                            # slice by symbol if present
                            try:
                                df_raw = df_raw.xs(symbol, axis=1, level=-1, drop_level=True)
                            except Exception:
                                df_raw.columns = [c[0] for c in df_raw.columns]
                except Exception:
                    # last resort: flatten by taking top-level name
                    df_raw.columns = [c[0] if isinstance(c, tuple) else c for c in df_raw.columns]

            # standardize columns
            df_raw = df_raw.rename(columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }).reset_index(names="ts")

            # keep just the fields we use
            cols = ["ts", "open", "high", "low", "close", "volume"]
            df = df_raw[[c for c in cols if c in df_raw.columns]].copy()

            # ensure scalar Series (never DataFrames)
            for c in ("open","high","low","close","volume"):
                if c in df.columns and isinstance(df[c], pd.DataFrame):
                    df[c] = df[c].iloc[:, 0]

            self._cache_set(key, df)

        return df.tail(bars).reset_index(drop=True)
    
    def price_to_precision(self, symbol:str, price:float) -> float:
        return float(round(price, self.price_decimals))

# “Add-on pack” factories (top-20 defaults)
def ForexAdapter() -> BaseAdapter:
    return YFAdapter("forex", ["EURUSD=X","GBPUSD=X","USDJPY=X","AUDUSD=X","USDCAD=X","USDCHF=X","NZDUSD=X","EURGBP=X","EURJPY=X","GBPJPY=X","AUDJPY=X","EURAUD=X","EURCAD=X","EURCHF=X","GBPCHF=X","AUDCAD=X","CHFJPY=X","AUDNZD=X","CADJPY=X","NZDJPY=X"], price_decimals=5)

def CommoditiesAdapter() -> BaseAdapter:
    return YFAdapter("commodities", ["GC=F","CL=F","SI=F","BZ=F","NG=F","HO=F","RB=F","HG=F","ALI=F","PL=F","TIO=F","ZC=F","ZS=F","ZW=F","KC=F","SB=F","CC=F","CT=F","LBS=F"], price_decimals=2)

def StocksAdapter() -> BaseAdapter:
    return YFAdapter("stocks", ["AAPL","MSFT","NVDA","TSLA","SPY","QQQ","AMZN","INTC","PLTR","META","ORCL","AVGO","TQQQ","AMD","GOOGL","COST","IWM","MU","APP","GOOG"], price_decimals=2)

# ----- Mixed Top-Traded (All Markets) -----
class MixedTopTradedAdapter(BaseAdapter):
    """
    Cross-market universe (crypto/forex/commodities/stocks).
    Ranking metric is configurable:
      - TOPTRADED_METRIC=24h  -> sum over last ~24×1h of (close*volume) when available
      - TOPTRADED_METRIC=1d   -> last daily bar close*volume
      - TOPTRADED_METRIC=auto -> crypto=24h, stocks/commodities=1d, forex=atr_proxy
    Composition is configurable (balanced quotas vs pure global top):
      - TOPTRADED_MODE=balanced (default) with TOPTRADED_TARGETS (e.g. "stocks:8,crypto:6,forex:4,commodities:2")
      - TOPTRADED_MODE=pure     ignore quotas; take global top-N

    Tuning:
      - TOPTRADED_CACHE_TTL (seconds) default 600
      - TOPTRADED_TARGETS string must sum to ~20 if you want a 20-name universe
    """
    def __init__(self, mode: str | None = None, metric: str | None = None, cache_ttl: int | None = None, targets: dict | None = None):
        self.name = "top_traded"

        # Reuse existing adapters
        self._crypto = CryptoCCXT(os.getenv("EXCHANGE", "kraken"), os.getenv("QUOTE", "USD"),
                                  curated=[s.strip().upper() for s in os.getenv("CURATED","BTC,ETH,SOL,XRP,ADA,DOGE,LINK,LTC,BCH,TRX,DOT,ATOM,XLM,ETC,MATIC,UNI,APT,ARB,OP,AVAX,NEAR,ALGO,FIL,SUI,SHIB,USDC,USDT,XMR,AAVE,PAXG,ONDO,PEPE,SEI,IMX,TIA").split(",") if s.strip()])
        self._forex  = ForexAdapter()
        self._comm   = CommoditiesAdapter()
        self._stocks = StocksAdapter()

        self.mode   = (mode or os.getenv("TOPTRADED_MODE", "balanced")).lower()
        self.metric = (metric or os.getenv("TOPTRADED_METRIC", "auto")).lower()
        self.name = f"top_traded:{self.mode}"
        self.cache_ttl = int(cache_ttl or os.getenv("TOPTRADED_CACHE_TTL", "600"))
        self._cache: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}

        def _parse_targets(s: str) -> Dict[str, int]:
            out = {"stocks": 8, "crypto": 6, "forex": 4, "commodities": 2}
            if not s:
                return out
            for part in s.split(","):
                if ":" in part:
                    k, v = part.split(":", 1)
                    k = k.strip().lower()
                    try:
                        out[k] = max(0, int(v.strip()))
                    except Exception:
                        pass
            return out

        self.targets = targets or _parse_targets(os.getenv("TOPTRADED_TARGETS", "stocks:8,crypto:6,forex:4,commodities:2"))

    # ---- helpers ----
    def _guess_market(self, symbol: str) -> str:
        s = symbol.upper()
        if "/" in s:         return "crypto"
        if s.endswith("=X"): return "forex"
        if s.endswith("=F"): return "commodities"
        return "stocks"

    def _notional_24h(self, adapter: BaseAdapter, symbol: str) -> float:
        """Sum over last ~24×1h of close*volume (skips if volume missing)."""
        try:
            df = adapter.fetch_ohlcv(symbol, "1h", bars=30)
            close = pd.to_numeric(df["close"], errors="coerce")
            vol   = pd.to_numeric(df.get("volume"), errors="coerce").fillna(0.0)
            # Take last ~24 bars (use 26 to be safe for gaps)
            return float((close.tail(26) * vol.tail(26)).sum())
        except Exception:
            return 0.0

    def _notional_1d(self, adapter: BaseAdapter, symbol: str) -> float:
        """Use last daily bar close*volume (works well for stocks/commodities)."""
        try:
            df = adapter.fetch_ohlcv(symbol, "1d", bars=2)
            close = float(pd.to_numeric(df["close"], errors="coerce").iloc[-1])
            vol   = float(pd.to_numeric(df.get("volume"), errors="coerce").fillna(0.0).iloc[-1])
            return close * vol
        except Exception:
            return 0.0

    def _atr_proxy(self, adapter: BaseAdapter, symbol: str, lot: float = 100_000.0) -> float:
        """
        ATR-based activity proxy for markets with unreliable/no volume (e.g., forex).
        Scale by standard FX lot notional so majors rank sensibly.
        """
        try:
            df = adapter.fetch_ohlcv(symbol, "1h", bars=30)
            hi  = pd.to_numeric(df["high"], errors="coerce")
            lo  = pd.to_numeric(df["low"], errors="coerce")
            cl  = pd.to_numeric(df["close"], errors="coerce")
            prev = cl.shift(1)
            tr  = pd.concat([(hi-lo).abs(), (hi-prev).abs(), (lo-prev).abs()], axis=1).max(axis=1)
            atr = tr.rolling(14, min_periods=1).mean()
            # Sum last ~24 bars and scale by price*lot
            proxy = float((atr.tail(26) * cl.tail(26)).sum() * lot)
            return proxy
        except Exception:
            return 0.0

    def _score(self, mkey: str, symbol: str) -> float:
        met = self.metric
        if met == "24h":
            if mkey in ("stocks", "commodities"):
                # still okay to use 24h, but many instruments only have daily volume; fall back
                v = self._notional_24h(self._stocks if mkey=="stocks" else self._comm, symbol)
                return v if v > 0 else self._notional_1d(self._stocks if mkey=="stocks" else self._comm, symbol)
            if mkey == "crypto":
                return self._notional_24h(self._crypto, symbol)
            if mkey == "forex":
                return self._atr_proxy(self._forex, symbol)
        elif met == "1d":
            if mkey == "crypto":
                # use 24h on crypto even in 1d mode (exchanges are 24/7)
                v = self._notional_24h(self._crypto, symbol)
                return v if v > 0 else self._notional_1d(self._crypto, symbol)
            if mkey == "forex":
                return self._atr_proxy(self._forex, symbol)
            return self._notional_1d(self._stocks if mkey=="stocks" else self._comm, symbol)
        # auto: crypto=24h, stocks/comm=1d, forex=atr
        if mkey == "crypto":       return self._notional_24h(self._crypto, symbol)
        if mkey == "stocks":       return self._notional_1d(self._stocks, symbol)
        if mkey == "commodities":  return self._notional_1d(self._comm, symbol)
        return self._atr_proxy(self._forex, symbol)

    def _candidates(self, adapter: BaseAdapter, cap: int) -> List[str]:
        try:
            uni = adapter.list_universe(limit=cap)
            out = []
            for it in uni:
                if isinstance(it, dict):
                    out.append(it.get("symbol") or "")
                else:
                    out.append(getattr(it, "symbol", "") or "")
            return [s for s in out if s]
        except Exception:
            return []

    def _build_balanced(self, total:int) -> List[Dict[str, Any]]:
        # 1) score candidates by class
        pools = {
            "crypto":      (self._crypto,      40),
            "forex":       (self._forex,       40),
            "commodities": (self._comm,        30),
            "stocks":      (self._stocks,      50),
        }
        scored_by_class: Dict[str, List[Tuple[str, float]]] = {}
        for k, (ad, cap) in pools.items():
            syms = self._candidates(ad, cap)
            pairs = []
            for s in syms:
                sc = self._score(k, s)
                if sc > 0:
                    pairs.append((s, sc))
            pairs.sort(key=lambda t: t[1], reverse=True)
            scored_by_class[k] = pairs

        # 2) normalize quotas to the requested 'total' and available candidates
        raw = {k: max(0, int(self.targets.get(k, 0))) for k in scored_by_class.keys()}
        # zero quotas for empty classes
        for k in list(raw.keys()):
            if len(scored_by_class[k]) == 0:
                raw[k] = 0
        must = sum(raw.values()) or 1

        # start with at least 1 per eligible class (if there is room)
        alloc = {k: 0 for k in raw.keys()}
        eligible = [k for k, v in raw.items() if v > 0 and len(scored_by_class[k]) > 0]
        if total >= len(eligible):
            for k in eligible:
                alloc[k] = 1
            remaining = total - len(eligible)
        else:
            # fewer slots than classes: take the best 'total' classes by their top score
            ranked_classes = sorted(
                eligible,
                key=lambda k: scored_by_class[k][0][1] if scored_by_class[k] else 0.0,
                reverse=True
            )
            for k in ranked_classes[:total]:
                alloc[k] = 1
            remaining = 0

        # 3) distribute remaining proportionally to raw quotas
        if remaining > 0 and must > 0:
            shares = {k: (raw[k] / must) * remaining for k in raw}
            floors = {k: int(np.floor(shares[k])) for k in raw}
            # respect capacity per class
            for k in floors:
                floors[k] = min(floors[k], max(0, len(scored_by_class[k]) - alloc[k]))
            used = sum(floors.values())
            for k, v in floors.items():
                alloc[k] += v
            left = max(0, remaining - used)

            if left > 0:
                # give leftovers by largest fractional remainder, respecting capacity
                rema = sorted(
                    [(k, shares[k] - np.floor(shares[k])) for k in raw],
                    key=lambda x: x[1],
                    reverse=True
                )
                i = 0
                while left > 0 and i < len(rema):
                    k = rema[i][0]
                    if alloc[k] < len(scored_by_class[k]):
                        alloc[k] += 1
                        left -= 1
                    i += 1
                # still left? round-robin any remaining capacity
                if left > 0:
                    for k in list(raw.keys()):
                        while left > 0 and alloc[k] < len(scored_by_class[k]):
                            alloc[k] += 1
                            left -= 1
                            if left == 0:
                                break

        # 4) build final list exactly 'total' long
        chosen: List[Dict[str, Any]] = []
        leftovers: List[Dict[str, Any]] = []

        for k, pairs in scored_by_class.items():
            take = min(alloc.get(k, 0), len(pairs))
            for s, sc in pairs[:take]:
                chosen.append({"symbol": s, "market": k, "score": sc, "tf_supported": ["5m","15m","1h","1d"]})
            for s, sc in pairs[take:]:
                leftovers.append({"symbol": s, "market": k, "score": sc, "tf_supported": ["5m","15m","1h","1d"]})

        if len(chosen) < total and leftovers:
            leftovers.sort(key=lambda r: r["score"], reverse=True)
            chosen.extend(leftovers[: max(0, total - len(chosen))])

        chosen = sorted(chosen, key=lambda r: r["score"], reverse=True)[:total]
        return chosen

    def _build_pure(self, total:int) -> List[Dict[str, Any]]:
        # flatten all candidates and rank globally
        pools = {
            "crypto":      (self._crypto,      60),
            "forex":       (self._forex,       60),
            "commodities": (self._comm,        40),
            "stocks":      (self._stocks,      80),
        }
        scored: List[Dict[str, Any]] = []
        for k, (ad, cap) in pools.items():
            for s in self._candidates(ad, cap):
                sc = self._score(k, s)
                if sc > 0:
                    scored.append({"symbol": s, "market": k, "score": sc, "tf_supported": ["5m","15m","1h","1d"]})
        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:total]

    def _compute_top(self, limit:int=20) -> List[Dict[str, Any]]:
        targets_key = ",".join(f"{k}:{int(self.targets.get(k,0))}" for k in sorted(self.targets.keys()))
        key = f"{self.mode}:{self.metric}:targets:{targets_key}:top:{limit}"
        cached = self._cache.get(key)
        if cached and (time.time() - cached[0]) <= self.cache_ttl:
            return cached[1]

        total = max(6, min(20, int(limit)))
        out = self._build_balanced(total) if self.mode == "balanced" else self._build_pure(total)
        # strip score before returning
        cleaned = [{"symbol": r["symbol"], "name": r["symbol"].replace("=F","").replace("^",""), "market": r["market"], "tf_supported": r["tf_supported"]} for r in out]

        self._cache[key] = (time.time(), cleaned)
        return cleaned

    # ---- BaseAdapter API ----
    def list_universe(self, limit:int=20) -> List[Dict[str, Any]]:
        return self._compute_top(limit)

    def fetch_ohlcv(self, symbol: str, tf: str, bars:int) -> pd.DataFrame:
        mkey = self._guess_market(symbol)
        ad = {"crypto": self._crypto, "forex": self._forex, "commodities": self._comm, "stocks": self._stocks}.get(mkey, self._stocks)
        return ad.fetch_ohlcv(symbol, tf, bars)

    def fetch_window(self, symbol:str, tf:str, start_ms:int, end_ms:int) -> pd.DataFrame:
        mkey = self._guess_market(symbol)
        ad = {"crypto": self._crypto, "forex": self._forex, "commodities": self._comm, "stocks": self._stocks}.get(mkey, self._stocks)
        return ad.fetch_window(symbol, tf, start_ms, end_ms)
        
    def clear_cache(self):
        self._cache.clear()

# ---------- Registry & dynamic loading ----------
ADAPTERS: Dict[str, Callable[[], BaseAdapter]] = {
    "crypto": lambda: CryptoCCXT(
        os.getenv("EXCHANGE", "kraken"),
        os.getenv("QUOTE", "USD"),
        curated=[s.strip().upper() for s in os.getenv("CURATED","BTC,ETH,SOL,XRP,ADA,DOGE,LINK,LTC,BCH,TRX,DOT,ATOM,XLM,ETC,MATIC,UNI,APT,ARB,OP,AVAX,NEAR,ALGO,FIL,SUI,SHIB,USDC,USDT,XMR,AAVE,PAXG,ONDO,PEPE,SEI,IMX,TIA").split(",") if s.strip()]
    ),
    "binance_perps": lambda: BinancePerpsUSDT(),
    "forex": ForexAdapter,
    "commodities": CommoditiesAdapter,
    "stocks": StocksAdapter,
    "top_traded_pure": lambda: MixedTopTradedAdapter(mode="pure"),
    "top_traded_bal": lambda: MixedTopTradedAdapter(mode="balanced"),
}

# SINGLETON CACHE for adapters (so ccxt/yf clients + in-adapter caches are reused)
_ADAPTER_INSTANCES: Dict[str, BaseAdapter] = {}

def register_adapter(key: str, factory: Callable[[], BaseAdapter]):
    ADAPTERS[key.lower()] = factory

# --- replace your current get_adapter with this normalized + cached version ---
def get_adapter(name: Optional[str] = None) -> BaseAdapter:
    """
    Select adapter by query param or env ADAPTER; accept long-form names like 'crypto:kraken:usd'.
    Returns a singleton instance per logical adapter (crypto, binance_perps, forex, ...).
    """
    raw = (name or os.getenv("ADAPTER", "crypto")).lower().strip()
    key = raw.split(":", 1)[0]  # 'crypto:kraken:usd' -> 'crypto'

    aliases = {
        "crypto": "crypto",
        "binance_perps": "binance_perps",
        "futures": "binance_perps",
        "binance": "binance_perps",
        "forex": "forex",
        "commodities": "commodities",
        "stocks": "stocks",
        "top_traded_pure": "top_traded_pure",
        "top_traded_bal": "top_traded_bal",
    }
    key = aliases.get(key, key)

    inst = _ADAPTER_INSTANCES.get(key)
    if inst is None:
        factory = ADAPTERS.get(key) or ADAPTERS["crypto"]
        inst = factory()
        _ADAPTER_INSTANCES[key] = inst
    return inst