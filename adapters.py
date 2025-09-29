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

class AdapterError(Exception): ...

class BaseAdapter:
    """Minimal interface your app uses."""
    name: str = "base"
    def list_universe(self, limit:int) -> List[Dict[str,Any]]: raise NotImplementedError
    def fetch_ohlcv(self, symbol:str, tf:str, bars:int) -> pd.DataFrame: raise NotImplementedError

    # Optional fast path: fetch by time window (ms); default falls back to fetch_ohlcv + slicing
    def fetch_window(self, symbol:str, tf:str, start_ms:int, end_ms:int) -> pd.DataFrame:
        per = tf_ms(tf)
        need = max(2, int((end_ms - start_ms)/per) + 8)
        df = self.fetch_ohlcv(symbol, tf, bars=min(2000, max(need, 240)))
        # normalize 'ts' to ms
        if np.issubdtype(df["ts"].dtype, np.datetime64):
            ms = df["ts"].astype("int64") // 10**6
        else:
            ms = df["ts"].astype("int64")
        keep = (ms >= (start_ms - per)) & (ms <= (end_ms + per))
        return df.loc[keep].reset_index(drop=True)

    # Precision helpers used by build_trade
    def price_to_precision(self, symbol:str, price:float) -> float: return float(round(price, 2))
    def amount_to_precision(self, symbol:str, qty:float) -> float:  return float(qty)

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

    def _exh(self):
        if self._ex is None:
            self._ex = getattr(ccxt, self.exchange_id)({"enableRateLimit": True, "timeout": 20000})
        return self._ex

    def _load_markets(self):
        if self._markets is None:
            self._markets = self._exh().load_markets()
        return self._markets

    def list_universe(self, limit: int = None, top: int = None) -> List[Dict[str, Any]]:
        if top is not None and limit is None:
            limit = top
        limit = int(limit or 20)
        m = self._load_markets()
        avail: List[str] = []
        # curated first
        for base in (self.curated or []):
            sym = f"{base}/{self.quote}"
            if sym in m and m[sym].get("active"):
                avail.append(sym)
        # fill remainder
        if len(avail) < limit:
            for sym, info in m.items():
                if info.get("quote")==self.quote and info.get("active"):
                    if sym not in avail:
                        avail.append(sym)
                        if len(avail) >= limit: break
        return [{"symbol": s, "name": s.split('/')[0], "market": self.name,
                 "tf_supported": ["5m","15m","1h","1d"]} for s in avail[:limit]]

    def _cache_get(self, key:str) -> Optional[pd.DataFrame]:
        v = self._cache.get(key)
        if not v: return None
        ts, df = v
        return df.copy() if (time.time()-ts) <= self.cache_ttl else None

    def _cache_set(self, key:str, df:pd.DataFrame):
        self._cache[key] = (time.time(), df.copy())

    def fetch_ohlcv(self, symbol:str, tf:str, bars:int) -> pd.DataFrame:
        ex = self._exh()
        tf_ex = TF_MAP.get(tf); 
        if tf_ex is None: raise AdapterError(f"Unsupported tf: {tf}")
        key = f"ohlcv:{self.name}:{symbol}:{tf_ex}:{bars}"
        c = self._cache_get(key)
        if c is not None: return c
        data = ex.fetch_ohlcv(symbol, timeframe=tf_ex, limit=min(bars+40, 2000))
        if not data: raise AdapterError("no candles")
        df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        self._cache_set(key, df)
        return df

    def fetch_window(self, symbol:str, tf:str, start_ms:int, end_ms:int) -> pd.DataFrame:
        ex = self._exh()
        tf_ex = TF_MAP.get(tf); 
        if tf_ex is None: raise AdapterError(f"Unsupported tf: {tf}")
        per = tf_ms(tf)
        need = max(2, int((end_ms - start_ms)/per) + 4)
        since = max(0, start_ms - 2*per)
        data = ex.fetch_ohlcv(symbol, timeframe=tf_ex, since=since, limit=min(need+20, 2000))
        if not data: raise AdapterError("no candles for window")
        df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
        # NOTE: window path keeps numeric ms (faster slicing downstream)
        return df[(df["ts"] >= (start_ms - per)) & (df["ts"] <= (end_ms + per))].reset_index(drop=True)

    def price_to_precision(self, symbol:str, price:float) -> float:
        return float(self._exh().price_to_precision(symbol, price))
    def amount_to_precision(self, symbol:str, qty:float) -> float:
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
                "timeout": 20000,
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
        tf_ex = TF_MAP.get(tf); 
        if tf_ex is None: raise AdapterError(f"Unsupported tf: {tf}")
        data = ex.fetch_ohlcv(symbol, timeframe=tf_ex, limit=min(bars+40, 1500))
        if not data: raise AdapterError("no candles")
        df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        return df

    def fetch_window(self, symbol:str, tf:str, start_ms:int, end_ms:int) -> pd.DataFrame:
        ex = self._exh()
        per = tf_ms(tf)
        need = max(2, int((end_ms - start_ms)/per) + 4)
        since = max(0, start_ms - 2*per)
        data = ex.fetch_ohlcv(symbol, timeframe=TF_MAP[tf], since=since, limit=min(need+20, 1500))
        if not data: raise AdapterError("no candles for window")
        df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
        return df[(df["ts"] >= (start_ms - per)) & (df["ts"] <= (end_ms + per))].reset_index(drop=True)

    def price_to_precision(self, symbol:str, price:float) -> float:
        return float(self._exh().price_to_precision(symbol, price))
    def amount_to_precision(self, symbol:str, qty:float) -> float:
        return float(self._exh().amount_to_precision(symbol, qty))

# ---------- yfinance generic + thin wrappers ----------
class YFAdapter(BaseAdapter):
    def __init__(self, name:str, tickers:List[str], price_decimals:int=4):
        _ensure(yf is not None, "yfinance not installed; add yfinance>=0.2.44")
        self.name = name
        self.tickers = tickers
        self.price_decimals = price_decimals

    def list_universe(self, limit:int) -> List[Dict[str,Any]]:
        syms = self.tickers[:max(1, limit)]
        return [{"symbol": s, "name": s.replace("=F","").replace("^",""), "market": self.name,
                 "tf_supported": ["15m","1h","1d"]} for s in syms]

    def _interval(self, tf:str) -> str:
        return {"5m":"5m", "15m":"15m", "1h":"60m", "1d":"1d"}.get(tf, "60m")

    def fetch_ohlcv(self, symbol:str, tf:str, bars:int) -> pd.DataFrame:
        df = yf.download(symbol, period="60d", interval=self._interval(tf), progress=False, prepost=False, threads=False, auto_adjust=True, actions=False)
        if df is None or df.empty:
            raise AdapterError("no candles from yfinance")
        df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
        df = df.reset_index(names="ts").tail(bars)
        return df[["ts","open","high","low","close","volume"]].copy()

    def price_to_precision(self, symbol:str, price:float) -> float:
        return float(round(price, self.price_decimals))

# “Add-on pack” factories (top-20 defaults)
def ForexAdapter() -> BaseAdapter:
    return YFAdapter("forex", ["EURUSD=X","GBPUSD=X","USDJPY=X","AUDUSD=X","USDCAD=X","USDCHF=X","NZDUSD=X","EURGBP=X","EURJPY=X","GBPJPY=X","AUDJPY=X","EURAUD=X","EURCAD=X","EURCHF=X","GBPCHF=X","AUDCAD=X","CHFJPY=X","AUDNZD=X","CADJPY=X","NZDJPY=X"], price_decimals=5)

def CommoditiesAdapter() -> BaseAdapter:
    return YFAdapter("commodities", ["GC=F","CL=F","SI=F","BZ=F","LCO=F","NG=F","HO=F","RB=F","HG=F","AH=F","PL=F","TIO=F","ZC=F","ZS=F","ZW=F","KC=F","SB=F","CC=F","CT=F","LBS=F"], price_decimals=2)

def StocksAdapter() -> BaseAdapter:
    return YFAdapter("stocks", ["AAPL","MSFT","NVDA","TSLA","SPY","QQQ","AMZN","INTC","PLTR","META","ORCL","AVGO","TQQQ","AMD","GOOGL","COST","IWM","MU","APP","GOOG"], price_decimals=2)

# ---------- Registry & dynamic loading ----------
ADAPTERS: Dict[str, Callable[[], BaseAdapter]] = {
    "crypto": lambda: CryptoCCXT(os.getenv("EXCHANGE","kraken"), os.getenv("QUOTE","USD"),
                                 curated=os.getenv("CURATED","BTC,ETH,SOL,XRP,ADA,DOGE").split(",")),
    "binance_perps": lambda: BinancePerpsUSDT(),
    "forex": ForexAdapter,
    "commodities": CommoditiesAdapter,
    "stocks": StocksAdapter,
}

def register_adapter(key:str, factory:Callable[[], BaseAdapter]):
    ADAPTERS[key.lower()] = factory

def get_adapter(name: Optional[str] = None) -> BaseAdapter:
    raw = (name or os.getenv("ADAPTER","crypto")).lower().strip()
    key = raw
    # accept descriptive variants
    if raw.startswith("crypto:"):            key = "crypto"
    elif raw.startswith("futures:binance"):  key = "binance_perps"
    # fallback
    factory = ADAPTERS.get(key) or ADAPTERS.get("crypto")
    return factory()