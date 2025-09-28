# adapters.py
from __future__ import annotations
from typing import List, Dict, Any
import time, pandas as pd, ccxt

class Adapter:
    def list_universe(self, limit:int) -> List[Dict[str,Any]]:
        raise NotImplementedError
    def fetch_ohlcv(self, symbol:str, tf:str, bars:int) -> pd.DataFrame:
        raise NotImplementedError
    def name(self) -> str:
        return "base"

# ---- Crypto (CCXT / Kraken by default) ----
class CryptoCCXT(Adapter):
    def __init__(self, exchange_id:str="kraken", quote:str="USD", curated:list[str]|None=None):
        self.exchange_id = exchange_id
        self.quote = quote
        self.curated = curated or []
        self._ex = None
        self._markets = None

    def _exh(self):
        if self._ex is None:
            self._ex = getattr(ccxt, self.exchange_id)({"enableRateLimit": True, "timeout": 20000})
        return self._ex

    def _load_markets(self):
        if self._markets is None:
            self._markets = self._exh().load_markets()
        return self._markets

    def list_universe(self, limit:int) -> List[Dict[str,Any]]:
        m = self._load_markets()
        avail = []
        # curated first
        for base in self.curated:
            sym = f"{base}/{self.quote}"
            if sym in m and m[sym].get("active"):
                avail.append(sym)
        # fill
        if len(avail) < limit:
            for sym, info in m.items():
                if info.get("quote")==self.quote and info.get("active"):
                    if sym not in avail:
                        avail.append(sym)
                        if len(avail) >= limit: break
        out = [{"symbol": s, "name": s.split('/')[0], "market": self.exchange_id, "tf_supported": ["5m","15m","1h","1d"]} for s in avail[:limit]]
        return out

    def fetch_ohlcv(self, symbol:str, tf:str, bars:int) -> pd.DataFrame:
        ex = self._exh()
        data = ex.fetch_ohlcv(symbol, timeframe=tf, limit=min(bars+40, 2000))
        df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
        return df

    def name(self) -> str:
        return f"crypto:{self.exchange_id}:{self.quote}"