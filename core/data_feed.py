import os, ccxt, pandas as pd

EXCHANGE = os.getenv("EXCHANGE", "mexc3")
DEFAULT_TYPE = os.getenv("DEFAULT_TYPE", "swap")

def _client():
    cls = getattr(ccxt, EXCHANGE) if hasattr(ccxt, EXCHANGE) else ccxt.mexc3
    return cls({"enableRateLimit": True, "options": {"defaultType": DEFAULT_TYPE}})

def get_ohlcv(symbol: str, timeframe: str = "5m", limit: int = 200) -> pd.DataFrame | None:
    ex = _client()
    data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not data:
        return None
    df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df
