import ccxt, pandas as pd, os

EXCHANGE = os.getenv("EXCHANGE", "mexc3")

def _client():
    cls = getattr(ccxt, EXCHANGE) if hasattr(ccxt, EXCHANGE) else ccxt.mexc3
    return cls({"enableRateLimit": True, "options": {"defaultType": os.getenv("DEFAULT_TYPE", "swap")}})

def get_ohlcv(symbol, timeframe="15m", limit=200):
    ex = _client()
    try:
        data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not data:
            return None
        df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        return df
    except Exception:
        return None
