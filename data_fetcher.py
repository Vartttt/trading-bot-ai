# data_fetcher.py
import ccxt
import pandas as pd
import time
from config import EXCHANGE_ID

def get_exchange():
    ex_class = getattr(ccxt, EXCHANGE_ID)
    # no keys required for public OHLCV
    return ex_class({"enableRateLimit": True})

def ohlcv_to_df(ohlcv):
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("datetime", inplace=True)
    df.drop(columns=["ts"], inplace=True)
    return df

def fetch_ohlcv(symbol, timeframe="5m", limit=200, retries=2):
    ex = get_exchange()
    for i in range(retries):
        try:
            ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            return ohlcv_to_df(ohlcv)
        except Exception as e:
            if i + 1 < retries:
                time.sleep(1)
                continue
            raise
