# optimizer.py
import numpy as np, json, os
from indicators import df_from_ohlcv, add_indicators
import ccxt
BEST_PATH = "best_params.json"

def optimize(symbol="BTC/USDT", timeframe="15m"):
    ex = ccxt.mexc({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=1000, params={"type":"swap"})
    df = add_indicators(df_from_ohlcv(ohlcv))
    best, best_win = {}, -1
    for ema_fast in range(10, 30, 5):
        for ema_slow in range(40, 80, 5):
            win = ((df["ema_fast"].shift(1) < df["ema_slow"].shift(1)) &
                   (df["ema_fast"] > df["ema_slow"])).sum()
            if win > best_win:
                best = {"ema_fast": ema_fast, "ema_slow": ema_slow}
                best_win = win
    with open(BEST_PATH, "w") as f:
        json.dump(best, f)
    return best
