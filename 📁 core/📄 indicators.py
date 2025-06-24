import pandas as pd
import requests
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

def get_klines(symbol, interval="15m", limit=100):
    r = requests.get(f"https://api.mexc.com/api/v3/klines", params={
        "symbol": symbol.replace("_", ""),
        "interval": interval,
        "limit": limit
    })
    data = r.json()
    df = pd.DataFrame(data, columns=[
        "time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["close"] = pd.to_numeric(df["close"])
    df["volume"] = pd.to_numeric(df["volume"])
    return df

def get_indicators(symbol):
    df = get_klines(symbol)
    close = df["close"]
    volume = df["volume"]

    ema_200 = EMAIndicator(close, 200).ema_indicator()
    macd = MACD(close).macd_diff()
    rsi = RSIIndicator(close).rsi()
    adx = ADXIndicator(df["high"], df["low"], df["close"]).adx()

    return {
        "ema_200": "above" if close.iloc[-1] > ema_200.iloc[-1] else "below",
        "macd": "bullish" if macd.iloc[-1] > 0 else "bearish",
        "rsi": rsi.iloc[-1],
        "volume": "up" if volume.iloc[-1] > volume.mean() else "down",
        "adx": adx.iloc[-1],
        "trend": "up" if close.iloc[-1] > ema_200.iloc[-1] else "down"
    }
