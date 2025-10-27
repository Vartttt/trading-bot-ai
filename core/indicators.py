import ta, pandas as pd

def enrich(df):
    if df is None or df.empty:
        return df
    x = df.copy()
    x["ema9"] = ta.trend.EMAIndicator(x["close"], 9).ema_indicator()
    x["ema21"] = ta.trend.EMAIndicator(x["close"], 21).ema_indicator()
    macd = ta.trend.MACD(x["close"])
    x["macd"] = macd.macd()
    x["macds"] = macd.macd_signal()
    x["rsi"] = ta.momentum.RSIIndicator(x["close"], 14).rsi()
    x["atr"] = ta.volatility.AverageTrueRange(x["high"], x["low"], x["close"], 14).average_true_range()
    x.dropna(inplace=True)
    return x
