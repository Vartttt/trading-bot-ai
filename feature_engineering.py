import numpy as np
import pandas as pd
import ta

def build_features(df: pd.DataFrame) -> np.ndarray:
    df = df.copy()
    df["ema_fast"] = ta.trend.ema_indicator(df["close"], 21)
    df["ema_slow"] = ta.trend.ema_indicator(df["close"], 55)
    df["rsi"] = ta.momentum.rsi(df["close"], 14)
    df["macd"] = ta.trend.macd_diff(df["close"])
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], 14)
    df["vol_change"] = df["volume"].pct_change()
    df["price_change"] = df["close"].pct_change()
    df.dropna(inplace=True)
    features = df[["ema_fast","ema_slow","rsi","macd","atr","vol_change","price_change"]].iloc[-1].values
    return features

