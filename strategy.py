# strategy.py
import numpy as np
import ta
from config import TP_MULTIPLIERS, SL_ATR_MULT, MIN_STRENGTH
import pandas as pd

def compute_indicators(df):
    df = df.copy()
    df["close"] = df["close"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["volume"] = df["volume"].astype(float)

    df["ema9"] = ta.trend.EMAIndicator(df["close"], window=9).ema_indicator()
    df["ema21"] = ta.trend.EMAIndicator(df["close"], window=21).ema_indicator()
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    df["vol_z"] = (df["volume"] - df["volume"].rolling(50).mean()) / (df["volume"].rolling(50).std() + 1e-9)
    df.dropna(inplace=True)
    return df

def score_signal(latest, previous):
    score = 0
    # EMA cross (30)
    if previous["ema9"] < previous["ema21"] and latest["ema9"] > latest["ema21"]:
        score += 30
    if previous["ema9"] > previous["ema21"] and latest["ema9"] < latest["ema21"]:
        score += 30
    # RSI (15)
    if latest["rsi"] < 45:
        score += 15
    if latest["rsi"] > 55:
        score += 15
    # MACD (20)
    if latest["macd"] > latest["macd_signal"]:
        score += 20
    if latest["macd"] < latest["macd_signal"]:
        score += 20
    # volume (15)
    if latest["vol_z"] > 0.5:
        score += 15
    return min(score, 100)

def generate_signal(df_5m, df_15m=None):
    if len(df_5m) < 3:
        return None
    latest = df_5m.iloc[-1]
    previous = df_5m.iloc[-2]

    long_cross = previous["ema9"] < previous["ema21"] and latest["ema9"] > latest["ema21"]
    short_cross = previous["ema9"] > previous["ema21"] and latest["ema9"] < latest["ema21"]

    tf_confirm = True
    if df_15m is not None and len(df_15m) >= 2:
        last15 = df_15m.iloc[-1]
        if long_cross:
            tf_confirm = last15["ema9"] > last15["ema21"]
        if short_cross:
            tf_confirm = last15["ema9"] < last15["ema21"]

    strength = score_signal(latest, previous)
    if not tf_confirm:
        strength = int(strength * 0.8)

    if long_cross and strength >= MIN_STRENGTH:
        entry = float(latest["close"])
        atr = float(latest["atr"] or 0.0)
        sl = entry - SL_ATR_MULT * atr
        tps = [entry * m for m in TP_MULTIPLIERS]
        return {"signal": "LONG", "entry": entry, "sl": sl, "tps": tps, "strength": int(strength)}
    if short_cross and strength >= MIN_STRENGTH:
        entry = float(latest["close"])
        atr = float(latest["atr"] or 0.0)
        sl = entry + SL_ATR_MULT * atr
        tps = [entry * (2 - m) for m in TP_MULTIPLIERS]
        return {"signal": "SHORT", "entry": entry, "sl": sl, "tps": tps, "strength": int(strength)}
    return None

def predict_cross_eta(df_1m):
    if len(df_1m) < 6:
        return None
    if "ema9" not in df_1m.columns or "ema21" not in df_1m.columns:
        df_1m = compute_indicators(df_1m)
    series = (df_1m["ema9"] - df_1m["ema21"]).dropna().values[-6:]
    if len(series) < 4:
        return None
    x = np.arange(len(series))
    a, b = np.polyfit(x, series, 1)
    if abs(a) < 1e-8:
        return None
    t_future = - (series[-1]) / a
    if t_future < 0 or t_future > 30:
        return None
    return float(t_future)
