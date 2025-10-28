import pandas as pd, ta
from ai.transformer_trainer import predict_strength

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: 
        return df
    x = df.copy()
    x["ema9"]  = ta.trend.EMAIndicator(x["close"], 9).ema_indicator()
    x["ema21"] = ta.trend.EMAIndicator(x["close"], 21).ema_indicator()
    macd = ta.trend.MACD(x["close"])
    x["macd"]  = macd.macd()
    x["macds"] = macd.macd_signal()
    x["rsi"] = ta.momentum.RSIIndicator(x["close"], 14).rsi()
    x["atr"] = ta.volatility.AverageTrueRange(x["high"], x["low"], x["close"], 14).average_true_range()
    x.dropna(inplace=True)

    # 🧩 додаємо формування features для моделі
    last = x.iloc[-1]
    features = {
        "ema_diff5": float(last["ema9"] - last["ema21"]),
        "rsi5": float(last["rsi"]),
        "atr": float(last["atr"]),
        "volz5": float((last["volume"] - x["volume"].tail(20).mean()) / (x["volume"].tail(20).std() or 1))
    }

    # 🧠 передбачення сили сигналу
    strength = int(predict_strength(features))
    x["signal_strength"] = strength

    return x


