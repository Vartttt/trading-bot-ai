import pandas as pd
import ta
from ai.transformer_trainer import predict_strength

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        print("⚠️ enrich: отримано порожній DataFrame")
        return df

    required_cols = {"close", "high", "low", "volume"}
    if not required_cols.issubset(df.columns):
        print(f"⚠️ enrich: відсутні необхідні колонки: {required_cols - set(df.columns)}")
        return df

    x = df.copy()
    try:
        x["ema9"]  = ta.trend.EMAIndicator(close=x["close"], window=9).ema_indicator()
        x["ema21"] = ta.trend.EMAIndicator(close=x["close"], window=21).ema_indicator()
        ema_diff = x["ema9"] - x["ema21"]

        macd = ta.trend.MACD(close=x["close"])
        x["macd"]  = macd.macd()
        x["macds"] = macd.macd_signal()

        x["rsi"] = ta.momentum.RSIIndicator(close=x["close"], window=14).rsi()
        atr = ta.volatility.AverageTrueRange(high=x["high"], low=x["low"], close=x["close"], window=14)
        x["atr"] = atr.average_true_range()

        if x["volume"].notna().sum() >= 20:
            m = x["volume"].tail(20).mean()
            s = x["volume"].tail(20).std() or 1.0
            x["volz"] = (x["volume"] - m) / s
        else:
            x["volz"] = 0.0

        x.dropna(inplace=True)
        if x.empty:
            print("⚠️ enrich: після обчислень залишилось 0 рядків.")
            return df

        trend_accel = float(ema_diff.diff().iloc[-1]) if len(ema_diff) > 1 else 0.0
        features = {
            "ema_diff5": float(ema_diff.iloc[-1]),
            "rsi5":       float(x["rsi"].iloc[-1]),
            "atr":        float(x["atr"].iloc[-1]),
            "volz5":      float(x["volz"].iloc[-1]),
            "trend_accel": trend_accel,
        }

        # ВАЖЛИВО: список рядків
        strength = float(predict_strength([features]))
        x.loc[:, "signal_strength"] = strength
        print(f"🤖 AI signal_strength: {strength:.2f}")
        return x

    except Exception as e:
        print(f"❌ enrich: помилка у розрахунках — {e}")
        return df


