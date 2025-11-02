import pandas as pd
import ta
from ai.transformer_trainer import predict_strength

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """
    🧩 Обробляє вхідні дані з біржі:
    додає EMA, MACD, RSI, ATR, vol-z та силу сигналу (AI).
    """
    if df is None or df.empty:
        print("⚠️ enrich: отримано порожній DataFrame")
        return df

    required_cols = {"close", "high", "low", "volume"}
    if not required_cols.issubset(df.columns):
        print(f"⚠️ enrich: відсутні необхідні колонки: {required_cols - set(df.columns)}")
        return df

    x = df.copy()

    try:
        # 📈 EMA
        x["ema9"]  = ta.trend.EMAIndicator(close=x["close"], window=9).ema_indicator()
        x["ema21"] = ta.trend.EMAIndicator(close=x["close"], window=21).ema_indicator()
        ema_diff = x["ema9"] - x["ema21"]

        # 📊 MACD
        macd = ta.trend.MACD(close=x["close"])
        x["macd"]  = macd.macd()
        x["macds"] = macd.macd_signal()

        # 💪 RSI
        x["rsi"] = ta.momentum.RSIIndicator(close=x["close"], window=14).rsi()

        # 📉 ATR
        atr = ta.volatility.AverageTrueRange(
            high=x["high"], low=x["low"], close=x["close"], window=14
        )
        x["atr"] = atr.average_true_range()

        # 🔊 Volume Z-score (volz)
        if x["volume"].notna().sum() >= 20:
            vol_mean = x["volume"].tail(20).mean()
            vol_std  = x["volume"].tail(20).std() or 1.0
            x["volz"] = (x["volume"] - vol_mean) / vol_std
        else:
            x["volz"] = 0.0

        x.dropna(inplace=True)
        if x.empty:
            print("⚠️ enrich: після обчислень залишилось 0 рядків.")
            return df

        # 🧩 features для AI-моделі (імена узгоджені з тренером)
        trend_accel = float(ema_diff.diff().iloc[-1]) if len(ema_diff) > 1 else 0.0
        features = {
            "ema_diff5": float(ema_diff.iloc[-1]),
            "rsi5":       float(x["rsi"].iloc[-1]),
            "atr":        float(x["atr"].iloc[-1]),
            "volz5":      float(x["volz"].iloc[-1]),
            "trend_accel": trend_accel,
        }

        # 🧠 інференс — ПЕРЕДАЄМО СПИСОК рядків (або DataFrame)
        try:
            strength = float(predict_strength([features]))
            x.loc[:, "signal_strength"] = strength
            print(f"🤖 AI signal_strength: {strength:.2f}")
        except Exception as ai_error:
            print(f"⚠️ enrich: помилка при виклику predict_strength: {ai_error}")
            x.loc[:, "signal_strength"] = 50.0  # дефолтне значення

        return x

    except Exception as e:
        print(f"❌ enrich: помилка у розрахунках — {e}")
        return df


