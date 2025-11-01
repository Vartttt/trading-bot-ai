import pandas as pd
import ta
import requests
import json
import os

def load_training_data(symbol="BTCUSDT", interval="15m", limit=20000):
    print(f"📊 Завантажую {limit} свічок з MEXC для {symbol} ({interval})...")
    url = f"https://api.mexc.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"

    try:
        r = requests.get(url, timeout=15)
        data = r.json()

        if not isinstance(data, list):
            print("❌ Некоректна відповідь API MEXC:", data)
            return []

        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "trades", "taker_base",
            "taker_quote", "ignore"
        ])

        df = df.astype({
            "open": float, "high": float, "low": float, "close": float, "volume": float
        })
        df.dropna(inplace=True)

        # --- Індикатори ---
        df["ema9"] = ta.trend.EMAIndicator(df["close"], 9).ema_indicator()
        df["ema21"] = ta.trend.EMAIndicator(df["close"], 21).ema_indicator()
        df["ema_diff5"] = df["ema9"] - df["ema21"]

        df["rsi5"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()
        df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], 14).average_true_range()
        df["volz5"] = (df["volume"] - df["volume"].rolling(20).mean()) / (df["volume"].rolling(20).std() + 1e-9)

        # 🧠 Додаємо нову фічу — прискорення тренду
        df["trend_accel"] = df["ema_diff5"].diff()

        df.dropna(inplace=True)

        # Останні 20000 рядків
        df = df.tail(limit)

        # Зберігаємо у JSON
        df_out = df[["ema_diff5", "rsi5", "atr", "volz5", "trend_accel"]].to_dict(orient="records")

        os.makedirs("models", exist_ok=True)
        with open("models/train_data.json", "w") as f:
            json.dump(df_out, f, indent=2)

        print(f"✅ Дані збережено: models/train_data.json ({len(df_out)} рядків)")
        return df_out

    except Exception as e:
        print(f"❌ Помилка при завантаженні історії: {e}")
        return []

if __name__ == "__main__":
    load_training_data()
