# ==============================
# ✅ UNIVERSAL IMPORT & MODEL_DIR HANDLER
# ==============================
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

try:
    from config.config import MODEL_DIR
except ModuleNotFoundError:
    print("⚠️ Не знайдено модуль 'config'. Додаю вручну...")
    sys.path.append(os.path.join(root_dir, "config"))
    from config import config
    MODEL_DIR = getattr(config, "MODEL_DIR", os.path.join(root_dir, "models"))
    print("✅ MODEL_DIR імпортовано вручну.")

os.makedirs(MODEL_DIR, exist_ok=True)
print(f"✅ MODEL_DIR активний шлях: {MODEL_DIR}")

# --- дефолтні фічі ---
DEFAULT_FEATURE_COLS = ["ema_diff5", "rsi5", "atr", "volz5", "trend_accel"]
TARGET_COLS = ["next_return", "target"]

# ==============================
# 🔚 END OF UNIVERSAL IMPORT FIX
# ==============================
import json
import time
import traceback
import requests
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import ta
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from notifier.telegram_notifier import send_message

def _resolve_feature_cols(df: pd.DataFrame):
    """
    Повертає список колонок-ознак, які реально присутні у df.
    1) намагаємось взяти перетин із DEFAULT_FEATURE_COLS;
    2) якщо нічого не знайшли — беремо всі числові колонки,
       крім цільових і сирих OHLCV/часу.
    """
    # 1) перетин із дефолтним списком
    cols = [c for c in DEFAULT_FEATURE_COLS if c in df.columns]

    # 2) фолбек — всі числові, крім заборонених
    if not cols:
        blacklist = set(TARGET_COLS + ["open_time", "open", "high", "low", "close", "volume"])
        cols = [c for c in df.columns
                if c not in blacklist and pd.api.types.is_numeric_dtype(df[c])]

    if not cols:
        raise ValueError("Не вдалося визначити feature_cols — у DataFrame немає придатних ознак.")
    return cols

# ============================================================
# ⚙️ Шляхи
# ============================================================
MODEL_PATH = os.path.join(MODEL_DIR, "transformer_signal_model.pt")
SCALER_PATH = os.path.join(MODEL_DIR, "transformer_scaler.joblib")
TRAIN_DATA_PATH = os.path.join(MODEL_DIR, "train_data.json")
FLAG_PATH = "/tmp/last_auto_retrain.txt"
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "feature_cols.json")

# ============================================================
# ⚙️ Завантаження історичних свічок
# ============================================================
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

        # 🧠 НОВА ФІЧА — прискорення тренду
        df["trend_accel"] = df["ema_diff5"].diff()

        df.dropna(inplace=True)

        # Останні 20 000 рядків
        df = df.tail(limit)

        # Зберігаємо у JSON
        df_out = df[["ema_diff5", "rsi5", "atr", "volz5", "trend_accel"]].to_dict(orient="records")

        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(TRAIN_DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(df_out, f, ensure_ascii=False, indent=2)

        print(f"✅ Дані збережено: models/train_data.json ({len(df_out)} рядків)")
        return df_out

    except Exception as e:
        print(f"❌ Помилка при завантаженні історії: {e}")
        return []

def ensure_artifacts():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # якщо feature_cols.json відсутній — створюємо з дефолтним списком
    if not os.path.exists(FEATURE_COLS_PATH):
        with open(FEATURE_COLS_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_FEATURE_COLS, f, ensure_ascii=False, indent=2)
        print(f"🆕 Створено {FEATURE_COLS_PATH} (дефолтні фічі).")
    else:
        print(f"✅ {os.path.basename(FEATURE_COLS_PATH)} існує.")

# ============================================================
# 🧠 Dataset
# ============================================================
class SignalDataset(Dataset):
    def __init__(self, data, seq_len=50):
        X, y = [], []
        for i in range(len(data) - seq_len):
            seq = data[i:i + seq_len, :-1]
            target = data[i + seq_len, -1]
            X.append(seq)
            y.append(target)
        self.X = np.array(X, dtype=np.float32)
        self.y = np.array(y, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================
# 🧩 Transformer Model
# ============================================================
class SignalTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=64, n_heads=4, ff_dim=128, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.embedding(x)
        encoded = self.encoder(x)
        out = encoded[:, -1, :]
        return self.fc(out)

# ============================================================
# 🏋️‍♂️ Train / Save  (виправлено)
# ============================================================
def train_transformer(epochs=20, batch_size=32, seq_len=50):
    try:
        ensure_artifacts()

        if not os.path.exists(TRAIN_DATA_PATH):
            print("⚠️ Немає train_data.json — створюю...")
            load_training_data(limit=20000)

        df = pd.read_json(TRAIN_DATA_PATH, orient="records")
        if df.empty:
            raise ValueError("train_data.json порожній!")

        # визначаємо та зберігаємо порядок фіч
        feature_cols = _resolve_feature_cols(df)
        with open(FEATURE_COLS_PATH, "w", encoding="utf-8") as f:
            json.dump(feature_cols, f, ensure_ascii=False, indent=2)
        print(f"✅ Використані фічі: {feature_cols}")

        # X — тільки фічі, y — окремо (тимчасовий таргет-заглушка)
        X = df[feature_cols].fillna(0).values
        y = np.random.uniform(0, 1, len(df)).reshape(-1, 1)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        dump(scaler, SCALER_PATH)

        # матриця для датасету: остання колонка — це таргет
        data_mat = np.hstack([X_scaled, y])

        dataset = SignalDataset(data_mat, seq_len)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = SignalTransformer(input_dim=len(feature_cols))
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        for epoch in range(epochs):
            total_loss = 0.0
            for Xb, yb in loader:
                optimizer.zero_grad()
                preds = model(Xb).squeeze()
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"🧠 Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.6f}")

        torch.save(model.state_dict(), MODEL_PATH)
        print(f"✅ Модель збережено: {MODEL_PATH}")
        send_message("🤖 Модель успішно перевчена та збережена! ✅")

        with open(FLAG_PATH, "w") as f:
            f.write(str(time.time()))

    except Exception as e:
        print(f"❌ Помилка тренування: {e}")
        traceback.print_exc()
        send_message(f"⚠️ Помилка при тренуванні моделі: {e}")

# ============================================================
# 🔮 Predict  (реальний інференс)
# ============================================================
def predict_strength(feature_rows, seq_len=50):
    """
    feature_rows: список dict'ів або DataFrame з тими самими фічами, що й під час тренування.
                  Має містити принаймні `seq_len` останніх рядків (буде паддінг, якщо менше).
    Повертає: відсоток сили сигналу (0..100).
    """
    try:
        ensure_artifacts()

        # 1) Завантажуємо порядок фіч
        if os.path.exists(FEATURE_COLS_PATH):
            with open(FEATURE_COLS_PATH, "r", encoding="utf-8") as f:
                feature_cols = json.load(f)
        else:
            feature_cols = DEFAULT_FEATURE_COLS
        if not feature_cols:
            raise ValueError("feature_cols.json порожній — немає списку фіч.")

        # 2) Завантажуємо scaler і модель
        scaler = load(SCALER_PATH)
        model = SignalTransformer(input_dim=len(feature_cols))
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()

        # 3) Готуємо дані
        if isinstance(feature_rows, pd.DataFrame):
            df_infer = feature_rows.copy()
        else:
            df_infer = pd.DataFrame(feature_rows)

        X = df_infer[feature_cols].fillna(0).values
        X_scaled = scaler.transform(X)

        # Паддінг до seq_len (повторюємо останній рядок)
        if X_scaled.shape[0] < seq_len:
            if X_scaled.shape[0] == 0:
                raise ValueError("Немає жодного рядка для інференсу.")
            pad = np.repeat(X_scaled[-1:], seq_len - X_scaled.shape[0], axis=0)
            X_scaled = np.vstack([X_scaled, pad])

        # 4) Інференс
        x_seq = torch.tensor(X_scaled[-seq_len:], dtype=torch.float32).unsqueeze(0)  # (1, seq_len, input_dim)
        with torch.no_grad():
            pred = float(model(x_seq).squeeze().item())

        pred_pct = max(0.0, min(1.0, pred)) * 100.0
        send_message(f"📊 Сила сигналу ШІ: {pred_pct:.2f}%")
        return round(pred_pct, 2)

    except Exception as e:
        print(f"⚠️ predict_strength error: {e}")
        traceback.print_exc()
        send_message(f"⚠️ Помилка під час прогнозу сили сигналу: {e}")
        return 50.0

# ============================================================
# 🚀 Main
# ============================================================
if __name__ == "__main__":
    send_message("🚀 AI Transformer запущений. Починаю тренування моделі...")
    if not os.path.exists(TRAIN_DATA_PATH):
        load_training_data(limit=20000)
    else:
        mtime = os.path.getmtime(TRAIN_DATA_PATH)
        if (time.time() - mtime) / 3600 > 24:
            print("🔁 Оновлюю train_data.json...")
            load_training_data(limit=20000)

    train_transformer(epochs=20, seq_len=10)
    send_message("✅ Навчання завершено, модель готова до роботи!")






