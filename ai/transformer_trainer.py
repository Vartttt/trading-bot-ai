# ==============================
# ✅ UNIVERSAL IMPORT & MODEL_DIR HANDLER
# ==============================
import os
import sys

# Поточна директорія файлу (ai/)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Коренева директорія проєкту (/workspaces/)
root_dir = os.path.abspath(os.path.join(current_dir, ".."))

# Додаємо root у sys.path
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Імпорт конфігурації (config/config.py)
try:
    from config.config import MODEL_DIR
except ModuleNotFoundError:
    print("⚠️  Не знайдено модуль 'config'. Спробую додати шлях вручну...")
    sys.path.append(os.path.join(root_dir, "config"))
    from config import config
    MODEL_DIR = getattr(config, "MODEL_DIR", os.path.join(root_dir, "models"))
    print("✅ MODEL_DIR імпортовано після ручного додавання шляху.")

# Перевіряємо існування MODEL_DIR
os.makedirs(MODEL_DIR, exist_ok=True)
print(f"✅ MODEL_DIR активний шлях: {MODEL_DIR}")

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

# ============================================================
# ⚙️ Шляхи
# ============================================================
MODEL_PATH = os.path.join(MODEL_DIR, "transformer_signal_model.pt")
SCALER_PATH = os.path.join(MODEL_DIR, "transformer_scaler.joblib")
TRAIN_DATA_PATH = os.path.join(MODEL_DIR, "train_data.json")

# ============================================================
# ⚙️ Завантаження історичних свічок
# ============================================================
def load_training_data(symbol="BTCUSDT", interval="15m", limit=20000):
    """
    Завантажує історичні свічки з MEXC і формує DataFrame
    """
    print(f"📊 Завантажую {limit} свічок з MEXC для {symbol} ({interval})...")
    url = f"https://api.mexc.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"

    try:
        r = requests.get(url, timeout=15)
        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            raise ValueError("Некоректна відповідь API MEXC")

        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "trades", "taker_base",
            "taker_quote", "ignore"
        ])
        df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
        df.dropna(inplace=True)

        # Технічні індикатори
        df["ema_diff5"] = df["close"].ewm(span=9).mean() - df["close"].ewm(span=21).mean()
        df["rsi5"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()
        df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], 14).average_true_range()
        df["volz5"] = (df["volume"] - df["volume"].rolling(20).mean()) / (df["volume"].rolling(20).std() + 1e-9)
        df.dropna(inplace=True)

        df = df.tail(limit)
        df_out = df[["ema_diff5", "rsi5", "atr", "volz5"]].to_dict(orient="records")

        os.makedirs(os.path.dirname(TRAIN_DATA_PATH), exist_ok=True)
        with open(TRAIN_DATA_PATH, "w") as f:
            json.dump(df_out, f, indent=2)

        print(f"✅ Дані збережено: {TRAIN_DATA_PATH} ({len(df_out)} рядків)")
        return df_out

    except Exception as e:
        print(f"❌ Помилка при завантаженні історії: {e}")
        return []


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
            d_model=embed_dim, nhead=n_heads, dim_feedforward=ff_dim, dropout=0.1, batch_first=True
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
# 🏋️‍♂️ Train / Save
# ============================================================
def train_transformer(epochs=15, batch_size=32, seq_len=50):
    if not os.path.exists(TRAIN_DATA_PATH):
        print("⚠️ Немає train_data.json — спочатку згенеруй історію.")
        return

    with open(TRAIN_DATA_PATH, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    if df.empty:
        print("❌ train_data.json порожній!")
        return

    feature_cols = ["ema_diff5", "rsi5", "atr", "volz5"]
    df = df[feature_cols].fillna(0)
    df["strength"] = np.random.uniform(0, 1, len(df))  # тимчасова мітка

    print(f"📊 Рядків до тренування: {len(df)}")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    dump(scaler, SCALER_PATH)

    y = df["strength"].values.reshape(-1, 1)
    data_mat = np.hstack([X_scaled, y])
    dataset = SignalDataset(data_mat, seq_len)

    if len(dataset) < 50:
        print("⚠️ Dataset занадто малий — потрібно мінімум 100 рядків.")
        return

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = SignalTransformer(input_dim=len(feature_cols))
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        total_loss = 0
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


# ============================================================
# 🔮 Predict
# ============================================================
def predict_strength(features_dict):
    try:
        feature_cols = ["ema_diff5", "rsi5", "atr", "volz5"]
        scaler = load(SCALER_PATH)
        model = SignalTransformer(input_dim=len(feature_cols))
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()

        x = np.array([[features_dict[c] for c in feature_cols]], dtype=float)
        x_scaled = scaler.transform(x)
        x_t = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred = model(x_t).item()
        return float(pred * 100)

    except Exception as e:
        print(f"⚠️ predict_strength error: {e}")
        return 50.0


# ============================================================
# 🚀 Main (ініціалізація)
# ============================================================
if __name__ == "__main__":
    if not os.path.exists(TRAIN_DATA_PATH):
        load_training_data(limit=20000)
    else:
        mtime = os.path.getmtime(TRAIN_DATA_PATH)
        age_hours = (time.time() - mtime) / 3600
        if age_hours > 24:
            print("🔁 Оновлюю train_data.json...")
            load_training_data(limit=20000)

    train_transformer(epochs=20, seq_len=10)




