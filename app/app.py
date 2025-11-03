import os
import sys

# Додаємо корінь проєкту до системного шляху
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# ==============================
# ✅ UNIVERSAL IMPORT & MODEL_DIR HANDLER
# ==============================
import os
import sys
import time
import json
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
from ai.load_training_data import load_training_data

# Поточна директорія
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))

if root_dir not in sys.path:
    sys.path.append(root_dir)

MODEL_DIR = os.path.join(root_dir, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
print(f"✅ MODEL_DIR активний шлях: {MODEL_DIR}")

# ============================================================
# ⚙️ Шляхи
# ============================================================
MODEL_PATH = os.path.join(MODEL_DIR, "transformer_signal_model.pt")
SCALER_PATH = os.path.join(MODEL_DIR, "transformer_scaler.joblib")
TRAIN_DATA_PATH = os.path.join(MODEL_DIR, "train_data.json")

FEATURE_COLS = ["ema_diff5", "rsi5", "atr", "volz5", "trend_accel"]

# ============================================================
# 🧠 Dataset
# ============================================================
class SignalDataset(DataLoader):
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
# 🏋️‍♂️ Train / Save
# ============================================================
def train_transformer(epochs=15, batch_size=32, seq_len=50):
    try:
        if not os.path.exists(TRAIN_DATA_PATH):
            print("⚠️ Немає train_data.json — створюю заново.")
            load_training_data(limit=20000)

        with open(TRAIN_DATA_PATH, "r") as f:
            data = json.load(f)

        df = pd.DataFrame(data)
        if any(col not in df.columns for col in FEATURE_COLS):
            print("⚠️ У train_data.json бракує фіч — перевантажую дані.")
            load_training_data(limit=20000)
            df = pd.DataFrame(json.load(open(TRAIN_DATA_PATH)))

        df["strength"] = np.random.uniform(0, 1, len(df))
        df = df[FEATURE_COLS + ["strength"]].fillna(0)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[FEATURE_COLS])
        dump(scaler, SCALER_PATH)

        y = df["strength"].values.reshape(-1, 1)
        data_mat = np.hstack([X_scaled, y])

        dataset = SignalDataset(data_mat, seq_len)
        if len(dataset) < 50:
            print("⚠️ Dataset занадто малий для тренування.")
            return

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model = SignalTransformer(input_dim=len(FEATURE_COLS))
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
        send_message("🤖 Модель успішно перевчена та збережена! ✅")

    except Exception as e:
        print(f"❌ Помилка під час тренування: {e}")
        traceback.print_exc()


# ============================================================
# 🔮 Predict
# ============================================================
def predict_strength(features_dict):
    try:
        scaler = load(SCALER_PATH)
        model = SignalTransformer(input_dim=len(FEATURE_COLS))
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()

        x = np.array([[features_dict[c] for c in FEATURE_COLS]], dtype=float)
        x_scaled = scaler.transform(x)
        x_t = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred = model(x_t).item()
        strength = float(pred * 100)
        send_message(f"📊 AI сигнал: {strength:.2f}%")
        return strength

    except Exception as e:
        print(f"⚠️ predict_strength error: {e}")
        send_message(f"⚠️ Помилка під час прогнозу сили сигналу: {e}")
        return 50.0


# ============================================================
# 🚀 Main (ініціалізація)
# ============================================================
if __name__ == "__main__":
    # Якщо дані відсутні або старі — оновлюємо
    if not os.path.exists(TRAIN_DATA_PATH):
        load_training_data(limit=20000)
    else:
        mtime = os.path.getmtime(TRAIN_DATA_PATH)
        age_hours = (time.time() - mtime) / 3600
        if age_hours > 24:
            print("🔁 Оновлюю train_data.json...")
            load_training_data(limit=20000)

    # 🧠 Тренування
    train_transformer(epochs=20, seq_len=10)

    # 🔍 Автоматичний тест після навчання
    try:
        with open(TRAIN_DATA_PATH, "r") as f:
            data = json.load(f)
            if len(data) > 0:
                sample = data[-1]
                send_message("🧪 Виконую тестове прогнозування після навчання...")
                result = predict_strength(sample)
                send_message(f"✅ Тестове прогнозування виконано. AI сила сигналу: {result:.2f}%")
    except Exception as e:
        print("⚠️ Не вдалося виконати автотест:", e)
        send_message(f"⚠️ Не вдалося виконати автотест: {e}")

