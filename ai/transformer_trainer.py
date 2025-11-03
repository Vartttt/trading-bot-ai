# ==============================
# ✅ UNIVERSAL IMPORT & MODEL_DIR HANDLER
# ==============================
import os
import sys
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

# --- Ініціалізація директорій ---
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

# ============================================================
# ⚙️ Шляхи
# ============================================================
MODEL_PATH = os.path.join(MODEL_DIR, "transformer_signal_model.pt")
SCALER_PATH = os.path.join(MODEL_DIR, "transformer_scaler.joblib")
TRAIN_DATA_PATH = os.path.join(MODEL_DIR, "train_data.json")
FLAG_PATH = "/tmp/last_auto_retrain.txt"
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "feature_cols.json")

# ============================================================
# 🧰 Службова функція: ensure_artifacts()
# ============================================================
def ensure_artifacts():
    """Перевіряє наявність базових артефактів і створює структуру models/ якщо її немає."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    for path in [MODEL_PATH, SCALER_PATH, TRAIN_DATA_PATH]:
        folder = os.path.dirname(path)
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

# ============================================================
# 🔍 Автоматичне визначення фіч
# ============================================================
def _resolve_feature_cols(df: pd.DataFrame):
    """Визначає, які фічі реально присутні у DataFrame."""
    cols = [c for c in DEFAULT_FEATURE_COLS if c in df.columns]
    if not cols:
        blacklist = set(TARGET_COLS + ["open_time", "open", "high", "low", "close", "volume"])
        cols = [c for c in df.columns if c not in blacklist and pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        raise ValueError("Не вдалося визначити feature_cols — у DataFrame немає придатних ознак.")
    return cols

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
# 🏋️‍♂️ Train / Save
# ============================================================
def train_transformer(epochs=20, batch_size=32, seq_len=50):
    try:
        ensure_artifacts()

        if not os.path.exists(TRAIN_DATA_PATH):
            print("⚠️ Немає train_data.json — створюю...")
            from ai.load_training_data import load_training_data
            load_training_data(limit=20000)

        df = pd.read_json(TRAIN_DATA_PATH, orient="records")
        if df.empty:
            raise ValueError("train_data.json порожній!")

        feature_cols = _resolve_feature_cols(df)
        with open(FEATURE_COLS_PATH, "w", encoding="utf-8") as f:
            json.dump(feature_cols, f, ensure_ascii=False, indent=2)
        print(f"✅ Використані фічі: {feature_cols}")

        X = df[feature_cols].fillna(0).values
        y = np.random.uniform(0, 1, len(df)).reshape(-1, 1)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        dump(scaler, SCALER_PATH)

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
# 🔮 Predict
# ============================================================
def predict_strength(feature_rows, seq_len=50):
    """Інференс моделі: повертає силу сигналу у відсотках."""
    try:
        ensure_artifacts()

        if os.path.exists(FEATURE_COLS_PATH):
            with open(FEATURE_COLS_PATH, "r", encoding="utf-8") as f:
                feature_cols = json.load(f)
        else:
            feature_cols = DEFAULT_FEATURE_COLS

        scaler = load(SCALER_PATH)
        model = SignalTransformer(input_dim=len(feature_cols))
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()

        df_infer = pd.DataFrame(feature_rows) if not isinstance(feature_rows, pd.DataFrame) else feature_rows
        X = df_infer[feature_cols].fillna(0).values
        X_scaled = scaler.transform(X)

        if X_scaled.shape[0] < seq_len:
            pad = np.repeat(X_scaled[-1:], seq_len - X_scaled.s_
