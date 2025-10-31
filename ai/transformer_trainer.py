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
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

MODEL_PATH = os.path.join(MODEL_DIR, "transformer_signal_model.pt")
SCALER_PATH = os.path.join(MODEL_DIR, "transformer_scaler.joblib")
TRAIN_DATA_PATH = os.path.join(MODEL_DIR, "train_data.json")


# ============================================================
# 📘 Dataset Definition
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
# 🧠 Transformer Model
# ============================================================
class SignalTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=64, n_heads=4, ff_dim=128, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=ff_dim, dropout=0.1
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
        x = x.permute(1, 0, 2)  # [seq, batch, features]
        encoded = self.encoder(x)
        out = encoded[-1]  # last time step
        return self.fc(out)


# ============================================================
# ⚙️ Train / Save
# ============================================================
def train_transformer(epochs=10, batch_size=32, seq_len=50):

    if not os.path.exists(TRAIN_DATA_PATH):
        print("⚠️ Немає train_data.json — спочатку згенеруй історію сигналів.")
        return

    with open(TRAIN_DATA_PATH, "r") as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        print("❌ train_data.json порожній або некоректний формат!")
        return

    df = pd.DataFrame(data)
    print(f"🧾 Структура DataFrame: {df.shape}")
    print("🔑 Колонки:", df.columns.tolist())
    print(df.head(3))

    # 🔧 Нормалізація strength у діапазон [0, 1]
    if df["strength"].max() > 1 or df["strength"].min() < 0:
        print("⚙️ Strength виходить за межі [0,1], виконуємо нормалізацію...")
        df["strength"] = (df["strength"] - df["strength"].min()) / (df["strength"].max() - df["strength"].min())

    # І навіть якщо не виходить — підстрахуємось
    df["strength"] = df["strength"].clip(0, 1)
    
    features = ["ema_diff5", "rsi5", "atr", "volz5", "strength"]
    df = df[features].fillna(0)
    print(f"📊 Рядків до тренування: {len(df)}")

    # Нормалізація значень strength у діапазон [0, 1]
    if df["strength"].max() > 1 or df["strength"].min() < 0:
        print("⚙️ Нормалізую strength у діапазон [0, 1] ...")
        df["strength"] = (df["strength"] - df["strength"].min()) / (df["strength"].max() - df["strength"].min())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.values)
    dump(scaler, SCALER_PATH)

    # Перевірка коректності даних strength
    if (df["strength"] < 0).any() or (df["strength"] > 1).any():
        print("⚠️ Виправляю значення strength у межах [0,1]")
        df["strength"] = np.clip(df["strength"], 0, 1)

    dataset = SignalDataset(X_scaled, seq_len)
    print(f"📏 Довжина dataset після формування: {len(dataset)}")

    if len(dataset) == 0:
        print("⚠️ Dataset порожній — збільши кількість рядків у train_data.json (мінімум 60-70).")
        return

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SignalTransformer(input_dim=len(features) - 1)
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
    print(f"✅ Модель збережено → {MODEL_PATH}")


# ============================================================
# 🔮 Predict
# ============================================================
def predict_strength(features_dict: dict) -> float:
    try:
        scaler = load(SCALER_PATH)
        model = SignalTransformer(input_dim=4)
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()

        x = np.array([[features_dict["ema_diff5"], features_dict["rsi5"], features_dict["atr"], features_dict["volz5"]]])
        x_scaled = scaler.transform(x)
        x_t = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred = model(x_t).item()
        return float(pred * 100)

        except Exception as e:
    print("⚠️ predict_strength error:", e)

    # 🧠 Якщо помилка пов'язана з фічами — спробувати автоперевчання (з кулдауном)
    if any(k in str(e).lower() for k in [
        "features", "n_features", "x has", "shape mismatch",
        "number of features", "feature_names_in_", "expected", "got"
    ]):
        import os, time, traceback

        COOLDOWN_SEC = int(os.getenv("RETRAIN_COOLDOWN_SEC", 6 * 60 * 60))  # 6 год за замовчуванням
        FLAG_PATH = "/tmp/last_auto_retrain.txt"

        # зчитати час останнього автоперевчання
        last = 0.0
        try:
            with open(FLAG_PATH, "r") as f:
                last = float(f.read().strip())
        except Exception:
            pass

        now = time.time()
        if now - last >= COOLDOWN_SEC:
            print("♻️ Перевчаю модель через зміну кількості фіч...")
            try:
                from ai.transformer_trainer import train_transformer
                train_transformer(epochs=10, seq_len=10)
                print("✅ Модель перевчена автоматично!")

                # оновити мітку часу кулдауну
                try:
                    with open(FLAG_PATH, "w") as f:
                        f.write(str(now))
                except Exception:
                    pass

                # 🔔 Повідомлення у Telegram
                try:
                    from notifier.telegram_bot import send_message
                    send_message("🤖 Модель була перевчена автоматично після зміни фіч ✅")
                except Exception as te:
                    print("⚠️ Не вдалося надіслати повідомлення в Telegram:", te)

            except Exception as retrain_error:
                print("❌ Помилка при автонавчанні:", retrain_error)
                traceback.print_exc()
        else:
            wait = int(COOLDOWN_SEC - (now - last))
            print(f"⏳ Автоперевчання пропущено: кулдаун ще {wait}s.")

    # безпечний фолбек
    return 50.0

if __name__ == "__main__":
    train_transformer(epochs=15, seq_len=10)

