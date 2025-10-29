import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from joblib import dump
import os

# Завантажуємо тренувальні дані
with open("models/train_data.json", "r") as f:
    data = json.load(f)

# Припустимо, у тебе там масив фіч
X = np.array(data["features"])  # змінюй під свою структуру

# Створюємо і навчаємо scaler
scaler = StandardScaler()
scaler.fit(X)

# Зберігаємо його
os.makedirs("models", exist_ok=True)
dump(scaler, "models/transformer_scaler.joblib")

print("✅ transformer_scaler.joblib створено!")
