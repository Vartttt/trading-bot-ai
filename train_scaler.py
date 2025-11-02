import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import dump

from ai.transformer_trainer import ensure_artifacts, TRAIN_DATA_PATH, SCALER_PATH, MODEL_DIR
    ensure_artifacts,
    MODEL_DIR,
    TRAIN_DATA_PATH,
    SCALER_PATH,
    FEATURE_COLS_PATH,
    DEFAULT_FEATURE_COLS,
)

# гарантуємо, що є базові файли
ensure_artifacts()

# читаємо тренувальні дані (список dict'ів з фічами)
with open(TRAIN_DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# визначаємо набір колонок
if os.path.exists(FEATURE_COLS_PATH):
    with open(FEATURE_COLS_PATH, "r", encoding="utf-8") as f:
        feature_cols = json.load(f) or DEFAULT_FEATURE_COLS
else:
    feature_cols = [c for c in DEFAULT_FEATURE_COLS if c in df.columns]
    if not feature_cols:
        raise RuntimeError("Не знайдено придатних фіч у train_data.json")

X = df[feature_cols].fillna(0).values

# тренуємо та зберігаємо scaler
scaler = StandardScaler()
scaler.fit(X)

os.makedirs(MODEL_DIR, exist_ok=True)
dump(scaler, SCALER_PATH)

print(f"✅ Scaler збережено: {SCALER_PATH}")
print(f"📦 Використані фічі: {feature_cols}")
