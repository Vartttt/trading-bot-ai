import json
from ai.transformer_trainer import ensure_artifacts, predict_strength

# гарантуємо наявність артефактів (feature_cols.json тощо)
ensure_artifacts()

sample_features = {
    "ema_diff5": 0.0032,
    "rsi5": 62.4,
    "atr": 0.015,
    "volz5": 1.08,
    "trend_accel": 0.0001
}

print("🔍 Тестові дані:", json.dumps(sample_features, indent=2, ensure_ascii=False))

# ВАЖЛИВО: передаємо СПИСОК рядків
strength = predict_strength([sample_features])
print(f"💪 Прогноз сили сигналу: {strength:.2f}%")
