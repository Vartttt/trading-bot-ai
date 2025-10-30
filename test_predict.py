import json
from ai.transformer_trainer import predict_strength

# 🔹 Тестові вхідні параметри (імітація поточних ринкових даних)
sample_features = {
    "ema_diff5": 0.0032,  # різниця EMA
    "rsi5": 62.4,         # RSI
    "atr": 0.015,         # волатильність
    "volz5": 1.08         # обсяг/тренд
}

print("🔍 Тестові дані:", json.dumps(sample_features, indent=2))

# 🔹 Отримуємо прогноз сили сигналу
strength = predict_strength(sample_features)
print(f"💪 Прогноз сили сигналу: {strength:.2f}%")
