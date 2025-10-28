import os

# Шлях до директорії з моделями (можна змінювати через .env)
MODEL_DIR = os.getenv("MODEL_DIR", "./models")

# Основні параметри (опціонально)
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", 60))  # інтервал у секундах
SYMBOLS = os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT").split(",")
MIN_STRENGTH = int(os.getenv("MIN_STRENGTH", 70))
