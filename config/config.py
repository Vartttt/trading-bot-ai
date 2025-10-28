import os

# === Модель і навчання ===
# Каталог, де зберігаються файли моделі та scaler
MODEL_DIR = os.getenv("MODEL_DIR", "./models")

# Файли моделі
MODEL_PATH = os.path.join(MODEL_DIR, "transformer_signal_model.pt")
SCALER_PATH = os.path.join(MODEL_DIR, "transformer_scaler.joblib")
TRAIN_DATA_PATH = os.path.join(MODEL_DIR, "train_data.json")

# === Основні параметри торгового бота ===
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "30"))  # інтервал у секундах
PHASE_REFRESH_MIN = int(os.getenv("PHASE_REFRESH_MIN", "30"))
DRY_RUN = os.getenv("DRY_RUN", "True").lower() == "true"
MIN_STRENGTH = int(os.getenv("MIN_STRENGTH", "72"))

# === Біржові налаштування ===
EXCHANGE = os.getenv("EXCHANGE", "MEXC")
SYMBOLS = os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT").split(",")
DYNSYM_TOPN = int(os.getenv("DYNSYM_TOPN", "12"))

# === Telegram ===
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# === API для біржі ===
API_KEY = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")

# === Оптимізація ===
OPT_REFRESH_SEC = int(os.getenv("OPT_REFRESH_SEC", "7200"))  # 2 години
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# === Веб-сервер ===
PORT = int(os.getenv("PORT", "8080"))
BASE_URL = os.getenv("URL_ADDRESS", "")

# === Ризик-менеджмент ===
MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", "0.05"))  # 5%
RISK_MODE = os.getenv("RISK_MODE", "adaptive")

# === Додаткові фільтри ===
ENABLE_NEWS_FILTER = os.getenv("ENABLE_NEWS_FILTER", "true").lower() == "true"
ENABLE_FUNDING_FILTER = os.getenv("ENABLE_FUNDING_FILTER", "true").lower() == "true"
ENABLE_SESSION_GUARD = os.getenv("ENABLE_SESSION_GUARD", "true").lower() == "true"

# === Створення директорії моделі (на випадок відсутності) ===
os.makedirs(MODEL_DIR, exist_ok=True)
