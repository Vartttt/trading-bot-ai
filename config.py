import os
from dotenv import load_dotenv
load_dotenv()

# 🔑 MEXC
MEXC_API_KEY = os.getenv("MEXC_API_KEY", "")
MEXC_API_SECRET = os.getenv("MEXC_API_SECRET", "")

# 💬 Telegram
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ⚙️ Торгівля
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT").split(",")]
TIMEFRAME = os.getenv("TIMEFRAME", "15m")
CONFIRM_TIMEFRAME = os.getenv("CONFIRM_TIMEFRAME", "1h")
CONFIRM_TIMEFRAME_2 = os.getenv("CONFIRM_TIMEFRAME_2", "4h")
CHECK_INTERVAL_SEC = int(os.getenv("CHECK_INTERVAL_SEC", "30"))

# 💰 Ризик
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "1.5")) / 100.0
MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", "3"))
DAILY_MAX_LOSS_PCT = 0.05
MAX_TRADES_PER_SYMBOL = 3
LOSS_COOLDOWN_MIN = 20

# 📈 Індикатори
EMA_FAST = 21
EMA_SLOW = 55
RSI_LEN = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ATR_PERIOD = 14
ATR_MULT = 1.5

# 🧩 TP/SL
TP_PCTS = [0.006, 0.012, 0.02]
TP_SIZES = [0.5, 0.3, 0.2]
SL_PCT = 0.004
TRAIL_START_PCT = 0.008
TRAIL_STEP_PCT = 0.004

# 💾 Файли
LOG_CSV_PATH = "trades_log.csv"
SQLITE_PATH = "trades.db"
MODEL_PATH = "ai_model.pkl"
BEST_PATH = "best_params.json"

# 🧠 AI / Dashboard
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 5000
RETRAIN_INTERVAL_HOURS = 6

# 🔧 Проче
LOG_LEVEL = "INFO"
DRY_RUN = os.getenv("DRY_RUN", "False").lower() == "true"

