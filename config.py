# config.py
import os

# Pairs (user-provided list)
TOP_MANUAL_PAIRS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "TON/USDT"
]

SYMBOL_QUOTE = os.getenv("SYMBOL_QUOTE", "USDT")
EXCHANGE_ID = os.getenv("EXCHANGE", "binance")
CHECK_INTERVAL_SECONDS = int(os.getenv("CHECK_INTERVAL_SECONDS", "30"))
DB_PATH = os.getenv("DB_PATH", "signals.db")

MIN_STRENGTH = 70  # min % to send signal
TP_MULTIPLIERS = [1.02, 1.08, 1.15]
SL_ATR_MULT = 1.0

# Telegram formatting
TIMEFRAME_LABEL = "5m"  # primary TF for signals

# ліміти ризику на сесію/день
DAILY_MAX_LOSS_PCT = 0.05        # 5% від балансу — стоп-день
MAX_TRADES_PER_SYMBOL = 3        # захист від овертрейдингу
LOSS_COOLDOWN_MIN = 20           # пауза в хвилинах після SL

# обмеження ковзання/ринкових умов
MAX_SPREAD_PCT = 0.002           # 0.2% макс спред
MAX_MARKET_IMPACT_PCT = 0.003    # не відправляти ордер, якщо рух аномальний

# лог рівень
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
