
---

# `config.py`
```python
# config.py
import os

# Pairs (user-provided list)
TOP_MANUAL_PAIRS = [p.strip() for p in os.getenv(
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
