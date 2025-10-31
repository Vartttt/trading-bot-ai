import csv
import os
from datetime import datetime

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "trades.csv")

os.makedirs(LOG_DIR, exist_ok=True)

def log_trade(symbol, side, entry_price, exit_price, pnl, status):
    """Зберігає результат угоди у CSV"""
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["time", "symbol", "side", "entry", "exit", "pnl", "status"])
        writer.writerow([datetime.now(), symbol, side, entry_price, exit_price, pnl, status])
