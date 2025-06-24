import csv
from datetime import datetime

def log_trade(symbol, direction, strength, result):
    with open("logs/trades.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().isoformat(),
            symbol,
            direction,
            strength,
            result.get("code"),
            result.get("msg")
        ])
