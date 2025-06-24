import time
from core.api import get_balance, open_position, close_position
from core.logic import generate_signal
from utils.logger import log_trade
from utils.risk import calculate_order_size

SYMBOL = "BTC_USDT"

while True:
    balance = get_balance()
    if balance < 1:
        print("Недостатній баланс для торгівлі")
        time.sleep(60)
        continue

    signal_data = generate_signal(SYMBOL)
    signal = signal_data["signal"]
    strength = signal_data["strength"]

    if signal == "none" or strength < 50:
        print("Немає сильного сигналу")
        time.sleep(30)
        continue

    leverage = 50 if strength >= 70 else 10
    usdt_amount = calculate_order_size(balance)

    result = open_position(SYMBOL, signal, usdt_amount, leverage)
    log_trade(SYMBOL, signal, strength, result)
    print(f"Відкрита позиція: {signal.upper()} зі силою {strength}%")
    time.sleep(60)
