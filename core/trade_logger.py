# -*- coding: utf-8 -*-
"""
trade_logger.py — універсальний логер угод.
Логує всі операції (LONG / SHORT), фіксує PnL, час, режим (симуляція або реальна торгівля),
та надсилає повідомлення у Telegram.
"""

import os
import json
import time
from datetime import datetime
from notifier.telegram_notifier import send_message
from core.trading_events import is_safe_mode

# Директорія для логів
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "trades.log")

os.makedirs(LOG_DIR, exist_ok=True)


def log_trade(symbol: str, side: str, entry: float, exit_price: float,
              pnl: float, status: str, balance: float, mode: str = "simulation"):
    """
    Логує інформацію про трейд у файл та Telegram.
    :param symbol: торговий інструмент (наприклад, BTCUSDT)
    :param side: LONG або SHORT
    :param entry: ціна входу
    :param exit_price: ціна виходу
    :param pnl: відсотковий прибуток/збиток
    :param status: WIN / LOSS
    :param balance: поточний баланс після угоди
    :param mode: 'simulation' або 'real'
    """
    try:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        record = {
            "time": timestamp,
            "symbol": symbol,
            "side": side,
            "entry": round(entry, 4),
            "exit": round(exit_price, 4),
            "pnl_percent": round(pnl, 2),
            "status": status,
            "balance": round(balance, 2),
            "mode": mode,
        }

        # Запис у файл
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Вивід у консоль
        print(f"[TRADE] {timestamp} | {symbol} {side} {status} | PnL={pnl:.2f}% | Баланс={balance:.2f}$ | {mode}")

        # Повідомлення у Telegram
        emoji = "💰" if pnl > 0 else "📉"
        message = (
            f"{emoji} <b>{'Симуляція' if mode == 'simulation' else 'Реальна торгівля'}</b>\n"
            f"📊 Пара: <b>{symbol}</b>\n"
            f"🧭 Напрям: <b>{side}</b>\n"
            f"💵 Вхід: {entry:.2f}\n"
            f"🏁 Вихід: {exit_price:.2f}\n"
            f"📈 Результат: <b>{pnl:.2f}%</b> {'✅ Прибуток' if pnl > 0 else '❌ Збиток'}\n"
            f"💰 Баланс: <b>{balance:.2f}$</b>\n"
            f"🕒 UTC: {timestamp}"
        )
        send_message(message)

    except Exception as e:
        print(f"❌ Помилка логування трейду: {e}")


def safe_trade_check() -> bool:
    """
    Перевіряє безпечний режим перед відкриттям позиції.
    Якщо safe_mode активний — надсилає повідомлення у Telegram і повертає False.
    """
    if is_safe_mode():
        print("🛡️ Торгівля заблокована — безпечний режим увімкнено.")
        send_message("🛡️ Безпечний режим увімкнено — відкриття позицій заблоковано.")
        return False
    return True


def load_trades(limit: int = 50):
    """
    Повертає останні N угод із логу.
    """
    if not os.path.exists(LOG_FILE):
        return []

    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()[-limit:]
            return [json.loads(line) for line in lines]
    except Exception:
        return []


def summarize_trades(limit: int = 100):
    """
    Обчислює короткий підсумок останніх угод.
    """
    trades = load_trades(limit)
    if not trades:
        return {"trades": 0, "avg_pnl": 0, "wins": 0, "losses": 0}

    pnls = [t["pnl_percent"] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    losses = len(pnls) - wins
    avg_pnl = sum(pnls) / len(pnls)

    return {
        "trades": len(pnls),
        "wins": wins,
        "losses": losses,
        "avg_pnl": round(avg_pnl, 2)
    }


if __name__ == "__main__":
    # 🔧 Тест симуляції
    if safe_trade_check():
        log_trade("BTCUSDT", "LONG", 68200, 68950, 1.1, "WIN", 1011.0, mode="simulation")

    # 🔍 Вивід підсумку
    print(summarize_trades(10))

