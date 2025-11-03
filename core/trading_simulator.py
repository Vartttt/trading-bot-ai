"""
TradingSimulator — емулятор угод для тестування логіки бота.
Використовується для симуляційних угод без ризику.
"""

import random
from notifier.telegram_notifier import send_message
from core.trade_logger import log_trade
from core.trading_events import (
    notify_open_position,
    notify_close_position,
    is_safe_mode,
)

class TradingSimulator:
    def __init__(self, balance=1000.0):
        self.balance = balance
        self.trades = []

    # ============================================================
    # 💹 Симуляція однієї угоди
    # ============================================================
    def simulate_trade(self, symbol: str, side: str, entry: float):
        """Імітація однієї торгової угоди з повідомленнями."""
        if is_safe_mode():
            send_message("🛡️ Безпечний режим увімкнено — відкриття позицій заблоковано.")
            print("🛡️ Торгівля заблокована (Safe Mode).")
            return

        # 🔹 Повідомлення про відкриття позиції
        notify_open_position(symbol, side, entry, leverage=10, mode="simulation")

        # 📉 Імітуємо рух ціни (±2%)
        exit_price = entry * random.uniform(0.98, 1.03)

        # 📊 Розрахунок прибутку/збитку (%)
        if side.upper() == "LONG":
            pnl = (exit_price - entry) / entry * 100
        else:
            pnl = (entry - exit_price) / entry * 100

        self.trades.append(pnl)
        self.balance *= (1 + pnl / 100)

        # 💰 Розрахунок реального прибутку в USDT
        profit_usdt = round((self.balance * pnl / 100), 2)

        # 📜 Логування
        log_trade(symbol, side, entry, round(exit_price, 2), round(pnl, 2), "WIN" if pnl > 0 else "LOSS")

        # 🔔 Повідомлення про закриття
        notify_close_position(symbol, profit_usdt, mode="simulation")

        # 🧾 Додаткове резюме
        send_message(
            f"📊 Симуляція | {symbol}\n"
            f"📈 {side} | PnL: {pnl:.2f}%\n"
            f"💵 Баланс: {self.balance:.2f} USDT\n"
            f"{'✅ Прибуток' if pnl > 0 else '❌ Збиток'}"
        )

        print(f"[SIM] {symbol} {side} | Entry: {entry} → Exit: {exit_price:.2f} | PnL={pnl:.2f}%")

    # ============================================================
    # 📈 Підсумки
    # ============================================================
    def summary(self):
        """Повертає підсумок усіх угод."""
        wins = len([t for t in self.trades if t > 0])
        losses = len([t for t in self.trades if t <= 0])
        avg_pnl = sum(self.trades) / len(self.trades) if self.trades else 0

        summary_data = {
            "trades": len(self.trades),
            "wins": wins,
            "losses": losses,
            "avg_pnl": round(avg_pnl, 2),
            "balance": round(self.balance, 2)
        }

        # 🔔 Повідомлення в Telegram
        send_message(
            f"📊 <b>Підсумок симуляції</b>\n"
            f"🔹 Угод: {summary_data['trades']}\n"
            f"✅ Виграно: {wins} | ❌ Програно: {losses}\n"
            f"📈 Середній PnL: {avg_pnl:.2f}%\n"
            f"💰 Баланс: {self.balance:.2f} USDT"
        )

        return summary_data


