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

    def simulate_trade(self, symbol: str, side: str, entry: float):
        """Симуляція однієї угоди"""
        if is_safe_mode():
            send_message("🛡️ Безпечний режим увімкнено — відкриття позицій заблоковано.")
            print("🛡️ Торгівля заблокована (Safe Mode).")
            return

        # відкриваємо позицію
        notify_open_position(symbol, side, entry, leverage=10, mode="simulation")

        # випадкове коливання (імітація торгівлі)
        exit_price = entry * random.uniform(0.98, 1.03)

        # розрахунок PnL
        pnl = (exit_price - entry) / entry * 100 if side.upper() == "LONG" else (entry - exit_price) / entry * 100
        self.trades.append(pnl)
        self.balance *= (1 + pnl / 100)

        # лог + повідомлення
        log_trade(symbol, side, entry, round(exit_price, 2), round(pnl, 2), "WIN" if pnl > 0 else "LOSS")
        profit_value = round(self.balance * (pnl / 100), 2)
        notify_close_position(symbol, profit_value, mode="simulation")

        send_message(
            f"💹 <b>Симуляція:</b> {symbol}\n"
            f"📈 {side} | PnL: {pnl:.2f}%\n"
            f"💵 Баланс: {self.balance:.2f}$"
        )

    def summary(self):
        wins = len([t for t in self.trades if t > 0])
        losses = len([t for t in self.trades if t <= 0])
        avg_pnl = sum(self.trades) / len(self.trades) if self.trades else 0
        return {
            "trades": len(self.trades),
            "wins": wins,
            "losses": losses,
            "avg_pnl": round(avg_pnl, 2),
            "balance": round(self.balance, 2)
        }


