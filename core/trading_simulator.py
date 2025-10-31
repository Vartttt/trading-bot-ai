import random
from notifier.telegram_bot import send_message
from core.trade_logger import log_trade

class TradingSimulator:
    def __init__(self, balance=1000.0):
        self.balance = balance
        self.trades = []

    def simulate_trade(self, symbol: str, side: str, entry: float):
        """Симуляція однієї угоди"""
        # випадкове коливання для прикладу
        exit_price = entry * random.uniform(0.98, 1.03)

        if side.upper() == "LONG":
            pnl = (exit_price - entry) / entry * 100
        else:
            pnl = (entry - exit_price) / entry * 100

        self.trades.append(pnl)
        self.balance *= (1 + pnl / 100)

        status = "WIN" if pnl > 0 else "LOSS"

        log_trade(symbol, side, entry, round(exit_price, 2), round(pnl, 2), status)

        send_message(
            f"💹 Симуляція | {symbol}\n📈 {side}\n💰 PnL: {pnl:.2f}%\n💵 Баланс: {self.balance:.2f}$\n{'✅' if pnl > 0 else '❌'} {status}"
        )

        print(f"[SIM] {symbol} {side} | Entry: {entry} → Exit: {exit_price:.2f} | PnL={pnl:.2f}%")

    def summary(self):
        """Підсумок усіх угод"""
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
