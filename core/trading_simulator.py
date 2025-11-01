import random
from notifier.telegram_notifier import send_message
from core.trade_logger import log_trade
from core.trading_events import notify_open_position, notify_close_position, is_safe_mode


class TradingSimulator:
    def __init__(self, balance=1000.0):
        self.balance = balance
        self.trades = []

    def simulate_trade(self, symbol: str, side: str, entry: float, leverage: int = 1):
        """Симуляція однієї угоди з урахуванням безпечного режиму"""

        # 🛡️ Перевірка безпечного режиму
        if is_safe_mode():
            print("🛡️ Торгівля заблокована — безпечний режим увімкнено.")
            send_message("🛡️ Безпечний режим увімкнено — відкриття позицій заблоковано.")
            return

        # 🔔 Повідомлення про відкриття позиції
        notify_open_position(symbol, side, entry, leverage, mode="simulation")

        # 🧮 Симуляція руху ціни
        exit_price = entry * random.uniform(0.98, 1.03)

        # 💹 Розрахунок PnL (%)
        if side.upper() == "LONG":
            pnl = (exit_price - entry) / entry * 100
        else:
            pnl = (entry - exit_price) / entry * 100

        # 💰 Оновлення балансу
        self.trades.append(pnl)
        self.balance *= (1 + pnl / 100)

        status = "WIN" if pnl > 0 else "LOSS"

        # 🧾 Логування
        log_trade(symbol, side, entry, round(exit_price, 2), round(pnl, 2), status)

        # 🔔 Повідомлення про закриття позиції
        profit = (self.balance - 1000) if len(self.trades) > 0 else 0
        notify_close_position(symbol, profit, mode="simulation")

        # 📩 Детальний звіт у Telegram
        send_message(
            f"💹 Симуляція | {symbol}\n"
            f"📈 {side}\n"
            f"💰 PnL: {pnl:.2f}%\n"
            f"💵 Баланс: {self.balance:.2f}$\n"
            f"{'✅' if pnl > 0 else '❌'} {status}"
        )

        # 💻 Консоль
        print(f"[SIM] {symbol} {side} | Entry: {entry} → Exit: {exit_price:.2f} | PnL={pnl:.2f}%")

    def summary(self):
        """Підсумок усіх угод"""
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

        # 🧾 Відправити короткий звіт у Telegram
        send_message(
            f"📊 Підсумок симуляції:\n"
            f"🔹 Угод: {summary_data['trades']}\n"
            f"✅ Перемог: {summary_data['wins']}\n"
            f"❌ Поразок: {summary_data['losses']}\n"
            f"📈 Середній PnL: {summary_data['avg_pnl']}%\n"
            f"💰 Баланс: {summary_data['balance']}$"
        )

        return summary_data

