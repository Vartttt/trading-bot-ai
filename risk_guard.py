# risk_guard.py
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from config import DAILY_MAX_LOSS_PCT, MAX_TRADES_PER_SYMBOL, LOSS_COOLDOWN_MIN

class RiskGuard:
    def __init__(self):
        self.start_equity = None
        self.realized_pnl = 0.0
        self.trades_count = defaultdict(int)
        self.last_sl_time = None

    def set_start_equity(self, eq):
        if self.start_equity is None and eq:
            self.start_equity = float(eq)

    def add_pnl(self, pnl):
        self.realized_pnl += float(pnl)
        if pnl < 0:
            self.last_sl_time = datetime.now(timezone.utc)

    def daily_stop(self):
        if not self.start_equity:
            return False
        return (self.realized_pnl <= - self.start_equity * DAILY_MAX_LOSS_PCT)

    def can_trade_symbol(self, symbol: str):
        return self.trades_count[symbol] < MAX_TRADES_PER_SYMBOL

    def register_entry(self, symbol: str):
        self.trades_count[symbol] += 1

    def cooldown_active(self):
        if not self.last_sl_time:
            return False
        return datetime.now(timezone.utc) - self.last_sl_time < timedelta(minutes=LOSS_COOLDOWN_MIN)
