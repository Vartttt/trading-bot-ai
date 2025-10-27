"""
Підключення Prometheus метрик у реальному часі.
"""
from prometheus_client import Gauge, Counter

g_equity = Gauge("stb_equity", "Current equity multiplier (1.0=start)")
g_daily_pnl = Gauge("stb_daily_pnl", "Daily realized PnL fraction")
c_trades = Counter("stb_trades_total", "Total trades executed")
