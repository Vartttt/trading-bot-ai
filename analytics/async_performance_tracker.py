"""
Async Performance Tracker — телеметрія, Prometheus-метрики і автозвіти в Telegram.
Працює з асинхронним циклом v8.4 Pro, не блокує основний трейдинг.

Функції:
  ✅ record_md_fetch(symbol, dt_sec, bars) — час отримання OHLCV
  ✅ record_signal(symbol, strength) — зафіксувати сформований сигнал
  ✅ record_order(status, latency_sec, symbol, side) — угоди (успіх/помилка)
  ✅ record_error(source, message) — будь-які помилки
  ✅ heartbeat() — оновлення "живості" рушія
  ✅ hourly_report_loop() — щогодинний звіт у Telegram (PnL/обсяг/latency/помилки)
  ✅ expose Prometheus gauges/counters (якщо /metrics вже є в Flask — вони підтягнуться)
"""

import os
import time
import math
import asyncio
from collections import deque, defaultdict
from statistics import mean, median

from prometheus_client import Gauge, Counter

from notifier.telegram_notifier import send_message
from core.trade_switch import is_trading_enabled
# daily PnL (фракція від equity) якщо є; інакше за замовчуванням 0
try:
    from core.risk_daily_guard import _load as _load_daily
    def _daily_pnl_frac():
        try:
            s = _load_daily()
            return float(s.get("pnl", 0.0))
        except Exception:
            return 0.0
except Exception:
    def _daily_pnl_frac():
        return 0.0

DRY_RUN = os.getenv("DRY_RUN", "True").lower() == "true"
REPORT_EVERY_MIN = int(os.getenv("PERF_REPORT_MIN", "60"))      # щогодини
MAX_KEEP = int(os.getenv("PERF_KEEP", "500"))                    # скільки подій тримати в пам'яті

# ── Prometheus метрики
g_last_heartbeat   = Gauge("stb_async_last_heartbeat_ts", "Last heartbeat ts (epoch)")
g_md_latency_ms    = Gauge("stb_md_fetch_latency_ms", "Last market data fetch latency (ms)")
g_trade_latency_ms = Gauge("stb_trade_submit_latency_ms", "Last trade submit latency (ms)")
g_signals_per_min  = Gauge("stb_signals_per_min", "Signals per minute (rolling)")
g_orders_success   = Counter("stb_orders_success_total", "Orders successfully submitted")
g_orders_failed    = Counter("stb_orders_failed_total", "Orders failed")
g_errors_total     = Counter("stb_errors_total", "Errors total (tracked by perf tracker)")
g_mode_live        = Gauge("stb_mode_live", "1 if REAL trading (DRY_RUN=False), else 0")
g_trading_enabled  = Gauge("stb_trading_enabled", "1 if trading enabled by switch, else 0")

g_mode_live.set(0 if DRY_RUN else 1)
g_trading_enabled.set(1 if is_trading_enabled() else 0)


class AsyncPerfTracker:
    def __init__(self):
        # rolling windows
        self._md_latencies = deque(maxlen=MAX_KEEP)       # seconds
        self._trade_lat    = deque(maxlen=MAX_KEEP)       # seconds
        self._signals_ts   = deque(maxlen=MAX_KEEP)       # timestamps
        self._errors       = deque(maxlen=MAX_KEEP)       # (ts, source, msg)
        self._orders       = deque(maxlen=MAX_KEEP)       # (ts, ok:bool, symbol, side, lat)
        self._bars_fetched = 0

        self._last_report_ts = 0

    # ── ІНТЕРФЕЙС ЗАПИСУ ПОДІЙ ──────────────────────────────────────────────
    def heartbeat(self):
        g_trading_enabled.set(1 if is_trading_enabled() else 0)
        g_last_heartbeat.set(time.time())

    def record_md_fetch(self, symbol: str, dt_sec: float, bars: int):
        self._md_latencies.append(float(max(dt_sec, 0.0)))
        self._bars_fetched += int(max(bars, 0))
        g_md_latency_ms.set(self._md_latencies[-1] * 1000.0)

    def record_signal(self, symbol: str, strength: int):
        self._signals_ts.append(time.time())

    def record_order(self, ok: bool, latency_sec: float, symbol: str, side: str):
        self._orders.append((time.time(), bool(ok), symbol, side, float(max(latency_sec, 0.0))))
        if ok:
            g_orders_success.inc()
            g_trade_latency_ms.set(latency_sec * 1000.0)
        else:
            g_orders_failed.inc()

    def record_error(self, source: str, message: str):
        self._errors.append((time.time(), source, str(message)))
        g_errors_total.inc()

    # ── АГРЕГАЦІЯ ───────────────────────────────────────────────────────────
    @staticmethod
    def _p95(vals):
        if not vals:
            return 0.0
        s = sorted(vals)
        k = int(math.ceil(0.95 * len(s))) - 1
        return float(s[max(0, min(k, len(s)-1))])

    def _signals_per_min(self):
        now = time.time()
        # чистимо старіші за 60с
        while self._signals_ts and now - self._signals_ts[0] > 60:
            self._signals_ts.popleft()
        g_signals_per_min.set(len(self._signals_ts))
        return len(self._signals_ts)

    def snapshot(self):
        # latency
        md_avg = mean(self._md_latencies) if self._md_latencies else 0.0
        md_p95 = self._p95(self._md_latencies)
        tr_avg = mean(self._trade_lat) if self._trade_lat else 0.0
        tr_p95 = self._p95(self._trade_lat)

        # orders
        succ = len([x for x in self._orders if x[1]])
        fail = len([x for x in self._orders if not x[1]])

        # errors (групуємо по джерелу)
        err_by_src = defaultdict(int)
        for _, src, _ in self._errors:
            err_by_src[src] += 1

        sig_pm = self._signals_per_min()
        return {
            "md_avg": md_avg, "md_p95": md_p95,
            "tr_avg": tr_avg, "tr_p95": tr_p95,
            "orders_ok": succ, "orders_fail": fail,
            "signals_per_min": sig_pm,
            "bars_fetched": self._bars_fetched,
            "errors_by_src": dict(err_by_src),
            "pnl_day": _daily_pnl_frac(),
            "live_mode": not DRY_RUN,
            "enabled": is_trading_enabled(),
        }

    # ── РЕПОРТИ ─────────────────────────────────────────────────────────────
    def _format_report(self, snap):
        err_txt = "—"
        if snap["errors_by_src"]:
            err_txt = ", ".join(f"{k}:{v}" for k,v in snap["errors_by_src"].items())

        mode_txt = "💰 LIVE" if snap["live_mode"] else "🧪 DRY-RUN"
        en_txt   = "✅ ON" if snap["enabled"] else "⛔️ OFF"

        return (
            "📊 <b>Async Performance (за годину)</b>\n"
            f"Mode: {mode_txt} | Trading: {en_txt}\n"
            f"MD latency: avg <b>{snap['md_avg']*1000:.0f}ms</b> | p95 <b>{snap['md_p95']*1000:.0f}ms</b>\n"
            f"Trade submit: avg <b>{snap['tr_avg']*1000:.0f}ms</b> | p95 <b>{snap['tr_p95']*1000:.0f}ms</b>\n"
            f"Signals/min: <b>{snap['signals_per_min']}</b> | Bars fetched: <b>{snap['bars_fetched']}</b>\n"
            f"Orders: ✅ <b>{snap['orders_ok']}</b> / ❌ <b>{snap['orders_fail']}</b>\n"
            f"Errors: {err_txt}\n"
            f"PnL (доба): <b>{snap['pnl_day']*100:.2f}%</b>"
        )

    async def hourly_report_loop(self):
        # чекаємо перший повний інтервал, щоб накопичити статистику
        await asyncio.sleep(60)
        while True:
            try:
                snap = self.snapshot()
                send_message(self._format_report(snap))
            except Exception as e:
                # не завалюємо цикл
                try:
                    send_message(f"⚠️ Perf report error: {e}")
                except Exception:
                    pass
            await asyncio.sleep(REPORT_EVERY_MIN * 60)
