"""
Async Executor — високопродуктивний асинхронний рушій для SmartTraderBot.

Можливості:
  • Паралельна обробка 10–50+ пар без блокувань (asyncio, queues).
  • Безпечний пул клієнтів ccxt.async_support (MEXC/Binance, spot/swap).
  • Рейт-лімітер, експон. backoff + jitter, circuit-breaker на символ/біржу.
  • Черги: market-data, trade-intents, tca-логування.
  • Граційний shutdown і “at-least-once” обробка завдань.

Використання:
  from engine.async_executor import AsyncEngine, TradeIntent, MDRequest

  engine = AsyncEngine()
  await engine.start()
  await engine.submit_md(MDRequest(symbol="BTC/USDT", timeframe="15m", limit=200))
  await engine.submit_trade(TradeIntent(symbol="BTC/USDT", side="buy", qty=0.01, type="market"))
  ...
  await engine.stop()
"""

import os
import asyncio
import time
import random
import math
import json
from dataclasses import dataclass, field
from typing import Optional, Callable, Any, Dict, List, Tuple

# ccxt async
import ccxt.async_support as ccxt_async  # потребує aiohttp
import aiohttp

# локальні
from notifier.telegram_notifier import send_message
from analytics.tca import log_tca, estimate_fee

# -------- ENV / CONFIG --------
EXCHANGE_NAME = os.getenv("EXCHANGE", "mexc3")  # mexc3/binance
DEFAULT_TYPE  = os.getenv("DEFAULT_TYPE", "swap")  # "swap" або "spot"
API_KEY       = os.getenv("API_KEY", "")
API_SECRET    = os.getenv("API_SECRET", "")

DRY_RUN       = os.getenv("DRY_RUN", "True").lower() == "true"

# Конкурентність
MD_CONCURRENCY      = int(os.getenv("ASYNC_MD_CONCURRENCY", "8"))
TRADE_CONCURRENCY   = int(os.getenv("ASYNC_TRADE_CONCURRENCY", "4"))

# Рейт-ліміти (запобіжні, поверх ccxt.enableRateLimit)
MAX_CALLS_PER_SEC   = float(os.getenv("ASYNC_MAX_CALLS_PER_SEC", "8.0"))

# Circuit breaker
CB_FAIL_WINDOW_SEC  = int(os.getenv("ASYNC_CB_WINDOW_SEC", "60"))
CB_FAILS_THRESHOLD  = int(os.getenv("ASYNC_CB_FAILS_THRESHOLD", "5"))
CB_COOLDOWN_SEC     = int(os.getenv("ASYNC_CB_COOLDOWN_SEC", "45"))

# Технічні
REQUEST_TIMEOUT_SEC = int(os.getenv("ASYNC_REQ_TIMEOUT_SEC", "20"))
RETRY_MAX_TRIES     = int(os.getenv("ASYNC_RETRY_MAX_TRIES", "4"))
RETRY_BASE_DELAY    = float(os.getenv("ASYNC_RETRY_BASE_DELAY", "0.7"))

# -------- Data classes --------
@dataclass
class MDRequest:
    symbol: str
    timeframe: str = "15m"
    limit: int = 200
    # колбек: async def on_data(df_like) -> None
    on_data: Optional[Callable[[Any], Any]] = None
    # не обов’язково: контекст
    ctx: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeIntent:
    symbol: str
    side: str  # "buy"|"sell"
    qty: float
    type: str = "market"  # "market"|"limit"
    price: Optional[float] = None  # обов’язково для limit
    post_only: bool = False
    # колбек: async def on_exec(result_dict) -> None
    on_exec: Optional[Callable[[Dict[str, Any]], Any]] = None
    ctx: Dict[str, Any] = field(default_factory=dict)


class RateLimiter:
    """
    Найпростіший токен-бакет для MAX_CALLS_PER_SEC.
    """
    def __init__(self, rate_per_sec: float):
        self.interval = 1.0 / max(rate_per_sec, 0.1)
        self._lock = asyncio.Lock()
        self._next_free = 0.0

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            if now < self._next_free:
                await asyncio.sleep(self._next_free - now)
            self._next_free = time.monotonic() + self.interval


class CircuitBreaker:
    """
    Circuit breaker на символьному/загальному рівні.
    """
    def __init__(self):
        self._fails: Dict[str, List[float]] = {}
        self._open_until: Dict[str, float] = {}

    def _key(self, symbol: Optional[str]) -> str:
        return symbol or "__global__"

    def record_success(self, symbol: Optional[str] = None):
        k = self._key(symbol)
        self._fails.pop(k, None)

    def record_fail(self, symbol: Optional[str] = None):
        k = self._key(symbol)
        arr = self._fails.setdefault(k, [])
        now = time.time()
        arr.append(now)
        # чистка вікна
        self._fails[k] = [t for t in arr if now - t <= CB_FAIL_WINDOW_SEC]
        if len(self._fails[k]) >= CB_FAILS_THRESHOLD:
            self._open_until[k] = now + CB_COOLDOWN_SEC

    def is_open(self, symbol: Optional[str] = None) -> bool:
        k = self._key(symbol)
        until = self._open_until.get(k, 0)
        if until and time.time() < until:
            return True
        # авто-закриття
        self._open_until.pop(k, None)
        return False


# -------- AsyncEngine --------
class AsyncEngine:
    """
    Основний асинхронний рушій:
      - створює async-клієнт ccxt
      - піднімає worker-и для market-data і trades
      - експорт submit_* методів
    """
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._ex = None  # ccxt async client
        self._md_q: asyncio.Queue[MDRequest] = asyncio.Queue()
        self._trade_q: asyncio.Queue[TradeIntent] = asyncio.Queue()

        self._md_workers: List[asyncio.Task] = []
        self._trade_workers: List[asyncio.Task] = []

        self._rl = RateLimiter(MAX_CALLS_PER_SEC)
        self._cb = CircuitBreaker()
        self._running = False

    # ---------- lifecycle ----------
    async def start(self):
        if self._running:
            return
        self._running = True
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SEC))
        # ініціалізація клієнта
        self._ex = await self._create_exchange()
        # воркери
        for _ in range(MD_CONCURRENCY):
            self._md_workers.append(asyncio.create_task(self._md_worker()))
        for _ in range(TRADE_CONCURRENCY):
            self._trade_workers.append(asyncio.create_task(self._trade_worker()))
        send_message("⚙️ AsyncEngine запущено.")

    async def stop(self):
        self._running = False
        # дренування черг
        await self._md_q.join()
        await self._trade_q.join()
        # зупинка воркерів
        for t in self._md_workers + self._trade_workers:
            t.cancel()
        await asyncio.gather(*self._md_workers, return_exceptions=True)
        await asyncio.gather(*self._trade_workers, return_exceptions=True)
        self._md_workers.clear(); self._trade_workers.clear()
        # закриття біржі/сесії
        try:
            if self._ex is not None:
                await self._ex.close()
        except Exception:
            pass
        try:
            if self._session:
                await self._session.close()
        except Exception:
            pass
        send_message("🛑 AsyncEngine зупинено.")

    # ---------- public API ----------
    async def submit_md(self, req: MDRequest):
        """Додати запит маркет-даних у чергу."""
        await self._md_q.put(req)

    async def submit_trade(self, ti: TradeIntent):
        """Додати торговий намір у чергу."""
        await self._trade_q.put(ti)

    # ---------- workers ----------
    async def _md_worker(self):
        while self._running:
            try:
                req: MDRequest = await self._md_q.get()
                try:
                    if self._cb.is_open(req.symbol):
                        # пропускаємо символ у фазі cooldown
                        await asyncio.sleep(1.0)
                        continue

                    data = await self._retry_call(self._fetch_ohlcv, req)
                    if req.on_data:
                        if asyncio.iscoroutinefunction(req.on_data):
                            await req.on_data(data)
                        else:
                            # sync колбек у threadpool?
                            req.on_data(data)
                finally:
                    self._md_q.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                send_message(f"⚠️ MD worker error: {e}")

    async def _trade_worker(self):
        while self._running:
            try:
                ti: TradeIntent = await self._trade_q.get()
                try:
                    if DRY_RUN:
                        # симуляція
                        exec_px = await self._last_price(ti.symbol)
                        res = {
                            "status": "dry-run", "symbol": ti.symbol, "side": ti.side,
                            "qty": ti.qty, "price": exec_px, "ts": time.time()
                        }
                        # TCA — прикинемо фі
                        try:
                            notional = abs(float(ti.qty) * float(exec_px or 0))
                            fee = estimate_fee(notional)
                            log_tca("state/tca_events.json", {"event": "entry", "symbol": ti.symbol,
                                                              "side": ti.side, "px": exec_px,
                                                              "notional": notional, "fee": fee,
                                                              "ts": int(time.time())})
                        except Exception:
                            pass
                        if ti.on_exec:
                            await self._safe_cb(ti.on_exec, res)
                        continue

                    # live-режим
                    if self._cb.is_open("__trade__"):
                        await asyncio.sleep(1.0)
                        continue

                    res = await self._retry_call(self._place_order, ti)
                    if ti.on_exec:
                        await self._safe_cb(ti.on_exec, res)
                finally:
                    self._trade_q.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                send_message(f"⚠️ Trade worker error: {e}")

    # ---------- low-level ops ----------
    async def _create_exchange(self):
        """
        Створює async клієнт ccxt з потрібними опціями.
        """
        cls = getattr(ccxt_async, EXCHANGE_NAME) if hasattr(ccxt_async, EXCHANGE_NAME) else ccxt_async.mexc3
        ex = cls({
            "apiKey": API_KEY,
            "secret": API_SECRET,
            "enableRateLimit": True,
            "aiohttp_session": self._session,
            "options": {"defaultType": DEFAULT_TYPE}
        })
        # простий пінг
        try:
            await self._rl.acquire()
            await ex.fetch_time()
        except Exception as e:
            send_message(f"⚠️ async exchange init warning: {e}")
        return ex

    async def _fetch_ohlcv(self, req: MDRequest):
        await self._rl.acquire()
        if self._cb.is_open(req.symbol):
            raise RuntimeError("circuit open")
        # ccxt: returns list[list[ms, o,h,l,c,v]]
        ohlcv = await self._ex.fetch_ohlcv(req.symbol, timeframe=req.timeframe, limit=req.limit)
        self._cb.record_success(req.symbol)
        # повернемо просту структуру (щоб не тягнути pandas тут)
        return {
            "symbol": req.symbol,
            "timeframe": req.timeframe,
            "data": ohlcv,
            "ctx": req.ctx
        }

    async def _last_price(self, symbol: str) -> Optional[float]:
        try:
            await self._rl.acquire()
            t = await self._ex.fetch_ticker(symbol)
            return float(t.get("last") or t.get("close") or 0.0)
        except Exception:
            return None

    async def _place_order(self, ti: TradeIntent) -> Dict[str, Any]:
        await self._rl.acquire()
        if self._cb.is_open("__trade__"):
            raise RuntimeError("trade circuit open")

        side = ti.side
        ttype = ti.type
        params = {}
        if ti.post_only:
            params["postOnly"] = True

        if ttype == "limit":
            if ti.price is None:
                raise ValueError("limit order requires price")
            order = await self._ex.create_order(ti.symbol, "limit", side, ti.qty, ti.price, params)
            px = float(ti.price)
        else:
            order = await self._ex.create_order(ti.symbol, "market", side, ti.qty, None, params)
            # спробуємо взяти ціну
            px = float(order.get("price") or (await self._last_price(ti.symbol)) or 0.0)

        self._cb.record_success("__trade__")

        # TCA логування
        try:
            notional = abs(float(ti.qty) * float(px))
            fee = estimate_fee(notional)
            log_tca("state/tca_events.json", {"event": "entry", "symbol": ti.symbol,
                                              "side": side, "px": px,
                                              "notional": notional, "fee": fee,
                                              "ts": int(time.time())})
        except Exception:
            pass

        return {"status": "ok", "order": order, "px": px}

    # ---------- utils ----------
    async def _retry_call(self, fn: Callable, arg) -> Any:
        """
        Загальний ретраєр з експоненціальним backoff + jitter.
        Також взаємодіє з circuit breaker.
        """
        last_exc = None
        for i in range(RETRY_MAX_TRIES):
            try:
                return await fn(arg)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                last_exc = e
                # circuit breaker
                key = None
                if isinstance(arg, MDRequest):
                    key = arg.symbol
                elif isinstance(arg, TradeIntent):
                    key = "__trade__"
                self._cb.record_fail(key)

                delay = RETRY_BASE_DELAY * (2 ** i) + random.uniform(0, 0.25)
                await asyncio.sleep(min(delay, 8.0))
        # остаточний фейл
        raise last_exc

    async def _safe_cb(self, cb: Callable, *args, **kwargs):
        try:
            if asyncio.iscoroutinefunction(cb):
                await cb(*args, **kwargs)
            else:
                cb(*args, **kwargs)
        except Exception as e:
            send_message(f"⚠️ callback error: {e}")
