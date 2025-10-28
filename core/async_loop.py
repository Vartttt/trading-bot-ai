"""
Async Trading Loop — головний асинхронний цикл для SmartTraderBot v8.4 Pro.

⚙️ Особливості:
  ✅ Паралельна обробка 10–30 символів без блокувань.
  ✅ Повна інтеграція з AsyncEngine (engine/async_executor.py).
  ✅ Використовує ті ж індикатори, ризик, TP/SL, Telegram.
  ✅ DRY_RUN / LIVE режим сумісний із попередніми версіями.
"""

import os
import asyncio
import time
from datetime import datetime

# 🔗 Імпорти основних модулів
from engine.async_executor import AsyncEngine, MDRequest, TradeIntent
from core.indicators import enrich
from core.data_feed import get_ohlcv
from core.trade_switch import is_trading_enabled
from core.trade_manager import open_signal_trade
from indicators.signal_strength import compute_signal_strength
from risk.smart_risk_curve import get_dynamic_risk
from risk.smart_tp_sl_curve import calc_smart_tp_sl
from core.phase_filter import filter_symbol_phase
from core.market_phase import load_phase_cache
from notifier.telegram_notifier import send_message

# --- Конфігурація
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", 30))
SYMBOLS = os.getenv(
    "SYMBOLS",
    "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT,TON/USDT,PEPE/USDT"
).split(",")
MIN_STRENGTH = int(os.getenv("MIN_STRENGTH", "72"))
DRY_RUN = os.getenv("DRY_RUN", "True").lower() == "true"

# --- AsyncEngine (спільний для всієї сесії)
engine = AsyncEngine()


# ============================================================
# 📊 Обробка нових барів
# ============================================================
async def on_market_data(md):
    """
    Колбек для обробки даних по кожному символу.
    Отримує {"symbol","timeframe","data":[ [ts,o,h,l,c,v], ... ]}
    """
    try:
        sym = md["symbol"]
        rows = md["data"]
        if not rows or len(rows) < 50:
            return

        import pandas as pd
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        x = enrich(df)
        if x is None or x.empty:
            return

        # фазовий фільтр (рівень тренду/регім)
        df1h = get_ohlcv(sym, timeframe="1h", limit=200)
        df4h = get_ohlcv(sym, timeframe="4h", limit=200)
        global_phase = load_phase_cache() or {}
        mult, comment, local_phase, local_regime = filter_symbol_phase(
            enrich(df1h) if df1h is not None else None,
            enrich(df4h) if df4h is not None else None,
            global_phase
        )

        # поточні дані
        last = x.iloc[-1]
        data = {
            "rsi": float(last.get("rsi", 50)),
            "macd": float(last.get("macd", 0)),
            "macd_signal": float(last.get("macds", 0)),
            "ema_fast": float(last.get("ema9", last.close)),
            "ema_slow": float(last.get("ema21", last.close)),
            "volume": float(last.get("volume", 1)),
            "avg_volume": float(x["volume"].tail(50).mean() or 1),
            "price": float(last.close),
            "atr": float(last.get("atr", last.close * 0.01)),
            "momentum": float(last.close - x["close"].iloc[-5])
        }

        # сила сигналу
        weights = {}
        s = compute_signal_strength(data, weights)
        strength = int(s["strength"] * mult)
        direction = s["direction"]

        # ризик / TP / SL
        risk_pct, risk_mode = get_dynamic_risk()
        tp_off, sl_off = calc_smart_tp_sl(data["atr"], strength, risk_mode)

        if strength < MIN_STRENGTH:
            return

        msg = (
            f"📈 <b>{sym}</b> | Сила сигналу: <b>{strength}%</b> ({direction})\n"
            f"Фаза: {local_phase} ({local_regime})\n"
            f"Ризик: {risk_mode} ({risk_pct*100:.2f}%)\n"
            f"TP≈{tp_off:.5f} | SL≈{sl_off:.5f}"
        )
        send_message(msg)

        # --- Торгівельна дія
        if not is_trading_enabled() or DRY_RUN:
            send_message(f"🧪 DRY_RUN → без відправки ордера ({sym})")
            return

        await engine.submit_trade(
            TradeIntent(
                symbol=sym,
                side="buy" if direction == "long" else "sell",
                qty=0.02,  # приблизно $20/поз
                type="market",
                on_exec=lambda res: send_message(f"✅ Ордер виконано: {res['symbol']} ({res['status']})")
            )
        )

    except Exception as e:
        send_message(f"⚠️ on_market_data error: {e}")


# ============================================================
# 🚀 Основний асинхронний цикл
# ============================================================
async def async_main():
    send_message("🚀 Async Trading Loop стартує...")
    await engine.start()

    try:
        # постійна подача даних
        while True:
            tasks = []
            for sym in SYMBOLS:
                req = MDRequest(symbol=sym, timeframe="15m", limit=200, on_data=on_market_data)
                tasks.append(engine.submit_md(req))
            await asyncio.gather(*tasks)
            await asyncio.sleep(CHECK_INTERVAL)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        send_message(f"💥 Main loop error: {e}")
    finally:
        await engine.stop()
        send_message("🛑 Async Loop зупинено.")


if __name__ == "__main__":
    asyncio.run(async_main())
