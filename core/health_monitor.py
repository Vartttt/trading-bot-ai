"""
Health Monitor v8.4 Pro — перевіряє:
  ✅ latency біржі
  ✅ баланс акаунта
  ✅ перевищення API rate-limit
  ✅ авто-перезапуск контейнера при збоях
  ✅ Telegram-сповіщення
"""

import os, time, statistics, traceback, subprocess
from notifier.telegram_notifier import send_message

# Зберігаємо історію вимірів для динамічного порогу
PING_HISTORY = []
API_CALLS = []
BALANCE_CACHE = {"ts": 0, "value": 0.0}

# --- CONFIG (ENV або дефолти)
MAX_LATENCY_SEC = float(os.getenv("MAX_LATENCY_SEC", "2.5"))
MAX_API_CALLS_PER_MIN = int(os.getenv("MAX_API_CALLS_PER_MIN", "180"))
MIN_BALANCE_USDT = float(os.getenv("MIN_BALANCE_USDT", "15.0"))
RESTART_FAILS = int(os.getenv("RESTART_FAILS", "3"))  # після 3 фейлів поспіль — restart

def ping_exchange(ex):
    """
    Перевірка доступності біржі й latency.
    Повертає True/False.
    """
    try:
        t0 = time.time()
        ex.fetch_ticker("BTC/USDT")
        dt = time.time() - t0
        PING_HISTORY.append(dt)
        if len(PING_HISTORY) > 25:
            PING_HISTORY.pop(0)
        avg = statistics.mean(PING_HISTORY)

        # оцінка відхилення
        if dt > MAX_LATENCY_SEC or dt > avg * 2.5:
            send_message(f"⚠️ <b>High latency</b>: {dt:.2f}s (avg={avg:.2f}s)")
        return True
    except Exception as e:
        send_message(f"🚫 <b>Exchange ping failed</b>: {e}")
        return False

def check_balance(ex):
    """
    Перевіряє, чи є баланс достатнім для продовження торгів.
    """
    global BALANCE_CACHE
    try:
        if time.time() - BALANCE_CACHE["ts"] < 60:
            return True, BALANCE_CACHE["value"]
        bal = ex.fetch_balance()
        usdt = float(((bal or {}).get("total") or {}).get("USDT", 0))
        BALANCE_CACHE = {"ts": time.time(), "value": usdt}
        if usdt < MIN_BALANCE_USDT:
            send_message(f"💰 <b>Low balance alert</b>: {usdt:.2f} USDT — trading paused.")
            return False, usdt
        return True, usdt
    except Exception as e:
        send_message(f"⚠️ Balance fetch error: {e}")
        return False, 0.0

def track_api_call():
    """
    Реєструє API виклики, щоб виявити rate-limit.
    """
    t = time.time()
    API_CALLS.append(t)
    while API_CALLS and t - API_CALLS[0] > 60:
        API_CALLS.pop(0)
    count = len(API_CALLS)
    if count > MAX_API_CALLS_PER_MIN * 0.9:
        send_message(f"⚠️ <b>API load high</b>: {count}/min (limit {MAX_API_CALLS_PER_MIN})")
    return count

def exchange_ok(ex):
    """
    Головна функція моніторингу:
    - викликає ping_exchange()
    - перевіряє баланс
    - перевіряє API rate
    - перезапускає контейнер при 3 поспіль збоях
    """
    try:
        track_api_call()
        fails = 0

        for _ in range(RESTART_FAILS):
            if ping_exchange(ex):
                ok, usdt = check_balance(ex)
                if ok:
                    return True
                else:
                    send_message("⛔️ Trading halted — balance check failed.")
                    return False
            else:
                fails += 1
                time.sleep(3)

        # Якщо 3 поспіль невдалі пінги → auto-restart
        if fails >= RESTART_FAILS:
            send_message("🚨 Exchange unreachable 3 times. Restarting container...")
            try:
                subprocess.run(["kill", "1"])
            except Exception as e:
                send_message(f"⚙️ Restart attempt failed: {e}")
            return False

        return True

    except Exception as e:
        send_message(f"💥 Health monitor error: {traceback.format_exc()}")
        return False

