"""
Health Monitor — перевіряє доступність біржі й API latency.
"""
import time, statistics
from notifier.telegram_notifier import send_message

PING_HISTORY = []

def ping_exchange(ex):
    try:
        t0 = time.time()
        ex.fetch_ticker("BTC/USDT")
        dt = (time.time() - t0)
        PING_HISTORY.append(dt)
        if len(PING_HISTORY) > 20:
            PING_HISTORY.pop(0)
        avg = statistics.mean(PING_HISTORY)
        if dt > avg * 2 or dt > 2.0:
            send_message(f"⚠️ High latency detected: {dt:.2f}s (avg={avg:.2f}s)")
        return True
    except Exception as e:
        send_message(f"🚫 Exchange connection error: {e}")
        return False
