import time, os
from notifier.telegram_notifier import send_message

MAKER_FIRST = os.getenv("MAKER_FIRST", "True").lower() == "true"
MAKER_TIMEOUT_SEC = int(os.getenv("MAKER_TIMEOUT_SEC", "6"))
FEE_RATE_BPS = float(os.getenv("FEE_RATE_BPS", "7")) / 10000.0  # 0.07%

def open_market_future(ex, symbol, side, usdt_notional, price):
    try:
        amount = usdt_notional / max(price, 1e-9)
        if MAKER_FIRST:
            # спроба maker (postOnly)
            try:
                order = ex.ex.create_limit_order(symbol, side, amount, price, {"postOnly": True})
                t0 = time.time()
                while time.time() - t0 < MAKER_TIMEOUT_SEC:
                    status = ex.ex.fetch_order(order["id"], symbol)
                    if status.get("status") == "closed":
                        send_message(f"✅ Maker fill {symbol} {side} {amount:.4f}")
                        return status, amount
                    time.sleep(0.5)
                # не встигло заповнитись → маркет
                ex.ex.cancel_order(order["id"], symbol)
                send_message(f"⚠️ Maker не заповнився → маркет {symbol}")
            except Exception:
                pass
        order = ex.ex.create_market_order(symbol, side, amount)
        return order, amount
    except Exception as e:
        send_message(f"❌ Помилка open_market_future: {e}")
        return None, 0.0
