"""
Smart Order Router: maker-first (postOnly) з ребіддингом і fallback на market.
Ретраї з лінійним backoff. Орієнтовано на MEXC/Binance USDT-M.
"""
import os, time
from notifier.telegram_notifier import send_message

MAKER_FIRST = os.getenv("MAKER_FIRST","True").lower() == "true"
MAKER_TIMEOUT_SEC = int(os.getenv("MAKER_TIMEOUT_SEC","6"))
MAKER_REBID_SEC = int(os.getenv("MAKER_REBID_SEC","3"))
MAX_RETRY = int(os.getenv("MAX_RETRY","3"))
RETRY_SLEEP_SEC = float(os.getenv("RETRY_SLEEP_SEC","1.2"))
DEFAULT_LEVERAGE = int(os.getenv("DEFAULT_LEVERAGE","10"))

def _rebid_price(side, ref_price, step=0.0005):
    # крок ~ 5 bps, можна адаптувати від спреду
    return ref_price * (1 - step if side=='buy' else 1 + step)

def smart_entry(ex, symbol, side, amount, ref_price):
    """
    1) Установлює leverage
    2) Maker-first: postOnly ліміт з ребіддингом
    3) Якщо не вдалось — market з ретраями
    """
    ex.set_leverage(symbol, DEFAULT_LEVERAGE)

    if MAKER_FIRST:
        price = ref_price * (0.9995 if side=='buy' else 1.0005)
        start = time.time()
        try:
            o = ex.limit_order(symbol, side, amount, price, params={"postOnly": True})
            while time.time() - start < MAKER_TIMEOUT_SEC:
                # DRY_RUN вважаємо filled
                if o.get("status") in ("filled","closed") or str(o.get("id","")).startswith("dry"):
                    return o, float(o.get("price") or price)
                time.sleep(0.4)
                # ребід кожні MAKER_REBID_SEC
                if time.time() - start > MAKER_REBID_SEC:
                    try:
                        ex.cancel_order(o["id"], symbol)
                    except Exception:
                        pass
                    price = _rebid_price(side, price)
                    o = ex.limit_order(symbol, side, amount, price, params={"postOnly": True})
            # timeout -> fallback
            try:
                ex.cancel_order(o["id"], symbol)
            except Exception:
                pass
        except Exception:
            pass

    # taker fallback з ретраями
    last_err = None
    for i in range(MAX_RETRY):
        try:
            o = ex.market_order(symbol, side, amount)
            return o, float(o.get("price") or ref_price)
        except Exception as e:
            last_err = e
            time.sleep(RETRY_SLEEP_SEC * (i+1))
    send_message(f"❌ Market entry failed ({symbol}): {last_err}")
    raise last_err
