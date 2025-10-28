# core/execution.py
"""
Smart Entry — виконує maker-first ордери з fallback на market.
Підтримує DRY_RUN режим, леверидж, контроль часу й повідомлення.
"""

import os, time
from notifier.telegram_notifier import send_message

DEFAULT_LEVERAGE = int(os.getenv("DEFAULT_LEVERAGE", "10"))
MAKER_FIRST = os.getenv("MAKER_FIRST", "True").lower() == "true"
MAKER_TIMEOUT_SEC = int(os.getenv("MAKER_TIMEOUT_SEC", "6"))
DRY_RUN = os.getenv("DRY_RUN", "True").lower() == "true"

def smart_entry(ex, symbol, side, amount, ref_price):
    """
    Першочергово намагається поставити postOnly (maker) ордер,
    якщо не виконується за timeout — скасовує й переходить на market.
    """
    ex.set_leverage(symbol, DEFAULT_LEVERAGE)

    if not MAKER_FIRST:
        o = ex.market_order(symbol, side, amount)
        return o, float(o.get("price") or ref_price)

    tick_price = ref_price * (0.999 if side == 'buy' else 1.001)

    try:
        o = ex.limit_order(symbol, side, amount, tick_price, params={"postOnly": True})
        start = time.time()
        while time.time() - start < MAKER_TIMEOUT_SEC:
            # У DRY_RUN — миттєве виконання
            if o.get("status") in ("filled", "closed") or str(o.get("id", "")).startswith("dry"):
                return o, float(o.get("price") or tick_price)
            time.sleep(0.5)
        try:
            ex.cancel_order(o["id"], symbol)
        except Exception:
            pass
    except Exception as e:
        send_message(f"⚠️ Maker order error: {e}")

    # fallback на market
    o = ex.market_order(symbol, side, amount)
    return o, float(o.get("price") or ref_price)
