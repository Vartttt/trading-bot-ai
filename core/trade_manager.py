import os, json, time
from core.execution import open_market_future
from core.position_sizer import position_size
from notifier.telegram_notifier import send_message
from risk.smart_risk_curve import update_pnl
from core.risk_daily_guard import add_trade_pnl
from optimizer.smart_auto_optimizer import optimize_weights

STATE_FILE = "state/open_positions.json"
SHADOW_FILE = "state/shadow_positions.json"
ENABLE_SHADOW = os.getenv("ENABLE_SHADOW", "True").lower() == "true"

TP1_RATIO = float(os.getenv("TP1_RATIO", "0.5"))
TP2_RATIO = float(os.getenv("TP2_RATIO", "0.3"))
RUNNER_RATIO = float(os.getenv("RUNNER_RATIO", "0.2"))

def _load(p):
    if not os.path.exists(p): return {}
    try: return json.load(open(p))
    except: return {}

def _save(p, d):
    os.makedirs("state", exist_ok=True)
    json.dump(d, open(p,"w"), indent=2)

def open_signal_trade(ex, symbol, direction, price, atr, risk_pct, tp_off, sl_off, factors):
    bal = ex.fetch_balance()
    usdt = float(((bal or {}).get("total") or {}).get("USDT", 0))
    notional = position_size(usdt, factors.get("strength", 80), atr, price)
    side = "buy" if direction.upper()=="LONG" else "sell"
    order, amount = open_market_future(ex, symbol, side, notional, price)
    if not order: return False

    pos = _load(STATE_FILE)
    pos[symbol] = {
        "symbol": symbol, "side": side, "entry": price,
        "amount": amount, "tp": tp_off, "sl": sl_off,
        "factors": factors, "opened": time.time()
    }
    _save(STATE_FILE, pos)
    send_message(f"📊 Opened {symbol} {side} {amount:.4f} @ {price:.5f}")

    if ENABLE_SHADOW:
        sh = _load(SHADOW_FILE)
        sh[symbol] = {"symbol":symbol, "side":"opposite", "entry":price,
                      "amount":amount, "strategy":"shadow", "opened":time.time()}
        _save(SHADOW_FILE, sh)
    return True

def _close(symbol, price, reason, pnl):
    pos = _load(STATE_FILE)
    if symbol not in pos: return
    pos.pop(symbol)
    _save(STATE_FILE, pos)
    update_pnl(pnl)
    add_trade_pnl(pnl)
    send_message(f"🔴 Close {symbol} ({reason}) Δ={pnl*100:.2f}%")

def tick_manage_positions(ex):
    pos = _load(STATE_FILE)
    for sym, p in list(pos.items()):
        try:
            price = float(ex.fetch_ticker(sym)["last"])
            side = p["side"]
            entry = p["entry"]
            pnl = (price-entry)/entry if side=="buy" else (entry-price)/entry
            if abs(pnl) >= 0.03:  # 3% move
                reason = "TP" if pnl>0 else "SL"
                _close(sym, price, reason, pnl)
                optimize_weights()
        except Exception as e:
            print("manage", e)
    return len(pos)
