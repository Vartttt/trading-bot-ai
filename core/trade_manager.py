import os, json, time
from notifier.telegram_notifier import send_message
from core.order_router import smart_entry
from core.exposure_guard import check_exposure
from risk.portfolio_risk import can_open_new
from analytics.tca import estimate_fee, log_tca

OPEN_POS_FILE = "state/open_positions.json"
TRADE_HISTORY_FILE = "state/trade_history.json"
TCA_FILE = "state/tca_events.json"

TP1_RATIO = float(os.getenv("TP1_RATIO", "0.5"))
TP2_RATIO = float(os.getenv("TP2_RATIO", "0.3"))
RUNNER_RATIO = float(os.getenv("RUNNER_RATIO", "0.2"))
TRAIL_ATR_K = float(os.getenv("TRAIL_ATR_K", "1.2"))
TRAIL_MIN_SECONDS = int(os.getenv("TRAIL_MIN_SECONDS", "120"))
MOVE_SL_TO_BE_AFTER_TP1 = os.getenv("MOVE_SL_TO_BE_AFTER_TP1", "True").lower() == "true"

def _load_json(path, default):
    if not os.path.exists(path): return default
    try: return json.load(open(path))
    except: return default

def _save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(data, open(path,"w"), indent=2)

def open_signal_trade(ex, symbol, direction, price, atr, base_risk, strength, tp_off, sl_off, factors):
    bal = ex.fetch_balance()
    usdt = float(((bal or {}).get("total") or {}).get("USDT", 0))
    if usdt <= 0:
        send_message("❌ No USDT balance available for trading.")
        return False, {}

    ok, reason = check_exposure(symbol)
    if not ok:
        send_message(f"⚠️ ExposureGuard: {reason}. Skip {symbol}.")
        return False, {}

    ok, reason = can_open_new(usdt)
    if not ok:
        send_message(f"⚠️ PortfolioRisk: {reason}. Skip {symbol}.")
        return False, {}

    risk_frac = base_risk
    notional = max(usdt * risk_frac, 5.0)
    side = "buy" if direction.upper()=="LONG" else "sell"

    amount_guess = notional / max(price,1e-9)
    order, exec_price = smart_entry(ex, symbol, side, amount_guess, price)

    fee_entry = estimate_fee(notional)
    log_tca(TCA_FILE, {"event":"entry","symbol":symbol,"side":side,"ref_price":price,"exec_price":exec_price,
                       "notional":notional,"fee":fee_entry,"ts":int(time.time())})

    amt_tp1 = amount_guess * TP1_RATIO
    amt_tp2 = amount_guess * TP2_RATIO
    amt_runner = max(amount_guess - amt_tp1 - amt_tp2, 0.0)

    pos = _load_json(OPEN_POS_FILE, {})
    pos[symbol] = {
        "symbol": symbol,
        "side": "long" if side=="buy" else "short",
        "entry": float(exec_price),
        "amount": amount_guess,
        "atr": atr,
        "opened_at": time.time(),
        "tp_levels": [
            {"target": float(exec_price) + (tp_off if side=='buy' else -tp_off), "amount": amt_tp1, "hit": False, "label": "TP1"},
            {"target": float(exec_price) + (tp_off*1.8 if side=='buy' else -tp_off*1.8), "amount": amt_tp2, "hit": False, "label": "TP2"}
        ],
        "runner": {"active": amt_runner>0, "amount": amt_runner},
        "sl": float(exec_price) - (sl_off if side=='buy' else -sl_off),
        "sl_last_update": time.time(),
        "sl_base": sl_off,
        "be_set": False,
        "factors": factors
    }
    _save_json(OPEN_POS_FILE, pos)
    send_message(f"📌 {symbol} {pos[symbol]['side']} entry={exec_price:.5f} TP/SL set.")
    return True, {"risk_frac":risk_frac,"notional":notional,"exec_price":exec_price}

def _get_last_price(ex, symbol):
    try:
        return float(ex.fetch_ticker(symbol)["last"])
    except Exception:
        return None

def tick_manage_positions(ex, on_close_pnl=None):
    pos = _load_json(OPEN_POS_FILE, {})
    changed=False
    for sym, p in list(pos.items()):
        px = _get_last_price(ex, sym)
        if px is None: continue
        # TP/SL обробка
        for t in p.get("tp_levels", []):
            if t["hit"]: continue
            hit = (p["side"]=="long" and px>=t["target"]) or (p["side"]=="short" and px<=t["target"])
            if hit:
                p["amount"] -= t["amount"]; t["hit"]=True; changed=True
                pnl_rel = (px-p["entry"])/p["entry"] if p["side"]=="long" else (p["entry"]-px)/p["entry"]
                if on_close_pnl: on_close_pnl(pnl_rel*(t["amount"]/max(p["amount"]+t["amount"],1e-9)))
        sl_hit = (p["side"]=="long" and px<=p["sl"]) or (p["side"]=="short" and px>=p["sl"])
        if sl_hit:
            pnl_rel = (px-p["entry"])/p["entry"] if p["side"]=="long" else (p["entry"]-px)/p["entry"]
            if on_close_pnl: on_close_pnl(pnl_rel)
            pos.pop(sym, None); changed=True; continue
        if p["amount"]<=1e-9:
            pos.pop(sym,None); changed=True
    if changed: _save_json(OPEN_POS_FILE,pos)
    return len(pos)
