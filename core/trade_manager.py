# core/trade_manager.py
# v8.4 Boosted — Partial Add-ons, BE-перенос, ATR-трейл, TCA, авто-тюнінг TP/SL
# Сумісно з v8.3: open_signal_trade(), tick_manage_positions()

import os, json, time
from typing import Dict, Any, Tuple

from notifier.telegram_notifier import send_message
from core.execution import smart_entry
from core.position_sizer import compute_risk_fraction, compute_position_notional, choose_leverage
from analytics.tca import estimate_fee, log_tca
from optimizer.smart_auto_optimizer import optimize_weights
from risk.smart_tp_sl_curve import tuned_tp_sl, update_tp_sl_stats

OPEN_POS_FILE = "state/open_positions.json"
TRADE_HISTORY_FILE = "state/trade_history.json"
TCA_FILE = "state/tca_events.json"

# Split & trailing config
TP1_RATIO = float(os.getenv("TP1_RATIO", "0.5"))
TP2_RATIO = float(os.getenv("TP2_RATIO", "0.3"))
RUNNER_RATIO = float(os.getenv("RUNNER_RATIO", "0.2"))

TRAIL_ATR_K = float(os.getenv("TRAIL_ATR_K", "1.2"))
TRAIL_MIN_SECONDS = int(os.getenv("TRAIL_MIN_SECONDS", "120"))
MOVE_SL_TO_BE_AFTER_TP1 = os.getenv("MOVE_SL_TO_BE_AFTER_TP1", "True").lower() == "true"

# Add-on (pyramiding winners)
ADDON_ENABLE = os.getenv("ADDON_ENABLE", "True").lower() == "true"
ADDON_MAX = int(os.getenv("ADDON_MAX", "2"))                 # скільки разів можна добирати
ADDON_STEP_ATR = float(os.getenv("ADDON_STEP_ATR", "1.0"))   # кожні +1 ATR від останнього add-point
ADDON_PORTION_RATIO = float(os.getenv("ADDON_PORTION_RATIO", "0.3"))  # 30% від початкового розміру

# Safety minimums
MIN_NOTIONAL = float(os.getenv("MIN_NOTIONAL", "5"))

def _load_json(path, default):
    if not os.path.exists(path): return default
    try: return json.load(open(path))
    except: return default

def _save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(data, open(path, "w"), indent=2)

def _append_history(rec: Dict[str, Any]):
    hist = _load_json(TRADE_HISTORY_FILE, [])
    hist.append(rec)
    _save_json(TRADE_HISTORY_FILE, hist)

def _now() -> int:
    return int(time.time())

def _get_last_price(ex, symbol) -> float | None:
    try:
        return float(ex.fetch_ticker(symbol)["last"])
    except Exception:
        return None

def _close_market(ex, symbol: str, side_long: bool, amount: float):
    side = "sell" if side_long else "buy"
    try:
        ex.market_order(symbol, side, amount)
    except Exception:
        # in DRY mode wrapper already “fills”
        pass

def open_signal_trade(
    ex,
    symbol: str,
    direction: str,     # "LONG"|"SHORT"
    price: float,
    atr: float,
    base_risk: float,
    strength: int,
    tp_off: float,
    sl_off: float,
    factors: Dict[str, float],
    phase: str | None = None,
    vol_1h: float | None = None,
    recent_pnl_avg_pct: float = 0.0,
    day_drawdown_pct: float | None = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Відкриває позицію з адаптивним плечем і авто-тюнінгом TP/SL.
    """
    bal = ex.fetch_balance()
    usdt = float(((bal or {}).get("total") or {}).get("USDT", 0))
    if usdt <= 0:
        send_message("❌ No USDT balance available for trading.")
        return False, {}

    # risk sizing (fraction of equity)
    notional, risk_frac = compute_position_notional(
        equity_usdt=usdt, price=price, atr=atr, strength=strength, base_risk=base_risk
    )
    notional = max(notional, MIN_NOTIONAL)

    # leverage decision (optional; wrapper will try to set on exchange)
    lev = choose_leverage(
        strength=strength, phase=phase, vol_1h=vol_1h,
        recent_pnl_avg_pct=recent_pnl_avg_pct, safety_drawdown_pct=day_drawdown_pct
    )
    try:
        ex.set_leverage(symbol, lev)
    except Exception:
        pass

    # TP/SL tuned by history (per symbol)
    tp_off_tuned, sl_off_tuned, stats_used = tuned_tp_sl(
        atr_value=atr, signal_strength=strength, symbol=symbol, regime=phase or "UNKNOWN",
        base_tp=tp_off, base_sl=sl_off
    )

    side_long = (direction.upper() == "LONG")
    side_ccxt = "buy" if side_long else "sell"
    # crude amount by notional/price — futures leverage is set by exchange
    amount_guess = notional / max(price, 1e-9)

    # smart maker-first entry
    order, exec_price = smart_entry(ex, symbol, side_ccxt, amount_guess, price)

    # TCA entry
    fee_entry = estimate_fee(notional)
    log_tca(TCA_FILE, {
        "event": "entry", "symbol": symbol, "side": side_ccxt,
        "ref_price": price, "exec_price": exec_price,
        "notional": notional, "fee": fee_entry, "lev": lev, "ts": _now()
    })

    # split sizes
    total_amount = amount_guess
    amt_tp1 = total_amount * TP1_RATIO
    amt_tp2 = total_amount * TP2_RATIO
    runner_amt = max(total_amount - amt_tp1 - amt_tp2, 0.0)

    # initial SL
    sl = float(exec_price) - (sl_off_tuned if side_long else -sl_off_tuned)

    pos = _load_json(OPEN_POS_FILE, {})
    pos[symbol] = {
        "symbol": symbol,
        "side": "long" if side_long else "short",
        "entry": float(exec_price),
        "amount": float(total_amount),
        "atr": float(atr),
        "opened_at": _now(),
        "factors": factors,
        "tp_levels": [
            {"target": float(exec_price) + (tp_off_tuned if side_long else -tp_off_tuned),     "amount": float(amt_tp1), "hit": False, "label": "TP1"},
            {"target": float(exec_price) + (tp_off_tuned*1.8 if side_long else -tp_off_tuned*1.8), "amount": float(amt_tp2), "hit": False, "label": "TP2"},
        ],
        "runner": {"active": runner_amt > 0, "amount": float(runner_amt)},
        "sl": float(sl),
        "sl_last_update": _now(),
        "sl_base": float(sl_off_tuned),
        "be_set": False,

        # v8.4 — pyramiding winners (add-ons)
        "addons": [],
        "addon_cfg": {
            "enabled": ADDON_ENABLE,
            "max": ADDON_MAX,
            "step_atr": ADDON_STEP_ATR,
            "portion_ratio": ADDON_PORTION_RATIO
        },
        "next_addon_price": float(exec_price + (ADDON_STEP_ATR * atr if side_long else -ADDON_STEP_ATR * atr)),
        "phase": phase or "UNKNOWN",
        "lev": int(lev),
        "risk_frac": float(risk_frac),
        "tp_sl_stats_used": stats_used
    }
    _save_json(OPEN_POS_FILE, pos)

    send_message(
        f"📌 <b>{symbol}</b> {pos[symbol]['side'].upper()} @ {exec_price:.6f}\n"
        f"TP1/TP2 set • SL {pos[symbol]['sl']:.6f}\n"
        f"ATR {atr:.6f} • Lev x{lev} • Risk {risk_frac*100:.2f}%"
    )
    return True, {"risk_frac": risk_frac, "notional": notional, "exec_price": exec_price, "lev": lev}

# ------------------------ runtime management ----------------------------------

def _maybe_move_sl_to_be(p) -> bool:
    if p.get("be_set"): 
        return False
    if MOVE_SL_TO_BE_AFTER_TP1 and any(t["label"]=="TP1" and t["hit"] for t in p.get("tp_levels", [])):
        p["sl"] = p["entry"]
        p["be_set"] = True
        return True
    return False

def _trail_sl(p, px) -> bool:
    now = _now()
    if now - p.get("sl_last_update", 0) < max(TRAIL_MIN_SECONDS, 60):
        return False
    atr_move = p["atr"] * TRAIL_ATR_K
    moved = False
    if p["side"] == "long":
        new_sl = max(p["sl"], px - atr_move)
        if new_sl > p["sl"]:
            p["sl"] = new_sl; moved = True
    else:
        new_sl = min(p["sl"], px + atr_move)
        if new_sl < p["sl"]:
            p["sl"] = new_sl; moved = True
    if moved:
        p["sl_last_update"] = now
    return moved

def _record_close(p, symbol, exit_price, portion, reason) -> float:
    # portion PnL as fraction of original position notional (approx by amount ratio)
    signed = (exit_price - p["entry"]) if p["side"] == "long" else (p["entry"] - exit_price)
    portion_frac = portion / max(p["amount"] + portion, 1e-9)
    pnl_rel = signed / max(p["entry"], 1e-9) * portion_frac

    _append_history({
        "symbol": symbol, "side": p["side"], "entry": p["entry"], "exit": exit_price,
        "result": round(pnl_rel, 6), "reason": reason, "portion": portion, "ts": _now(),
        "factors": p.get("factors", {}), "phase": p.get("phase"), "lev": p.get("lev", None)
    })
    # update stats for auto TP/SL tuning
    update_tp_sl_stats(symbol=symbol, event=reason, entry=p["entry"], exit=exit_price, side=p["side"])
    return pnl_rel

def _try_addon(ex, symbol: str, p: Dict[str, Any], px: float) -> bool:
    cfg = p.get("addon_cfg", {})
    if not cfg.get("enabled", False): 
        return False
    done = len(p.get("addons", []))
    if done >= int(cfg.get("max", 0)):
        return False

    side_long = (p["side"] == "long")
    trigger_ok = (px >= p["next_addon_price"]) if side_long else (px <= p["next_addon_price"])
    if not trigger_ok:
        return False

    # compute portion amount
    portion_ratio = float(cfg.get("portion_ratio", 0.3))
    add_amount = p["amount"] * portion_ratio
    side_ccxt = "buy" if side_long else "sell"

    order, exec_price = smart_entry(ex, symbol, side_ccxt, add_amount, px)
    # log TCA
    log_tca(TCA_FILE, {
        "event": "addon", "symbol": symbol, "side": side_ccxt,
        "exec_price": exec_price, "portion": add_amount, "ts": _now()
    })
    p["amount"] += add_amount
    p.setdefault("addons", []).append({"price": exec_price, "amount": add_amount, "ts": _now()})

    # next trigger
    step = float(cfg.get("step_atr", 1.0)) * p["atr"]
    p["next_addon_price"] = float(exec_price + (step if side_long else -step))
    send_message(f"➕ Add-on {symbol} {p['side'].upper()} @ {exec_price:.6f} | next @ {p['next_addon_price']:.6f}")
    return True

def tick_manage_positions(ex, on_close_pnl=None) -> int:
    pos: Dict[str, Any] = _load_json(OPEN_POS_FILE, {})
    changed = False

    for sym, p in list(pos.items()):
        px = _get_last_price(ex, sym)
        if px is None:
            continue

        # Handle TP1/TP2 partials
        for t in p.get("tp_levels", []):
            if t["hit"]:
                continue
            hit = (p["side"] == "long" and px >= t["target"]) or (p["side"] == "short" and px <= t["target"])
            if hit and t["amount"] > 0:
                _close_market(ex, sym, side_long=(p["side"] == "long"), amount=t["amount"])
                pnl_rel = _record_close(p, sym, px, t["amount"], t["label"])
                notional = t["amount"] * px
                log_tca(TCA_FILE, {"event": "tp", "symbol": sym, "label": t["label"], "px": px,
                                   "portion": t["amount"], "fee": estimate_fee(notional), "ts": _now()})
                t["hit"] = True
                changed = True
                if on_close_pnl: on_close_pnl(pnl_rel)

        # Runner trailing
        if p.get("runner", {}).get("active") and p.get("runner", {}).get("amount", 0) > 0:
            if _trail_sl(p, px):
                changed = True

        # Move SL → BE after TP1
        if _maybe_move_sl_to_be(p):
            send_message(f"🛡 SL→BE {sym} @ {p['sl']:.6f}")
            changed = True

        # Add-on winners
        if _try_addon(ex, sym, p, px):
            changed = True

        # Stop-loss check
        sl_hit = (p["side"] == "long" and px <= p["sl"]) or (p["side"] == "short" and px >= p["sl"])
        if sl_hit:
            portion = p["amount"]
            if portion > 0:
                _close_market(ex, sym, side_long=(p["side"] == "long"), amount=portion)
                pnl_rel = _record_close(p, sym, px, portion, "SL")
                notional = portion * px
                log_tca(TCA_FILE, {"event": "sl", "symbol": sym, "px": px,
                                   "portion": portion, "fee": estimate_fee(notional), "ts": _now()})
                if on_close_pnl: on_close_pnl(pnl_rel)
            pos.pop(sym, None)
            changed = True
            continue

        # Remove if fully closed by TPs
        if p["amount"] <= 1e-9:
            pos.pop(sym, None)
            changed = True
            continue

        pos[sym] = p

    if changed:
        _save_json(OPEN_POS_FILE, pos)
    return len(pos)
