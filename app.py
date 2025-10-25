from validate import ensure_env
# ...
def main():
    ensure_env()
    # далі як у тебе

import time, traceback, math
from datetime import datetime, timezone

from config import SYMBOLS, TIMEFRAME, CHECK_INTERVAL_SEC, MAX_OPEN_TRADES, TP_PCTS, TP_SIZES, SL_PCT, TRAIL_START_PCT, TRAIL_STEP_PCT, ATR_MULT
from exchange import create_exchange
from indicators import df_from_ohlcv, add_indicators, latest_row
from risk import position_size_usd, side_to_ccxt
from strategy import compute_signal_with_mtf, choose_leverage
from telegram_bot import notify
from learner import record_tp, record_sl
from logger import log_event
from market_utils import quantize_amount, quantize_price   # ✅ ДОДАНО

def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def atr_trailing_stop(entry, side, atr_value, last):
    """ATR-базований стоп: entry -/+ ATR_MULT * ATR."""
    if atr_value is None or math.isnan(atr_value):
        return None
    if side == "LONG":
        return max(entry * (1 - SL_PCT), last - ATR_MULT * atr_value)
    else:
        return min(entry * (1 + SL_PCT), last + ATR_MULT * atr_value)

def make_targets(entry: float, side: str):
    sign = 1 if side == "LONG" else -1
    tps = [entry * (1 + sign * p) for p in TP_PCTS]
    sl  = entry * (1 - sign * SL_PCT)
    return tps, sl

def update_trailing(trail: dict, last: float, side: str, atr_stop: float | None):
    """trail = {"active":bool,"stop":price} — якщо є atr_stop, віддаємо пріоритет йому."""
    if atr_stop is not None:
        trail["stop"] = atr_stop
        trail["active"] = True
        return trail

    # fallback: фіксований крок
    if not trail["active"]:
        return trail
    step = TRAIL_STEP_PCT
    if side == "LONG":
        desired = last * (1 - step)
        if trail["stop"] is None or desired > trail["stop"]:
            trail["stop"] = desired
    else:
        desired = last * (1 + step)
        if trail["stop"] is None or desired < trail["stop"]:
            trail["stop"] = desired
    return trail

def main():
    ex = create_exchange()
    notify("🤖 MEXC Smart Bot started (TPs, ATR-Trailing, MTF 15m/1h/4h, SQLite logs).")
    log_event("START", extra="bot online")

    open_positions = {}

    while True:
        try:
            balance = ex.fetch_balance(params={"type":"swap"})
            total_usdt = balance["total"].get("USDT", 0)
            free_usdt = balance["free"].get("USDT", 0)

            for symbol in SYMBOLS:
                ticker = ex.fetch_ticker(symbol, params={"type":"swap"})
                last = ticker["last"]

                # === Менеджмент існуючої позиції
                if symbol in open_positions:
                    pos = open_positions[symbol]
                    side = pos["side"]

                    # ATR stop оновлення
                    atr_val = pos.get("atr")
                    if atr_val is not None:
                        atr_stop = atr_trailing_stop(pos["entry"], side, atr_val, last)
                    else:
                        atr_stop = None

                    # активація трейла при профіті
                    if not pos["trail"]["active"]:
                        if (side == "LONG" and last >= pos["entry"] * (1 + TRAIL_START_PCT)) or \
                           (side == "SHORT" and last <= pos["entry"] * (1 - TRAIL_START_PCT)):
                            pos["trail"]["active"] = True

                    pos["trail"] = update_trailing(pos["trail"], last, side, atr_stop)

                    # Часткові TP
                    for i, tp_price in enumerate(pos["tps"]):
                        if not pos["tp_hit"][i]:
                            hit = (last >= tp_price) if side == "LONG" else (last <= tp_price)
                            if hit and pos["amount"] > 0:
                                size = pos["amount"] * TP_SIZES[i]
                                size = float(quantize_amount(ex, symbol, size))   # ✅ округлення під крок
                                if size > 0:
                                    reduce_side = "sell" if side == "LONG" else "buy"
                                    ex.create_order(symbol, "market", reduce_side, size, None, params={"reduceOnly": True, "type":"swap"})
                                    pos["amount"] -= size
                                    pos["tp_hit"][i] = True
                                    notify(f"✅ TP{i+1} hit {symbol} {side} @ {last:.4f}")
                                    log_event(f"TP{i+1}", symbol, side, last, size)
                                    record_tp()

                    # Закриття по трейлу/SL
                    stop = pos["trail"]["stop"] if pos["trail"]["active"] and pos["trail"]["stop"] else pos["sl"]
                    stop_hit = (last <= stop) if side == "LONG" else (last >= stop)
                    if stop_hit and pos["amount"] > 0:
                        reduce_side = "sell" if side == "LONG" else "buy"
                        amt = float(quantize_amount(ex, symbol, pos["amount"]))  # ✅ округлення
                        if amt > 0:
                            ex.create_order(symbol, "market", reduce_side, amt, None, params={"reduceOnly": True, "type":"swap"})
                        pnl_note = "TRAIL" if (pos["trail"]["active"] and pos["trail"]["stop"]) else "SL"
                        notify(f"⛔ {pnl_note} exit {symbol} {side} @ {last:.4f}")
                        log_event(pnl_note, symbol, side, last, amt)
                        record_sl() if pnl_note == "SL" else record_tp()
                        del open_positions[symbol]
                        continue

                    if pos["amount"] <= 1e-12:
                        notify(f"📘 Position fully closed {symbol}")
                        log_event("FLAT", symbol, side)
                        del open_positions[symbol]
                        continue

                    continue  # менеджмент завершено

                # === Відкриття нової позиції
                if len(open_positions) >= MAX_OPEN_TRADES or free_usdt < 10:
                    continue

                ohlcv = ex.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=400, params={"type":"swap"})
                df15 = add_indicators(df_from_ohlcv(ohlcv))
                side, conf, (bias1, bias2) = compute_signal_with_mtf(ex, symbol, df15)
                lev, thr = choose_leverage(conf)

                notional = position_size_usd(total_usdt) * lev
                price = last
                amount = notional / price

                # округлення amount під крок біржі
                amount = float(quantize_amount(ex, symbol, amount))       # ✅
                price_q = float(quantize_price(ex, symbol, price))        # (на майбутнє для ліміток)

                if amount <= 0:
                    continue

                try:
                    ex.set_leverage(lev, symbol, params={"type":"swap"})
                except Exception:
                    pass

                order_side = side_to_ccxt(side)
                ex.create_order(symbol, "market", order_side, amount, None, params={"type":"swap"})

                # ATR зі свічок 15m
                atr_val = float(latest_row(df15)["atr"]) if "atr" in df15.columns else None
                tps, sl = make_targets(price, side)
                trail_stop = atr_trailing_stop(price, side, atr_val, last)

                open_positions[symbol] = {
                    "side": side,
                    "entry": price,
                    "amount": amount,
                    "leverage": lev,
                    "tps": tps,
                    "tp_hit": [False]*len(tps),
                    "sl": sl,
                    "trail": {"active": False, "stop": trail_stop},  # одразу ATR-стоп, якщо є
                    "atr": atr_val,
                }

                notify(f"📈 Open {side} {symbol} @ {price:.4f} | conf={conf:.2f} thr={thr:.2f} lev=x{lev} | bias {bias1}/{bias2}")
                log_event("OPEN", symbol, side, price, amount, extra=f"conf={conf:.2f},lev=x{lev},bias={bias1}/{bias2}")

            time.sleep(CHECK_INTERVAL_SEC)

        except Exception as e:
            notify(f"⚠️ Loop error: {e}")
            log_event("ERROR", extra=str(e))
            time.sleep(CHECK_INTERVAL_SEC)

if __name__ == "__main__":
    main()



