import os, time, threading
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from flask import Flask, jsonify, Response
from prometheus_client import Gauge, Counter, generate_latest, CONTENT_TYPE_LATEST

from core.market_phase import compute_phase_from_df, save_phase_cache, load_phase_cache
from core.symbol_scanner import get_dynamic_symbols
from core.phase_filter import filter_symbol_phase
from core.position_synchronizer import synchronize_positions
from core.exchange_wrapper import ExchangeWrapper
from core.data_feed import get_ohlcv
from core.indicators import enrich
from indicators.signal_strength import compute_signal_strength
from risk.smart_risk_curve import get_dynamic_risk
from risk.smart_tp_sl_curve import tuned_tp_sl
from optimizer.smart_auto_optimizer import load_weights, optimize_weights
from notifier.telegram_notifier import send_message
from core.trade_manager import open_signal_trade, tick_manage_positions
from core.trade_switch import is_trading_enabled
from core.alpha_guards import session_guard, news_guard, funding_guard
from risk.risk_daily_guard import daily_risk_ok, report_trade_pnl
from core.health_monitor import exchange_ok

# ------------------ ENV CONFIG ------------------
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "30"))
PHASE_REFRESH_MIN = int(os.getenv("PHASE_REFRESH_MIN", "30"))
DRY_RUN = os.getenv("DRY_RUN", "True").lower() == "true"
MIN_STRENGTH = int(os.getenv("MIN_STRENGTH", "72"))

# ------------------ PROMETHEUS ------------------
g_last_tick = Gauge("stb_last_tick_ts", "Last background tick timestamp")
g_open_positions = Gauge("stb_open_positions", "Number of tracked open positions")
c_signals = Counter("stb_signals_total", "Signals seen (strength >= MIN_STRENGTH)")
c_trades = Counter("stb_trades_opened_total", "Trades opened")
c_trades_blocked = Counter("stb_trades_blocked_total", "Trades blocked by guards")
c_errors = Counter("stb_errors_total", "Errors encountered")

# ------------------ FLASK APP ------------------
app = Flask(__name__)

@app.route("/")
def home():
    return "SmartTraderBot v8.4 Pro Boosted — OK"

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

# ------------------ CORE FUNCTIONS ------------------
def refresh_market_phase(exchange):
    """Оновлює фазу ринку BTC/USDT (1h + 4h)"""
    try:
        df1h = get_ohlcv("BTC/USDT", timeframe="1h", limit=300)
        df4h = get_ohlcv("BTC/USDT", timeframe="4h", limit=300)
        if df1h is None or df1h.empty:
            return
        x1 = enrich(df1h)
        x4 = enrich(df4h) if df4h is not None and not df4h.empty else None
        rec = compute_phase_from_df(x1, x4)
        save_phase_cache(rec)
        send_message(f"🛰 Market Phase updated: {rec['phase']} | Regime: {rec['regime']}")
    except Exception as e:
        c_errors.inc()
        print("phase refresh error:", e)

# ------------------ MAIN LOOP ------------------
def background_loop():
    ex = ExchangeWrapper()
    last_phase_ts = 0
    last_opt_ts = 0

    send_message("🤖 SmartTraderBot v8.4 Boosted started successfully.")

    while True:
        try:
            # ✅ HEALTH MONITOR (повна перевірка біржі, latency, балансу та API rate)
            if not exchange_ok(ex):
                c_errors.inc()
                send_message("⛔️ Exchange health failed. Pausing one interval.")
                time.sleep(CHECK_INTERVAL)
                continue

            # 🔁 MARKET PHASE UPDATE
            if time.time() - last_phase_ts > PHASE_REFRESH_MIN * 60:
                refresh_market_phase(ex)
                synchronize_positions(ex)
                last_phase_ts = time.time()

            # 🧠 AUTO OPTIMIZATION (раз на 2 год)
            if time.time() - last_opt_ts > 7200:
                try:
                    new_w = optimize_weights()
                    send_message(f"🧠 Auto-optimized indicator weights: {new_w}")
                    last_opt_ts = time.time()
                except Exception as oe:
                    c_errors.inc()
                    print("optimizer error:", oe)

            # ⚙️ SYMBOL SCANNING
            symbols = get_dynamic_symbols(top_n=int(os.getenv("DYNSYM_TOPN", "12")))
            global_phase = load_phase_cache() or {}

            for sym in symbols:
                try:
                    # fetch candles
                    df15 = get_ohlcv(sym, timeframe="15m", limit=200)
                    if df15 is None or df15.empty:
                        continue
                    x = enrich(df15)

                    df1h = get_ohlcv(sym, timeframe="1h", limit=200)
                    df4h = get_ohlcv(sym, timeframe="4h", limit=200)
                    mult, comment, local_phase, local_regime = filter_symbol_phase(
                        enrich(df1h) if df1h is not None else None,
                        enrich(df4h) if df4h is not None else None,
                        global_phase
                    )

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

                    # compute signal
                    weights = load_weights()
                    s = compute_signal_strength(data, weights, phase=global_phase.get("phase"))
                    strength = int(s["strength"] * mult)
                    direction = s["direction"]

                    # risk + TP/SL (tuned)
                    risk_pct, risk_mode = get_dynamic_risk()
                    tp_off, sl_off, stats = tuned_tp_sl(data["atr"], strength, sym, regime=global_phase.get("phase","UNKNOWN"))

                    if strength >= MIN_STRENGTH:
                        c_signals.inc()
                        # Guards
                        if not session_guard() or not news_guard() or not daily_risk_ok() or not funding_guard(ex, sym):
                            c_trades_blocked.inc()
                            continue

                        send_message(
                            f"📊 <b>{sym}</b> | Strength: <b>{strength}%</b> dir={direction}\n"
                            f"Phase: {local_phase} ({mult}x {comment}) | Global: {global_phase.get('phase')}\n"
                            f"Risk: {risk_mode} ({risk_pct*100:.2f}%) | ATR={data['atr']:.5f}\n"
                            f"TP≈{tp_off:.5f} | SL≈{sl_off:.5f} | stats={stats}"
                        )

                        # TRADE EXECUTION
                        if is_trading_enabled() and not DRY_RUN:
                            try:
                                ok, trade_meta = open_signal_trade(
                                    ex, symbol=sym, direction=direction, price=data["price"], atr=data["atr"],
                                    base_risk=risk_pct, strength=strength,
                                    tp_off=tp_off, sl_off=sl_off, factors=s.get("factors", {}),
                                    phase=global_phase.get("phase")
                                )
                                if ok:
                                    c_trades.inc()
                                    report_trade_pnl(0.0)
                            except Exception as te:
                                c_errors.inc()
                                send_message(f"❌ Trade open error for {sym}: {te}")
                except Exception as se:
                    c_errors.inc()
                    print("symbol error", sym, se)

            # 🧩 POSITION MANAGEMENT
            try:
                open_count = tick_manage_positions(ex, on_close_pnl=report_trade_pnl)
                g_open_positions.set(open_count)
            except Exception as me:
                c_errors.inc()
                print("manage error:", me)

            g_last_tick.set(time.time())
            time.sleep(CHECK_INTERVAL)

        except Exception as e:
            c_errors.inc()
            print("main loop error:", e)
            send_message(f"⚠️ Main loop exception: {e}")
            time.sleep(5)

# ------------------ THREAD STARTER ------------------
def start_bg():
    th = threading.Thread(target=background_loop, daemon=True)
    th.start()

# ------------------ MAIN ------------------
if __name__ == "__main__":
    start_bg()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), debug=False)
