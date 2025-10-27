import os, time, threading
from flask import Flask, jsonify, Response
from prometheus_client import Gauge, Counter, generate_latest, CONTENT_TYPE_LATEST

from core.exchange_wrapper import ExchangeWrapper
from core.data_feed import get_ohlcv
from core.indicators import enrich
from core.market_phase import compute_phase_from_df, save_phase_cache, load_phase_cache
from core.symbol_scanner import get_dynamic_symbols
from core.phase_filter import filter_symbol_phase
from indicators.signal_strength import compute_signal_strength
from optimizer.smart_auto_optimizer import load_weights
from core.trade_manager import open_signal_trade, tick_manage_positions
from core.trade_switch import is_trading_enabled
from risk.smart_risk_curve import get_dynamic_risk
from risk.smart_tp_sl_curve import calc_smart_tp_sl
from notifier.telegram_notifier import send_message
from core.alpha_guards import check_guards
from core.risk_daily_guard import check_daily_loss_limit

# Flask app
app = Flask(__name__)

CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "30"))
PHASE_REFRESH_MIN = int(os.getenv("PHASE_REFRESH_MIN", "30"))
DRY_RUN = os.getenv("DRY_RUN", "True").lower() == "true"
MIN_STRENGTH = int(os.getenv("MIN_STRENGTH", "70"))

# Prometheus metrics
g_last_tick = Gauge("stb_last_tick_ts", "Last background tick timestamp")
g_open_positions = Gauge("stb_open_positions", "Number of tracked open positions")
c_signals = Counter("stb_signals_total", "Signals >= MIN_STRENGTH")
c_trades = Counter("stb_trades_opened_total", "Trades opened")

@app.route("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route("/")
def home():
    return "SmartTraderBot v8.3 — OK"

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

def refresh_market_phase(exchange):
    try:
        df1h = get_ohlcv("BTC/USDT", "1h", 300)
        df4h = get_ohlcv("BTC/USDT", "4h", 300)
        if df1h is None or df1h.empty:
            return
        x1 = enrich(df1h)
        x4 = enrich(df4h)
        rec = compute_phase_from_df(x1, x4)
        save_phase_cache(rec)
        send_message(f"🛰 Market Phase: {rec['phase']} | Regime: {rec['regime']}")
    except Exception as e:
        print("Phase error:", e)

def background_loop():
    ex = ExchangeWrapper()
    last_phase_ts = 0

    while True:
        try:
            if check_daily_loss_limit():
                time.sleep(60)
                continue

            if not check_guards():
                time.sleep(60)
                continue

            # refresh market phase
            if time.time() - last_phase_ts > PHASE_REFRESH_MIN * 60:
                refresh_market_phase(ex)
                last_phase_ts = time.time()

            # dynamic symbol scanning
            symbols = get_dynamic_symbols(10)
            global_phase = load_phase_cache() or {}

            for sym in symbols:
                try:
                    df15 = get_ohlcv(sym, "15m", 200)
                    if df15 is None or df15.empty:
                        continue
                    x = enrich(df15)

                    df1h = get_ohlcv(sym, "1h", 200)
                    df4h = get_ohlcv(sym, "4h", 200)
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

                    weights = load_weights()
                    s = compute_signal_strength(data, weights)
                    strength = int(s["strength"] * mult)
                    direction = s["direction"]

                    risk_pct, risk_mode = get_dynamic_risk()
                    tp_off, sl_off = calc_smart_tp_sl(data["atr"], strength, risk_mode)

                    if strength >= MIN_STRENGTH:
                        c_signals.inc()
                        send_message(
                            f"📈 <b>{sym}</b> | Strength: <b>{strength}%</b> | {direction} "
                            f"({comment})\\nATR={data['atr']:.4f} | Risk={risk_mode}"
                        )

                        if is_trading_enabled() and not DRY_RUN:
                            try:
                                ok = open_signal_trade(
                                    ex, sym, direction, data["price"], data["atr"],
                                    risk_pct, tp_off, sl_off,
                                    factors={
                                        "rsi": data["rsi"],
                                        "macd": data["macd"],
                                        "ema": abs((data["ema_fast"]-data["ema_slow"])/max(data["ema_slow"],1e-9)),
                                        "volume": data["volume"]/max(data["avg_volume"],1e-9),
                                        "volatility": abs(data["momentum"])/max(data["atr"],1e-9)
                                    }
                                )
                                if ok:
                                    c_trades.inc()
                            except Exception as te:
                                send_message(f"❌ Trade open error for {sym}: {te}")

                except Exception as se:
                    print("symbol error", sym, se)

            try:
                count = tick_manage_positions(ex)
                g_open_positions.set(count)
            except Exception as me:
                print("manage error:", me)

            g_last_tick.set(time.time())
            time.sleep(CHECK_INTERVAL)

        except Exception as e:
            print("Main loop error:", e)
            time.sleep(5)

def start_bg():
    th = threading.Thread(target=background_loop, daemon=True)
    th.start()

if __name__ == "__main__":
    start_bg()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), debug=False)
