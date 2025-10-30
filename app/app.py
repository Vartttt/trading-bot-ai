import os, sys, time, threading
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import requests

def download_if_missing(url, save_path):
    """Завантажує файл із GitHub, якщо його немає локально."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if not os.path.exists(save_path):
        print(f"⬇️ Завантажую {os.path.basename(save_path)} з {url}")
        r = requests.get(url)
        if r.status_code != 200:
            raise Exception(f"❌ Не вдалося завантажити файл ({r.status_code}): {url}")
        with open(save_path, "wb") as f:
            f.write(r.content)
        print(f"✅ Файл збережено у {save_path}")
    else:
        print(f"✅ Файл {os.path.basename(save_path)} вже існує")

# 📂 Шляхи збереження
MODEL_PATH = "models/transformer_signal_model.pt"
SCALER_PATH = "models/transformer_scaler.joblib"

# 🔗 Прямі raw-посилання на GitHub (змінено на raw.githubusercontent.com)
GITHUB_MODEL_URL = "https://raw.githubusercontent.com/Vartttt/-/95a6ab24de8c306bb7e22f0c233edaaa1dedba8b/models/transformer_signal_model.pt"
GITHUB_SCALER_URL = "https://raw.githubusercontent.com/Vartttt/-/95a6ab24de8c306bb7e22f0c233edaaa1dedba8b/models/transformer_scaler.joblib"

# ⬇️ Завантаження обох файлів
download_if_missing(GITHUB_MODEL_URL, MODEL_PATH)
download_if_missing(GITHUB_SCALER_URL, SCALER_PATH)

from flask import Flask, jsonify, Response, request  # + request
import telebot                                       # + telebot
from notifier.bot_listener import run_bot, bot, BOT_TOKEN  # + bot, BOT_TOKEN
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
from notifier.bot_listener import run_bot, bot, BOT_TOKEN
BASE_URL = os.getenv("URL_ADDRESS", "")  # з середовища Railway                                    
run_bot()

# ------------------ ADAPTIVE PROTECTION LAYER ------------------

import statistics

SAFE_LATENCY_LIMIT = 0.6    # межа, коли вмикається Safe Mode
LATENCY_RECOVERY = 0.25     # коли стабілізується — вимикаємо Safe Mode
COOLDOWN_SECONDS = 600      # 10 хвилин паузи після збиткової угоди
MAX_DRAWDOWN_DAY = -3.0     # % денної просадки для зниження ризику

safe_mode = False
latency_log = []
cooldowns = {}
phase_stats = {}

def update_latency(latency: float):
    """Оновлює середню затримку і керує Safe Mode."""
    global safe_mode
    latency_log.append(latency)
    if len(latency_log) > 20:
        latency_log.pop(0)
    avg_latency = statistics.mean(latency_log)

    if avg_latency > SAFE_LATENCY_LIMIT and not safe_mode:
        safe_mode = True
        send_message(f"⚠️ <b>Безпечний режим увімкнено</b> — висока затримка ({avg_latency:.2f} с). Торгівля призупинена.")
    elif avg_latency < LATENCY_RECOVERY and safe_mode:
        safe_mode = False
        send_message(f"✅ <b>Безпечний режим вимкнено</b> — стабільна затримка ({avg_latency:.2f} с).")

    return avg_latency, safe_mode


def can_trade(symbol: str) -> bool:
    """Перевіряє, чи можна відкривати нову угоду."""
    if safe_mode:
        send_message("⏸ Торгівля тимчасово призупинена через високу затримку.")
        return False

    now = int(time.time())
    if symbol in cooldowns and now - cooldowns[symbol] < COOLDOWN_SECONDS:
        left = COOLDOWN_SECONDS - (now - cooldowns[symbol])
        send_message(f"🕒 Пауза для {symbol}: очікування {int(left)} с після збиткової угоди.")
        return False
    return True


def register_trade_result(symbol: str, phase: str, profit_pct: float):
    """Реєструє результат угоди для статистики winrate."""
    global phase_stats
    phase = phase or "UNKNOWN"
    if phase not in phase_stats:
        phase_stats[phase] = {"win": 0, "loss": 0}

    if profit_pct >= 0:
        phase_stats[phase]["win"] += 1
    else:
        phase_stats[phase]["loss"] += 1
        cooldowns[symbol] = int(time.time())

    total = phase_stats[phase]["win"] + phase_stats[phase]["loss"]
    winrate = 100 * phase_stats[phase]["win"] / max(total, 1)
    send_message(f"📊 Фаза {phase}: {winrate:.1f}% виграшних угод ({total} угод).")


def adjust_risk_on_drawdown(day_drawdown_pct: float, base_risk: float) -> float:
    """Знижує ризик при великій просадці."""
    if day_drawdown_pct is not None and day_drawdown_pct < MAX_DRAWDOWN_DAY:
        new_risk = base_risk * 0.5
        send_message(f"⚠️ Просадка {day_drawdown_pct:.2f}% → ризик знижено до {new_risk*100:.2f}%.")
        return new_risk
    return base_risk

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
    return "🤖 SmartTraderBot v8.4 Pro Boosted — працює стабільно ✅"

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route(f"/{BOT_TOKEN}", methods=["POST"])
def telegram_webhook():
    if not bot:
        return "bot disabled", 200
    update = telebot.types.Update.de_json(request.stream.read().decode("utf-8"))
    bot.process_new_updates([update])
    return "ok", 200

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
        send_message(f"🛰 Фаза ринку оновлена: {rec['phase']} | Режим: {rec['regime']}")
    except Exception as e:
        c_errors.inc()
        print("phase refresh error:", e)

# ------------------ MAIN LOOP ------------------
def background_loop():
    ex = ExchangeWrapper()
    last_phase_ts = 0
    last_opt_ts = 0

    send_message("🤖 SmartTraderBot v8.4 Boosted успішно запущено та працює стабільно ✅")

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
                    send_message(f"🧠 Автоматично оптимізовані ваги індикаторів: {new_w}")
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
                        # 🧠 Перевірка Safe Mode або Cooldown
                        if not can_trade(sym):
                            continue
                        if not session_guard() or not news_guard() or not daily_risk_ok() or not funding_guard(ex, sym):
                            c_trades_blocked.inc()
                            continue

                        send_message(
                            f"📊 <b>{sym}</b> | Сила сигналу: <b>{strength}%</b> напрям: {direction}\n"
                            f"Фаза: {local_phase} ({mult}x {comment}) | Глобальна: {global_phase.get('phase')}\n"
                            f"Ризик: {risk_mode} ({risk_pct*100:.2f}%) | ATR={data['atr']:.5f}\n"
                            f"TP≈{tp_off:.5f} | SL≈{sl_off:.5f} | статистика={stats}"
                        )

                        # TRADE EXECUTION
                        if is_trading_enabled() and not DRY_RUN:
                            try:
                                # ⚖️ Адаптація ризику при просадці
                                risk_pct = adjust_risk_on_drawdown(None, risk_pct)
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
                                send_message(f"❌ Помилка відкриття угоди для {sym}: {te}")
                except Exception as se:
                    c_errors.inc()
                    print("symbol error", sym, se)

            # 🧩 POSITION MANAGEMENT
            try:
                # 📊 Реєстрація результатів після кожної угоди
                open_count = tick_manage_positions(
                    ex,
                    on_close_pnl=lambda pnl: register_trade_result(sym, global_phase.get("phase"), pnl)
                )
                
                g_open_positions.set(open_count)
            except Exception as me:
                c_errors.inc()
                print("manage error:", me)

            g_last_tick.set(time.time())
            time.sleep(CHECK_INTERVAL)

            # 🕒 Контроль затримки (latency)
            latency = time.time() - last_phase_ts  # або заміни на реальний час виконання циклу
            avg_lat, _ = update_latency(latency)

        except Exception as e:
            c_errors.inc()
            print("main loop error:", e)
            send_message(f"⚠️ Main loop exception: {e}")
            time.sleep(5)

def start_bg():
    th = threading.Thread(target=background_loop, daemon=True)
    th.start()

# ------------------ MAIN ------------------
if __name__ == "__main__":
    start_bg()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), debug=False)
