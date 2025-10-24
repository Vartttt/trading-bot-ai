# app.py
import threading
import time
import os
from datetime import datetime, timezone
from flask import Flask, jsonify

app = Flask(__name__)

# --- безпечне підключення модулів ---
try:
    from data_fetcher import fetch_ohlcv
    from strategy import compute_indicators, generate_signal
    from notifier.telegram_notifier import send_message, send_photo
    from persistence import init_db, save_signal
    from utils import plot_signal_chart
    from config import TOP_MANUAL_PAIRS, CHECK_INTERVAL_SECONDS, MIN_STRENGTH
    print("✅ Modules imported successfully")
except Exception as e:
    print("❌ Import error:", e)
    # Заглушки, якщо щось не імпортується
    fetch_ohlcv = compute_indicators = generate_signal = lambda *a, **k: None
    send_message = send_photo = lambda *a, **k: None
    init_db = save_signal = plot_signal_chart = lambda *a, **k: None
    TOP_MANUAL_PAIRS, CHECK_INTERVAL_SECONDS, MIN_STRENGTH = [], 60, 70

# --- Web ---
@app.route("/")
def home():
    return "🚀 Trading Signal Bot is running!"

@app.route("/health")
def health():
    return jsonify({"status": "ok", "time": datetime.now(timezone.utc).isoformat()})


# --- Формування сигналу ---
def format_signal_message(symbol, timeframe, sig):
    if sig["signal"] == "LONG":
        head = f"🟢🔥 <b>{symbol} {timeframe}</b>\n<b>СТАТУС:</b> LONG {sig['strength']}%\n"
        accent = "🟢🚀"
    else:
        head = f"🔴💥 <b>{symbol} {timeframe}</b>\n<b>СТАТУС:</b> SHORT {sig['strength']}%\n"
        accent = "🔴⚡️"

    msg = (
        f"{head}\n"
        f"<b>Вхід:</b> <code>{sig['entry']:.6g}</code>\n"
        f"<b>SL:</b> <code>{sig['sl']:.6g}</code>\n"
    )

    # Додаємо take-profit рівні
    probs = [sig["strength"], max(sig["strength"]-5, 60), max(sig["strength"]-15, 50)]
    for i, tp in enumerate(sig["tps"], start=1):
        msg += f"<b>TP{i}:</b> <code>{tp:.6g}</code> ({probs[i-1]}%)\n"

    msg += "\n<b>Тривалість:</b> 3h15m ⏱\n"
    msg += f"{accent} <b>Good luck — trade safe!</b>\n"

    return msg


# --- Основний процес ---
def background_loop():
    try:
        init_db()
    except Exception as e:
        print("DB init error:", e)

    sent = set()

    while True:
        for sym in TOP_MANUAL_PAIRS:
            try:
                df5 = fetch_ohlcv(sym, timeframe="5m", limit=200)
                df15 = fetch_ohlcv(sym, timeframe="15m", limit=200)
            except Exception as e:
                print("Fetch error", sym, e)
                continue

            try:
                df5i = compute_indicators(df5)
                df15i = compute_indicators(df15)
                sig = generate_signal(df5i, df15i)
            except Exception as e:
                print("Signal error", sym, e)
                continue

            if not sig or sig["strength"] < MIN_STRENGTH:
                continue

            uid = f"{sym}:{sig['signal']}:{int(sig['entry']*100000)}"
            if uid in sent:
                continue

            save_signal(sym, "5m", sig)
            os.makedirs("charts", exist_ok=True)
            chart_path = f"charts/{sym.replace('/','_')}.png"

            try:
                plot_signal_chart(df5i.tail(200), sym, entry=sig['entry'],
                                  sl=sig['sl'], tps=sig['tps'], out_path=chart_path)
            except Exception as e:
                print("Chart error", e)
                chart_path = None

            msg = format_signal_message(sym, "5m", sig)
            try:
                if chart_path:
                    send_photo(chart_path, caption=msg)
                else:
                    send_message(msg)
                print(f"✅ Sent signal: {sym} {sig['signal']} ({sig['strength']}%)")
            except Exception as e:
                print("Telegram send error:", e)

            sent.add(uid)

        time.sleep(CHECK_INTERVAL_SECONDS)


# --- Запуск ---
if __name__ == "__main__":
    t = threading.Thread(target=background_loop, daemon=True)
    t.start()
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


