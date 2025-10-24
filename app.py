# app.py
import threading
import time
import os
from datetime import datetime, UTC
from flask import Flask, jsonify

app = Flask(__name__)

# --- безпечне підключення модулів ---
try:
    from data_fetcher import fetch_ohlcv
    from strategy import compute_indicators, generate_signal, predict_cross_eta
    from notifier.telegram_notifier import send_message, send_photo
    from persistence import init_db, save_signal
    from utils import plot_signal_chart
    from config import TOP_MANUAL_PAIRS, CHECK_INTERVAL_SECONDS, MIN_STRENGTH, TIMEFRAME_LABEL
    print("✅ Modules imported successfully")
except Exception as e:
    print("❌ Import error:", e)
    # щоб Gunicorn не впав — створимо заглушки
    fetch_ohlcv = compute_indicators = generate_signal = predict_cross_eta = lambda *a, **k: None
    send_message = send_photo = lambda *a, **k: None
    init_db = save_signal = plot_signal_chart = lambda *a, **k: None
    TOP_MANUAL_PAIRS, CHECK_INTERVAL_SECONDS, MIN_STRENGTH, TIMEFRAME_LABEL = [], 60, 70, "5m"


@app.route("/")
def home():
    return "Trading signal bot is running ✅"


@app.route("/health")
def health():
    # тепер час у форматі UTC без попереджень
    return jsonify({"status": "ok", "time": datetime.now(UTC).isoformat()})


def format_signal_message(symbol, timeframe, sig):
    if sig["signal"] == "LONG":
        head = f"🟢🔥 <b>СТАТУС   LONG   {sig['strength']}%</b>\n"
        accent = "🟢"
    else:
        head = f"🔴💥 <b>СТАТУС   SHORT   {sig['strength']}%</b>\n"
        accent = "🔴"

    probs = [sig["strength"], max(sig["strength"] - 4, 50), max(sig["strength"] - 20, 40)]
    tps_lines = []
    for i, tp in enumerate(sig["tps"]):
        minutes = (i + 1) * 60
        tps_lines.append(f"ТР. {i+1} : {tp:.6g} ({probs[i]}%).     (0d.0h.{minutes}m)")

    msg = (
        f"<b>{symbol}  {timeframe}</b>\n"
        f"{head}\n"
        f"Вхід : <code>{sig['entry']:.6g}</code>\n"
        f"SL. : <code>{sig['sl']:.6g}</code>\n"
    )
    for line in tps_lines:
        msg += line + "\n"
    msg += "\n⏱ <b>Тривалість:</b> 3h15m\n"
    msg += f"\n{accent} Good luck — trade safe!"
    return msg


def format_pre_entry_message(symbol, timeframe, minutes_to_entry, pred_strength, est_entry):
    return (
        f"⏳ <b>PRE-ENTRY ALERT</b>\n"
        f"<b>{symbol}  {timeframe}</b>\n"
        f"Очікувано вхід через: <b>{int(round(minutes_to_entry))}m</b>\n"
        f"Попередня ціна: <code>{est_entry:.6g}</code>\n"
        f"СТАТУС: {pred_strength}%\n"
        f"⚠️ Готуйтесь до входу!"
    )


def background_loop():
    try:
        init_db()
    except Exception as e:
        print("DB init error:", e)

    sent = set()
    last_pre = {}

    while True:
        for sym in TOP_MANUAL_PAIRS:
            try:
                df5 = fetch_ohlcv(sym, timeframe="5m", limit=200)
                df15 = fetch_ohlcv(sym, timeframe="15m", limit=200)
                df1 = fetch_ohlcv(sym, timeframe="1m", limit=120)
            except Exception as e:
                print("Fetch error", sym, e)
                continue

            try:
                df5i = compute_indicators(df5)
                df15i = compute_indicators(df15)
                df1i = compute_indicators(df1)
            except Exception as e:
                print("Indicator error", sym, e)
                continue

            try:
                # прогноз за 5 хв до входу
                eta = predict_cross_eta(df1i)
                if eta is not None and eta <= 6:
                    est_entry = float(df1i["close"].iloc[-1])
                    sig5 = generate_signal(df5i, df15i)
                    strength = sig5["strength"] if sig5 else 75
                    key = f"pre:{sym}:{int(datetime.now(UTC).timestamp())//300}"
                    if key not in last_pre and strength >= MIN_STRENGTH:
                        msg = format_pre_entry_message(sym, "5m", eta, strength, est_entry)
                        send_message(msg)
                        last_pre[key] = datetime.now(UTC)

                # основний сигнал
                sig = generate_signal(df5i, df15i)
                if sig:
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
                    if chart_path:
                        send_photo(chart_path, caption=msg)
                    else:
                        send_message(msg)
                    sent.add(uid)
            except Exception as e:
                print("Main loop error for", sym, e)

        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    t = threading.Thread(target=background_loop, daemon=True)
    t.start()
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


