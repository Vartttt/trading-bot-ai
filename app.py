# app.py
import os
import requests
from flask import Flask, request, jsonify
from formatter import format_signal_message
from learning import record_result

app = Flask(__name__)

# ✅ Telegram налаштування
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
if TELEGRAM_TOKEN and CHAT_ID:
    TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
else:
    TELEGRAM_API = None
    print("⚠️ WARNING: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set!")

@app.route("/")
def health():
    return "✅ OK", 200

@app.route("/signal", methods=["POST"])
def signal():
    if not TELEGRAM_API:
        return jsonify({"error": "Telegram not configured"}), 500

    try:
        payload = request.get_json()
        if not payload:
            return jsonify({"error": "No JSON payload"}), 400

        message = format_signal_message(payload)
        resp = requests.post(
            TELEGRAM_API,
            json={
                "chat_id": CHAT_ID,
                "text": message,
                "parse_mode": "HTML",
                "disable_web_page_preview": True
            },
            timeout=15
        )

        success = resp.status_code == 200
        record_result(success)

        if not success:
            return jsonify({"error": "Telegram API error", "details": resp.text}), 500

        return jsonify({"status": "sent"}), 200

    except Exception as e:
        record_result(False)
        return jsonify({"error": str(e)}), 500

# ⚙️ Railway/Gunicorn використовують свій сервер, але локально можна запускати
if __name__ == "__main__":
    try:
        port = int(os.environ.get("PORT", 8080))
    except ValueError:
        print("⚠️ Invalid PORT, using 8080")
        port = 8080

    print(f"✅ Bot running on port {port}")
    app.run(host="0.0.0.0", port=port)




