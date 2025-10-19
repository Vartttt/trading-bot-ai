# app.py
import os
import requests
from flask import Flask, request, jsonify
from formatter import format_signal_message

app = Flask(__name__)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")  # може бути channel id або user id
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

if not TELEGRAM_TOKEN or not CHAT_ID:
    print("WARNING: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set in environment!")

@app.route("/")
def health():
    return "OK", 200

@app.route("/signal", methods=["POST"])
def signal():
    """
    Очікує JSON payload. Приклад повного payload нижче в README.
    Форматує повідомлення і шле в Telegram (HTML).
    """
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
        if resp.status_code != 200:
            return jsonify({"error": "Telegram API error", "details": resp.text}), 500

        return jsonify({"status": "sent"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
