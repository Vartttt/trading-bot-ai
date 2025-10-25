# telegram_bot.py
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
from http_client import post_json

def notify(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        post_json(url, {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode":"HTML"}, timeout=8, retries=3)
    except Exception:
        pass  # не роняємо цикл торгівлі через збій Telegram
