# notifier/telegram_notifier.py
import os
import requests
from dotenv import load_dotenv
load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not TOKEN or not CHAT_ID:
    raise EnvironmentError("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set in env")

BASE = f"https://api.telegram.org/bot{TOKEN}"

def send_message(text, parse_mode="HTML"):
    url = f"{BASE}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True
    }
    resp = requests.post(url, data=payload, timeout=10)
    try:
        resp.raise_for_status()
    except Exception:
        print("Telegram send error:", resp.status_code, resp.text)
    return resp.json()

def send_photo(photo_path, caption=None, parse_mode="HTML"):
    url = f"{BASE}/sendPhoto"
    files = {"photo": open(photo_path, "rb")}
    data = {"chat_id": CHAT_ID}
    if caption:
        data["caption"] = caption
        data["parse_mode"] = parse_mode
    resp = requests.post(url, files=files, data=data, timeout=20)
    try:
        resp.raise_for_status()
    except Exception:
        print("Telegram photo send error:", resp.status_code, resp.text)
    return resp.json()
