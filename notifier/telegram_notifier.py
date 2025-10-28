"""
Telegram Notifier — універсальний відправник повідомлень у канал / чат.
Підтримує автоперевірку конфігурації і повідомлення при старті.
"""

import os, requests, time

# --- Підтримка різних назв змінних середовища ---
TELEGRAM_BOT_TOKEN = (
    os.getenv("TELEGRAM_BOT_TOKEN")
    or os.getenv("TOKEN_BOT")
    or os.getenv("ТОКЕН_БОТА")
    or ""
)
TELEGRAM_CHAT_ID = (
    os.getenv("TELEGRAM_CHAT_ID")
    or os.getenv("CHAT_ID")
    or os.getenv("Ідентифікатор_Чату")
    or ""
)

def send_message(text, parse_mode="HTML", silent=False):
    """
    Відправляє повідомлення у Telegram.
    Якщо токен або ID відсутні — просто виводить у лог.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[TG:disabled]", text)
        return
    try:
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": parse_mode,
            "disable_notification": silent,
        }
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        r = requests.post(url, json=payload, timeout=10)
        if not r.ok:
            print("Telegram error:", r.text)
    except Exception as e:
        print("Telegram exception:", e)

def send_startup_message():
    ""

