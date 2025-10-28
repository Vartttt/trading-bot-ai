"""
Telegram Notifier — універсальний відправник повідомлень у канал або чат.
Підтримує автоперевірку конфігурації та надсилає повідомлення при старті бота.
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
    Надсилає повідомлення у Telegram.
    Якщо токен або ID не вказані — виводить повідомлення у консоль.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[TG вимкнено]", text)
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
            print("⚠️ Помилка Telegram:", r.text)
    except Exception as e:
        print("❌ Виняток Telegram:", e)


def send_startup_message():
    """
    Надсилає стартове повідомлення у Telegram після запуску бота.
    """
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    send_message(
        f"🚀 <b>SmartTraderBot запущено</b>\n⏰ UTC: {start_time}\n"
        f"✅ Telegram повідомлення активні.",
        parse_mode="HTML",
        silent=True,
    )

