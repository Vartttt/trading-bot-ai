"""
Telegram Notifier — універсальний відправник повідомлень у канал або чат.
Підтримує автоперевірку конфігурації, обробку команд і надсилає повідомлення при старті бота.
"""

import os
import requests
import time

# ============================================================
# ⚙️ Конфігурація
# ============================================================
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

# ============================================================
# 📤 Надсилання повідомлення
# ============================================================
def send_message(text: str, parse_mode: str = "HTML", silent: bool = False):
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
        else:
            print(f"[TG] {text[:60]}{'...' if len(text) > 60 else ''}")
    except Exception as e:
        print("❌ Виняток Telegram:", e)

# ============================================================
# 🚀 Стартове повідомлення
# ============================================================
def send_startup_message():
    """
    Надсилає стартове повідомлення у Telegram після запуску бота.
    """
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    send_message(
        f"🚀 <b>SmartTraderBot запущено</b>\n"
        f"⏰ UTC: {start_time}\n"
        f"✅ Telegram повідомлення активні.",
        parse_mode="HTML",
        silent=True,
    )

# ============================================================
# 🧠 Обробка Telegram-команд (/safe_on, /safe_off, /safe_status)
# ============================================================
def handle_command(command: str):
    """
    Обробляє команди керування безпечним режимом:
    /safe_on — увімкнути безпечний режим
    /safe_off — вимкнути безпечний режим
    /safe_status — перевірити стан
    """
    try:
        from core.trading_events import set_safe_mode, is_safe_mode

        cmd = command.lower().strip()
        if cmd == "/safe_on":
            set_safe_mode(True)
        elif cmd == "/safe_off":
            set_safe_mode(False)
        elif cmd == "/safe_status":
            state = "🟢 Увімкнено" if is_safe_mode() else "🔴 Вимкнено"
            send_message(f"🛡️ Безпечний режим: {state}")
        else:
            send_message("❓ Невідома команда.\nДоступні: /safe_on /safe_off /safe_status")

        print(f"[CMD] Оброблено команду: {command}")

    except Exception as e:
        print(f"❌ Помилка обробки команди '{command}': {e}")
        send_message(f"⚠️ Помилка при виконанні команди: {e}")

# ============================================================
# 💡 Приклад використання (у симуляції або реальній торгівлі)
# ============================================================
"""
from core.trading_events import notify_open_position, notify_close_position, is_safe_mode

if not is_safe_mode():
    notify_open_position("BTCUSDT", "LONG", 68200, leverage=50)
    # ... виконати трейд
    profit = 12.3
    notify_close_position("BTCUSDT", profit)
else:
    send_message("🛡️ Безпечний режим увімкнено — відкриття позицій заблоковано.")
"""

