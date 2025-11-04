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
            print(f"⚠️ Telegram error: {r.text}")
        else:
            print(f"[TG] {text[:70]}{'...' if len(text) > 70 else ''}")

    except Exception as e:
        print(f"❌ Telegram exception: {e}")


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


# -*- coding: utf-8 -*-
"""
bot_listener.py — обробник команд Telegram для SmartTraderBot v8.4 Pro


# ============================================================
# 🧠 Обробник Telegram-команд
# ============================================================
def handle_command(command: str):
    """Головна функція для обробки команд із Telegram."""
    from core.trading_events import set_safe_mode, is_safe_mode

        cmd = command.strip().lower()
        dry_run = os.getenv("DRY_RUN", "True").lower() == "true"

        # --- /mode — показує поточний режим
        if cmd == "/mode":
            if dry_run:
                send_message(
                    "🧪 Поточний режим: <b>СИМУЛЯЦІЯ</b>\n"
                    "DRY_RUN=True — ордери не надсилаються на біржу.\n"
                    "Використовується для тестів без ризику."
                )
            else:
                send_message(
                    "💰 Поточний режим: <b>РЕАЛЬНА ТОРГІВЛЯ</b>\n"
                    "DRY_RUN=False — угоди виконуються через MEXC API.\n"
                    "⚠️ Використовуйте обережно!"
                )

        # --- /safe_on — увімкнути безпечний режим
        elif cmd == "/safe_on":
            from core.trading_events import set_safe_mode
            set_safe_mode(True)
            send_message("🛡️ Безпечний режим увімкнено. Торгівля призупинена.")

        # --- /safe_off — вимкнути безпечний режим
        elif cmd == "/safe_off":
            from core.trading_events import set_safe_mode
            set_safe_mode(False)
            send_message("⚙️ Безпечний режим вимкнено. Торгівля активна.")

        # --- /safe_status — перевірити стан
        elif cmd == "/safe_status":
            from core.trading_events import is_safe_mode
            state = "🟢 Увімкнено" if is_safe_mode() else "🔴 Вимкнено"
            send_message(f"🛡️ Безпечний режим: {state}")

        # --- /help — список усіх команд
        elif cmd == "/help":
            send_message(
                "📘 <b>Доступні команди:</b>\n"
                "• /mode — показати режим (симуляція / реальна торгівля)\n"
                "• /safe_on — увімкнути безпечний режим\n"
                "• /safe_off — вимкнути безпечний режим\n"
                "• /safe_status — перевірити стан безпечного режиму\n"
                "• /help — цей список команд"
            )

        else:
            send_message("❓ Невідома команда.\nВведіть /help щоб побачити список доступних команд.")

        print(f"[CMD] Оброблено команду: {command}")

    except Exception as e:
        print(f"❌ Помилка обробки команди '{command}': {e}")
        send_message(f"⚠️ Помилка при виконанні команди: {e}")


# ============================================================
# 💡 Приклад використання (у симуляції або реальній торгівлі)
# ============================================================
"""
from core.trading_events import notify_open_position, notify_close_position, is_safe_mode
from notifier.telegram_notifier import send_message

if not is_safe_mode():
    notify_open_position("BTCUSDT", "LONG", 68200, leverage=50)
    # ... виконати трейд
    profit = 12.3
    notify_close_position("BTCUSDT", profit)
else:
    send_message("🛡️ Безпечний режим увімкнено — відкриття позицій заблоковано.")
"""

