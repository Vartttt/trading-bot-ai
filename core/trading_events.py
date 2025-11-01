# core/trading_events.py
import os
import json
import time
from notifier.telegram_notifier import send_message

SAFE_MODE_PATH = os.path.join("/tmp", "safe_mode.json")

# ======================================================
# 🛡️ Безпечний режим
# ======================================================
def set_safe_mode(enabled: bool):
    """Увімкнути або вимкнути безпечний режим."""
    data = {"enabled": enabled, "timestamp": time.time()}
    os.makedirs(os.path.dirname(SAFE_MODE_PATH), exist_ok=True)
    with open(SAFE_MODE_PATH, "w") as f:
        json.dump(data, f)

    state = "УВІМКНЕНО" if enabled else "ВИМКНЕНО"
    send_message(f"🛡️ Безпечний режим {state}.")
    print(f"[SAFE MODE] {state}")


def is_safe_mode():
    """Перевіряє стан безпечного режиму."""
    if not os.path.exists(SAFE_MODE_PATH):
        return False
    try:
        with open(SAFE_MODE_PATH, "r") as f:
            data = json.load(f)
        return bool(data.get("enabled", False))
    except Exception:
        return False


# ======================================================
# 💰 Події відкриття / закриття позицій
# ======================================================
def notify_open_position(symbol: str, side: str, price: float, leverage: int = 1, mode: str = "simulation"):
    """Повідомлення при відкритті позиції."""
    msg = (
        f"🚀 *{mode.upper()}*: Відкрито позицію\n"
        f"Монета: {symbol}\n"
        f"Напрям: {side}\n"
        f"Ціна: {price:.2f}\n"
        f"Плече: x{leverage}\n"
    )
    send_message(msg)


def notify_close_position(symbol: str, profit: float, mode: str = "simulation"):
    """Повідомлення при закритті позиції."""
    emoji = "💰" if profit > 0 else "📉"
    status = "прибутком" if profit > 0 else "збитком"
    msg = (
        f"{emoji} *{mode.upper()}*: Позицію {symbol} закрито з {status}\n"
        f"Результат: {profit:+.2f} USDT"
    )
    send_message(msg)
