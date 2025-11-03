import time
from notifier.telegram_notifier import send_message

def daily_risk_ok(current_loss_pct, max_daily_loss_pct=5.0):
    """
    Перевіряє, чи не перевищено добовий ліміт втрат.
    Якщо перевищено — надсилає повідомлення і повертає False.
    """
    if current_loss_pct > max_daily_loss_pct:
        send_message(
            f"⛔ <b>Добовий ліміт втрат перевищено:</b> {current_loss_pct:.2f}%\n"
            f"Торгівля призупинена до наступного дня.",
            parse_mode="HTML"
        )
        return False
    return True
