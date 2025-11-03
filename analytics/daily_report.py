from datetime import datetime
from notifier.telegram_notifier import send_message

def send_daily_report(balance, profit, trades, winrate):
    """
    Надсилає щоденний Telegram-звіт про роботу бота.
    """
    msg = (
        f"📊 <b>Щоденний звіт ({datetime.utcnow().strftime('%Y-%m-%d')})</b>\n"
        f"💰 Баланс: <code>{balance:.2f} USDT</code>\n"
        f"📈 Прибуток: <code>{profit:.2f}%</code>\n"
        f"🧠 Кількість угод: <code>{trades}</code>\n"
        f"🎯 Winrate: <code>{winrate:.1f}%</code>\n"
        f"⏰ UTC: {datetime.utcnow().strftime('%H:%M:%S')}"
    )
    send_message(msg)
