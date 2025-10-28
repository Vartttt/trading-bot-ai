"""
Telegram Бот — головний інтерфейс керування та аналітики.
Команди: /start /status /phase /equity /report /tca /risk /tradeon /tradeoff /mode
"""
import os, json, telebot
from analytics.phase_report import current_phase, plot_phase_timeline
from analytics.equity_report import compute_kpis, plot_equity
from analytics.tca import log_tca
from risk.smart_risk_curve import load_stats
from core.trade_switch import set_trading, is_trading_enabled

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
bot = telebot.TeleBot(BOT_TOKEN, parse_mode="HTML") if BOT_TOKEN else None

def _load(path, default):
    if not os.path.exists(path): 
        return default
    try:
        return json.load(open(path))
    except:
        return default

if bot:
    @bot.message_handler(commands=["start"])
    def start(msg):
        bot.reply_to(msg, (
            "🤖 <b>SmartTraderBot v8.3 Pro</b> активний ✅\n\n"
            "📋 <b>Доступні команди:</b>\n"
            "/phase — поточна фаза ринку\n"
            "/equity — графік балансу\n"
            "/report — короткий KPI-звіт\n"
            "/tca — комісії та витрати\n"
            "/risk — стан ризиків\n"
            "/tradeon /tradeoff — вмикання або вимкнення торгівлі\n"
            "/mode — показати режим (реальний або симуляція)"
        ))

    @bot.message_handler(commands=["status"])
    def status(msg):
        bot.reply_to(msg, f"⚙️ <b>Режим торгівлі:</b> {'✅ УВІМКНЕНО' if is_trading_enabled() else '⛔️ ВИМКНЕНО'}")

    @bot.message_handler(commands=["tradeon"])
    def trade_on(msg):
        set_trading(True)
        bot.reply_to(msg, "✅ Торгівля <b>УВІМКНЕНА</b>")

    @bot.message_handler(commands=["tradeoff"])
    def trade_off(msg):
        set_trading(False)
        bot.reply_to(msg, "🛑 Торгівля <b>ВИМКНЕНА</b>")

    @bot.message_handler(commands=["phase"])
    def phase_cmd(msg):
        rec = current_phase()
        if not rec:
            bot.reply_to(msg, "❌ Фаза ринку наразі невідома.")
            return
        phase, regime = rec.get("phase"), rec.get("regime")
        scores = rec.get("scores", {})
        txt = (
            f"🌍 <b>Фаза ринку</b>: {phase} ({regime})\n"
            f"📈 vol1h={scores.get('vol1h','?')} | "
            f"trend={scores.get('trend_slope','?')} | "
            f"spikeZ={scores.get('spike_z','?')}"
        )
        bot.reply_to(msg, txt)
        p = plot_phase_timeline(7)
        if p:
            with open(p, "rb") as f:
                bot.send_photo(msg.chat.id, f, caption="📊 Історія фази ринку за 7 днів")

    @bot.message_handler(commands=["equity"])
    def equity(msg):
        path = plot_equity()
        kpis = compute_kpis()
        if not path:
            bot.reply_to(msg, "⚠️ Історія торгів відсутня.")
            return
        caption = (
            f"💰 <b>Звіт по балансу</b>\n"
            f"📊 Угод: {kpis.get('trades',0)}\n"
            f"🏆 Виграшних: {kpis.get('winrate',0):.1f}%\n"
            f"📈 Прибутковість: {kpis.get('gross',0)*100:.2f}%\n"
            f"📉 Максимальна просадка: {kpis.get('max_dd',0)*100:.2f}%"
        )
        with open(path, "rb") as f:
            bot.send_photo(msg.chat.id, f, caption=caption)

    @bot.message_handler(commands=["report"])
    def report(msg):
        kpis = compute_kpis()
        hist = _load("state/trade_history.json", [])
        txt = (
            f"📊 <b>KPI Звіт по торгівлі</b>\n"
            f"Угод: {kpis.get('trades',0)} | Виграшних: {kpis.get('winrate',0):.1f}%\n"
            f"Середній прибуток: {kpis.get('avg',0)*100:.3f}% | Макс. просадка: {kpis.get('max_dd',0)*100:.2f}%\n"
            f"Загальний PnL: {kpis.get('gross',0)*100:.2f}%"
        )
        bot.reply_to(msg, txt)

    @bot.message_handler(commands=["tca"])
    def tca_report(msg):
        tca = _load("state/tca_events.json", [])
        if not tca:
            bot.reply_to(msg, "— Немає збережених TCA-даних.")
            return
        total_fee = sum(e.get("fee",0) for e in tca)
        avg_fee = total_fee / max(len(tca),1)
        txt = (
            f"🧾 <b>Звіт по комісіях (TCA)</b>\n"
            f"Кількість операцій: {len(tca)}\n"
            f"Загальна комісія: ${total_fee:.2f}\n"
            f"Середня комісія на операцію: ${avg_fee:.2f}"
        )
        bot.reply_to(msg, txt)

    @bot.message_handler(commands=["risk"])
    def risk_cmd(msg):
        s = load_stats()
        txt = (
            f"⚖️ <b>Стан ризиків</b>\n"
            f"📈 Поточний PnL: {s.get('pnl',0)*100:.2f}%\n"
            f"🔹 Серія виграшів: {s.get('win_streak',0)} | 🔸 Серія програшів: {s.get('loss_streak',0)}"
        )
        bot.reply_to(msg, txt)

    @bot.message_handler(commands=["mode"])
    def mode_cmd(msg):
        dry = os.getenv("DRY_RUN", "False").lower() == "true"
        if dry:
            bot.reply_to(msg, "🧪 <b>Режим симуляції</b> — DRY_RUN=True\n(ордера не надсилаються на біржу)")
        else:
            bot.reply_to(msg, "💰 <b>Реальний режим</b> — DRY_RUN=False\n(угоди виконуються на MEXC)")

def run_bot():
    if not bot:
        print("Telegram бот вимкнено (немає TOKEN).")
        return

    print("Telegram бот працює через webhook…")

    # Зняти старий webhook, якщо був
    bot.remove_webhook()

    # Встановити новий webhook для Railway
    railway_url = os.getenv("RAILWAY_URL")  # наприклад: https://your-app.up.railway.app
    if not railway_url:
        print("⚠️ Відсутня змінна середовища RAILWAY_URL.")
        return

    bot.set_webhook(url=f"{railway_url}/{BOT_TOKEN}")


