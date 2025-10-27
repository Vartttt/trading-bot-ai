"""
Telegram Bot — головний інтерфейс керування та аналітики.
Команди: /start /status /phase /equity /report /tca /risk /tradeon /tradeoff
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
    if not os.path.exists(path): return default
    try: return json.load(open(path))
    except: return default

if bot:
    @bot.message_handler(commands=["start"])
    def start(msg):
        bot.reply_to(msg, (
            "🤖 <b>SmartTraderBot v8.3 Pro</b> Online\n"
            "/phase — ринок і режим\n"
            "/equity — графік балансу\n"
            "/report — короткий KPI звіт\n"
            "/tca — комісії та витрати\n"
            "/risk — стан ризиків\n"
            "/tradeon /tradeoff — включити або зупинити торгівлю"
        ))

    @bot.message_handler(commands=["status"])
    def status(msg):
        bot.reply_to(msg, f"⚙️ Trading mode: {'✅ ON' if is_trading_enabled() else '⛔️ OFF'}")

    @bot.message_handler(commands=["tradeon"])
    def trade_on(msg):
        set_trading(True)
        bot.reply_to(msg, "✅ Торгівля УВІМКНЕНА")

    @bot.message_handler(commands=["tradeoff"])
    def trade_off(msg):
        set_trading(False)
        bot.reply_to(msg, "🛑 Торгівля ВИМКНЕНА")

    @bot.message_handler(commands=["phase"])
    def phase_cmd(msg):
        rec = current_phase()
        if not rec:
            bot.reply_to(msg, "❌ Фаза ринку невідома.")
            return
        phase, regime = rec.get("phase"), rec.get("regime")
        scores = rec.get("scores", {})
        txt = (f"🌍 <b>Market Phase</b>: {phase} ({regime})\n"
               f"vol1h={scores.get('vol1h','?')} slope={scores.get('trend_slope','?')} spikeZ={scores.get('spike_z','?')}")
        bot.reply_to(msg, txt)
        p = plot_phase_timeline(7)
        if p:
            with open(p, "rb") as f:
                bot.send_photo(msg.chat.id, f, caption="📈 Timeline 7d")

    @bot.message_handler(commands=["equity"])
    def equity(msg):
        path = plot_equity()
        kpis = compute_kpis()
        if not path:
            bot.reply_to(msg, "— Немає історії торгів.")
            return
        caption = (
            f"💰 <b>Equity Report</b>\n"
            f"Trades: {kpis.get('trades',0)}\n"
            f"Winrate: {kpis.get('winrate',0):.1f}%\n"
            f"Gross: {kpis.get('gross',0)*100:.2f}%\n"
            f"Max DD: {kpis.get('max_dd',0)*100:.2f}%"
        )
        with open(path, "rb") as f:
            bot.send_photo(msg.chat.id, f, caption=caption)

    @bot.message_handler(commands=["report"])
    def report(msg):
        kpis = compute_kpis()
        hist = _load("state/trade_history.json", [])
        txt = (
            f"📊 <b>Trading KPI</b>\n"
            f"Trades: {kpis.get('trades',0)} | Winrate: {kpis.get('winrate',0):.1f}%\n"
            f"Avg/trade: {kpis.get('avg',0)*100:.3f}% | MaxDD: {kpis.get('max_dd',0)*100:.2f}%\n"
            f"Gross PnL: {kpis.get('gross',0)*100:.2f}%"
        )
        bot.reply_to(msg, txt)

    @bot.message_handler(commands=["tca"])
    def tca_report(msg):
        tca = _load("state/tca_events.json", [])
        if not tca:
            bot.reply_to(msg, "— Немає TCA даних.")
            return
        total_fee = sum(e.get("fee",0) for e in tca)
        avg_fee = total_fee / max(len(tca),1)
        txt = (
            f"🧾 <b>TCA Summary</b>\n"
            f"Events: {len(tca)}\n"
            f"Total Fees: ${total_fee:.2f}\n"
            f"Avg Fee/event: ${avg_fee:.2f}"
        )
        bot.reply_to(msg, txt)

    @bot.message_handler(commands=["risk"])
    def risk_cmd(msg):
        s = load_stats()
        txt = (
            f"⚖️ <b>Risk Status</b>\n"
            f"PnL: {s.get('pnl',0)*100:.2f}%\n"
            f"Win streak: {s.get('win_streak',0)} | Loss streak: {s.get('loss_streak',0)}"
        )
        bot.reply_to(msg, txt)

def run_bot():
    if not bot:
        print("Telegram bot disabled (no TOKEN).")
        return
    print("Telegram bot polling…")
    bot.infinity_polling()
