# bot_listener.py
import telebot
import os
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

bot = telebot.TeleBot(TOKEN)

@bot.message_handler(commands=['start'])
def start_message(message):
    bot.reply_to(
        message,
        "👋 Привіт! Бот активний і готовий надсилати сигнали 📈\n"
        "Використай /signal щоб отримати останні можливості для входу."
    )

@bot.message_handler(commands=['signal'])
def signal_message(message):
    bot.reply_to(
        message,
        "⚙️ Пошук торгових можливостей...\n⏳ Зачекай кілька секунд..."
    )
    # тут можна вставити логіку, щоб відправити останній сигнал або фейковий тест
    bot.send_message(
        message.chat.id,
        "🟢 LONG сигнал (приклад)\n\nМонета: BTC/USDT\nВхід: 68250\nSL: 67700\nTP1: 68900\nTP2: 69600"
    )

print("✅ Telegram listener started.")
bot.infinity_polling()
