# bot_listener.py
import telebot
import os
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

bot = telebot.TeleBot(TOKEN)

@bot.message_handler(commands=['start'])
def start_message(message):
    bot.reply_to(
        message,
        "👋 Привіт! Бот запущений і готовий до роботи.\n"
        "Натисни /signal, щоб отримати останній торговий сигнал 📈"
    )

@bot.message_handler(commands=['signal'])
def signal_message(message):
    bot.reply_to(
        message,
        "🟢 Приклад LONG сигналу (демо):\n\n"
        "Монета: BTC/USDT\nВхід: 68,200\nSL: 67,700\nTP1: 68,900\nTP2: 69,600\n"
        "⚙️ Алгоритм шукає реальні сигнали..."
    )

print("✅ Telegram listener started")
bot.infinity_polling()
