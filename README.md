# Signal Telegram Bot (HTTP -> Telegram)

## Опис
Цей сервіс приймає POST /signal з JSON payload та надсилає красиво відформатоване повідомлення в Telegram.

## Налаштування (Railway)
1. Підготуй репозиторій з файлами.
2. В Railway створюй проект → Deploy from GitHub.
3. Додай Environment variables:
   - TELEGRAM_BOT_TOKEN
   - TELEGRAM_CHAT_ID
4. Деплой.

## Приклад payload (curl)
```bash
curl -X POST "https://your-project.up.railway.app/signal" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "WIFUSDT",
    "timeframe": "15m",
    "side": "LONG",
    "entry": "0.4825",
    "add": "0.4478",
    "tp": [
      {"price": "0.4897", "achieved": true, "minutes": 15, "percent": 70},
      {"price": "0.4946", "achieved": true, "minutes": 15, "percent": 50},
      {"price": "0.4994", "achieved": true, "minutes": 15, "percent": 40},
      {"price": "0.5042", "achieved": false, "minutes": 17, "percent": 30}
    ],
    "note": "Trade with risk management. Ризик 1.5%",
    "author": "@Uprime"
  }'
