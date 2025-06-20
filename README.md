# -
import ccxt
import time
import pandas as pd
import numpy as np
from datetime import datetime
import requests

# === КОНФІГУРАЦІЯ ===
API_KEY = 'your_api_key'
SECRET_KEY = 'your_secret_key'
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'SOL/USDT', 'TON/USDT']
TIMEFRAMES = ['15m', '1h', '2h', '4h']
CAPITAL = 1000  # загальний капітал у USDT
MIN_BALANCE = 20
CHECK_INTERVAL = 30  # у секундах
MAX_LEVERAGE = 50
MIN_LEVERAGE = 10

exchange = ccxt.mexc({
    'apiKey': API_KEY,
    'secret': SECRET_KEY,
    'enableRateLimit': True
})

# === ІНДИКАТОРИ ===
def ema(df, period):
    return df['close'].ewm(span=period, adjust=False).mean()

def rsi(df, period=14):
    delta = df['close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def macd(df):
    ema12 = ema(df, 12)
    ema26 = ema(df, 26)
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

# === СИЛА СИГНАЛУ ===
def calculate_signal_strength(df):
    rsi_val = rsi(df).iloc[-1]
    ema_fast = ema(df, 10).iloc[-1]
    ema_slow = ema(df, 21).iloc[-1]
    macd_line, signal_line, _ = macd(df)

    score = 0
    if rsi_val < 30:
        score += 1
    elif rsi_val > 70:
        score -= 1

    if ema_fast > ema_slow:
        score += 1
    elif ema_fast < ema_slow:
        score -= 1

    if macd_line.iloc[-1] > signal_line.iloc[-1]:
        score += 1
    else:
        score -= 1

    # сила сигналу у %
    return max(min((score + 3) / 6 * 100, 100), 0)

# === ТРЕЙДИНГ ЛОГІКА ===
def fetch_ohlcv(symbol, timeframe):
    data = exchange.fetch_ohlcv(symbol, timeframe)
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def decide_and_trade():
    for symbol in SYMBOLS:
        strength_scores = []

        for tf in TIMEFRAMES:
            df = fetch_ohlcv(symbol, tf)
            score = calculate_signal_strength(df)
            strength_scores.append(score)

        avg_strength = sum(strength_scores) / len(strength_scores)
        direction = 'long' if avg_strength > 50 else 'short'
        leverage = MAX_LEVERAGE if avg_strength > 70 else MIN_LEVERAGE

        # === УМОВИ ВХОДУ ===
        if avg_strength > 55 or avg_strength < 45:
            print(f"🚀 {datetime.now()} | {symbol} | {direction.upper()} | Strength: {avg_strength:.2f}% | Leverage: x{leverage}")
            # Тут можна додати ордер через API, наприклад:
            # exchange.create_market_buy_order(symbol, amount)
        else:
            print(f"⏸ {datetime.now()} | {symbol} | No strong signal ({avg_strength:.2f}%)")

# === ГОЛОВНИЙ ЦИКЛ ===
while True:
    try:
        balance = exchange.fetch_balance()['total']['USDT']
        if balance < MIN_BALANCE:
            print("❗ Недостатньо балансу для торгівлі")
        else:
            decide_and_trade()
    except Exception as e:
        print(f"⚠️ Помилка: {e}")

    time.sleep(CHECK_INTERVAL)

import os
import time
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("MEXC_API_KEY")
api_secret = os.getenv("MEXC_API_SECRET")

def main():
    if not api_key or not api_secret:
        print("❌ API ключі не знайдено!")
        return

    print("✅ Бот підключено до MEXC API")
    print(f"API KEY: {api_key[:4]}...")
    # Тут логіка торгівлі

if __name__ == "__main__":
    main()
# === .env ===
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("mx0vglRO7tZcBeeq3y")
api_secret = os.getenv("b64311ce406c4baf91f775563484746b")

# === python-dotenv ===
pip install python-dotenv
python intelligent_trading_bot.py
