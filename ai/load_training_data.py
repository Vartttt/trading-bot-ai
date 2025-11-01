import pandas as pd
import ta
import requests
import json
import os

def load_training_data(symbol="BTCUSDT", interval="15m", limit=20000):
    print(f"📊 Завантажую {limit} свічок з MEXC для {symbol} ({interval})...")
    url = f"https://api.mexc.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"

    try:
        r = requests.get(url, timeout=15)
        data = r.json()

        if not isinstance(data, list):
            print("❌ Некоректна відповідь API MEXC:", data)
            return []

        # ✅ Автоматичне визначенн

