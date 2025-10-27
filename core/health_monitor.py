"""
ExchangeHealthGuard: простий перевіряльник доступності біржі.
- ping через fetch_ticker BTC/USDT
- якщо помилка/затримка — повертає False (пауза циклу)
"""
import time

def exchange_ok(ex, symbol="BTC/USDT", timeout=2.5):
    t0 = time.time()
    try:
        ex.fetch_ticker(symbol)
        return (time.time() - t0) < timeout
    except Exception:
        return False
