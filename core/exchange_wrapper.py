import os, ccxt

EXCHANGE = os.getenv("EXCHANGE", "mexc3")

class ExchangeWrapper:
    def __init__(self):
        cls = getattr(ccxt, EXCHANGE)
        self.ex = cls({
            "enableRateLimit": True,
            "apiKey": os.getenv("API_KEY"),
            "secret": os.getenv("API_SECRET"),
            "options": {"defaultType": os.getenv("DEFAULT_TYPE","swap")}
        })

    def fetch_ticker(self, symbol):
        return self.ex.fetch_ticker(symbol)

    def fetch_balance(self):
        try:
            return self.ex.fetch_balance()
        except Exception:
            return {}

    def fetch_positions(self):
        try:
            return self.ex.fetch_positions()
        except Exception:
            return []

    def market_order(self, symbol, side, amount):
        return self.ex.create_market_order(symbol, side, amount)
