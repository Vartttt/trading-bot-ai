import os
import ccxt

EXCHANGE = os.getenv("EXCHANGE", "mexc")

# вибір класу біржі з fallback
if hasattr(ccxt, EXCHANGE):
    cls = getattr(ccxt, EXCHANGE)
else:
    # fallback для старої назви або кастомних
    cls = getattr(ccxt, "mexc", None) or getattr(ccxt, "mexc3", None)


class ExchangeWrapper:
    def __init__(self):
        api_key = os.getenv("API_KEY")
        secret = os.getenv("API_SECRET")
        params = {
            "enableRateLimit": True,
            "apiKey": api_key,
            "secret": secret,
            "options": {"defaultType": DEFAULT_TYPE},
        }
        cls = getattr(ccxt, EXCHANGE) if hasattr(ccxt, EXCHANGE) else ccxt.mexc3
        self.ex = cls(params)

    def fetch_ticker(self, symbol):
        return self.ex.fetch_ticker(symbol)

    def fetch_order_book(self, symbol, limit=10):
        return self.ex.fetch_order_book(symbol, limit=limit)

    def fetch_positions(self):
        try:
            return self.ex.fetch_positions()
        except Exception:
            return []

    def market_order(self, symbol, side, amount):
        if DRY_RUN:
            return {"id":"dry_"+str(time.time()), "status":"filled", "symbol":symbol, "side":side, "amount":amount, "price": self.fetch_ticker(symbol).get("last")}
        return self.ex.create_order(symbol=symbol, type="market", side=side, amount=amount)

    def limit_order(self, symbol, side, amount, price, params=None):
        params = params or {}
        if DRY_RUN:
            return {"id":"dryl_"+str(time.time()), "status":"open", "symbol":symbol, "side":side, "amount":amount, "price":price, "postOnly":params.get("postOnly")}
        return self.ex.create_order(symbol=symbol, type="limit", side=side, amount=amount, price=price, params=params)

    def cancel_order(self, order_id, symbol):
        if DRY_RUN:
            return True
        return self.ex.cancel_order(order_id, symbol)

    def set_leverage(self, symbol, leverage=10):
        try:
            if hasattr(self.ex, "set_leverage"):
                return self.ex.set_leverage(leverage, symbol=symbol)
        except Exception:
            pass

    def fetch_balance(self):
        try:
            return self.ex.fetch_balance()
        except Exception:
            return {"total":{"USDT":0}}

