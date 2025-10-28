import os
import time
import ccxt

# --- Конфігурація ---
EXCHANGE = os.getenv("EXCHANGE", "mexc")
DEFAULT_TYPE = os.getenv("DEFAULT_TYPE", "swap")   # або "spot"
DRY_RUN = os.getenv("DRY_RUN", "True").lower() == "true"

# вибір класу біржі з fallback
if hasattr(ccxt, EXCHANGE):
    cls = getattr(ccxt, EXCHANGE)
else:
    cls = getattr(ccxt, "mexc", None) or getattr(ccxt, "mexc3", None)

if cls is None:
    raise ValueError(f"❌ Exchange '{EXCHANGE}' not found in ccxt module")


class ExchangeWrapper:
    def __init__(self):
        api_key = os.getenv("MEXC_API_KEY") or os.getenv("API_KEY")
        secret = os.getenv("MEXC_SECRET_KEY") or os.getenv("API_SECRET")

        params = {
            "enableRateLimit": True,
            "apiKey": api_key,
            "secret": secret,
            "options": {"defaultType": DEFAULT_TYPE},
        }

        self.ex = cls(params)
        try:
            self.ex.load_markets()
            print(f"✅ Connected to exchange: {EXCHANGE.upper()} ({DEFAULT_TYPE})")
        except Exception as e:
            print(f"⚠️ Exchange init error: {e}")

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
            return {
                "id": "dry_" + str(time.time()),
                "status": "filled",
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "price": self.fetch_ticker(symbol).get("last"),
            }
        return self.ex.create_order(symbol=symbol, type="market", side=side, amount=amount)

    def limit_order(self, symbol, side, amount, price, params=None):
        params = params or {}
        if DRY_RUN:
            return {
                "id": "dryl_" + str(time.time()),
                "status": "open",
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "price": price,
                "postOnly": params.get("postOnly"),
            }
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
        # якщо DRY_RUN=True → використовуємо фіктивний баланс
        if DRY_RUN:
            sim_balance = float(os.getenv("SIM_BALANCE", "1000"))  # значення з .env
            return {"total": {"USDT": sim_balance}, "free": {"USDT": sim_balance}}
        # якщо реальний режим → отримуємо з біржі
        return self.ex.fetch_balance()
    except Exception:
        # якщо сталася помилка — повертаємо нульовий баланс
        return {"total": {"USDT": 0}, "free": {"USDT": 0}}



