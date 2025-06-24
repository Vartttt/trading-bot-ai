import os, time, hmac, hashlib, requests

API_KEY = os.getenv("MEXC_API_KEY")
SECRET_KEY = os.getenv("MEXC_SECRET_KEY")
BASE_URL = "https://contract.mexc.com"

def sign(params):
    sorted_params = sorted(params.items())
    query = '&'.join(f"{k}={v}" for k, v in sorted_params)
    return hmac.new(SECRET_KEY.encode(), query.encode(), hashlib.sha256).hexdigest()

def get_price(symbol):
    r = requests.get(f"{BASE_URL}/api/v1/contract/ticker", params={"symbol": symbol})
    return float(r.json()["data"]["lastPrice"])

def get_balance():
    params = {
        "api_key": API_KEY,
        "req_time": int(time.time() * 1000)
    }
    params["sign"] = sign(params)
    r = requests.get(f"{BASE_URL}/api/v1/private/account/assets", params=params)
    data = r.json()["data"]
    for d in data:
        if d["currency"] == "USDT":
            return float(d["availableBalance"])
    return 0

def open_position(symbol, side, usdt_amount, leverage):
    price = get_price(symbol)
    volume = round(usdt_amount / price, 3)
    side_code = 1 if side == "buy" else 2

    params = {
        "api_key": API_KEY,
        "req_time": int(time.time() * 1000),
        "symbol": symbol,
        "price": price,
        "vol": volume,
        "side": side_code,
        "type": 1,
        "open_type": 1,
        "leverage": leverage,
        "position_id": 0
    }
    params["sign"] = sign(params)
    r = requests.post(f"{BASE_URL}/api/v1/private/order/submit", json=params)
    return r.json()

def close_position(symbol):
    params = {
        "api_key": API_KEY,
        "req_time": int(time.time() * 1000),
        "symbol": symbol,
        "type": 1
    }
    params["sign"] = sign(params)
    r = requests.post(f"{BASE_URL}/api/v1/private/position/close", json=params)
    return r.json()
