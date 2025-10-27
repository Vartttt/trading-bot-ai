import os, json

FEE_RATE_BPS = float(os.getenv("FEE_RATE_BPS","7"))  # taker default

def estimate_fee(notional):
    return abs(notional) * (FEE_RATE_BPS / 10000.0)

def log_tca(event_path, record):
    os.makedirs(os.path.dirname(event_path), exist_ok=True)
    data = []
    if os.path.exists(event_path):
        try: data = json.load(open(event_path))
        except Exception: data = []
    data.append(record)
    json.dump(data, open(event_path,"w"), indent=2)
