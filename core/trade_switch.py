import os, json
SW_FILE = "state/trade_switch.json"

def _load():
    if not os.path.exists(SW_FILE): return {"enabled": False}
    try: return json.load(open(SW_FILE))
    except: return {"enabled": False}

def is_trading_enabled():
    return bool(_load().get("enabled", False))

def set_trading(flag: bool):
    st = _load(); st["enabled"] = bool(flag)
    os.makedirs("state", exist_ok=True)
    json.dump(st, open(SW_FILE,"w"), indent=2)
    return st["enabled"]
