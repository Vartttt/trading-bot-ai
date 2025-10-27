import os, json, time
from notifier.telegram_notifier import send_message

STATE_FILE = "state/daily_risk.json"
MAX_DAILY_LOSS_PCT = float(os.getenv("MAX_DAILY_LOSS_PCT", "0.02"))
DAILY_COOLDOWN_MIN = int(os.getenv("DAILY_COOLDOWN_MIN", "180"))

def _load_state():
    if not os.path.exists(STATE_FILE): 
        return {"last_pnl": 0, "date": "", "cooldown_until": 0}
    try:
        return json.load(open(STATE_FILE))
    except:
        return {"last_pnl": 0, "date": "", "cooldown_until": 0}

def _save_state(s): json.dump(s, open(STATE_FILE, "w"), indent=2)

def check_daily_loss_limit():
    s = _load_state()
    today = time.strftime("%Y-%m-%d")
    now = time.time()

    if s.get("date") != today:
        s = {"date": today, "last_pnl": 0, "cooldown_until": 0}
        _save_state(s)
        return False

    # cooldown активний
    if now < s.get("cooldown_until", 0):
        return True

    pnl = s.get("last_pnl", 0)
    if pnl <= -MAX_DAILY_LOSS_PCT:
        s["cooldown_until"] = now + DAILY_COOLDOWN_MIN * 60
        _save_state(s)
        send_message(f"🛑 Daily loss limit reached ({pnl*100:.2f}%), пауза {DAILY_COOLDOWN_MIN} хв.")
        return True
    return False

def add_trade_pnl(pnl_delta: float):
    s = _load_state()
    today = time.strftime("%Y-%m-%d")
    if s.get("date") != today:
        s = {"date": today, "last_pnl": pnl_delta, "cooldown_until": 0}
    else:
        s["last_pnl"] += pnl_delta
    _save_state(s)
