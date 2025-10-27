"""
Щоденний ліміт збитків (Daily Risk Cap) + cooldown.
"""
import os, json, time, datetime
from notifier.telegram_notifier import send_message

STATE_FILE = "state/daily_risk.json"
MAX_DAILY_LOSS_PCT = float(os.getenv("MAX_DAILY_LOSS_PCT","0.02"))
DAILY_COOLDOWN_MIN = int(os.getenv("DAILY_COOLDOWN_MIN","180"))

def _load():
    if not os.path.exists(STATE_FILE):
        return {"day": None, "pnl": 0.0, "halted_at": None}
    try:
        return json.load(open(STATE_FILE))
    except:
        return {"day": None, "pnl": 0.0, "halted_at": None}

def _save(s):
    os.makedirs("state", exist_ok=True)
    json.dump(s, open(STATE_FILE,"w"), indent=2)

def _current_day_utc():
    return datetime.datetime.utcnow().strftime("%Y-%m-%d")

def daily_risk_ok(equity=1.0):
    s = _load()
    day = _current_day_utc()
    if s.get("day") != day:
        s = {"day": day, "pnl": 0.0, "halted_at": None}
        _save(s)
        return True
    if s.get("halted_at"):
        if time.time() - s["halted_at"] < DAILY_COOLDOWN_MIN * 60:
            return False
        else:
            s["halted_at"] = None
            _save(s)
            return True
    if equity <= 0: return True
    if s["pnl"] <= -MAX_DAILY_LOSS_PCT:
        if not s.get("halted_at"):
            s["halted_at"] = time.time()
            _save(s)
            send_message(f"⛔️ Daily loss cap hit ({s['pnl']*100:.2f}%). Cooling down for {DAILY_COOLDOWN_MIN} min.")
        return False
    return True

def report_trade_pnl(pnl_fraction):
    s = _load()
    day = _current_day_utc()
    if s.get("day") != day:
        s = {"day": day, "pnl": 0.0, "halted_at": None}
    s["pnl"] = round(s.get("pnl",0.0) + float(pnl_fraction), 6)
    _save(s)
    return s
