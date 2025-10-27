"""
Adaptive Risk Curve — динамічний ризик залежно від поточного PnL і серії.
"""
import json, os

STATE_FILE = "state/trade_stats.json"
DEFAULT_RISK = float(os.getenv("DEFAULT_RISK", "0.015"))
MAX_RISK = float(os.getenv("MAX_RISK", "0.025"))
MIN_RISK = float(os.getenv("MIN_RISK", "0.004"))

def load_stats():
    if not os.path.exists(STATE_FILE): 
        return {"pnl":0.0,"win_streak":0,"loss_streak":0}
    try:
        return json.load(open(STATE_FILE))
    except:
        return {"pnl":0.0,"win_streak":0,"loss_streak":0}

def save_stats(s): 
    os.makedirs("state", exist_ok=True)
    json.dump(s, open(STATE_FILE,"w"), indent=2)

def get_dynamic_risk():
    s = load_stats()
    pnl = s.get("pnl", 0.0)
    if pnl >= 0.05:
        return MAX_RISK, "AGGRESSIVE"
    if pnl <= -0.10:
        return MIN_RISK, "DEFENSE"
    if pnl <= -0.05:
        return 0.008, "LOW-RISK"
    return DEFAULT_RISK, "NORMAL"

def update_pnl(trade_pnl):
    s = load_stats()
    s["pnl"] = round(s.get("pnl",0)+trade_pnl,4)
    if trade_pnl>0:
        s["win_streak"]=s.get("win_streak",0)+1
        s["loss_streak"]=0
    else:
        s["loss_streak"]=s.get("loss_streak",0)+1
        s["win_streak"]=0
    save_stats(s)
    return s
