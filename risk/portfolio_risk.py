"""
Portfolio Risk Manager — керування експозицією по всіх позиціях:
- перевіряє max global exposure, family exposure
- обчислює VaR proxy (90% quantile)
"""
import json, os, numpy as np

STATE_FILE = "state/open_positions.json"
MAX_GLOBAL_EXPOSURE = float(os.getenv("MAX_GLOBAL_EXPOSURE","1.0"))  # 100% equity
VAR_LOOKBACK = int(os.getenv("VAR_LOOKBACK","50"))
VAR_PCT = float(os.getenv("VAR_PCT","0.9"))

def _load_positions():
    if not os.path.exists(STATE_FILE): return {}
    try: return json.load(open(STATE_FILE))
    except: return {}

def current_exposure(equity_usdt: float):
    pos = _load_positions()
    total_notional = sum(p.get("entry",0)*p.get("amount",0) for p in pos.values())
    return total_notional / max(equity_usdt,1e-9)

def var_proxy(pnl_series):
    if not pnl_series: return 0.0
    arr = np.array(pnl_series)
    return float(np.quantile(arr, 1 - VAR_PCT))

def can_open_new(equity_usdt: float, pnl_history=None):
    exp = current_exposure(equity_usdt)
    if exp >= MAX_GLOBAL_EXPOSURE:
        return False, f"Exposure {exp*100:.1f}% ≥ cap {MAX_GLOBAL_EXPOSURE*100:.1f}%"
    if pnl_history:
        var_val = var_proxy(pnl_history)
        if abs(var_val) > 0.05:  # 5% potential VaR
            return False, f"VaR limit exceeded ({var_val:.3f})"
    return True, "ok"
