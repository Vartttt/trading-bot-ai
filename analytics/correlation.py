import os, json, pandas as pd
from core.data_feed import get_ohlcv

OPEN_POS_FILE = "state/open_positions.json"
CORR_LOOKBACK_HOURS = int(os.getenv("CORR_LOOKBACK_HOURS", "24"))

def load_active_symbols():
    if not os.path.exists(OPEN_POS_FILE): return []
    try:
        pos = json.load(open(OPEN_POS_FILE))
        return list(pos.keys())
    except Exception:
        return []

def _returns(symbol, hours=CORR_LOOKBACK_HOURS):
    df = get_ohlcv(symbol, timeframe="1h", limit=hours+1)
    if df is None or df.empty: return None
    ret = df["close"].pct_change().dropna()
    if ret.empty: return None
    return ret

def recent_correlation(sym_a, sym_b, hours=CORR_LOOKBACK_HOURS):
    ra = _returns(sym_a, hours); rb = _returns(sym_b, hours)
    if ra is None or rb is None: return 0.0
    n = min(len(ra), len(rb))
    if n < 5: return 0.0
    ca = ra.tail(n).reset_index(drop=True)
    cb = rb.tail(n).reset_index(drop=True)
    return float(ca.corr(cb))
