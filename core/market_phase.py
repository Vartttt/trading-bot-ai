import os, json, time, pandas as pd, numpy as np

CACHE_PATH = os.getenv("PHASE_CACHE_PATH", "state/market_phase.json")

def compute_phase_from_df(df1h, df4h=None):
    if df1h is None or len(df1h) < 50:
        return {"phase":"UNKNOWN","regime":"range"}

    ret = df1h["close"].pct_change()
    vol = ret.rolling(24).std().iloc[-1]
    ema_s = df1h["close"].ewm(span=96).mean().iloc[-1]
    ema_f = df1h["close"].ewm(span=24).mean().iloc[-1]
    slope = (ema_f - ema_s)/ema_s
    if abs(slope) > 0.002:
        phase = "BULL_TREND" if slope>0 else "BEAR_TREND"
    else:
        phase = "RANGE_HIGH_VOL" if vol>0.015 else "RANGE_LOW_VOL"
    regime = "trend" if "TREND" in phase else "range"
    return {"phase":phase,"regime":regime}

def save_phase_cache(rec):
    os.makedirs("state", exist_ok=True)
    rec["ts"] = int(time.time())
    with open(CACHE_PATH,"w") as f:
        json.dump(rec,f,indent=2)

def load_phase_cache():
    if not os.path.exists(CACHE_PATH):
        return {}
    return json.load(open(CACHE_PATH))
