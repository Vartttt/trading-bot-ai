import os, json, time, numpy as np, pandas as pd

CACHE_PATH = os.getenv("PHASE_CACHE_PATH", "state/market_phase.json")
HISTORY_PATH = os.getenv("PHASE_HISTORY_PATH", "state/market_phase_history.jsonl")

TH_VOL_LOW   = float(os.getenv("PHASE_TH_VOL_LOW",   0.010))
TH_VOL_HIGH  = float(os.getenv("PHASE_TH_VOL_HIGH",  0.025))
TH_TREND_MIN = float(os.getenv("PHASE_TH_TREND_MIN", 0.002))
TH_SPIKE_Z   = float(os.getenv("PHASE_TH_SPIKE_Z",   2.5))

def _ema(s: pd.Series, span: int): return s.ewm(span=span, adjust=False).mean()
def _zscore(s: pd.Series, window=48):
    return (s - s.rolling(window).mean()) / (s.rolling(window).std() + 1e-9)

def _advice_defaults():
    return {"risk_mult":1.0,"tp_mult":1.0,"sl_mult":1.0,"scan_bias":"balanced","trade_freq":"normal"}

def compute_phase_from_df(df1h: pd.DataFrame, df4h: pd.DataFrame | None = None) -> dict:
    x = df1h.rename(columns={"open":"o","high":"h","low":"l","close":"c","volume":"v"}).copy()
    if len(x) < 120: return {"phase":"UNKNOWN","regime":"range","scores":{},"advice":_advice_defaults()}
    ret = x["c"].pct_change()
    vol1h = float(ret.rolling(24).std().iloc[-1] or 0.0)
    ema_f, ema_s = _ema(x["c"], 24), _ema(x["c"], 96)
    trend_slope = float((ema_f.iloc[-1] - ema_s.iloc[-1]) / (ema_s.iloc[-1] + 1e-9))
    spike_score = float(_zscore(x["c"].pct_change().abs(), 48).iloc[-1])
    if spike_score >= TH_SPIKE_Z and vol1h >= TH_VOL_HIGH:
        phase, regime = "SPIKE_EVENT","spike"
    else:
        if vol1h <= TH_VOL_LOW:
            if abs(trend_slope) >= TH_TREND_MIN: phase,regime = ("BULL_TREND" if trend_slope>0 else "BEAR_TREND","trend")
            else: phase,regime = "RANGE_LOW_VOL","range"
        elif vol1h >= TH_VOL_HIGH:
            if abs(trend_slope) >= TH_TREND_MIN: phase,regime = ("BULL_TREND" if trend_slope>0 else "BEAR_TREND","trend")
            else: phase,regime = "RANGE_HIGH_VOL","range"
        else:
            if abs(trend_slope) >= TH_TREND_MIN: phase,regime = ("BULL_TREND" if trend_slope>0 else "BEAR_TREND","trend")
            else: phase,regime = "RANGE_HIGH_VOL","range"
    scores = {"vol1h":round(vol1h,5),"trend_slope":round(trend_slope,5),"spike_z":round(spike_score,2)}
    advice = _advice_defaults()
    return {"phase":phase,"regime":regime,"scores":scores,"advice":advice}

def save_phase_cache(record: dict):
    os.makedirs("state", exist_ok=True)
    record["ts"] = int(time.time())
    with open(CACHE_PATH,"w") as f: json.dump(record,f,indent=2)
    with open(HISTORY_PATH,"a") as f: f.write(json.dumps(record)+"\n")

def load_phase_cache() -> dict | None:
    if not os.path.exists(CACHE_PATH): return None
    try: return json.load(open(CACHE_PATH))
    except: return None
