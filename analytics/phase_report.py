import os, json, datetime, matplotlib.pyplot as plt

CACHE = "state/market_phase.json"
HISTORY = "state/market_phase_history.jsonl"

def current_phase():
    if not os.path.exists(CACHE): return None
    try: return json.load(open(CACHE))
    except: return None

def load_history(days=7):
    if not os.path.exists(HISTORY): return []
    cutoff = datetime.datetime.utcnow().timestamp() - days*86400
    out = []
    with open(HISTORY) as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("ts",0) >= cutoff: out.append(rec)
            except: pass
    return out

def plot_phase_timeline(days=7, out_path="charts/phase_timeline.png"):
    hist = load_history(days)
    if not hist: return None
    os.makedirs("charts", exist_ok=True)
    phase_map = {"BULL_TREND":3,"BEAR_TREND":-3,"RANGE_HIGH_VOL":1,"RANGE_LOW_VOL":0,"SPIKE_EVENT":2,"UNKNOWN":0}
    xs = [r["ts"] for r in hist]; ys = [phase_map.get(r.get("phase","UNKNOWN"),0) for r in hist]
    plt.figure(figsize=(9,4))
