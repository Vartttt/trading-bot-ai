"""
Equity / PnL репорт із графіком — використовується для Telegram /report та /equity.
"""
import os, json, matplotlib.pyplot as plt

HIST_PATH = "state/trade_history.json"
OUT_PATH = "charts/equity_curve.png"

def load_pnls():
    if not os.path.exists(HIST_PATH): return []
    try: hist = json.load(open(HIST_PATH))
    except: return []
    return [h.get("result",0.0) for h in hist]

def plot_equity():
    pnls = load_pnls()
    if not pnls: return None
    eq = [1.0]
    for p in pnls:
        eq.append(eq[-1]*(1+p))
    xs = list(range(len(eq)))
    os.makedirs("charts", exist_ok=True)
    plt.figure(figsize=(8,4))
    plt.plot(xs, eq, linewidth=1.8)
    plt.title("Equity Curve")
    plt.xlabel("Trades")
    plt.ylabel("Equity (x)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PATH)
    plt.close()
    return OUT_PATH

def compute_kpis():
    pnls = load_pnls()
    if not pnls: return {"trades":0}
    wins = [p for p in pnls if p>0]; losses = [p for p in pnls if p<=0]
    gross = sum(pnls)
    winrate = len(wins)/len(pnls)*100
    avg = gross/len(pnls)
    max_dd = 0; peak = 1.0; eq = 1.0
    for p in pnls:
        eq *= (1+p)
        if eq>peak: peak=eq
        dd = 1 - eq/peak
        max_dd = max(max_dd, dd)
    return {
        "trades": len(pnls),
        "gross": gross,
        "winrate": winrate,
        "avg": avg,
        "max_dd": max_dd,
    }
