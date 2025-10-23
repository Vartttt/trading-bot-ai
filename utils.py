# utils.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

def plot_signal_chart(df, symbol, entry=None, sl=None, tps=None, out_path="chart.png"):
    plt.figure(figsize=(10,5))
    plt.plot(df.index, df["close"], label="close")
    if entry is not None:
        plt.axhline(entry, color="green", linestyle="--", label=f"entry {entry:.6g}")
    if sl is not None:
        plt.axhline(sl, color="red", linestyle="--", label=f"sl {sl:.6g}")
    if tps:
        colors = ["orange", "purple", "brown"]
        for i,tp in enumerate(tps):
            plt.axhline(tp, color=colors[i%len(colors)], linestyle=":", label=f"tp{i+1} {tp:.6g}")
    plt.title(symbol)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path
