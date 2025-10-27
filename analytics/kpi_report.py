import os, json, math, datetime
import pandas as pd

HIST = "state/trade_history.json"
TCA  = "state/tca_events.json"

def _load_json(path, default):
    if not os.path.exists(path): return default
    try: return json.load(open(path))
    except: return default

def _to_df(trades):
    if not trades: return pd.DataFrame(columns=["ts","result","symbol","side","entry","exit","reason"])
    df = pd.DataFrame(trades)
    if "ts" in df: df["dt"] = pd.to_datetime(df["ts"], unit="s")
    else: df["dt"] = pd.NaT
    return df

def max_drawdown(series):
    if len(series)==0: return 0.0
    cum = (1+series).cumprod()
    peak = cum.cummax()
    dd = (cum/peak) - 1.0
    return float(dd.min())

def sharpe_ratio(returns, periods_per_year=365):
    if len(returns)==0: return 0.0
    r = returns.mean() * periods_per_year
    vol = returns.std(ddof=1) * math.sqrt(periods_per_year)
    return 0.0 if vol==0 else float(r/vol)

def calmar_ratio(returns, periods_per_year=365):
    if len(returns)==0: return 0.0
    cum = (1+returns).cumprod()
    dd = max_drawdown(returns)
    cagr = (cum.iloc[-1])**(periods_per_year/len(returns)) - 1 if len(returns)>0 else 0
    return 0.0 if dd==0 else float(cagr/abs(dd))

def heatmap_by_hour(df):
    if df.empty or "dt" not in df: return pd.DataFrame()
    df["hour"] = df["dt"].dt.hour
    return df.groupby("hour")["result"].mean().reindex(range(24), fill_value=0.0).to_frame("avg_return")

def heatmap_by_weekday(df):
    if df.empty or "dt" not in df: return pd.DataFrame()
    df["weekday"] = df["dt"].dt.weekday
    return df.groupby("weekday")["result"].mean().reindex(range(7), fill_value=0.0).to_frame("avg_return")

def kpi_summary(days=30):
