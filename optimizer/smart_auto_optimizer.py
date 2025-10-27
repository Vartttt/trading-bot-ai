# optimizer/smart_auto_optimizer.py
# v8.4 Boosted — ваги індикаторів через Sharpe-подібну метрику і експоненційну реценсію

import json, os, math, numpy as np
from typing import Dict, Any, List

HISTORY_FILE = "state/trade_history.json"
WEIGHTS_FILE = "state/signal_weights.json"

DEFAULT_WEIGHTS = {"rsi":0.20,"macd":0.25,"ema":0.25,"volume":0.15,"volatility":0.15}

# параметри реценсії (останні новішим — більша вага)
RECENCY_HALF_LIFE = int(os.getenv("OPT_RECENCY_HALF_LIFE", "150"))  # у трейдах

def _load_json(path, default):
    if not os.path.exists(path): return default
    try: return json.load(open(path))
    except: return default

def save_weights(weights: Dict[str, float]):
    os.makedirs("state", exist_ok=True)
    json.dump(weights, open(WEIGHTS_FILE,"w"), indent=2)

def load_weights() -> Dict[str, float]:
    w = _load_json(WEIGHTS_FILE, None)
    return w if w else DEFAULT_WEIGHTS.copy()

def load_trade_history() -> List[Dict[str, Any]]:
    return _load_json(HISTORY_FILE, [])

def _recency_weights(n: int) -> np.ndarray:
    """
    Експоненційне згасання ваг по індексу трейду (новіші важать більше).
    """
    if n <= 0: 
        return np.array([])
    # λ з half-life
    lam = math.log(2) / max(RECENCY_HALF_LIFE, 1)
    idx = np.arange(n)
    w = np.exp(lam * idx)  # зростає до новіших
    w /= w.sum()
    return w

def optimize_weights(window: int = 400) -> Dict[str, float]:
    hist = load_trade_history()
    if len(hist) < 20:
        return load_weights()

    data = hist[-window:]
    inds = list(DEFAULT_WEIGHTS.keys())
    contrib = {k: [] for k in inds}

    # готуємо матрицю факторів і вектор результатів
    pnl = []
    for tr in data:
        pnl.append(float(tr.get("result", 0.0)))
        f = (tr.get("factors") or {})
        for k in inds:
            contrib[k].append(float(f.get(k, 0.0)))

    pnl = np.array(pnl)
    n = len(pnl)
    R = _recency_weights(n)  # w_i

    # “Sharpe-подібна” оцінка внеску індикатора: E[f*k * pnl] / std
    score = {}
    total_score = 0.0
    for k in inds:
        x = np.array(contrib[k])  # нормалізовані фактори з signal_strength
        # зсув у [-1..1] навколо 0.5, щоб уникнути тривіальної кореляції
        x_c = (x - 0.5) * 2.0
        # зважене очікування та std
        mu = float(np.sum(R * (x_c * pnl)))
        sd = float(np.sqrt(np.sum(R * (x_c - np.sum(R * x_c))**2) + 1e-9))
        s = mu / (sd + 1e-9)
        s = max(s, 0.0)   # негативні обнуляємо
        score[k] = s
        total_score += s

    if total_score <= 0:
        return load_weights()

    new_w = {k: round(score[k] / total_score, 3) for k in inds}
    save_weights(new_w)
    return new_w
