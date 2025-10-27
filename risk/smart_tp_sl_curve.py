# risk/smart_tp_sl_curve.py
# v8.4 Boosted — Auto TP/SL tuning per symbol (зберігає статистику в state/tp_sl_stats.json)

import os, json, math
from typing import Dict, Any

STATS_FILE = "state/tp_sl_stats.json"

# Базові коефіцієнти (як у v8.3, але можуть бути адаптовані)
BASE_TP_MULT = float(os.getenv("TP_BASE_MULT", "2.0"))
BASE_SL_MULT = float(os.getenv("SL_BASE_MULT", "1.0"))

# Межі адаптації
TP_MIN_MULT = float(os.getenv("TP_MIN_MULT", "1.0"))
TP_MAX_MULT = float(os.getenv("TP_MAX_MULT", "3.5"))
SL_MIN_MULT = float(os.getenv("SL_MIN_MULT", "0.6"))
SL_MAX_MULT = float(os.getenv("SL_MAX_MULT", "1.6"))

# Параметри тюнінгу
HIT_RATE_WINDOW = int(os.getenv("TP_HIT_WINDOW", "200"))     # останніх N часткових закриттів
TP2_TARGET_RATE = float(os.getenv("TP2_TARGET_RATE", "0.40"))
SL_PENALTY = float(os.getenv("SL_PENALTY", "0.10"))          # підсилення SL при високій частоті SL
REGIME_TP_BONUS = float(os.getenv("REGIME_TP_BONUS", "0.15"))# бонус TP у трендових режимах

def _load() -> Dict[str, Any]:
    if not os.path.exists(STATS_FILE): 
        return {}
    try: 
        return json.load(open(STATS_FILE))
    except Exception:
        return {}

def _save(s: Dict[str, Any]):
    os.makedirs("state", exist_ok=True)
    json.dump(s, open(STATS_FILE, "w"), indent=2)

def update_tp_sl_stats(symbol: str, event: str, entry: float, exit: float, side: str):
    """
    Логує події 'TP1'/'TP2'/'SL' для Auto-tuning.
    """
    s = _load()
    sym = s.setdefault(symbol, {"tp1_hits": 0, "tp2_hits": 0, "sl_hits": 0, "N": 0})
    if event == "TP1": sym["tp1_hits"] += 1
    elif event == "TP2": sym["tp2_hits"] += 1
    elif event == "SL": sym["sl_hits"] += 1
    sym["N"] = min(sym.get("N", 0) + 1, HIT_RATE_WINDOW)
    s[symbol] = sym
    _save(s)

def _regime_bias(regime: str) -> float:
    """
    Повертає бонус до TP множника у тренді (bull/bear).
    """
    if not regime: return 0.0
    r = regime.upper()
    if r.startswith("BULL") or r.startswith("BEAR"):
        return REGIME_TP_BONUS
    return 0.0

def tuned_tp_sl(
    atr_value: float,
    signal_strength: int,
    symbol: str,
    regime: str = "UNKNOWN",
    base_tp: float | None = None,
    base_sl: float | None = None
) -> tuple[float, float, Dict[str, float]]:
    """
    Обчислює TP/SL офсети з урахуванням історичних хіт-рейтів по монеті.
    Повертає (tp_off, sl_off, stats_used).
    """
    s = _load()
    stats = s.get(symbol, {"tp1_hits": 0, "tp2_hits": 0, "sl_hits": 0, "N": 0})
    N = max(stats.get("N", 0), 1)
    tp2_rate = stats.get("tp2_hits", 0) / N
    sl_rate  = stats.get("sl_hits", 0)  / N

    tp_mult = BASE_TP_MULT if base_tp is None else base_tp / max(atr_value, 1e-9)
    sl_mult = BASE_SL_MULT if base_sl is None else base_sl / max(atr_value, 1e-9)

    # якщо TP2 рідко досягається — трохи зменшуємо TP
    if tp2_rate < TP2_TARGET_RATE:
        # лінійний зсув до -15%
        tp_mult *= (1.0 - 0.15 * (1.0 - tp2_rate / max(TP2_TARGET_RATE, 1e-9)))

    # якщо SL частий — підсилюємо SL (робимо ближче, щоб різати втрати швидше)
    if sl_rate > 0.45:
        sl_mult *= (1.0 - SL_PENALTY * (sl_rate - 0.45) / 0.55)  # до ~10% ближче

    # бонус у тренді
    tp_mult *= (1.0 + _regime_bias(regime))

    # сила сигналу: для сильних сетапів дозволяємо трохи ширший TP
    if signal_strength >= 85:
        tp_mult *= 1.12; sl_mult *= 0.95
    elif signal_strength <= 65:
        tp_mult *= 0.92; sl_mult *= 1.05

    # клэмпи
    tp_mult = max(TP_MIN_MULT, min(TP_MAX_MULT, tp_mult))
    sl_mult = max(SL_MIN_MULT, min(SL_MAX_MULT, sl_mult))

    tp_off = round(tp_mult * atr_value, 6)
    sl_off = round(sl_mult * atr_value, 6)
    used = {
        "tp2_rate": round(tp2_rate, 3),
        "sl_rate": round(sl_rate, 3),
        "tp_mult": round(tp_mult, 3),
        "sl_mult": round(sl_mult, 3),
        "N": int(N)
    }
    return tp_off, sl_off, used

# Збережено сумісний API з v8.3 (простий виклик)
def calc_smart_tp_sl(atr_value, signal_strength, risk_mode="NORMAL"):
    base_tp, base_sl = 2.0, 1.0
    if risk_mode == "AGGRESSIVE": base_tp, base_sl = 3.0, 1.0
    elif risk_mode == "DEFENSE": base_tp, base_sl = 1.2, 0.6
    if signal_strength >= 80:
        base_tp *= 1.2; base_sl *= 0.9
    elif signal_strength <= 60:
        base_tp *= 0.8; base_sl *= 1.1
    return round(base_tp * atr_value, 6), round(base_sl * atr_value, 6)

