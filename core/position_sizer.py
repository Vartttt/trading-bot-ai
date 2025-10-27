# core/position_sizer.py
# v8.4 Boosted — Adaptive Leverage + сумісний risk sizing з v8.3
# Сумісно з v8.3: compute_risk_fraction(strength, atr, price, base_risk)
# Нове:
#  - choose_leverage(): підбір плеча з урахуванням сили сигналу, фази, волатильності та «форми» (avg PnL)
#  - compute_position_notional(): зручна обгортка для розрахунку нотації угоди

import os
import math

# Базові параметри ризику (як у v8.3, але збережено сумісність ENV)
BASE_RISK        = float(os.getenv("BASE_RISK",        "0.015"))  # ~1.5% від equity
TARGET_VOL       = float(os.getenv("TARGET_VOL",       "0.015"))  # vol targeting per trade
SIZE_STRENGTH_MIN= int(os.getenv("SIZE_STRENGTH_MIN",  "70"))
SIZE_STRENGTH_MAX= int(os.getenv("SIZE_STRENGTH_MAX",  "95"))
SIZE_CAP_RISK    = float(os.getenv("SIZE_CAP_RISK",    "0.03"))   # верхня межа ризику на одну угоду

# Нові параметри для Adaptive Leverage
LEV_MIN   = int(os.getenv("LEV_MIN",   "3"))
LEV_MAX   = int(os.getenv("LEV_MAX",   "30"))          # консервативна «стеля» за замовчуванням
LEV_S_HIGH= int(os.getenv("LEV_S_HIGH","88"))          # сила сигналу для high
LEV_S_MED = int(os.getenv("LEV_S_MED", "75"))
LEV_DEF   = int(os.getenv("DEFAULT_LEVERAGE", "10"))   # як у v8.3

# Вплив режимів/форми
LEV_CAP_SPIKE = int(os.getenv("LEV_CAP_SPIKE", "12"))
LEV_CAP_RANGE = int(os.getenv("LEV_CAP_RANGE", "18"))
LEV_CAP_TREND = int(os.getenv("LEV_CAP_TREND", "30"))

PNL_BAD   = float(os.getenv("LEV_PNL_BAD",  "-0.30"))  # середній % за останні N угод — «погана форма»
PNL_GOOD  = float(os.getenv("LEV_PNL_GOOD",  "0.80"))  # «добра форма»
VOL_HOT   = float(os.getenv("LEV_VOL_HOT",   "0.025")) # 2.5% 1h std
VOL_COLD  = float(os.getenv("LEV_VOL_COLD",  "0.010")) # 1.0% 1h std

def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def scale_from_strength(strength: int) -> float:
    """
    Перетворює силу сигналу у множник розміру ризику (0.6..1.2).
    """
    smin, smax = SIZE_STRENGTH_MIN, SIZE_STRENGTH_MAX
    s = max(min(int(strength), smax), smin)
    if smax == smin:
        return 1.0
    x = (s - smin) / (smax - smin)
    return 0.6 + 0.6 * x  # [0.6..1.2]

def vol_target_multiplier(atr: float, price: float) -> float:
    """
    Нормує розмір відносно реалізованої волатильності (ATR/price) до TARGET_VOL.
    Обмеження х2 для уникнення «oversize».
    """
    realized_vol = max(atr / max(price, 1e-9), 1e-6)
    return min(TARGET_VOL / realized_vol, 2.0)

def compute_risk_fraction(strength: int, atr: float, price: float, base_risk: float = BASE_RISK) -> float:
    """
    Повертає частку equity для ризику на угоду (0..SIZE_CAP_RISK).
    Це сумісна з v8.3 функція, яку використовує open_signal_trade().
    """
    rf = base_risk * scale_from_strength(strength) * vol_target_multiplier(atr, price)
    return float(_clip(rf, 0.0, SIZE_CAP_RISK))

# -------- Adaptive Leverage (нове в v8.4) ------------------------------------

def _cap_by_phase(phase: str | None) -> int:
    if not phase:
        return LEV_CAP_TREND
    p = phase.upper()
    if p.startswith("SPIKE"):
        return LEV_CAP_SPIKE
    if p.startswith("RANGE"):
        return LEV_CAP_RANGE
    # BULL/BEAR — трендові
    return LEV_CAP_TREND

def _shape_factor(recent_pnl_avg_pct: float) -> float:
    """
    Форм-фактор (0.6..1.2): при серіях лосів — знижує плече, при гарній формі — підвищує.
    """
    if recent_pnl_avg_pct <= PNL_BAD:
        return 0.6
    if recent_pnl_avg_pct >= PNL_GOOD:
        return 1.2
    span = max(PNL_GOOD - PNL_BAD, 1e-9)
    t = (recent_pnl_avg_pct - PNL_BAD) / span  # 0..1
    return 0.8 + 0.4 * _clip(t, 0.0, 1.0)      # 0.8..1.2

def _vol_factor(vol_1h: float) -> float:
    """
    Вол-фактор (0.6..1.1): гарячий ринок → нижче плече; холодний → трохи вище бази.
    """
    if vol_1h >= VOL_HOT:
        return 0.6
    if vol_1h <= VOL_COLD:
        return 1.1
    t = (vol_1h - VOL_COLD) / max(VOL_HOT - VOL_COLD, 1e-9)
    return 1.1 - 0.4 * _clip(t, 0.0, 1.0)      # 1.1..0.7

def _base_from_strength(strength: int) -> int:
    if strength >= LEV_S_HIGH:
        return 20
    if strength >= LEV_S_MED:
        return 12
    return 7

def choose_leverage(
    strength: int,
    phase: str | None = None,      # глобальна фаза з market_phase
    vol_1h: float | None = None,   # 1h std returns (або проксі)
    recent_pnl_avg_pct: float = 0.0,  # середній % останніх N угод
    safety_drawdown_pct: float | None = None  # денна/поточна просадка (напр., ≤ -5%)
) -> int:
    """
    Повертає рекомендоване плече у межах [LEV_MIN..LEV_MAX] та обмежує за фазою.
    """
    try:
        base = _base_from_strength(int(strength))
        f_shape = _shape_factor(float(recent_pnl_avg_pct))
        f_vol = _vol_factor(float(vol_1h)) if (vol_1h is not None) else 1.0
        lev = base * f_shape * f_vol

        cap = _cap_by_phase(phase)
        lev = min(lev, cap)

        if safety_drawdown_pct is not None and safety_drawdown_pct <= -5.0:
            lev = max(LEV_MIN, min(lev * 0.6, cap))  # сильне зниження у drawdown

        return int(_clip(round(lev), LEV_MIN, LEV_MAX))
    except Exception:
        return LEV_DEF

# -------- Додатково: зручний калькулятор нотації позиції ----------------------

def compute_position_notional(
    equity_usdt: float,
    price: float,
    atr: float,
    strength: int,
    base_risk: float = BASE_RISK,
    cap_single_trade_risk: float = SIZE_CAP_RISK
) -> tuple[float, float]:
    """
    Розраховує (notional_usdt, risk_fraction) для угоди.
    Залишено для зручної інтеграції в open_signal_trade().
    """
    rf = compute_risk_fraction(strength, atr, price, base_risk=base_risk)
    rf = float(_clip(rf, 0.0, cap_single_trade_risk))
    notional = equity_usdt * rf
    return float(notional), float(rf)
