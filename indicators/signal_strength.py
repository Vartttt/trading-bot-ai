# indicators/signal_strength.py
# v8.4 Boosted — Phase Bias + детальні компоненти для аналітики/авто-оптимізації
# Сумісний з v8.2/v8.3: compute_signal_strength(data, weights=None)
# Нове:
#  - phase-aware bias: підсилює LONG у bull-фазі, SHORT у bear-фазі, нейтрально у range
#  - повертає деталі компонентів для TCA/оптимізатора (factors), + phase_mult

import os
import numpy as np

# Базові ваги (як у v8.3) — оптимізуються через optimizer/smart_auto_optimizer.py
DEFAULT_WEIGHTS = {"rsi": 0.20, "macd": 0.25, "ema": 0.25, "volume": 0.15, "volatility": 0.15}

# Керування фазовим підсиленням через ENV
PHASE_BIAS_LONG_BULL   = float(os.getenv("PHASE_BIAS_LONG_BULL",   "1.15"))  # LONG у bull
PHASE_BIAS_SHORT_BEAR  = float(os.getenv("PHASE_BIAS_SHORT_BEAR",  "1.15"))  # SHORT у bear
PHASE_BIAS_RANGE_NEUTR = float(os.getenv("PHASE_BIAS_RANGE_NEUTR", "1.00"))  # range — нейтрально
PHASE_BIAS_MAX         = float(os.getenv("PHASE_BIAS_MAX",         "1.25"))  # глобальна «стеля»
PHASE_BIAS_MIN         = float(os.getenv("PHASE_BIAS_MIN",         "0.80"))  # глобальна «підлога»

def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))

def _phase_multiplier(phase: str | None, direction: str) -> float:
    """
    phase: рядок з core/market_phase (наприклад, 'BULL_TREND', 'BEAR_TREND', 'RANGE_HIGH_VOL', 'SPIKE_EVENT', 'UNKNOWN')
    direction: 'LONG' або 'SHORT' (визначено з ema_fast vs ema_slow)
    """
    if not phase or phase == "UNKNOWN":
        return 1.0
    p = phase.upper()
    if p.startswith("BULL"):
        return _clip(PHASE_BIAS_LONG_BULL if direction == "LONG" else 1.0)
    if p.startswith("BEAR"):
        return _clip(PHASE_BIAS_SHORT_BEAR if direction == "SHORT" else 1.0)
    if p.startswith("RANGE"):
        return _clip(PHASE_BIAS_RANGE_NEUTR)
    if p.startswith("SPIKE"):
        # у спайку не підсилюємо (ризик), залишаємо 1.0
        return 1.0
    return 1.0

def _clip(x: float) -> float:
    return float(np.clip(x, PHASE_BIAS_MIN, PHASE_BIAS_MAX))

def compute_signal_strength(data: dict, weights: dict | None = None, phase: str | None = None) -> dict:
    """
    Вхід:
      data: {
        'rsi', 'macd', 'macd_signal', 'ema_fast', 'ema_slow',
        'volume', 'avg_volume', 'atr', 'momentum'
      }
      weights: ваги компонентів (або DEFAULT_WEIGHTS)
      phase: глобальна фаза ринку ('BULL_TREND', 'BEAR_TREND', 'RANGE_*', 'SPIKE_EVENT') — опційно

    Вихід:
      {
        'strength': int[0..100],       # фінальна сила з урахуванням phase bias (clamped)
        'direction': 'LONG'|'SHORT',   # базово за ema_fast > ema_slow
        'raw_strength': float[0..100], # сила без фазового підсилення
        'phase_mult': float,           # застосований фазовий множник
        'factors': {                   # нормалізовані фактори для журналу/оптимізації
            'rsi','macd','ema','volume','volatility'
        }
      }
    """
    weights = weights or DEFAULT_WEIGHTS

    rsi = float(data.get("rsi", 50.0))
    # RSI score: середина — слабше, крайні зони — сильніше
    if 45 <= rsi <= 55:
        rsi_score = 0.40
    elif rsi < 30 or rsi > 70:
        rsi_score = 1.00
    else:
        rsi_score = 0.70

    macd = float(data.get("macd", 0.0))
    macd_signal = float(data.get("macd_signal", 0.0))
    macd_score = _clip01(abs(macd - macd_signal) / 0.5)

    ema_fast = float(data.get("ema_fast", 1.0))
    ema_slow = float(data.get("ema_slow", 1.0))
    ema_score = _clip01(abs((ema_fast - ema_slow) / max(ema_slow, 1e-9)))

    vol = float(data.get("volume", 1.0))
    avg_vol = float(data.get("avg_volume", vol))
    vol_score = 1.0 if (avg_vol > 0 and (vol / avg_vol) >= 1.0) else 0.50

    atr = float(data.get("atr", 0.01))
    mom = float(data.get("momentum", 0.0))
    volat_score = _clip01(abs(mom) / max(atr, 1e-9))

    # агрегована сирова сила
    raw_total = (
        weights.get("rsi", 0.20)        * rsi_score +
        weights.get("macd", 0.25)       * macd_score +
        weights.get("ema", 0.25)        * ema_score +
        weights.get("volume", 0.15)     * vol_score +
        weights.get("volatility", 0.15) * volat_score
    )
    raw_strength = float(np.clip(raw_total * 100.0, 0.0, 100.0))

    direction = "LONG" if ema_fast > ema_slow else "SHORT"

    # фазовий множник (нова логіка v8.4)
    phase_mult = _phase_multiplier(phase, direction)
    strength = float(np.clip(raw_strength * phase_mult, 0.0, 100.0))

    return {
        "strength": int(round(strength)),
        "direction": direction,
        "raw_strength": round(raw_strength, 2),
        "phase_mult": round(phase_mult, 3),
        "factors": {
            "rsi": round(rsi_score, 4),
            "macd": round(macd_score, 4),
            "ema": round(ema_score, 4),
            "volume": round(vol_score, 4),
            "volatility": round(volat_score, 4),
        },
    }
