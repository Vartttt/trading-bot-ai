import os, math

BASE_RISK = float(os.getenv("BASE_RISK", "0.015"))
TARGET_VOL = float(os.getenv("TARGET_VOL", "0.015"))
SIZE_STRENGTH_MIN = int(os.getenv("SIZE_STRENGTH_MIN", "70"))
SIZE_STRENGTH_MAX = int(os.getenv("SIZE_STRENGTH_MAX", "95"))
SIZE_CAP_RISK = float(os.getenv("SIZE_CAP_RISK", "0.03"))

def position_size(balance_usdt: float, strength: int, atr: float, price: float):
    """Розрахунок USDT розміру позиції"""
    risk = BASE_RISK
    # Масштабування ризику від сили сигналу
    if strength > SIZE_STRENGTH_MIN:
        scale = min((strength - SIZE_STRENGTH_MIN) / (SIZE_STRENGTH_MAX - SIZE_STRENGTH_MIN), 1)
        risk *= 1 + 0.5 * scale
    # Волатильність таргетинг
    vol_adj = max(min(TARGET_VOL / max(atr/price, 1e-6), 2), 0.5)
    risk *= vol_adj
    risk = min(risk, SIZE_CAP_RISK)
    return balance_usdt * risk
