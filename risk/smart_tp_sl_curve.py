"""
Розумна генерація TP/SL від ATR і режиму ризику.
"""
def calc_smart_tp_sl(atr_value, signal_strength, risk_mode="NORMAL"):
    base_tp, base_sl = 2.0, 1.0
    if risk_mode == "AGGRESSIVE":
        base_tp, base_sl = 3.0, 1.0
    elif risk_mode == "DEFENSE":
        base_tp, base_sl = 1.2, 0.6
    if signal_strength >= 80:
        base_tp *= 1.2; base_sl *= 0.9
    elif signal_strength <= 60:
        base_tp *= 0.8; base_sl *= 1.1
    return round(base_tp * atr_value, 6), round(base_sl * atr_value, 6)
