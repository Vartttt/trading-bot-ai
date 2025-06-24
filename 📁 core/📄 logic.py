from core.indicators import get_indicators

def generate_signal(symbol):
    indicators = get_indicators(symbol)
    strength = 0

    if indicators["ema_200"] == "above":
        strength += 20
    if indicators["macd"] == "bullish":
        strength += 20
    if indicators["rsi"] < 40:
        strength += 20
    if indicators["volume"] == "up":
        strength += 20
    if indicators["adx"] > 20:
        strength += 20

    signal = "none"
    if strength >= 50:
        signal = "buy" if indicators["trend"] == "up" else "sell"

    return {
        "signal": signal,
        "strength": strength,
        "trend": indicators["trend"]
    }
