from feature_engineering import build_features
from ai_model import AIMarketModel
from indicators import add_indicators, df_from_ohlcv, latest_row

ai_model = AIMarketModel()

def compute_ai_signal(df):
    X = build_features(df)
    probs = ai_model.predict_signal(X)
    up_p, down_p = probs["up"], probs["down"]
    if up_p > 0.65:
        return "LONG", up_p
    elif down_p > 0.65:
        return "SHORT", down_p
    else:
        return "FLAT", max(up_p, down_p)

def hybrid_signal(df):
    side_ai, conf_ai = compute_ai_signal(df)
    row = latest_row(df)
    side_ta = "LONG" if row["ema_fast"] > row["ema_slow"] else "SHORT"
    # узгодження напрямків
    if side_ai == side_ta:
        conf = min(1.0, conf_ai + 0.15)
    else:
        conf = max(0.0, conf_ai - 0.10)
    return side_ai if side_ai != "FLAT" else side_ta, conf

