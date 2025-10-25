import numpy as np
from ai_model import AIMarketModel
from feature_engineering import build_features

ai_model = AIMarketModel()

def record_trade_result(df, side, result):
    """
    result: 1 якщо TP, 0 якщо SL
    """
    X = build_features(df)
    y = np.array([result])
    ai_model.fit_online(X.reshape(1,-1), y)

