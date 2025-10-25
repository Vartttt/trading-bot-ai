import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import os

MODEL_PATH = "ai_model.pkl"

class AIMarketModel:
    def __init__(self):
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
        else:
            self.model = GradientBoostingClassifier(n_estimators=100, max_depth=3)

    def predict_signal(self, X: np.ndarray) -> dict:
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        prob = self.model.predict_proba(X)[0]
        return {"up": prob[1], "down": prob[0]}

    def fit_online(self, X, y):
        # Псевдо оновлення: refit на об'єднаних даних (спрощений онлайн)
        try:
            self.model.fit(X, y)
            joblib.dump(self.model, MODEL_PATH)
        except Exception as e:
            print(f"[AI] fit_online error: {e}")
