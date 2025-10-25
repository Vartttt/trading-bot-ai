# auto_trainer.py
import pandas as pd, sqlite3, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import joblib
from config import SQLITE_PATH
MODEL_PATH = "ai_model.pkl"

def retrain_ai():
    con = sqlite3.connect(SQLITE_PATH)
    df = pd.read_sql("SELECT ts, event, symbol, side, price, pnl FROM events", con)
    con.close()
    if len(df) < 100:
        return "Too few samples"
    df["result"] = (df["pnl"] > 0).astype(int)
    # Прості фічі: норма прибутку, напрямок, таймінг
    X = np.column_stack([
        np.sign(df["pnl"]),
        np.log1p(abs(df["pnl"])),
        df["result"].rolling(5).mean().fillna(0)
    ])
    y = df["result"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    model = GradientBoostingClassifier(n_estimators=150, max_depth=3)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    joblib.dump(model, MODEL_PATH)
    return f"AI retrained. Accuracy={acc:.2%}"
