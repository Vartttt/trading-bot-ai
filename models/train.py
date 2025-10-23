# models/train.py
import sqlite3
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from config import DB_PATH

MODEL_PATH = "models/signal_model.joblib"

def load_training_data():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, entry, sl, tp1, tp2, tp3, strength FROM signals")
    rows = cur.fetchall()
    conn.close()
    # Placeholder: build X,y artificially (in real case need PnL labels)
    X = []
    y = []
    for r in rows:
        # naive features: [entry, sl, tp1, strength]
        _, entry, sl, tp1, tp2, tp3, strength = r
        if entry is None:
            continue
        X.append([entry or 0, sl or 0, tp1 or 0, strength or 0])
        # dummy label: if tp1>entry -> 1 else 0 (not real)
        y.append(1 if (tp1 and tp1>entry) else 0)
    if len(X)==0:
        return None,None
    return np.array(X), np.array(y)

def train_and_save():
    X,y = load_training_data()
    if X is None:
        print("No training data")
        return
    clf = LogisticRegression(max_iter=200)
    clf.fit(X,y)
    joblib.dump(clf, MODEL_PATH)
    print("Model trained and saved.")
