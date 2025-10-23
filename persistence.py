# persistence.py
import sqlite3
from datetime import datetime
from config import DB_PATH

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT,
        symbol TEXT,
        timeframe TEXT,
        signal TEXT,
        entry REAL,
        sl REAL,
        tp1 REAL,
        tp2 REAL,
        tp3 REAL,
        strength INTEGER,
        meta TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        signal_id INTEGER,
        closed_at TEXT,
        pnl REAL,
        reached_tp INTEGER,
        reached_sl INTEGER,
        FOREIGN KEY(signal_id) REFERENCES signals(id)
    )
    """)
    conn.commit()
    conn.close()

def save_signal(symbol, timeframe, signal_dict):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    ts = datetime.utcnow().isoformat()
    cur.execute("""
    INSERT INTO signals (ts, symbol, timeframe, signal, entry, sl, tp1, tp2, tp3, strength, meta)
    VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (
        ts,
        symbol,
        timeframe,
        signal_dict["signal"],
        signal_dict["entry"],
        signal_dict["sl"],
        signal_dict["tps"][0] if len(signal_dict["tps"])>0 else None,
        signal_dict["tps"][1] if len(signal_dict["tps"])>1 else None,
        signal_dict["tps"][2] if len(signal_dict["tps"])>2 else None,
        signal_dict["strength"],
        str(signal_dict.get("meta",""))
    ))
    conn.commit()
    conn.close()
