# db.py
import sqlite3
from contextlib import contextmanager
from config import SQLITE_PATH

DDL = """
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  event TEXT NOT NULL,
  symbol TEXT,
  side TEXT,
  price REAL,
  amount REAL,
  pnl REAL,
  extra TEXT
);
"""

@contextmanager
def conn():
    c = sqlite3.connect(SQLITE_PATH)
    try:
        yield c
        c.commit()
    finally:
        c.close()

def init_db():
    with conn() as c:
        c.execute(DDL)

def insert_event(ts, event, symbol="", side="", price=None, amount=None, pnl=None, extra=""):
    with conn() as c:
        c.execute(
            "INSERT INTO events(ts,event,symbol,side,price,amount,pnl,extra) VALUES(?,?,?,?,?,?,?,?)",
            (ts, event, symbol, side, price, amount, pnl, extra)
        )

from obs import jlog
jlog("INFO", "open_position", symbol=symbol, side=side, price=price)
jlog("ERROR", "telegram_fail", detail=str(e))
