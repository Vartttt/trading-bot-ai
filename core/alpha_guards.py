# core/alpha_guards.py
"""
Alpha Guards — контролює торгові сесії, новинні вікна та funding rate.
Блокує торгівлю, якщо:
  - поза дозволеним торговим часом;
  - активне вікно новин;
  - funding rate занадто високий.
"""

import os, datetime
from notifier.telegram_notifier import send_message

TRADING_SESSION_UTC = os.getenv("TRADING_SESSION_UTC", "7-22")      # UTC години дозволеної сесії
NEWS_WINDOWS_UTC = os.getenv("NEWS_WINDOWS_UTC", "").strip()        # "12:25-13:15,18:55-19:20"
MAX_FUNDING_ABS = float(os.getenv("MAX_FUNDING_ABS", "0.003"))      # funding rate обмеження (0.3%)

def _now_utc():
    """Поточний час у UTC."""
    return datetime.datetime.utcnow()

def session_guard():
    """
    Дозволяє торгівлю лише у вказаний UTC-діапазон годин.
    """
    try:
        start, end = TRADING_SESSION_UTC.split("-")
        h = _now_utc().hour
        ok = int(start) <= h <= int(end)
        if not ok:
            send_message(f"🕒 Session guard: outside trading window ({TRADING_SESSION_UTC} UTC).")
        return ok
    except Exception:
        return True

def _parse_windows(s):
    out = []
    if not s:
        return out
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "-" in p:
            a, b = p.split("-")
            out.append((a.strip(), b.strip()))
    return out

def news_guard():
    """
    Блокує торгівлю під час запланованих “вікон новин”.
    """
    windows = _parse_windows(NEWS_WINDOWS_UTC)
    if not windows:
        return True
    now = _now_utc().strftime("%H:%M")
    for a, b in windows:
        if a <= now <= b:
            send_message(f"📰 News guard active ({a}-{b} UTC). Trading paused.")
            return False
    return True

def funding_guard(ex, symbol):
    """
    Блокує торгівлю, якщо funding rate занадто високий.
    """
    try:
        t = ex.fetch_ticker(symbol)
        rate = float(t.get("fundingRate") or 0.0)
        if abs(rate) > MAX_FUNDING_ABS:
            send_message(f"💸 Funding guard: {symbol} rate={rate:.4f} exceeds limit.")
            return False
    except Exception:
        pass
    return True
