import os, datetime, ccxt
from notifier.telegram_notifier import send_message

def _parse_ranges(rng):
    if not rng: return []
    out = []
    for seg in rng.split(","):
        if "-" in seg:
            a,b = seg.split("-")
            out.append((a.strip(), b.strip()))
    return out

def check_session_guard():
    rng = os.getenv("TRADING_SESSION_UTC", "")
    if not rng: return True
    now_h = datetime.datetime.utcnow().hour
    for seg in rng.split(","):
        if "-" not in seg: continue
        a,b = map(int, seg.split("-"))
        if a <= now_h <= b:
            return True
    send_message(f"⏰ За межами торгової сесії ({rng})")
    return False

def check_news_guard():
    windows = os.getenv("NEWS_WINDOWS_UTC", "")
    if not windows: return True
    now = datetime.datetime.utcnow().time()
    for seg in _parse_ranges(windows):
        try:
            t1 = datetime.datetime.strptime(seg[0], "%H:%M").time()
            t2 = datetime.datetime.strptime(seg[1], "%H:%M").time()
            if t1 <= now <= t2:
                send_message(f"📰 News window active {seg}")
                return False
        except:
            continue
    return True

def check_funding_guard(symbol="BTC/USDT"):
    MAX_FUNDING_ABS = float(os.getenv("MAX_FUNDING_ABS", "0.003"))
    ex = ccxt.mexc3()
    try:
        funding = ex.fetch_funding_rate(symbol)
        rate = abs(funding.get("fundingRate", 0))
        if rate >= MAX_FUNDING_ABS:
            send_message(f"💰 Funding guard активний {rate:.4%}")
            return False
    except:
        pass
    return True

def check_guards():
    return check_session_guard() and check_news_guard() and check_funding_guard()
