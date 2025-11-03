import requests
from datetime import datetime, timedelta

def news_guard():
    """
    Перевіряє останні важливі новини (через Cryptopanic).
    Якщо були великі події за останні 30 хв — блокує вхід.
    """
    url = "https://cryptopanic.com/api/v1/posts/?auth_token=YOUR_API_TOKEN&filter=important"

    try:
        data = requests.get(url, timeout=5).json()
        recent = [
            n for n in data.get("results", [])
            if "published_at" in n
        ]
        for n in recent:
            t = datetime.fromisoformat(n["published_at"].replace("Z", ""))
            if (datetime.utcnow() - t) < timedelta(minutes=30):
                return False  # новини були нещодавно
        return True
    except Exception:
        return True  # якщо API недоступне — не блокуємо
