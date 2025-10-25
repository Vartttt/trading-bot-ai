# http_client.py
import requests, time
from typing import Optional

DEFAULT_TIMEOUT = 10
RETRY_STATUS = {429, 500, 502, 503, 504}

def post_json(url: str, json: dict, timeout: int = DEFAULT_TIMEOUT, retries: int = 3, backoff: float = 0.8):
    last_err = None
    for i in range(retries):
        try:
            r = requests.post(url, json=json, timeout=timeout)
            if r.status_code in RETRY_STATUS:
                last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            else:
                return r
        except Exception as e:
            last_err = e
        time.sleep(backoff * (2 ** i))
    raise last_err or RuntimeError("post_json failed")
