"""
Exposure Guard (повернуто з v7, спрощено):
- MAX_GLOBAL_POS
- MAX_FAMILY_POS (BTCFAM, SOLFAM, TONFAM, AI, DEFI)
"""
import os, json
from threading import Lock

STATE_FILE = "state/open_positions.json"
MAX_GLOBAL_POS = int(os.getenv("MAX_GLOBAL_POS", "6"))
MAX_FAMILY_POS = int(os.getenv("MAX_FAMILY_POS", "1"))

FAMILIES = {
    "BTCFAM": ["BTC","ETH","BNB","LTC"],
    "SOLFAM": ["SOL","AVAX","ADA","NEAR"],
    "TONFAM": ["TON","DOGE","PEPE","SHIB"],
    "AI": ["FET","RNDR","TAO","WLD"],
    "DEFI": ["UNI","AAVE","CAKE","MKR"],
}
_lock = Lock()

def _load_positions():
    if not os.path.exists(STATE_FILE): return {}
    try: return json.load(open(STATE_FILE))
    except: return {}

def get_family(symbol: str) -> str:
    base = symbol.split("/")[0].upper()
    for fam, members in FAMILIES.items():
        if base in members:
            return fam
    return base

def check_exposure(symbol: str) -> tuple[bool, str]:
    with _lock:
        pos = _load_positions()
        if len(pos) >= MAX_GLOBAL_POS:
            return False, "max_global_positions"
        fam = get_family(symbol)
        fam_count = sum(1 for _,p in pos.items() if get_family(p["symbol"]) == fam)
        if fam_count >= MAX_FAMILY_POS:
            return False, f"max_family_positions({fam})"
        return True, "ok"
