# obs.py
import json, sys, time
from typing import Any, Dict
from config import LOG_LEVEL

LEVELS = ["DEBUG","INFO","WARN","ERROR"]
LVL = LEVELS.index(LOG_LEVEL) if LOG_LEVEL in LEVELS else 1

def jlog(level: str, msg: str, **kv: Dict[str, Any]):
    if LEVELS.index(level) < LVL: 
        return
    rec = {"ts": int(time.time()), "level": level, "msg": msg}
    rec.update(kv)
    sys.stdout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    sys.stdout.flush()
