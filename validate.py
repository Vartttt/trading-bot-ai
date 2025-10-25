# validate.py
import os, sys

REQUIRED = ["MEXC_API_KEY", "MEXC_API_SECRET"]

def ensure_env():
    miss = [k for k in REQUIRED if not os.getenv(k)]
    if miss:
        sys.stderr.write(f"[FATAL] Missing env: {', '.join(miss)}\n")
        sys.exit(1)
