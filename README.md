# Trading Signal Bot (Railway-ready)

Simple prototype that:
- Scans defined pairs (top 5 + 5 active),
- Generates reversal LONG/SHORT signals (EMA/RSI/MACD multi-TF),
- Sends pre-entry (~5 min) and entry alerts to Telegram (text + chart),
- Persists signals into SQLite for later training,
- Background loop + Flask webserver (health endpoint) — suitable for Railway.

## Quick start (local)
1. Create virtualenv:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
