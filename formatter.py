# formatter.py
from datetime import datetime, timezone

def nice_check(achieved: bool):
    return "✅" if achieved else "🔴"

def format_tp_line(i, tp):
    """
    tp: dict with keys:
      - price (float or str)
      - achieved (bool) optional
      - minutes (int) optional -> time to reach in minutes or time passed
      - percent (int or float) optional -> confidence in %
    """
    price = tp.get("price")
    achieved = tp.get("achieved", False)
    minutes = tp.get("minutes")  # number of minutes (int)
    percent = tp.get("percent")
    check = nice_check(achieved)
    minutes_str = f" ({minutes}m)" if minutes is not None else ""
    percent_str = f" {percent}%" if percent is not None else ""
    return f"🎯 TP{i}: <b>{price}</b> - {check}{minutes_str}{percent_str}"

def format_signal_message(payload: dict) -> str:
    """
    payload expected keys:
      - symbol: "AVAXUSDT"
      - timeframe: "15m"
      - side: "LONG" or "SHORT"
      - entry: "20.36" (str or float)
      - add: "20.10" optional
      - tp: list of tp dicts (see format_tp_line)
      - note: optional additional text
      - author: optional name/tag
    """
    symbol = payload.get("symbol", "UNKNOWN")
    timeframe = payload.get("timeframe", "")
    side = payload.get("side", "LONG").upper()
    entry = payload.get("entry", "")
    add = payload.get("add")  # optional
    tp_list = payload.get("tp", [])
    note = payload.get("note", "")
    author = payload.get("author", "")

    lines = []
    # Header
    emoji = "💎" if side == "LONG" else "🔻"
    lines.append(f"<b>#{symbol} {timeframe}</b>")
    lines.append(f"{emoji} <b>СТАТУС : {side}</b> 🚀")
    lines.append("")  # empty line

    # Entry / Add
    lines.append(f"👉 ENTRY : <b>{entry}</b>")
    if add:
        lines.append(f"👉 ДОБОР : <b>{add}</b>")
    lines.append("")  # empty line

    # TPs
    for i, tp in enumerate(tp_list, start=1):
        lines.append(format_tp_line(i, tp))

    if note:
        lines.append("")
        lines.append(f"ℹ️ {note}")

    if author:
        lines.append(f"\n— <i>{author}</i>")

    # Footer with timestamp (UTC)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines.append(f"\n<code>{ts}</code>")
    return "\n".join(lines)
