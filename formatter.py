# formatter.py
from datetime import datetime, timezone
from learning import get_confidence_factor

def nice_check(achieved: bool):
    return "✅" if achieved else "🔴"

def format_tp_line(i, tp):
    price = tp.get("price")
    achieved = tp.get("achieved", False)
    minutes = tp.get("minutes")
    percent = tp.get("percent")
    check = nice_check(achieved)
    minutes_str = f" ({minutes}m)" if minutes is not None else ""
    percent_str = f" {percent}%" if percent is not None else ""
    return f"🎯 TP{i}: <b>{price}</b> - {check}{minutes_str}{percent_str}"

def format_signal_message(payload: dict) -> str:
    symbol = payload.get("symbol", "UNKNOWN")
    timeframe = payload.get("timeframe", "")
    side = payload.get("side", "LONG").upper()
    entry = payload.get("entry", "")
    add = payload.get("add")
    tp_list = payload.get("tp", [])
    note = payload.get("note", "")
    author = payload.get("author", "")

    factor = get_confidence_factor()
    emoji = "💎" if side == "LONG" else "🔻"
    strength_emoji = "🔥" if factor > 1.0 else "💤"

    lines = []
    lines.append(f"<b>#{symbol} {timeframe}</b>")
    lines.append(f"{emoji} <b>СТАТУС : {side}</b> {strength_emoji}")
    lines.append("")
    lines.append(f"👉 ENTRY : <b>{entry}</b>")
    if add:
        lines.append(f"👉 ДОБОР : <b>{add}</b>")
    lines.append("")

    for i, tp in enumerate(tp_list, start=1):
        lines.append(format_tp_line(i, tp))

    if note:
        lines.append("")
        lines.append(f"ℹ️ {note}")

    if author:
        lines.append(f"\n— <i>{author}</i>")

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines.append(f"\n<code>{ts}</code>")
    return "\n".join(lines)

