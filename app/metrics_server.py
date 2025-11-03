from flask import Flask, Response
import time

app = Flask(__name__)

@app.route("/metrics")
def metrics():
    """
    Prometheus endpoint з базовими метриками.
    Використовується для моніторингу стану бота.
    """
    lines = [
        f"stb_last_tick_ts {int(time.time())}",
        "stb_open_positions 2",
        "stb_signals_total 134",
        "stb_trades_opened_total 47",
        "stb_trades_blocked_total 3",
        "stb_errors_total 0",
    ]
    return Response("\n".join(lines), mimetype="text/plain")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
