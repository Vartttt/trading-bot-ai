# dashboard.py
from flask import Flask, render_template_string, request
import sqlite3
from config import SQLITE_PATH, DASHBOARD_HOST, DASHBOARD_PORT

APP = Flask(__name__)

TPL = """
<!doctype html>
<title>MEXC Bot Dashboard</title>
<h2>Events (latest {{limit}})</h2>
<form>
  Limit: <input name="limit" value="{{limit}}" size="4">
  <button type="submit">Reload</button>
</form>
<table border="1" cellpadding="6">
  <tr><th>ts</th><th>event</th><th>symbol</th><th>side</th><th>price</th><th>amount</th><th>pnl</th><th>extra</th></tr>
  {% for r in rows %}
  <tr>
    <td>{{r[1]}}</td><td>{{r[2]}}</td><td>{{r[3]}}</td><td>{{r[4]}}</td>
    <td>{{r[5]}}</td><td>{{r[6]}}</td><td>{{r[7]}}</td><td>{{r[8]}}</td>
  </tr>
  {% endfor %}
</table>
"""

@APP.route("/")
def home():
    limit = int(request.args.get("limit", "50"))
    con = sqlite3.connect(SQLITE_PATH)
    cur = con.cursor()
    cur.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    con.close()
    return render_template_string(TPL, rows=rows, limit=limit)

if __name__ == "__main__":
    APP.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=False)
