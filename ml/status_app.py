"""T3 — Status page (pinned tab).

Owner: status agent. Owned file: ml/status_app.py.

A zero-dependency (stdlib only) HTTP server that serves a live dashboard for
the render campaign.  The page auto-refreshes every 2 s and shows:

- scenes done (from manifest) / target
- measured rate (scenes per minute, computed from manifest timestamps — NOT estimated)
- ETA (remaining / measured-rate, in human-readable h:mm)
- control state (RUNNING / PAUSED / STOP)
- Pause / Resume / Stop buttons that write the control file

Acceptance (from spec Section 8 T3):
  "pause from UI, confirm generator halts at next scene boundary,
   resume, confirm continuation."

Usage (repo root, while the render harness runs separately)::

    python -m ml.status_app --root ~/swarm_ml [--port 8080]

The harness must share the same root.  The page URL is printed at startup.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from ml import control  # noqa: E402
from ml.render_harness import read_manifest  # noqa: E402


# ---------------------------------------------------------------------------
# data helpers
# ---------------------------------------------------------------------------


def _rate_and_eta(root, target):
    """Computed from manifest timestamps — never estimated."""
    recs = read_manifest(root)
    n_done = len(recs)
    if n_done >= 2:
        ts = []
        for rec in recs:
            fa = rec.get("finished_at")
            if fa:
                try:
                    ts.append(datetime.fromisoformat(fa.replace("Z", "+00:00")).timestamp())
                except (ValueError, TypeError):
                    pass
        if len(ts) >= 2:
            t_min, t_max = min(ts), max(ts)
            elapsed = t_max - t_min
            if elapsed > 0.1:
                rate_per_sec = (len(ts) - 1) / elapsed  # scenes/min
                rate_per_min = rate_per_sec * 60.0
                remaining = max(0, target - n_done)
                eta_s = remaining / rate_per_sec if rate_per_sec > 0 else None
                return n_done, rate_per_min, eta_s
    return n_done, None, None


def _fmt_eta(eta_s):
    if eta_s is None:
        return "—"
    h = int(eta_s // 3600)
    m = int((eta_s % 3600) // 60)
    s = int(eta_s % 60)
    if h > 0:
        return "%dh %dm" % (h, m)
    return "%dm %ds" % (m, s)


# ---------------------------------------------------------------------------
# HTML page
# ---------------------------------------------------------------------------

PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Render campaign status</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, -apple-system, sans-serif; background: #0d1117;
         color: #c9d1d9; padding: 2rem; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
          padding: 1.5rem; max-width: 640px; margin: 0 auto; }
  h1 { font-size: 1.25rem; margin-bottom: 1rem; color: #58a6ff; }
  table { width: 100%%; margin-bottom: 1rem; }
  td { padding: 0.35rem 0.5rem; vertical-align: top; }
  td:first-child { color: #8b949e; width: 40%%; }
  .big { font-size: 1.5rem; font-weight: 600; }
  .rate { color: #7ee787; }
  .eta  { color: #d2a8ff; }
  .state { font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
  .state-RUNNING { color: #7ee787; }
  .state-PAUSED  { color: #d29922; }
  .state-STOP    { color: #f85149; }
  .btns { display: flex; gap: 0.5rem; margin-top: 1rem; }
  button { padding: 0.5rem 1rem; border: 1px solid #30363d; border-radius: 6px;
           background: #21262d; color: #c9d1d9; cursor: pointer; font-size: 0.9rem; }
  button:hover { background: #30363d; }
  button.pause  { border-color: #d29922; color: #d29922; }
  button.resume { border-color: #7ee787; color: #7ee787; }
  button.stop   { border-color: #f85149; color: #f85149; }
  .ts { color: #484f58; font-size: 0.75rem; margin-top: 1rem; }
</style>
</head>
<body>
<div class="card">
  <h1>Render campaign status</h1>
  <table>
    <tr><td>Scenes done</td><td class="big" id="done">—</td></tr>
    <tr><td>Target</td><td id="target">—</td></tr>
    <tr><td>Measured rate</td><td class="rate" id="rate">—</td></tr>
    <tr><td>ETA</td><td class="eta" id="eta">—</td></tr>
    <tr><td>Control state</td><td class="state" id="state">—</td></tr>
  </table>
  <div class="btns">
    <button class="pause" onclick="setControl('PAUSED')">Pause</button>
    <button class="resume" onclick="setControl('RUNNING')">Resume</button>
    <button class="stop" onclick="setControl('STOP')">Stop</button>
  </div>
  <div class="ts" id="ts"></div>
</div>
<script>
async function poll() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    document.getElementById('done').textContent = d.manifest_count + ' / ' + d.target;
    document.getElementById('target').textContent = d.target;
    document.getElementById('rate').textContent = d.rate_per_min != null
      ? d.rate_per_min.toFixed(1) + ' scenes/min' : '—';
    document.getElementById('eta').textContent = d.eta_display;
    const st = document.getElementById('state');
    st.textContent = d.control_state;
    st.className = 'state state-' + d.control_state;
    document.getElementById('ts').textContent = 'updated ' + d.server_time;
  } catch (e) { /* retry next tick */ }
}
async function setControl(state) {
  await fetch('/api/control', {method:'POST', headers:{'Content-Type':'application/json'},
                               body: JSON.stringify({state})});
  poll();
}
setInterval(poll, 2000);
poll();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    root = "."
    target = 0

    def log_message(self, *args):
        pass  # suppress request logging

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/status":
            self.send_json(self._status())
        elif path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(PAGE_HTML.encode())
        else:
            self.send_error(404)

    def do_POST(self):
        if urlparse(self.path).path == "/api/control":
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state = body.get("state", "").upper()
            if state in control.STATES:
                control.write(control.control_path(self.root), state)
                self.send_json({"ok": True, "state": state})
            else:
                self.send_error(400, "invalid state")
        else:
            self.send_error(404)

    def _status(self):
        n, rate, eta_s = _rate_and_eta(self.root, self.target)
        return {
            "manifest_count": n,
            "target": self.target,
            "rate_per_min": round(rate, 1) if rate is not None else None,
            "eta_display": _fmt_eta(eta_s),
            "control_state": control.read(control.control_path(self.root)),
            "server_time": datetime.now().strftime("%H:%M:%S"),
        }

    def send_json(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description="T3 render campaign status page")
    parser.add_argument("--root", default=os.path.join(os.path.expanduser("~"), "swarm_ml"))
    parser.add_argument("--target", type=int, default=5000)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)

    Handler.root = args.root
    Handler.target = args.target

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    url = "http://127.0.0.1:%d" % args.port
    print("status page: %s" % url)
    print("root: %s | target: %d" % (args.root, args.target))
    print("Ctrl-C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
