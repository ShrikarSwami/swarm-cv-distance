"""ML overnight loop status page (pinned browser tab).

Owner: overnight status agent.  Owned file: ml/overnight_status_app.py.

A zero-dependency (stdlib only) HTTP server that serves a live dashboard for
scripts/ml_overnight.sh.  The page auto-refreshes every 10 s and shows:

- loop running or not (pgrep for scripts/ml_overnight.sh)
- control state read from .overnight/control.state, including the
  pause-at-boundary semantics: after Pause is clicked the page shows
  "PAUSE REQUESTED — will take effect when iteration N finishes" and only
  switches to "PAUSED" once the run log confirms the loop is actually waiting
  (its newest `control file PAUSED — pausing` line)
- current iteration (newest ml-iter-*.log, cross-checked against the newest
  `── iteration N ── next:` line in the run log) and the FIX task in flight
- elapsed active wall clock (paused time subtracted) and remaining against the
  8 h budget (WALL_CLOCK_MAX, read from the script)
- total time spent paused
- the FIX queue parsed from ml/FIX_QUEUE.md, colour coded
- predicted vs observed side by side for any task with a filled result block
- last 30 lines of the newest iteration log and last 10 lines of the run log

Controls: Pause / Resume / Stop write the single word PAUSED / RUNNING / STOP
to .overnight/control.state via ml.control.write — the same helper and file the
loop already polls.  Stop is guarded by a confirm() in the page (it ends the
run).  The page writes NOTHING else, ever; every log and the queue are read-only.

Usage (repo root)::

    python -m ml.overnight_status_app [--port 8081]

The page URL is printed at startup.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from ml import control  # noqa: E402

OVERNIGHT_DIRNAME = ".overnight"
RUN_LOG_GLOB = "ml-run-*.log"
ITER_LOG_GLOB = "ml-iter-*.log"
REFRESH_MS = 10000
ITER_TAIL = 30
RUN_TAIL = 10
DEFAULT_PORT = 8081
BUDGET_S = 8 * 3600
MAX_ITER_DEFAULT = 12

# ---------------------------------------------------------------------------
# filesystem helpers (read-only on everything except control.state)
# ---------------------------------------------------------------------------


def _overnight_dir(root: str) -> str:
    return os.path.join(root, OVERNIGHT_DIRNAME)


def _script_path(root: str) -> str:
    return os.path.join(root, "scripts", "ml_overnight.sh")


def _queue_path(root: str) -> str:
    return os.path.join(root, "ml", "FIX_QUEUE.md")


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def _tail(path: str, n: int) -> list:
    """Last n lines without loading a big log.  A partial first line is dropped."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            start = max(0, size - 65536)
            f.seek(start)
            data = f.read()
    except OSError:
        return []
    if not data:
        return []
    text = data.decode("utf-8", "replace")
    if start > 0:
        idx = text.find("\n")
        if idx != -1:
            text = text[idx + 1:]
    return text.splitlines()[-n:]


def _newest(odir: str, pattern: str):
    files = glob.glob(os.path.join(odir, pattern))
    if not files:
        return None
    return max(files, key=lambda p: (os.path.getmtime(p), p))


def _parse_var(path: str, name: str, default: int) -> int:
    """Read a MAX_ITER=/WALL_CLOCK_MAX= line from the (unmodified) script."""
    m = re.search(r"^\s*%s=(\d+)" % name, _read(path), re.M)
    return int(m.group(1)) if m else default


# ---------------------------------------------------------------------------
# process detection
# ---------------------------------------------------------------------------


def _loop_running() -> bool:
    """True if a live process matches scripts/ml_overnight.sh (the loop or
    its `caffeinate` wrapper).  pgrep excludes itself, so the page's own
    pgrep call is not a false positive."""
    try:
        out = subprocess.run(
            ["pgrep", "-f", r"scripts/ml_overnight\.sh"],
            capture_output=True, text=True, timeout=5)
        return out.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


# ---------------------------------------------------------------------------
# run-log parsing
# ---------------------------------------------------------------------------


def _line_epoch(ln: str):
    """Epoch for a `[HH:MM:SS]` log line, interpreted as today (local)."""
    m = re.match(r"\[(\d{2}):(\d{2}):(\d{2})\]", ln)
    if not m:
        return None
    now = datetime.now()
    try:
        return datetime(
            now.year, now.month, now.day,
            int(m.group(1)), int(m.group(2)), int(m.group(3)),
        ).timestamp()
    except ValueError:
        return None


def _run_start_epoch(path: str):
    """Run start comes from the run-log filename ml-run-YYYYMMDD-HHMMSS.log,
    which is date-aware (survives a midnight crossing)."""
    m = re.match(r"ml-run-(\d{8})-(\d{6})\.log$", os.path.basename(path))
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").timestamp()
    except ValueError:
        return None


def _empty_info() -> dict:
    return {"iteration": None, "next_task": None, "last_control": None,
            "pause_line_epoch": None, "total_paused": 0}


def _parse_run_log(path: str) -> dict:
    """Extract from the newest run log:
    - iteration number + next task (newest `── iteration N ── next:` line)
    - the newest control transition: "paused" | "running" | "stop" | None
    - total paused seconds (last `(Ns total)` line) and the epoch of the
      newest pausing line (for the in-progress pause duration)
    """
    info = _empty_info()
    for ln in _read(path).splitlines():
        m = re.search(r"── iteration (\d+) ── next: (.*)", ln)
        if m:
            info["iteration"] = int(m.group(1))
            info["next_task"] = m.group(2).strip().lstrip("#").strip()
            continue
        m = re.search(r"control file (PAUSED|RUNNING|STOP)\b", ln)
        if m:
            kind = m.group(1).lower()
            info["last_control"] = kind
            if kind == "paused":
                info["pause_line_epoch"] = _line_epoch(ln)
            # no continue: the resume line also carries the (Ns total) counter
        m = re.search(r"\((\d+)s total\)", ln)
        if m:
            info["total_paused"] = int(m.group(1))
    return info


def _final_wall_clock(text: str):
    """Elapsed printed in the morning summary: `wall clock: 123m45s`."""
    m = re.search(r"wall clock:\s*(\d+)m(\d+)s", text)
    if not m:
        return None
    return int(m.group(1)) * 60 + int(m.group(2))


def _live_timing(run_log_path: str, info: dict):
    start = _run_start_epoch(run_log_path)
    if start is None:
        return None, info["total_paused"]
    now = time.time()
    total = info["total_paused"]
    if info["last_control"] == "paused" and info["pause_line_epoch"]:
        ongoing = now - info["pause_line_epoch"]
        if 0 <= ongoing <= 12 * 3600:  # clamp: pauses don't span days
            total += int(ongoing)
    elapsed = max(0, int(now - start) - total)
    return elapsed, total


# ---------------------------------------------------------------------------
# control display (pause/stop take effect at the next iteration boundary)
# ---------------------------------------------------------------------------


def _boundary_msg(iteration):
    if iteration is None:
        return "will take effect when the current iteration finishes"
    return "will take effect when iteration %s finishes" % iteration


def _control_display(raw: str, last_control, loop_running: bool, iteration):
    """Return (display_label, detail) for the control state.

    The control file reflects what the USER requested; the run log reflects
    what the LOOP has acted on.  raw==PAUSED/STOP with no matching transition
    in the log means the request is still pending at the iteration boundary.

    Requested/confirmed nuance only applies while the loop is alive; once the
    process is gone the plain control state is shown (the "NOT RUNNING" chip
    carries the message).
    """
    if not loop_running:
        return raw, "loop is not running — control file shown for the next launch"
    if raw == "PAUSED":
        if last_control == "paused":
            return "PAUSED", "loop waiting at iteration boundary (polls every 30s)"
        return "PAUSE REQUESTED", _boundary_msg(iteration)
    if raw == "STOP":
        if last_control == "stop":
            return "STOPPING", "ending run cleanly"
        if last_control == "paused":
            return "STOP REQUESTED", "loop is paused — will end at the next poll"
        return "STOP REQUESTED", _boundary_msg(iteration)
    # RUNNING
    if last_control == "paused":
        return "RUNNING", "resume requested — waking from pause"
    return "RUNNING", "loop polls control at each iteration boundary"


# ---------------------------------------------------------------------------
# FIX queue parsing (read-only)
# ---------------------------------------------------------------------------


def _parse_queue(text: str) -> list:
    """Parse ml/FIX_QUEUE.md into [{id, title, status, result{key: value}}].

    Only `## FIX-NN` sections are collected; the result block is the fenced
    block following `**Result:**`, parsed as `key: value` lines.
    """
    tasks = []
    cur = None
    in_result = False
    fence_seen = False
    for ln in text.splitlines():
        m = re.match(r"^## FIX-(\d+)\s*(?:—|-)\s*(.*)$", ln)
        if m:
            cur = {"id": "FIX-%s" % m.group(1), "title": m.group(2).strip(),
                   "status": None, "result": {}}
            tasks.append(cur)
            in_result = False
            fence_seen = False
            continue
        if cur is None:
            continue
        m = re.match(r"^\*\*Status:\*\*\s*(\S+)", ln)
        if m:
            cur["status"] = m.group(1).upper()
            continue
        if re.match(r"^\*\*Result:\*\*", ln):
            in_result = True
            fence_seen = False
            continue
        if in_result:
            if not fence_seen:
                if ln.strip() == "```":  # opening fence
                    fence_seen = True
                continue
            if ln.strip() == "```":      # closing fence
                in_result = False
                continue
            m = re.match(r"^([A-Za-z_][\w ]*):\s*(.*)$", ln)
            if m:
                cur["result"][m.group(1).strip()] = m.group(2).strip()
    return tasks


# ---------------------------------------------------------------------------
# status assembly
# ---------------------------------------------------------------------------


def _collect_status(root: str) -> dict:
    odir = _overnight_dir(root)
    if not os.path.isdir(odir):
        return {"run_found": False}

    script = _script_path(root)
    max_iter = _parse_var(script, "MAX_ITER", MAX_ITER_DEFAULT)
    budget = _parse_var(script, "WALL_CLOCK_MAX", BUDGET_S)

    loop_running = _loop_running()
    raw = control.read(control.control_path(odir))

    queue = _parse_queue(_read(_queue_path(root)))

    run_log_path = _newest(odir, RUN_LOG_GLOB)
    info = _parse_run_log(run_log_path) if run_log_path else _empty_info()

    iteration = info["iteration"]
    if iteration is None:
        iter_path = _newest(odir, ITER_LOG_GLOB)
        if iter_path:
            iteration = _iter_number(iter_path)
    next_task = info["next_task"]
    if not next_task:
        for t in queue:
            if t["status"] == "IN_PROGRESS":
                next_task = "%s — %s" % (t["id"], t["title"])
                break

    label, detail = _control_display(raw, info["last_control"], loop_running, iteration)

    live_elapsed, paused_total = _live_timing(run_log_path, info) if run_log_path else (None, 0)
    final_fc = _final_wall_clock(_read(run_log_path)) if run_log_path else None
    if loop_running or final_fc is None:
        elapsed = live_elapsed
    else:
        elapsed = final_fc  # run is over; show its recorded elapsed, not a drifting value
    remaining = max(0, budget - elapsed) if elapsed is not None else None

    iter_path = _newest(odir, ITER_LOG_GLOB)
    return {
        "run_found": True,
        "loop_running": loop_running,
        "control_state": raw,
        "control_display": label,
        "control_detail": detail,
        "iteration": iteration,
        "iteration_max": max_iter,
        "next_task": next_task,
        "elapsed_s": elapsed,
        "remaining_s": remaining,
        "paused_s": paused_total,
        "budget_s": budget,
        "queue": queue,
        "run_tail": _tail(run_log_path, RUN_TAIL) if run_log_path else [],
        "iter_tail": _tail(iter_path, ITER_TAIL) if iter_path else [],
        "run_log_name": os.path.basename(run_log_path) if run_log_path else None,
        "iter_log_name": os.path.basename(iter_path) if iter_path else None,
        "server_time": datetime.now().strftime("%H:%M:%S"),
    }


def _iter_number(path: str):
    m = re.search(r"ml-iter-(\d+)\.log$", os.path.basename(path))
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# HTML page
# ---------------------------------------------------------------------------

PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ML overnight status</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, -apple-system, sans-serif; background: #0d1117;
         color: #c9d1d9; padding: 2rem; }
  .wrap { max-width: 1080px; margin: 0 auto; }
  h1 { font-size: 1.4rem; margin-bottom: 1rem; color: #58a6ff; }
  h2 { font-size: 1.05rem; margin-bottom: 0.75rem; color: #58a6ff; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
          padding: 1.25rem 1.5rem; margin-bottom: 1rem; }
  .top { display: flex; gap: 1rem; flex-wrap: wrap; }
  .top .card { flex: 1 1 420px; }
  table { width: 100%; border-collapse: collapse; }
  td { padding: 0.3rem 0.4rem; vertical-align: top; }
  td:first-child { color: #8b949e; width: 40%; white-space: nowrap; }
  .big { font-size: 1.35rem; font-weight: 600; }
  .chip { display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px;
          font-size: 0.75rem; font-weight: 600; letter-spacing: 0.05em;
          border: 1px solid #30363d; }
  .chip-run { color: #7ee787; border-color: #2ea043; background: rgba(46,160,67,0.12); }
  .chip-idle { color: #f85149; border-color: #f85149; background: rgba(248,81,73,0.1); }
  .state { font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; }
  .state-running { color: #7ee787; }
  .state-paused, .state-pause-requested { color: #d29922; }
  .state-stop, .state-stopping, .state-stop-requested { color: #f85149; }
  .state-none { color: #8b949e; }
  .detail { color: #8b949e; font-size: 0.85rem; margin-top: 0.25rem; }
  .muted { color: #484f58; }
  .btns { display: flex; gap: 0.5rem; margin: 0.75rem 0; }
  button { padding: 0.5rem 1rem; border: 1px solid #30363d; border-radius: 6px;
           background: #21262d; color: #c9d1d9; cursor: pointer; font-size: 0.9rem; }
  button:hover:not(:disabled) { background: #30363d; }
  button:disabled { opacity: 0.45; cursor: not-allowed; }
  button.pause  { border-color: #d29922; color: #d29922; }
  button.resume { border-color: #7ee787; color: #7ee787; }
  button.stop   { border-color: #f85149; color: #f85149; }
  .qwrap { max-height: 260px; overflow-y: auto; }
  table.qtbl th { text-align: left; color: #8b949e; font-weight: 600;
                  font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em;
                  padding: 0.3rem 0.4rem; border-bottom: 1px solid #30363d; }
  table.qtbl td { padding: 0.3rem 0.4rem; border-bottom: 1px solid #21262d; }
  td.qid { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
           color: #d2a8ff; white-space: nowrap; }
  .st { display: inline-block; padding: 0.1rem 0.5rem; border-radius: 999px;
        font-size: 0.7rem; font-weight: 700; letter-spacing: 0.05em; }
  .st-PENDING { color: #8b949e; background: #21262d; }
  .st-IN_PROGRESS { color: #d29922; background: rgba(210,153,34,0.15); }
  .st-PASSED { color: #7ee787; background: rgba(46,160,67,0.15); }
  .st-FAILED { color: #f85149; background: rgba(248,81,73,0.15); }
  .st-SKIPPED { color: #484f58; background: #161b22; }
  .st-SUSPICIOUS { color: #f0883e; background: rgba(240,136,62,0.15); }
  pre.log { background: #0d1117; border: 1px solid #30363d; border-radius: 6px;
            padding: 0.75rem; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 0.72rem; line-height: 1.45; white-space: pre-wrap;
            word-break: break-word; max-height: 320px; overflow-y: auto; color: #c9d1d9; }
  pre.log.tail30 { max-height: 480px; }
  .loglabel { color: #8b949e; font-size: 0.8rem; margin: 0 0 0.35rem; }
  .cmp { border: 1px solid #30363d; border-radius: 8px; margin-bottom: 0.75rem; overflow: hidden; }
  .cmp-head { padding: 0.5rem 0.75rem; background: #1c2128; border-bottom: 1px solid #30363d;
              display: flex; justify-content: space-between; align-items: center; gap: 0.5rem; }
  .cmp-cols { display: flex; flex-wrap: wrap; }
  .cmp-col { flex: 1 1 45%; padding: 0.75rem; min-width: 260px; }
  .cmp-col + .cmp-col { border-left: 1px solid #30363d; }
  .cmp-lbl { color: #8b949e; font-size: 0.7rem; text-transform: uppercase;
             letter-spacing: 0.06em; margin-bottom: 0.35rem; }
  .cmp-pre { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.75rem;
             white-space: pre-wrap; word-break: break-word; color: #c9d1d9; }
  .cmp-metrics { padding: 0.5rem 0.75rem; border-top: 1px solid #21262d;
                 color: #c9d1d9; font-size: 0.8rem; }
  .cmp-metrics .mkey { color: #8b949e; }
  .cmp-verdict { padding: 0 0.75rem 0.6rem; font-size: 0.8rem; color: #d2a8ff; }
  .empty { color: #8b949e; text-align: center; padding: 2rem 1rem; }
  code { background: #21262d; padding: 0.1rem 0.35rem; border-radius: 4px; font-size: 0.85em; }
  .ts { color: #484f58; font-size: 0.75rem; margin-top: 0.5rem; }
</style>
</head>
<body>
<div class="wrap">
  <h1>ML overnight status</h1>

  <div id="norun" class="card empty" style="display:none">
    No run found &mdash; <code>.overnight/</code> does not exist yet.<br>
    Start the loop from the repo root with
    <code>caffeinate -dims ./scripts/ml_overnight.sh</code>.
  </div>

  <div class="top">
    <div class="card">
      <h2>Run</h2>
      <table>
        <tr><td>Loop</td><td><span class="chip chip-idle" id="loop">—</span></td></tr>
        <tr><td>Control</td><td><span class="state" id="state">—</span>
            <div class="detail" id="detail"></div></td></tr>
        <tr><td>Iteration</td><td id="iter">—</td></tr>
        <tr><td>In flight</td><td id="task">—</td></tr>
        <tr><td>Elapsed (active)</td><td class="big" id="elapsed">—</td></tr>
        <tr><td>Remaining (8 h budget)</td><td id="remaining">—</td></tr>
        <tr><td>Time paused</td><td id="paused">—</td></tr>
      </table>
    </div>

    <div class="card">
      <h2>Controls</h2>
      <div class="btns">
        <button class="pause" id="btn-pause" onclick="setControl('PAUSED')">Pause</button>
        <button class="resume" id="btn-resume" onclick="setControl('RUNNING')">Resume</button>
        <button class="stop" id="btn-stop" onclick="setControl('STOP')">Stop</button>
      </div>
      <div class="detail">Pause / Resume / Stop write <code>.overnight/control.state</code>.
        Pause and Stop take effect at the next iteration boundary &mdash; an iteration can
        run up to 60&nbsp;min, so a request may sit pending for a while.</div>
    </div>
  </div>

  <div class="card">
    <h2>Fix queue</h2>
    <div class="qwrap"><table class="qtbl" id="queue"></table></div>
  </div>

  <div class="card">
    <h2>Predicted vs observed</h2>
    <div id="cmp"></div>
  </div>

  <div class="card">
    <h2>Iteration log &mdash; last 30 lines</h2>
    <div class="loglabel" id="itername2"></div>
    <pre class="log tail30" id="iterlog"></pre>
  </div>

  <div class="card">
    <h2>Run log &mdash; last 10 lines</h2>
    <div class="loglabel" id="runname2"></div>
    <pre class="log" id="runlog"></pre>
  </div>

  <div class="ts" id="ts"></div>
</div>
<script>
var $ = function (id) { return document.getElementById(id); };

function esc(t) {
  var d = document.createElement('div');
  d.textContent = (t == null ? '' : String(t));
  return d.innerHTML;
}

function fmt(s) {
  if (s == null || isNaN(s)) return '—';
  s = Math.max(0, Math.floor(s));
  var h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  if (h > 0) return h + 'h ' + m + 'm';
  if (m > 0) return m + 'm ' + sec + 's';
  return sec + 's';
}

function stateClass(label) {
  var c = (label || '').toLowerCase().replace(/[^a-z]+/g, '-').replace(/^-+|-+$/g, '');
  return c ? 'state-' + c : 'state-none';
}

function render(d) {
  var run = d.run_found === true;
  $('norun').style.display = run ? 'none' : 'block';

  var chip = $('loop');
  if (d.loop_running) { chip.className = 'chip chip-run'; chip.textContent = 'RUNNING'; }
  else { chip.className = 'chip chip-idle'; chip.textContent = 'NOT RUNNING'; }

  var st = $('state');
  st.textContent = run ? (d.control_display || '—') : '—';
  st.className = 'state ' + stateClass(d.control_display);
  $('detail').textContent = run ? (d.control_detail || '') : '';

  $('iter').textContent = run && d.iteration != null ? d.iteration + ' / ' + d.iteration_max : '—';
  $('task').textContent = run ? (d.next_task || '—') : '—';
  $('elapsed').textContent = fmt(run ? d.elapsed_s : null);
  $('remaining').textContent = fmt(run ? d.remaining_s : null);
  $('paused').textContent = fmt(run ? d.paused_s : null);

  var disp = d.control_display || '';
  $('btn-pause').disabled  = !run || /PAUSE|STOP/.test(disp);
  $('btn-resume').disabled = !run || /RUNNING|STOP/.test(disp);
  $('btn-stop').disabled   = !run || /STOP/.test(disp);

  renderQueue(d.queue || []);
  renderCmp(d.queue || []);

  $('iterlog').textContent = (d.iter_tail || []).join('\n');
  $('runlog').textContent = (d.run_tail || []).join('\n');
  $('itername2').textContent = d.iter_log_name ? 'File: ' + d.iter_log_name : '';
  $('runname2').textContent = d.run_log_name ? 'File: ' + d.run_log_name : '';
  $('ts').textContent = 'updated ' + d.server_time;
}

function renderQueue(q) {
  var t = $('queue');
  if (!q.length) { t.innerHTML = '<tr><td class="muted">queue unreadable or empty</td></tr>'; return; }
  var h = '<tr><th>ID</th><th>Task</th><th style="text-align:right">Status</th></tr>';
  for (var i = 0; i < q.length; i++) {
    var x = q[i], st = x.status || 'PENDING';
    h += '<tr><td class="qid">' + esc(x.id) + '</td><td>' + esc(x.title) + '</td>' +
         '<td style="text-align:right"><span class="st st-' + esc(st) + '">' + esc(st) + '</span></td></tr>';
  }
  t.innerHTML = h;
}

function renderCmp(q) {
  var el = $('cmp');
  var filled = q.filter(function (x) {
    return x.result && Object.keys(x.result).some(function (k) {
      return String(x.result[k] || '').trim();
    });
  });
  if (!filled.length) { el.innerHTML = '<div class="muted">No completed result blocks yet.</div>'; return; }
  el.innerHTML = filled.map(cmpCard).join('\n');
}

function cmpCard(x) {
  var r = x.result || {};
  var pred = String(r.predicted || '').trim() || '—';
  var obs = String(r.observed || '').trim() || '—';
  var metr = ['median_err_m', 'count_err per scene', 'peak_to_background']
    .map(function (k) {
      return (r[k] && String(r[k]).trim()) ? '<span class="mkey">' + esc(k) + ':</span> ' + esc(r[k]) : null;
    })
    .filter(function (v) { return v; })
    .join(' &nbsp; | &nbsp; ');
  var vd = String(r.verdict || '').trim();
  var status = x.status || '';
  return '<div class="cmp">' +
    '<div class="cmp-head"><b>' + esc(x.id) + ' — ' + esc(x.title) + '</b>' +
      '<span class="st st-' + esc(status) + '">' + esc(status) + '</span></div>' +
    '<div class="cmp-cols">' +
      '<div class="cmp-col"><div class="cmp-lbl">Predicted</div><pre class="cmp-pre">' + esc(pred) + '</pre></div>' +
      '<div class="cmp-col"><div class="cmp-lbl">Observed</div><pre class="cmp-pre">' + esc(obs) + '</pre></div>' +
    '</div>' +
    (metr ? '<div class="cmp-metrics">' + metr + '</div>' : '') +
    (vd ? '<div class="cmp-verdict">verdict: ' + esc(vd) + '</div>' : '') +
  '</div>';
}

function setControl(state) {
  if (state === 'STOP' && !confirm(
      'Stop the overnight run? It ends at the next iteration boundary and ' +
      'prints the morning summary. This ends the run.')) {
    return;
  }
  fetch('/api/control', {method: 'POST', headers: {'Content-Type': 'application/json'},
                         body: JSON.stringify({state: state})})
    .then(poll)["catch"](poll);
}

function poll() {
  return fetch('/api/status')
    .then(function (r) { return r.json(); })
    .then(render)
    ["catch"](function () { /* retry next tick */ });
}

setInterval(poll, __REFRESH_MS__);
poll();
</script>
</body>
</html>
""".replace("__REFRESH_MS__", str(REFRESH_MS))


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    root = REPO_ROOT

    def log_message(self, *args):
        pass  # suppress request logging

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/status":
            self.send_json(_collect_status(self.root))
        elif path == "/":
            body = PAGE_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/control":
            self.send_error(404)
            return
        odir = _overnight_dir(self.root)
        if not os.path.isdir(odir):
            self.send_error(400, "no run found (no .overnight directory)")
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length))
        except (ValueError, KeyError):
            self.send_error(400, "bad body")
            return
        state = str(body.get("state", "")).upper()
        if state not in control.STATES:
            self.send_error(400, "invalid state")
            return
        control.write(control.control_path(odir), state)
        self.send_json({"ok": True, "state": state})

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
    parser = argparse.ArgumentParser(description="ML overnight loop status page")
    parser.add_argument("--root", default=REPO_ROOT,
                        help="repo root containing .overnight/ and ml/FIX_QUEUE.md")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)

    Handler.root = os.path.abspath(args.root)

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    url = "http://127.0.0.1:%d" % args.port
    print("overnight status page: %s" % url)
    print("root: %s" % Handler.root)
    print("Ctrl-C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
