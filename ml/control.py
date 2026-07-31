"""T2 — control-file helpers (owner: harness agent).

The render harness polls a small state file at every scene boundary. Values:

    RUNNING  render the next scene
    PAUSED   halt at the next scene boundary, wait (status page "Pause")
    STOP     halt at the next scene boundary, exit cleanly (status page "Stop")

A missing control file means RUNNING (a fresh root starts un-paused). Writes
are atomic (temp + rename) so the harness never reads a torn state.
"""

from __future__ import annotations

import os
import tempfile

STATES = ("RUNNING", "PAUSED", "STOP")
DEFAULT_FILENAME = "control.state"


def control_path(root) -> str:
    return os.path.join(root, DEFAULT_FILENAME)


def read(path: str) -> str:
    """Return the current state; a missing/unreadable file means RUNNING."""
    try:
        with open(path) as f:
            state = f.read().strip().upper()
    except OSError:
        return "RUNNING"
    if state in STATES:
        return state
    return "RUNNING"


def write(path: str, state: str) -> None:
    if state not in STATES:
        raise ValueError("control state must be one of %s, got %r" % (STATES, state))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)),
                               prefix=".control.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(state + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def status(root: str) -> str:
    return read(control_path(root))


def set_running(root: str) -> None:
    write(control_path(root), "RUNNING")


def set_paused(root: str) -> None:
    write(control_path(root), "PAUSED")


def set_stop(root: str) -> None:
    write(control_path(root), "STOP")


def describe(root: str) -> dict:
    """Full status for the T3 page: state + control-file mtime."""
    path = control_path(root)
    state = read(path)
    mtime = None
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        pass
    return {"state": state, "control_file": path, "mtime": mtime}
