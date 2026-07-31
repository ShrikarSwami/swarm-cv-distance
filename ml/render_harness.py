"""T2 — resumable render harness (owner: harness agent).

Build and smoke-tested by an agent. The full campaign is launched by the human,
detached, outside any agent session.

Invariants
----------
- The manifest (`root/manifest.jsonl`) is the source of truth: a scene is
  "done" iff it has one line here. Resume skips scenes already in the manifest.
- No truncated scene can ever appear at the final path: a scene renders into
  `root/.tmp/rendering_<seed>`, is verified (all angles present and non-empty),
  then atomically renamed into place (`root/scenes/../<seed>/`), and only then
  is its manifest line appended. Rename-then-append ordering means a kill in
  the window between the two leaves a complete orphan scene (adopted on resume),
  never a manifest claim for a missing scene.
- Control file (`root/control.state`) is polled at every scene boundary:
  RUNNING renders the next scene, PAUSED waits, STOP exits cleanly. A missing
  or unwritable data root auto-pauses rather than crashing (the spec's
  drive-presence check, adapted: the external drive is dropped per Ruling 2,
  all data lives on the internal SSD).
- Scene generation is deterministic per seed (ml/scene_gen), so a killed scene
  is re-rendered byte-identical on resume.

Detached launch (human)::

    python -m ml.render_harness --root ~/swarm_ml --target 5000 --detach

or with a control file already placed. `--detach` re-execs the same run under
`caffeinate` with nohup semantics and a log file.

Usage::

    python -m ml.render_harness --root DIR --target N
                                [--n-drones N] [--cell primary|secondary]
                                [--start-seed 2000] [--control FILE]
                                [--detach] [--log FILE]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from ml import scene_gen  # noqa: E402
from ml import control  # noqa: E402

BLENDER_BIN = os.environ.get(
    "BLENDER_BIN", "/Applications/Blender.app/Contents/MacOS/blender"
)
RENDER_SCRIPT = os.path.join(REPO_ROOT, "ml", "_render_scene.py")
RENDER_TIMEOUT_S = 600
PAUSE_POLL_S = 2.0
TRAIN_START_SEED = scene_gen.SEED_TRAIN[0]


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


def read_manifest(root):
    """Return list of completed-scene records. Torn trailing lines (a kill
    mid-append) are dropped rather than treated as data."""
    records = []
    path = os.path.join(root, "manifest.jsonl")
    if not os.path.exists(path):
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # torn tail from a kill mid-append
    return records


def append_manifest(root, record):
    path = os.path.join(root, "manifest.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


def done_seeds(root):
    return {rec["seed"] for rec in read_manifest(root)}


def pending_seeds(root, start_seed, target):
    """Train-range seeds [start_seed, start_seed+target) not yet in the manifest
    (PATCH 7: campaign extension adds train seeds only, regenerates nothing)."""
    done = done_seeds(root)
    return [s for s in range(int(start_seed), int(start_seed) + int(target))
            if s not in done]


def cleanup_stale_tmp(root):
    """Remove temp dirs left by a killed run (their seeds re-render on resume)."""
    tmp_root = os.path.join(root, ".tmp")
    if not os.path.isdir(tmp_root):
        return 0
    import shutil

    n = 0
    for name in os.listdir(tmp_root):
        if name.startswith("rendering_"):
            shutil.rmtree(os.path.join(tmp_root, name), ignore_errors=True)
            n += 1
    return n


def root_available(root):
    """Data-root presence/writability check -> auto-pause when false."""
    if not os.path.isdir(root):
        return False
    if not os.access(root, os.W_OK):
        return False
    return True


def scene_dir_complete(final_dir):
    """True iff every declared angle PNG plus both JSONs exists and is non-empty."""
    cam_path = os.path.join(final_dir, "cameras.json")
    gt_path = os.path.join(final_dir, "ground_truth.json")
    if not (os.path.exists(cam_path) and os.path.exists(gt_path)):
        return False
    try:
        with open(cam_path) as f:
            n_views = json.load(f)["n_views"]
    except (OSError, KeyError, json.JSONDecodeError):
        return False
    for idx in range(int(n_views)):
        p = os.path.join(final_dir, "angle_%02d.png" % idx)
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            return False
    return True


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def render_one(root, seed, n_drones=None, cell=None, drone_size=scene_gen.DRONE_SIZE_M):
    """Render one scene end-to-end. Returns the manifest record.

    Raises RenderError on Blender failure (after one retry). Never returns a
    partial scene: on any failure the tmp dir is cleaned and nothing is renamed
    or manifested.
    """
    scene = scene_gen.generate_scene(seed=seed, cell=cell, n_drones=n_drones)
    final = scene_gen.scene_dir(root, seed)

    # Adopt an orphaned complete scene (rename done, manifest append not).
    if os.path.isdir(final):
        if scene_dir_complete(final):
            rec = _manifest_record(root, seed, scene, render_wall_s=None,
                                   png_bytes=None, scene_bytes=None, adopted=True)
            append_manifest(root, rec)
            return rec
        import shutil

        shutil.rmtree(final)  # defensive: never adopt a partial final dir

    tmp = os.path.join(root, ".tmp", "rendering_%05d" % seed)
    import shutil

    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)

    gt, cam = scene_gen.serialize_scene(scene)
    for name, payload in (("ground_truth.json", gt), ("cameras.json", cam)):
        with open(os.path.join(tmp, name), "w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

    last_err = None
    for attempt in (1, 2):
        try:
            recs, rc = _run_blender(tmp, tmp, drone_size)
            if rc != 0:
                raise RenderError("blender exited %d" % rc)
            done = next((r for r in recs if r.get("mode") == "done"), None)
            if done is None or done.get("angles") != scene_gen.N_VIEWS:
                raise RenderError("blender done record missing/incomplete: %r" % done)
            if not scene_dir_complete(tmp):
                missing = [i for i in range(scene_gen.N_VIEWS)
                           if not (os.path.isfile(os.path.join(tmp, "angle_%02d.png" % i))
                                   and os.path.getsize(os.path.join(tmp, "angle_%02d.png" % i)) > 0)]
                raise RenderError("missing/non-empty angle PNGs: %s" % missing[:8])
            break
        except RenderError as exc:
            last_err = exc
            import shutil as _sh

            _sh.rmtree(tmp, ignore_errors=True)
            os.makedirs(tmp)
            if attempt == 1:
                sys.stderr.write("scene %d attempt 1 failed (%s) — retrying\n"
                                 % (seed, exc))
    else:
        raise RenderError("scene %d failed twice: %s" % (seed, last_err))

    # Atomic publish: rename tmp -> final, THEN append manifest.
    os.makedirs(os.path.dirname(final), exist_ok=True)
    os.replace(tmp, final)
    png_bytes = sum(os.path.getsize(os.path.join(final, "angle_%02d.png" % i))
                    for i in range(scene_gen.N_VIEWS))
    scene_bytes = sum(os.path.getsize(os.path.join(final, n))
                      for n in os.listdir(final) if os.path.isfile(os.path.join(final, n)))
    rec = _manifest_record(root, seed, scene, render_wall_s=None,
                           png_bytes=png_bytes, scene_bytes=scene_bytes, adopted=False)
    append_manifest(root, rec)
    return rec


def _manifest_record(root, seed, scene, render_wall_s, png_bytes, scene_bytes, adopted):
    return {
        "scene_id": int(seed),
        "seed": int(seed),
        "split": scene["split"],
        "cell": scene["cell"],
        "cell_radius_m": scene["radius_m"],
        "a_max_px": scene["a_max_px"],
        "n_drones": scene["n_drones"],
        "n_views": scene_gen.N_VIEWS,
        "png_bytes_total": int(png_bytes) if png_bytes is not None else None,
        "scene_bytes": int(scene_bytes) if scene_bytes is not None else None,
        "render_wall_s": round(float(render_wall_s), 3) if render_wall_s is not None else None,
        "adopted": bool(adopted),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


class RenderError(RuntimeError):
    pass


def _run_blender(scene_dir, outdir, drone_size):
    cmd = [BLENDER_BIN, "--background", "--python", RENDER_SCRIPT, "--",
           "--scene-dir", scene_dir, "--outdir", outdir,
           "--drone-size", str(drone_size)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              errors="replace", timeout=RENDER_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise RenderError("blender timed out after %ds: %s" % (RENDER_TIMEOUT_S, exc))
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    records = []
    for line in combined.splitlines():
        if line.startswith("RENDER_JSON "):
            try:
                records.append(json.loads(line[len("RENDER_JSON "):]))
            except json.JSONDecodeError:
                pass
    if proc.returncode != 0:
        tail = "\n".join(combined.splitlines()[-15:])
        sys.stderr.write("blender tail on failure:\n%s\n" % tail)
    return records, proc.returncode


# ---------------------------------------------------------------------------
# run loop
# ---------------------------------------------------------------------------


def _log(log_path, msg):
    line = "%s %s" % (datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), msg)
    if log_path:
        with open(log_path, "a") as f:
            f.write(line + "\n")
            f.flush()
    print(line, flush=True)


def run(root, target, n_drones=None, cell=None, start_seed=TRAIN_START_SEED,
        control_path=None, log=None):
    os.makedirs(root, exist_ok=True)
    control_path = control_path or control.control_path(root)
    cleaned = cleanup_stale_tmp(root)
    if cleaned:
        _log(log, "cleaned %d stale tmp scene dir(s) from a prior run" % cleaned)

    rendered = 0
    while True:
        state = control.read(control_path)
        if state == "STOP":
            _log(log, "STOP received at scene boundary — exiting (rendered %d this run)" % rendered)
            return 0
        if state == "PAUSED":
            time.sleep(PAUSE_POLL_S)
            continue

        seeds = pending_seeds(root, start_seed, target)
        if not seeds:
            _log(log, "target reached: %d scenes done in manifest" % len(done_seeds(root)))
            return 0
        if not root_available(root):
            _log(log, "data root %r unavailable — AUTO-PAUSE (resume when it returns)"
                 % root)
            control.write(control_path, "PAUSED")
            continue

        seed = seeds[0]
        t0 = time.perf_counter()
        _log(log, "render seed %d (%d of %d, %d remaining)"
             % (seed, rendered + 1, target, len(seeds) - 1))
        try:
            rec = render_one(root, seed, n_drones=n_drones, cell=cell)
        except RenderError as exc:
            _log(log, "FATAL: %s — writing STOP so the failure is visible" % exc)
            control.write(control_path, "STOP")
            return 1
        dt = time.perf_counter() - t0
        rendered += 1
        _log(log, "done seed %d wall=%.2fs bytes=%s" % (seed, dt, rec["scene_bytes"]))
        # loop re-polls the control file at the scene boundary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argv(args, detached):
    argv = [sys.executable, "-m", "ml.render_harness",
            "--root", args.root, "--target", str(args.target)]
    if args.n_drones is not None:
        argv += ["--n-drones", str(args.n_drones)]
    if args.cell:
        argv += ["--cell", args.cell]
    argv += ["--start-seed", str(args.start_seed)]
    if args.control:
        argv += ["--control", args.control]
    if not detached and args.log:
        argv += ["--log", args.log]
    return argv


def main(argv=None):
    parser = argparse.ArgumentParser(description="T2 resumable render harness")
    parser.add_argument("--root", default=os.path.join(os.path.expanduser("~"), "swarm_ml"))
    parser.add_argument("--target", type=int, required=True, help="total scene count from --start-seed")
    parser.add_argument("--n-drones", type=int, default=None, help="fixed N override (pilot only)")
    parser.add_argument("--cell", default=None, choices=sorted(scene_gen.OPERATING_CELLS))
    parser.add_argument("--start-seed", type=int, default=TRAIN_START_SEED)
    parser.add_argument("--control", default=None, help="control file path (default root/control.state)")
    parser.add_argument("--log", default=None, help="append log file path")
    parser.add_argument("--detach", action="store_true",
                        help="re-exec under caffeinate with nohup semantics")
    args = parser.parse_args(argv)

    if args.target < 1:
        parser.error("--target must be >= 1")

    if args.detach:
        log = args.log or os.path.join(args.root, "render.log")
        os.makedirs(args.root, exist_ok=True)
        control.set_running(args.root)  # a previous STOP/PAUSE must not trap the new run
        argv = _build_argv(args, detached=True)
        with open(log, "a") as f:
            f.write("--- detached launch: %s ---\n" % " ".join(argv))
        proc = subprocess.Popen(["caffeinate", "-i"] + argv,
                                stdout=open(log, "a"), stderr=subprocess.STDOUT,
                                start_new_session=True)
        print("detached render harness pid=%d" % proc.pid)
        print("log: %s" % log)
        print("control: %s" % control.control_path(args.root))
        return 0

    return run(root=args.root, target=args.target, n_drones=args.n_drones,
               cell=args.cell, start_seed=args.start_seed,
               control_path=args.control, log=args.log)


if __name__ == "__main__":
    sys.exit(main())
