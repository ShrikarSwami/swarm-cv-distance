#!/usr/bin/env python3
"""T9 — ml/recon_app.py: interactive reconstruction UI (geometric baseline only).

OWNED FILE (Agent K): the ONLY file this module modifies is itself. Everything
it calls is FROZEN — `ml/baseline_adapter.process_scene` (the T5 geometric
control path) and `ml.metrics.evaluate` (the single frozen metric dict). No
torch, no ML model: the ML track is paused (G2 failed on a structural cause,
Session 15e), so reconstruction runs the pure geometric pipeline
detect -> correspond -> triangulate -> metrics on the rendered scenes in
`~/swarm_ml`.

What it does
------------
    1. Lists scenes from `~/swarm_ml/manifest.jsonl` with their metadata
       (seed, split, cell, n_drones, n_views).
    2. Lets the user pick a scene and an angle-selection mode:
         all          -> all 24 angles
         exact        -> a user-specified subset of angles 0..23
         random-n     -> a user-specified N, N random angles from 0..23
         random-random-> a user-specified max, 1..max random angles
    3. Runs the frozen baseline (`process_scene`) on that scene + view subset.
    4. Prints the frozen metric dict (mAP, median_err_m, chamfer_m, count_err,
       precision/recall per tau) and saves a 3D overlay PNG: ground truth
       (green), predictions (blue), red error vectors between Hungarian-matched
       pairs, ghost predictions (orange) and missed drones (hollow gray), and
       the used camera positions (gray triangles).

Usage
-----
    # Non-interactive (acceptance path): pick a scene + mode via flags.
    python3 -m ml.recon_app --scene 0 --mode random-n --n 6 --out /tmp/recon.png

    # Interactive: prompts for scene and angle mode.
    python3 -m ml.recon_app

    # List scenes from the manifest (optionally filtered).
    python3 -m ml.recon_app --list --split test --cell primary --limit 20

Constraints: numpy / scipy / matplotlib / PIL only (no torch, no bpy). Pure
geometry — MPS not needed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np

# ---------------------------------------------------------------------------
# sys.path bootstrap — make the repo root and the frozen stage-1 modules
# importable regardless of how the module is launched (python -m, direct
# script). Mirrors ml/baseline_adapter.py.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STAGE1 = os.path.join(_REPO_ROOT, "stage1_geometry")
for _p in (_STAGE1, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Headless backend before pyplot is imported: this is a CLI that saves PNGs.
import matplotlib  # noqa: E402
matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401  (registers "3d")

# --- Frozen entry point (read / call / never edit) ---
# `process_scene` runs the frozen control path and returns the frozen metric
# dict from `ml.metrics.evaluate` (detect -> correspond -> triangulate ->
# evaluate); that single dict is what we print. No ML model, no torch.
from ml.baseline_adapter import process_scene  # noqa: E402  frozen T5 control

N_VIEWS_TOTAL = 24            # scene_gen.N_VIEWS (8 per tier x 3 tiers)
DEFAULT_ROOT = os.path.join(os.path.expanduser("~"), "swarm_ml")
DEFAULT_RECALL_RADIUS_PX = 5.0

# Display taus for the precision/recall summary (frozen DEFAULT_TAUS order).
_TAU_ORDER = (0.5, 1.0, 2.0, 5.0)


# ---------------------------------------------------------------------------
# Scene selection (manifest)
# ---------------------------------------------------------------------------

def load_manifest(root: str) -> list[dict]:
    """Load every line of manifest.jsonl as a scene dict (seed -> metadata)."""
    path = os.path.join(root, "manifest.jsonl")
    if not os.path.isfile(path):
        raise SystemExit("manifest not found: %s" % path)
    scenes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            scenes.append(json.loads(line))
    scenes.sort(key=lambda s: int(s["seed"]))
    return scenes


def summarize_manifest(scenes: list[dict]) -> str:
    """One-line summary of the manifest composition (split x cell counts)."""
    counts: dict[tuple, int] = {}
    for s in scenes:
        counts[(s.get("split", "?"), s.get("cell", "?"))] = (
            counts.get((s.get("split", "?"), s.get("cell", "?")), 0) + 1
        )
    parts = ", ".join("%s/%s=%d" % (sp, ce, n)
                      for (sp, ce), n in sorted(counts.items()))
    return "%d scenes (%s)" % (len(scenes), parts)


def find_scene(scenes: list[dict], seed: int) -> dict:
    """Return the manifest entry for a seed, raising a helpful error if absent."""
    for s in scenes:
        if int(s["seed"]) == int(seed):
            return s
    raise SystemExit("seed %d not found in manifest (scenes present: %d)"
                     % (int(seed), len(scenes)))


def scene_metadata_line(s: dict) -> str:
    return ("seed=%d split=%-5s cell=%-9s n_drones=%d n_views=%d"
            % (int(s["seed"]), s.get("split", "?"), s.get("cell", "?"),
               int(s.get("n_drones", 0)), int(s.get("n_views", N_VIEWS_TOTAL))))


def pick_scene_interactive(scenes: list[dict]) -> dict:
    """Prompt the user for a seed (with a small listing helper)."""
    print("Manifest: %s" % summarize_manifest(scenes))
    while True:
        try:
            raw = input("scene seed (or 'list [split]' to browse, 'q' to quit): ").strip()
        except EOFError:
            raise SystemExit("no input")
        if raw.lower() in ("q", "quit", "exit"):
            raise SystemExit("aborted by user")
        if raw.lower().startswith("list"):
            parts = raw.split()
            split = parts[1] if len(parts) > 1 else None
            pool = [s for s in scenes if split is None or s.get("split") == split]
            print("first 30 of %d scenes (%s):" % (len(pool), split or "all"))
            for s in pool[:30]:
                print("  " + scene_metadata_line(s))
            continue
        try:
            seed = int(raw)
        except ValueError:
            print("  expected a seed number or 'list'")
            continue
        return find_scene(scenes, seed)


# ---------------------------------------------------------------------------
# Angle selection modes
# ---------------------------------------------------------------------------

def choose_angles(mode: str, rng: random.Random, args) -> list[int]:
    """Return the list of angle indices 0..23 for the selected mode.

    Modes:
      all           -> every angle 0..23
      exact         -> the comma-separated indices in `args.angles`
      random-n      -> `args.n` distinct random angles
      random-random -> k distinct random angles with k ~ U(1, args.max_views)
    """
    mode = (mode or "all").lower()
    if mode == "all":
        return list(range(N_VIEWS_TOTAL))
    if mode == "exact":
        if not args.angles:
            raise SystemExit("--mode exact requires --angles 'i,j,k,...' (0..23)")
        idxs = []
        for tok in args.angles.split(","):
            tok = tok.strip()
            if not tok:
                continue
            v = int(tok)
            if not (0 <= v < N_VIEWS_TOTAL):
                raise SystemExit("angle %d out of range 0..%d" % (v, N_VIEWS_TOTAL - 1))
            idxs.append(v)
        return sorted(set(idxs))
    if mode == "random-n":
        n = int(args.n)
        if not (1 <= n <= N_VIEWS_TOTAL):
            raise SystemExit("--n must be in 1..%d, got %d" % (N_VIEWS_TOTAL, n))
        return sorted(rng.sample(range(N_VIEWS_TOTAL), n))
    if mode == "random-random":
        maxv = int(args.max_views)
        if not (1 <= maxv <= N_VIEWS_TOTAL):
            raise SystemExit("--max-views must be in 1..%d, got %d"
                             % (N_VIEWS_TOTAL, maxv))
        k = rng.randint(1, maxv)
        return sorted(rng.sample(range(N_VIEWS_TOTAL), k))
    raise SystemExit("unknown angle mode %r (use all|exact|random-n|random-random)" % mode)


def pick_angles_interactive(rng: random.Random) -> list[int]:
    """Interactive angle-mode prompt. Returns the angle index list."""
    while True:
        try:
            mode = input("angle mode [all/exact/random-n/random-random]: ").strip().lower()
        except EOFError:
            mode = ""
        if mode in ("", "all"):
            return list(range(N_VIEWS_TOTAL))
        if mode == "exact":
            try:
                raw = input("angles (comma-separated 0..23, e.g. 0,3,5): ").strip()
            except EOFError:
                continue
            try:
                idxs = sorted(set(int(t) for t in raw.split(",") if t.strip()))
            except ValueError:
                print("  expected comma-separated integers")
                continue
            if any(not (0 <= i < N_VIEWS_TOTAL) for i in idxs):
                print("  angles must be in 0..%d" % (N_VIEWS_TOTAL - 1))
                continue
            return idxs
        if mode == "random-n":
            try:
                n = int(input("N (number of random angles): ").strip())
            except (EOFError, ValueError):
                continue
            if not (1 <= n <= N_VIEWS_TOTAL):
                print("  N must be in 1..%d" % N_VIEWS_TOTAL)
                continue
            return sorted(rng.sample(range(N_VIEWS_TOTAL), n))
        if mode == "random-random":
            try:
                maxv = int(input("max count (pick 1..max random angles): ").strip())
            except (EOFError, ValueError):
                continue
            if not (1 <= maxv <= N_VIEWS_TOTAL):
                print("  max must be in 1..%d" % N_VIEWS_TOTAL)
                continue
            k = rng.randint(1, maxv)
            return sorted(rng.sample(range(N_VIEWS_TOTAL), k))
        print("  expected all|exact|random-n|random-random")


# ---------------------------------------------------------------------------
# Scene data for the overlay (local loader — baseline_adapter has no exported
# loader, so this reads the scene JSONs directly; frozen files untouched).
# ---------------------------------------------------------------------------

def _scene_dir(root: str, seed: int) -> str:
    return os.path.join(root, "scenes", "%02d" % (int(seed) // 100), "%05d" % int(seed))


def _load_scene_data(root: str, seed: int) -> tuple[dict, dict]:
    """Return (ground_truth, cameras) dicts for a seed (PATCH-2 layout)."""
    sd = _scene_dir(root, seed)
    gt_path = os.path.join(sd, "ground_truth.json")
    cam_path = os.path.join(sd, "cameras.json")
    if not (os.path.isfile(gt_path) and os.path.isfile(cam_path)):
        raise SystemExit("scene data missing for seed %d (looked in %s)" % (int(seed), sd))
    with open(gt_path) as f:
        gt = json.load(f)
    with open(cam_path) as f:
        cam = json.load(f)
    return gt, cam


def camera_positions(cam: dict, view_idxs: list[int]) -> np.ndarray:
    """(V, 3) world ENU camera positions for the selected views (c2w[:3,3])."""
    pos = []
    for v in view_idxs:
        c2w = np.asarray(cam["views"][v]["c2w"], dtype=np.float64)
        pos.append(c2w[:3, 3])
    return np.asarray(pos, dtype=np.float64) if pos else np.empty((0, 3))


def selected_view_angles(cam: dict, view_idxs: list[int]) -> list[dict]:
    """Metadata for the selected views (angle_idx, tier, elevation, azimuth)."""
    out = []
    for v in view_idxs:
        w = cam["views"][v]
        out.append({"angle_idx": int(w["angle_idx"]), "tier": w["tier"],
                    "elevation_deg": float(w["elevation_deg"]),
                    "azimuth_deg": float(w["azimuth_deg"])})
    return out


# ---------------------------------------------------------------------------
# Matching (visualisation only — never feeds the frozen metrics)
# ---------------------------------------------------------------------------

def _match_pred_to_true(pred: np.ndarray, true: np.ndarray):
    """Unrestricted Hungarian match of pred rows to true rows (like metrics'
    `_median_matched_error` convention). Returns (rows, cols, ghost, missed)
    where rows/cols are matched pred/true indices and ghost/missed are the
    unmatched indices. Purely for drawing error vectors / ghosts / missed."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    finite = np.isfinite(pred).all(axis=1)
    pred = pred[finite]
    K, N = pred.shape[0], true.shape[0]
    if K == 0 or N == 0:
        return np.empty((0,), dtype=int), np.empty((0,), dtype=int), \
            np.arange(K), np.arange(N)
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    D = cdist(pred, true)
    rows, cols = linear_sum_assignment(D)
    pred_used = set(int(r) for r in rows)
    true_used = set(int(c) for c in cols)
    ghost = np.array([i for i in range(K) if i not in pred_used], dtype=int)
    missed = np.array([i for i in range(N) if i not in true_used], dtype=int)
    return rows, cols, ghost, missed


# ---------------------------------------------------------------------------
# 3D overlay plot
# ---------------------------------------------------------------------------

def plot_overlay(pred: np.ndarray, true: np.ndarray, cam_pos: np.ndarray,
                 view_idxs: list[int], out_path: str, title: str,
                 note: str | None = None) -> str:
    """Save a 3D overlay PNG: GT green, pred blue, red error vectors between
    Hungarian-matched pairs, ghosts (orange x), missed (hollow gray), and the
    selected camera positions (gray triangles). Returns the saved path."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)

    rows, cols, ghost_idx, missed_idx = _match_pred_to_true(pred, true)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Ground truth (green) and predictions (blue).
    if true.shape[0]:
        ax.scatter(true[:, 0], true[:, 1], true[:, 2], c="#1db954", s=42,
                   depthshade=False, label="ground truth (%d)" % true.shape[0])
    if pred.shape[0]:
        ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c="#4488ff", s=30,
                   depthshade=False, label="predicted (%d)" % pred.shape[0])

    # Error vectors between matched pairs (red lines).
    for r, c in zip(rows, cols):
        ax.plot([true[c, 0], pred[r, 0]], [true[c, 1], pred[r, 1]],
                [true[c, 2], pred[r, 2]], c="#d62728", lw=1.0, alpha=0.8)
    if len(rows):
        ax.plot([], [], [], c="#d62728", lw=1.0, label="error vectors (%d)" % len(rows))

    # Ghosts: unmatched predictions (orange x).
    if len(ghost_idx):
        g = pred[ghost_idx]
        ax.scatter(g[:, 0], g[:, 1], g[:, 2], c="#ff7f0e", marker="x", s=80,
                   depthshade=False, label="ghosts (%d)" % len(ghost_idx))

    # Missed: unmatched ground truth (hollow gray circles).
    if len(missed_idx):
        m = true[missed_idx]
        ax.scatter(m[:, 0], m[:, 1], m[:, 2], facecolors="none", edgecolors="#666666",
                   s=110, depthshade=False, label="missed (%d)" % len(missed_idx))

    # Camera positions (gray triangles) for the selected views.
    if cam_pos.shape[0]:
        ax.scatter(cam_pos[:, 0], cam_pos[:, 1], cam_pos[:, 2], c="#888888",
                   marker="^", s=60, depthshade=False,
                   label="cameras (%d)" % cam_pos.shape[0])

    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_zlabel("Up (m)")
    ax.set_title(title)

    # Square-ish aspect from the data bounds (all plotted points + cameras).
    all_pts = np.vstack([p for p in (true, pred, cam_pos) if p.shape[0]])
    if all_pts.shape[0]:
        lo = all_pts.min(axis=0)
        hi = all_pts.max(axis=0)
        span = np.maximum(hi - lo, 1e-9)
        ax.set_box_aspect((float(span[0]), float(span[1]), float(span[2])))

    ax.legend(loc="upper left", fontsize=9)
    if note:
        fig.text(0.02, 0.02, note, fontsize=8, color="#444444")

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Metrics printing
# ---------------------------------------------------------------------------

def _fmt_nan(v) -> str:
    if isinstance(v, float) and np.isnan(v):
        return "n/a"
    return "%.4f" % v


def print_metrics(result: dict) -> None:
    """Print the frozen metric dict plus run context in a readable block."""
    m = result["metrics"]
    print("=== Reconstruction metrics (ml.metrics.evaluate) ===")
    print("  n_true        : %d" % m["n_true"])
    print("  n_pred        : %d" % m["n_pred"])
    print("  count_err     : %+d" % m["count_err"])
    print("  mAP           : %s" % _fmt_nan(m["mAP"]))
    print("  median_err_m  : %s m" % _fmt_nan(m["median_err_m"]))
    print("  chamfer_m     : %s m" % _fmt_nan(m["chamfer_m"]))
    print("  per-tau (precision / recall / f1 / ap / n_matched):")
    for tau in _TAU_ORDER:
        pt = m["per_tau"].get(str(tau)) or m["per_tau"].get(tau)
        if pt is None:
            continue
        print("    tau=%-4s  precision=%.4f recall=%.4f f1=%.4f ap=%.4f n_matched=%d"
              % (tau, pt["precision"], pt["recall"], pt["f1"], pt["ap"], pt["n_matched"]))
    if result.get("detector_recall") is not None:
        print("  detector_recall (24 views, %.1f px): %.4f"
              % (result.get("recall_radius_px", DEFAULT_RECALL_RADIUS_PX),
                 result["detector_recall"]))
    if result.get("n_tracks") is not None:
        print("  n_tracks      : %d" % result["n_tracks"])
    if result.get("empty_reason"):
        print("  note          : %s" % result["empty_reason"])
    print("  wall_clock_s  : %.2f" % result.get("wall_clock_s", float("nan")))


# ---------------------------------------------------------------------------
# Main / CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Geometric method — multi-view triangulation (no learned "
                    "model).  Frozen geometric baseline on rendered ML scenes "
                    "(~/swarm_ml).")
    p.add_argument("--root", default=DEFAULT_ROOT,
                   help="data root with manifest.jsonl + scenes/ (default %s)"
                        % DEFAULT_ROOT)
    p.add_argument("--list", action="store_true",
                   help="list scenes from the manifest and exit")
    p.add_argument("--split", default=None, help="filter list by split (test/train/val)")
    p.add_argument("--cell", default=None, help="filter list by cell (primary/secondary)")
    p.add_argument("--limit", type=int, default=40, help="max rows when --list (default 40)")

    p.add_argument("--scene", type=int, default=None,
                   help="scene seed to reconstruct (0..4999); omit for interactive")
    p.add_argument("--mode", default="all",
                   help="angle mode: all (default) | exact | random-n | random-random")
    p.add_argument("--angles", default=None,
                   help="comma-separated angle indices 0..23 for --mode exact")
    p.add_argument("--n", type=int, default=6,
                   help="N random angles for --mode random-n (default 6)")
    p.add_argument("--max-views", type=int, default=8,
                   help="max count for --mode random-random (default 8)")
    p.add_argument("--rng-seed", type=int, default=0,
                   help="RNG seed for random angle modes (default 0)")

    p.add_argument("--recall-radius-px", type=float, default=DEFAULT_RECALL_RADIUS_PX,
                   help="detector-recall match radius in px (default 5.0)")
    p.add_argument("--out", default=None,
                   help="output PNG path (default ml/recon_overlay_<seed>_<n>.png)")
    p.add_argument("--no-plot", action="store_true",
                   help="skip the 3D overlay plot (metrics only)")
    return p.parse_args(argv)


def _fmt_views(view_idxs: list[int], views_meta: list[dict]) -> str:
    if views_meta and len(views_meta) == len(view_idxs) and all(
            m["angle_idx"] == v for m, v in zip(views_meta, view_idxs)):
        detail = ", ".join("angle %02d %s(%+.0f deg elev)" % (
            m["angle_idx"], m["tier"], m["elevation_deg"]) for m in views_meta)
    else:
        detail = ", ".join("angle %02d" % v for v in view_idxs)
    return detail


def run_recon(args) -> int:
    """Select scene + angles, run the frozen baseline, print, plot."""
    scenes = load_manifest(args.root)
    if args.scene is None:
        scene = pick_scene_interactive(scenes)
        rng = random.Random(args.rng_seed)
        view_idxs = pick_angles_interactive(rng)
    else:
        scene = find_scene(scenes, args.scene)
        rng = random.Random(args.rng_seed)
        view_idxs = choose_angles(args.mode, rng, args)

    seed = int(scene["seed"])

    # Scene data (read directly for display only — frozen JSONs untouched).
    gt, cam = _load_scene_data(args.root, seed)
    true = np.asarray(gt["positions"], dtype=np.float64)

    print("\n═══ GEOMETRIC METHOD — multi-view triangulation (no learned "
          "model) ═══")
    print("")
    print("Scene: " + scene_metadata_line(scene))
    print("Angles: %d views -> %s"
          % (len(view_idxs), _fmt_views(view_idxs, selected_view_angles(cam, view_idxs))))
    if len(view_idxs) < 2:
        print("WARNING: fewer than 2 views cannot triangulate; the baseline "
              "will return an empty reconstruction (by design).")
    print("Running frozen geometric baseline (detect -> correspond -> "
          "triangulate -> metrics)...", flush=True)

    result = process_scene(args.root, seed, view_idxs, args.recall_radius_px)
    result["recall_radius_px"] = args.recall_radius_px

    print("")
    print_metrics(result)

    n_pred_metric = result["metrics"]["n_pred"]
    # For the overlay, replay the frozen pipeline to recover the predicted 3D
    # positions: `process_scene` returns only the metric dict, not positions.
    # This is display-only (never scored again); the frozen metrics printed
    # above come exclusively from `process_scene`.
    pred_positions = _overlay_pred_positions(args.root, seed, view_idxs, cam=cam)
    cam_pos = camera_positions(cam, view_idxs)

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "recon_overlay_%05d_v%d.png" % (seed, len(view_idxs)))
    note = None
    if not args.no_plot:
        title = ("Reconstruction seed %d (%s, %s) | %d/%d views | median err %s m"
                 % (seed, scene.get("split", "?"), scene.get("cell", "?"),
                    len(view_idxs), N_VIEWS_TOTAL,
                    _fmt_nan(result["metrics"]["median_err_m"])))
        note = ("GT green, predicted blue, red = matched error vectors, orange x = "
                "ghosts, hollow gray = missed, gray triangles = cameras")
        if not os.path.isdir(os.path.dirname(out_path)):
            raise SystemExit("output directory missing: %s" % os.path.dirname(out_path))
        plot_overlay(pred_positions, true, cam_pos, view_idxs, out_path,
                     title, note=note)
        print("")
        print("3D overlay saved to: %s" % out_path)

    print("pred count from frozen metrics: %d (overlay draws %d)"
          % (n_pred_metric, len(pred_positions)))
    return 0


def _overlay_pred_positions(root: str, seed: int, view_idxs: list[int],
                            cam: dict | None = None) -> np.ndarray:
    """Replay the frozen pipeline to recover predicted 3D positions for the plot.

    `process_scene` returns only the metric dict, not positions, so the overlay
    replays the exact frozen sub-steps the adapter wires up (detect on all 24
    views -> subset -> epipolar correspondence -> DLT triangulation), using the
    adapter's internal helpers and its frozen constants
    (EPIPOLAR_THRESHOLD_PX etc.) so the positions are bit-identical to what
    `process_scene` triangulates. Display-only: the printed metrics come
    exclusively from `process_scene`/`ml.metrics.evaluate`."""
    import ml.baseline_adapter as ba  # read-only; no frozen file is modified

    sd = ba.scene_dir_for_seed(os.path.expanduser(root), seed)
    if cam is None:
        cam = _load_scene_data(os.path.expanduser(root), seed)[1]

    rig = ba._build_rig(cam)
    standoff_m = float(cam["standoff_m"])
    all_view_idxs = list(range(len(cam["views"])))
    dets = ba._detect_views(sd, cam, rig, standoff_m, all_view_idxs)
    if len(view_idxs) < 2:
        return np.empty((0, 3), dtype=np.float64)
    if len(view_idxs) == len(all_view_idxs):
        rig_sel, dets_sel = rig, dets
    else:
        rig_sel = ba._subset_rig(rig, view_idxs)
        dets_sel = ba.Detections(
            points_per_view=[dets.points_per_view[i] for i in view_idxs],
            image_size=ba.IMAGE_SIZE,
        )
    tracks = ba.solve_correspondence(
        detections=dets_sel, rig=rig_sel,
        epipolar_threshold=ba.EPIPOLAR_THRESHOLD_PX,
    )
    recon = ba.triangulate_dlt(tracks, rig_sel, dets_sel)
    return np.asarray(recon.positions_3d, dtype=np.float64)


def cmd_list(args) -> int:
    """--list: print manifest scenes filtered by --split/--cell, capped by --limit."""
    scenes = load_manifest(args.root)
    pool = scenes
    if args.split:
        pool = [s for s in pool if s.get("split") == args.split]
    if args.cell:
        pool = [s for s in pool if s.get("cell") == args.cell]
    print("Scenes (split=%s cell=%s): %d shown of %d in manifest"
          % (args.split or "all", args.cell or "all", min(len(pool), args.limit),
             len(pool)))
    for s in pool[: args.limit]:
        print("  " + scene_metadata_line(s))
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    root = os.path.expanduser(args.root)
    if args.list:
        return cmd_list(args)
    return run_recon(args)


if __name__ == "__main__":
    sys.exit(main())
