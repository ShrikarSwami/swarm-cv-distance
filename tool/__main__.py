#!/usr/bin/env python3
"""Swarm Reconstruction Tool — CLI entry point.

Usage:
    python -m tool --list
    python -m tool --scene 0 --mode all --d-max 15.0
    python -m tool --scene 5 --mode random-n --n 8 --backend geometric
    python -m tool --scene 10 --mode exact --angles 0,3,7,12 --export results/

Modes:
    all            — all 24 camera angles
    exact          — user-specified comma-separated angle indices
    random-n       — N random angles
    random-random  — k random angles where k ~ U(1, max_views)

Backends:
    geometric      — blob detection + epipolar correspondence + DLT (default)
    learned        — T6 voxel-fusion model (NOT YET PASSING G2)
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# sys.path — make the repo root importable for frozen pipeline modules
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tool.model_interface import list_backends, reconstruct
from tool.scene_loader import (
    list_scenes, load_images, load_cameras, load_ground_truth, camera_positions,
)
from tool.adjacency import compute_adjacency, adjacency_stats
from tool.visualizer import plot_reconstruction_overlay
from tool.export import export_results

# Import the frozen metrics module
_STAGE1 = os.path.join(_REPO_ROOT, "stage1_geometry")
if _STAGE1 not in sys.path:
    sys.path.insert(0, _STAGE1)
from ml.metrics import evaluate

N_VIEWS_TOTAL = 24
DEFAULT_D_MAX = 15.0  # metres (assumed comms range)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Swarm Reconstruction Tool — 3D drone positions from "
                    "multi-camera views, plus adjacency matrix for GA/PSO.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tool --list
  python -m tool --scene 0 --mode all --d-max 15.0
  python -m tool --scene 5 --mode random-n --n 8 --export results/
  python -m tool --scene 5 --mode exact --angles 0,3,7,12,18

Backends:
  geometric  — blob detection + epipolar correspondence + DLT (DEFAULT)
  learned    — T6 voxel-fusion model (NOT YET PASSING G2)
""",
    )

    # Scene selection
    p.add_argument("--list", action="store_true",
                   help="list bundled scenes and exit")
    p.add_argument("--split", default=None,
                   help="filter --list by split (e.g. test)")
    p.add_argument("--scene", type=int, default=None,
                   help="scene seed to reconstruct (0-19); omit for interactive")

    # Angle / view selection
    p.add_argument("--mode", default="all",
                   help="angle mode: all | exact | random-n | random-random "
                        "(default: all)")
    p.add_argument("--angles", default=None,
                   help="comma-separated angle indices 0..23 for --mode exact")
    p.add_argument("--n", type=int, default=8,
                   help="N random angles for --mode random-n (default: 8)")
    p.add_argument("--max-views", type=int, default=12,
                   help="max views for --mode random-random (default: 12)")
    p.add_argument("--rng-seed", type=int, default=0,
                   help="RNG seed for random modes (default: 0)")

    # Backend
    p.add_argument("--backend", default="geometric",
                   help="reconstruction backend (default: geometric)")

    # Adjacency
    p.add_argument("--d-max", type=float, default=DEFAULT_D_MAX,
                   help="comms-range threshold for adjacency matrix in metres "
                        "(default: %.1f)" % DEFAULT_D_MAX)

    # Output
    p.add_argument("--out", default=None,
                   help="output overlay PNG path (default: tool/recon_<scene>_v<N>.png)")
    p.add_argument("--no-plot", action="store_true",
                   help="skip the 3D overlay plot")
    p.add_argument("--export", default=None,
                   help="export directory for adjacency matrix + positions + metrics")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Angle selection
# ---------------------------------------------------------------------------

def choose_angles(mode: str, rng: random.Random, args: argparse.Namespace) -> list[int]:
    """Return sorted list of angle indices 0..23 for the selected mode."""
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
        if not (1 <= args.n <= N_VIEWS_TOTAL):
            raise SystemExit("--n must be in 1..%d, got %d" % (N_VIEWS_TOTAL, args.n))
        return sorted(rng.sample(range(N_VIEWS_TOTAL), args.n))
    if mode == "random-random":
        if not (1 <= args.max_views <= N_VIEWS_TOTAL):
            raise SystemExit("--max-views must be in 1..%d, got %d"
                             % (N_VIEWS_TOTAL, args.max_views))
        k = rng.randint(1, args.max_views)
        return sorted(rng.sample(range(N_VIEWS_TOTAL), k))
    raise SystemExit("unknown angle mode %r (use all|exact|random-n|random-random)" % mode)


# ---------------------------------------------------------------------------
# Interactive prompts
# ---------------------------------------------------------------------------

def pick_scene_interactive(scenes: list[dict]) -> str:
    """Prompt for a scene."""
    print("Bundled scenes: %d" % len(scenes))
    for s in scenes[:20]:
        print("  %s  n_drones=%d  split=%s"
              % (s["id"], s["n_drones"], s.get("split", "?")))
    while True:
        try:
            raw = input("scene ID (e.g. 00000, or 'q' to quit): ").strip()
        except EOFError:
            raise SystemExit("no input")
        if raw.lower() in ("q", "quit", "exit"):
            raise SystemExit("aborted")
        ids = {s["id"] for s in scenes}
        if raw in ids:
            return raw
        print("  scene %r not found; available: %s" % (raw, ", ".join(sorted(ids)[:10])))


def pick_angles_interactive(rng: random.Random) -> list[int]:
    """Interactive angle-mode prompt."""
    while True:
        try:
            mode = input("angle mode [all/exact/random-n/random-random]: ").strip().lower()
        except EOFError:
            mode = ""
        if mode in ("", "all"):
            return list(range(N_VIEWS_TOTAL))
        if mode == "exact":
            try:
                raw = input("angles (comma-separated 0..23): ").strip()
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
                maxv = int(input("max count (1..24): ").strip())
            except (EOFError, ValueError):
                continue
            if not (1 <= maxv <= N_VIEWS_TOTAL):
                print("  max must be in 1..%d" % N_VIEWS_TOTAL)
                continue
            k = rng.randint(1, maxv)
            return sorted(rng.sample(range(N_VIEWS_TOTAL), k))
        print("  expected all|exact|random-n|random-random")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _fmt_views(view_indices: list[int], cameras: list[dict]) -> str:
    """Format view list for display."""
    parts = []
    for v, c in zip(view_indices, cameras):
        parts.append("angle %02d %s(%+.0f° elev)"
                     % (c["angle_idx"], c["tier"], c["elevation_deg"]))
    return ", ".join(parts)


def cmd_list(args: argparse.Namespace) -> int:
    """--list: print bundled scenes."""
    scenes = list_scenes(split=args.split)
    print("Bundled scenes (%s): %d" % (args.split or "all", len(scenes)))
    for s in scenes:
        print("  %s  n_drones=%d  split=%s" % (s["id"], s["n_drones"], s.get("split", "?")))
    return 0


def run_recon(args: argparse.Namespace) -> int:
    """Select scene + angles, reconstruct, evaluate, export."""
    scenes = list_scenes()
    rng = random.Random(args.rng_seed)

    if args.scene is None:
        scene_id = pick_scene_interactive(scenes)
        view_idxs = pick_angles_interactive(rng)
    else:
        scene_id = "%05d" % args.scene
        valid_ids = {s["id"] for s in scenes}
        if scene_id not in valid_ids:
            raise SystemExit("scene %s not found; bundled IDs: %s"
                             % (scene_id, ", ".join(sorted(valid_ids))))
        view_idxs = choose_angles(args.mode, rng, args)

    # Validate backend
    backends = list_backends()
    if args.backend not in backends:
        raise SystemExit("unknown backend %r; available: %s"
                         % (args.backend, ", ".join(backends.keys())))

    print("")
    print("═══ Swarm Reconstruction Tool v1.0 ═══")
    print("")
    print("Scene:    %s" % scene_id)
    print("Backend:  %s — %s" % (args.backend, backends[args.backend]))
    print("Views:    %d — %s" % (len(view_idxs), ",".join(str(v) for v in view_idxs)))
    if len(view_idxs) < 2:
        print("WARNING: fewer than 2 views cannot triangulate.")
    print("d_max:    %.2f m" % args.d_max)
    print("")

    # Load scene data
    print("Loading scene data...", end=" ", flush=True)
    images = load_images(scene_id, view_idxs)
    cameras = load_cameras(scene_id, view_idxs)
    true_positions, n_drones = load_ground_truth(scene_id)
    print("done (%d images, %d ground-truth drones)" % (len(images), n_drones))

    # Run reconstruction
    t_start = time.perf_counter()
    print("Reconstructing via %s backend..." % args.backend, end=" ", flush=True)
    pred_positions, confidences = reconstruct(images, cameras, backend=args.backend)
    wall_s = time.perf_counter() - t_start
    print("done (%.2f s, %d positions)" % (wall_s, len(pred_positions)))

    # Evaluate against ground truth (frozen metric)
    metrics = evaluate(pred_positions, true_positions)

    # Compute adjacency matrix
    adjacency = compute_adjacency(pred_positions, args.d_max)
    adj_stats = adjacency_stats(adjacency)

    # --- Print results ---
    print("")
    print("═══ Reconstruction Metrics (frozen ml.metrics.evaluate) ═══")
    print("  n_true:       %d" % metrics["n_true"])
    print("  n_pred:       %d" % metrics["n_pred"])
    print("  count_err:    %+d" % metrics["count_err"])
    print("  median_err:   %.4f m" % metrics["median_err_m"])
    print("  chamfer:      %.4f m" % metrics["chamfer_m"])
    print("  mAP:          %.4f" % metrics["mAP"])
    print("  wall_clock:   %.2f s" % wall_s)
    print("")
    for tau in sorted(metrics["per_tau"].keys(), key=float):
        pt = metrics["per_tau"][tau]
        print("  tau=%-4s  precision=%.4f  recall=%.4f  f1=%.4f  n_matched=%d"
              % (tau, pt["precision"], pt["recall"], pt["f1"], pt["n_matched"]))

    print("")
    print("═══ Adjacency Matrix (d_max = %.2f m) ═══" % args.d_max)
    print("  n_nodes:      %d" % adj_stats["n_nodes"])
    print("  n_edges:      %d" % adj_stats["n_edges"])
    print("  density:      %.3f" % adj_stats["edge_density"])
    print("  mean_degree:  %.1f" % adj_stats["mean_degree"])
    print("  min_degree:   %d" % adj_stats["min_degree"])
    print("  max_degree:   %d" % adj_stats["max_degree"])
    print("  n_isolated:   %d" % adj_stats["n_isolated"])
    print("  n_components: %d" % adj_stats["n_components"])
    print("")
    print("  GA/PSO input:  adjacency.json  (edge list format)")
    print("  -> drop-in replacement for simulation-derived adjacency matrix")

    # Export
    if args.export:
        print("")
        print("Exporting results to %s ..." % args.export, end=" ", flush=True)
        saved = export_results(
            positions=pred_positions,
            adjacency=adjacency,
            true_positions=true_positions,
            metrics=metrics,
            out_dir=args.export,
            scene_id=scene_id,
            d_max=args.d_max,
            backend=args.backend,
        )
        print("done (%d files)" % len(saved))
        for name, path in sorted(saved.items()):
            print("  %s -> %s" % (name, path))

    # 3D overlay plot
    if not args.no_plot:
        cam_pos = camera_positions(cameras)
        out_path = args.out or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "recon_%s_v%d.png" % (scene_id, len(view_idxs)))
        title = ("Reconstruction %s (%s, %d/%d views) | median err %.4f m"
                 % (scene_id, args.backend, len(view_idxs), N_VIEWS_TOTAL,
                    metrics["median_err_m"]))
        plot_reconstruction_overlay(
            pred_positions, true_positions, cam_pos, view_idxs,
            out_path, title=title, metrics=metrics)
        print("")
        print("3D overlay saved to: %s" % out_path)

    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.list:
        return cmd_list(args)
    return run_recon(args)


if __name__ == "__main__":
    sys.exit(main())
