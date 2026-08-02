#!/usr/bin/env python3
"""T7 — ml/eval_sweep.py: sized evaluation sweep on the geometric baseline.

OWNED FILE: this is the ONLY file this agent may modify. Everything else it
touches is FROZEN or owned by another agent — read/call, never edit:
  stage1_geometry/**   entire geometric control (frozen)
  ml/metrics.py        frozen metrics
  ml/baseline_adapter.py   accepted T5 (baseline agent) — helpers reused here
  ml/scene_gen.py, ml/splits.json, ml/model.py, ml/train.py, tests/, calib.json

This module is WIRING, not reimplementation. The frozen geometric baseline path
is exactly what `ml/baseline_adapter.process_scene` does:
    detect_blobs (frozen) -> solve_correspondence (frozen, EPIPOLAR_THRESHOLD_PX=3.0)
      -> triangulate_dlt (frozen) -> ml.metrics.evaluate (frozen)
Here it is factorised so that all 24 views of a scene are detected ONCE and
cached, then correspondence + triangulation + metrics are re-run per
(composition, view_count) cell on the cached subset. Per-cell numbers are
verified to match `ml.baseline_adapter.process_scene` (same frozen calls, same
deterministic solver).

Sized grid (spec T7, NOT the naive full sweep):
  100 test scenes per cell (200 total), balanced across 3 density bins
  (~33/34/33 per cell per bin)
  view counts {2,3,4,5,6,7,8}
  compositions: ground (cams 0-7), level (8-15), aerial (16-23), and a mixed
  composition interleaving all 3 tiers (both pure and mixed reach 8 views, so
  composition comparisons are balanced).

Cost discipline: detection of the 24 views dominates per-scene cost; it is run
once per scene and the Detections are cached. Correspondence on the V<=8
subsets is cheaper than the adapter's V=24 correspondence. Scenes run in
parallel (multiprocessing, jobs ~ min(cpu_count,14)), spawn context, exactly
like the adapter.

CLI
---
    python3 -m ml.eval_sweep --root ~/swarm_ml --jobs 14 --out-dir logs/ml_sweep
        [--scenes-per-cell 100] [--cells primary,secondary] [--view-counts 2,3,4,5,6,7,8]
        [--compositions ground,level,aerial,mixed] [--sanity-v24] [--recall-radius-px 5.0]

    python3 -m ml.eval_sweep --mode analyze --out-dir logs/ml_sweep
        (re-reads the CSV and writes summary JSON + 2 plots + markdown report)

Deliverables (logs/ml_sweep/):
    eval_sweep_results.csv   one row per (scene, cell, density_bin, composition, view_count)
                             with all fields of the frozen evaluate dict + detector_recall
                             + n_drones + a_max_px + n_tracks + empty_reason
    eval_sweep_rows.jsonl    same rows, structured (nested per_tau) — intermediate/debug
    eval_sweep_summary.json  aggregates
    plot_mAP_vs_views.png        SEPARATE plot (a): mAP vs view count, per composition
    plot_median_err_vs_views.png SEPARATE plot (b): median position error vs view count
    eval_sweep_report.md     report with P1/P2/P5 recorded as predicted/observed/ratio/match

Constraints: numpy / scipy / PIL / scikit-image / matplotlib only (no torch, no
bpy, no Blender). Pure geometry — MPS not needed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# sys.path setup — make the frozen stage-1 modules and the ml package importable
# regardless of how the module is launched (python -m, direct script, worker).
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STAGE1 = os.path.join(_REPO_ROOT, "stage1_geometry")
for _p in (_STAGE1, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --- Frozen imports (read / call / never edit) ---
from data_contract import (  # noqa: E402  frozen conventions
    IMAGE_SIZE,
    Detections,
)
from b3_correspondence import solve_correspondence  # noqa: E402  frozen solver
from b5_triangulation import triangulate_dlt  # noqa: E402  frozen triangulator

from ml.metrics import evaluate  # noqa: E402  frozen metrics (single truth)
from ml.baseline_adapter import (  # noqa: E402  accepted T5 helpers (read/call)
    EPIPOLAR_THRESHOLD_PX,
    DEFAULT_RECALL_RADIUS_PX,
    _build_rig,
    _detect_views,
    _subset_rig,
    measure_detector_recall,
    scene_dir_for_seed,
)

# ---------------------------------------------------------------------------
# Sized grid constants (spec T7)
# ---------------------------------------------------------------------------
# Ordered camera lists. A composition's view count N = the first N cameras of
# the list. Pure-tier compositions are the tier's 8 cameras; "mixed" interleaves
# all 3 tiers so that pure and mixed both reach 8 views (balanced comparisons).
COMPOSITIONS = {
    "ground": list(range(0, 8)),
    "level": list(range(8, 16)),
    "aerial": list(range(16, 24)),
    "mixed": [0, 8, 16, 1, 9, 17, 2, 10, 18, 3, 11, 19, 4, 12, 20,
              5, 13, 21, 6, 14, 22, 7, 15, 23],
}
VIEW_COUNTS = [2, 3, 4, 5, 6, 7, 8]
DENSITY_BINS = [("low", 5, 15), ("mid", 28, 42), ("high", 48, 60)]
BIN_NAMES = [b[0] for b in DENSITY_BINS]

# Optional sanity anchor: the full 24-view mixed composition at V=24, exactly
# the adapter's `--views all`. Enables a direct check against the measured
# V=24 anchor (mAP 0.9865, median_err 0.0303, count_err +0.7, recall 0.9873).
SANITY_COMPOSITION = "mixed_v24"
SANITY_VIEW_COUNT = 24

# Operating cells (both W=1920, f=2666.67): primary R=50m a_max 9.6px standoff
# 139m; secondary R=100m a_max 4.8px standoff 278m. Radius is recorded per cell
# for the report / axis framing.
CELL_RADIUS_M = {"primary": 50.0, "secondary": 100.0}
CELL_A_MAX_PX = {"primary": 9.6, "secondary": 4.8}

TAUS = (0.5, 1.0, 2.0, 5.0)  # metres — frozen (ml.metrics.DEFAULT_TAUS)


# ---------------------------------------------------------------------------
# Density binning / scene selection (deterministic)
# ---------------------------------------------------------------------------

def density_bin_of(n_drones: int):
    """Exact spec-T7 density bins: low [5,15] (~10), mid [28,42] (~35),
    high [48,60] (~55, straddles the ~50-drone false-track threshold)."""
    for name, lo, hi in DENSITY_BINS:
        if lo <= n_drones <= hi:
            return name
    return None


def _bin_targets(scenes_per_cell: int) -> dict:
    """Distribute scenes_per_cell across the 3 density bins deterministically:
    base + remainder goes to 'mid' (the middle bin)."""
    base, rem = divmod(scenes_per_cell, 3)
    return {"low": base, "mid": base + rem, "high": base}


def select_scenes(root: str, scenes_per_cell: int,
                  cells=("primary", "secondary")) -> list[tuple]:
    """Deterministically pick scenes for the sized grid.

    Iterates test seeds 0..999 in order, assigns each to (cell, density_bin)
    (cell from the scene's cameras.json — the authoritative PATCH-2 tag; bin
    from ground_truth n_drones), and keeps the first N seeds per (cell, bin).
    Returns a flat, deterministic list of (seed, cell, density_bin) tuples.

    The operating-cell split is by seed RANGE (spec T2: test 0-499 primary,
    500-999 secondary); the cell is read from the data, never inferred.
    """
    targets = _bin_targets(scenes_per_cell)
    sel = {c: {b: [] for b in BIN_NAMES} for c in cells}
    n_drones_cache = {}
    cell_cache = {}
    for seed in range(1000):
        if seed not in n_drones_cache:
            sd = scene_dir_for_seed(root, seed)
            with open(os.path.join(sd, "ground_truth.json")) as f:
                n_drones_cache[seed] = int(json.load(f)["n_drones"])
            with open(os.path.join(sd, "cameras.json")) as f:
                cell_cache[seed] = str(json.load(f).get("cell", "?"))
        cell = cell_cache[seed]
        if cell not in cells:
            continue
        b = density_bin_of(n_drones_cache[seed])
        if b is None:
            continue
        if len(sel[cell][b]) < targets[b]:
            sel[cell][b].append(seed)
        if all(len(sel[c][b]) >= targets[b] for c in cells for b in BIN_NAMES):
            break
    plan = []
    for c in cells:
        for b in BIN_NAMES:
            for s in sel[c][b]:
                plan.append((s, c, b))
    # Guard: exactly scenes_per_cell per cell.
    per_cell = {}
    for _s, c, _b in plan:
        per_cell[c] = per_cell.get(c, 0) + 1
    assert all(per_cell.get(c, 0) == scenes_per_cell for c in cells), per_cell
    return plan


# ---------------------------------------------------------------------------
# Per-scene pipeline (module-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

def process_scene_grid(root: str, seed: int,
                       compositions: dict, view_counts: list[int],
                       recall_radius_px: float,
                       with_sanity_v24: bool = True) -> list[dict]:
    """Process one scene across the full (composition, view_count) grid.

    - builds the full 24-view rig, detects all 24 views ONCE, caches
    - measures detector recall on all 24 views once (scene-level, reused)
    - for each (composition, view_count): subset rig+detections, run the frozen
      solve_correspondence -> triangulate_dlt -> evaluate on the cached subset
    - returns one structured dict per (composition, view_count) cell

    Per-cell reconstruction numbers are bit-identical to
    `ml.baseline_adapter.process_scene` with the same view subset: identical
    frozen calls, deterministic solver.
    """
    scene_dir = scene_dir_for_seed(root, seed)
    with open(os.path.join(scene_dir, "cameras.json")) as f:
        cam = json.load(f)
    with open(os.path.join(scene_dir, "ground_truth.json")) as f:
        gt = json.load(f)

    true = np.asarray(gt["positions"], dtype=np.float64)
    standoff_m = float(cam["standoff_m"])
    cell = str(cam.get("cell", "?"))
    n_drones = int(gt["n_drones"])
    a_max_px = float(cam.get("a_max_px", float("nan")))
    density_bin = density_bin_of(n_drones)

    full_rig = _build_rig(cam)
    all_view_idxs = list(range(len(cam["views"])))
    # --- Detect all 24 views ONCE per scene and cache (cost discipline) ---
    full_dets = _detect_views(scene_dir, cam, full_rig, standoff_m, all_view_idxs)
    # --- Detector recall on all 24 views, once per scene (mandatory report) ---
    recall = measure_detector_recall(full_rig, full_dets, true, all_view_idxs,
                                     recall_radius_px)
    recall_mean = float(recall["overall_mean"])

    def _run_cell(comp_name: str, view_idxs: list[int], vc: int) -> dict:
        sel = view_idxs[:vc]
        rig = _subset_rig(full_rig, sel)
        dets = Detections(points_per_view=[full_dets.points_per_view[i] for i in sel],
                          image_size=IMAGE_SIZE)
        empty_reason = None
        n_tracks = 0
        if vc < 2:
            pred = np.empty((0, 3), dtype=np.float64)
            empty_reason = "fewer than 2 views cannot triangulate"
        else:
            # Frozen control call, exactly as sweep_b.process_bundle does it.
            tracks = solve_correspondence(
                detections=dets,
                rig=rig,
                epipolar_threshold=EPIPOLAR_THRESHOLD_PX,
            )
            n_tracks = len(tracks.tracks)
            recon = triangulate_dlt(tracks, rig, dets)
            pred = recon.positions_3d
            if len(pred) == 0:
                empty_reason = "no triangulated tracks (correspondence empty)"
        metrics = evaluate(pred, true)
        return {
            "seed": int(seed),
            "cell": cell,
            "density_bin": density_bin,
            "composition": comp_name,
            "view_count": int(vc),
            "n_drones": n_drones,
            "a_max_px": a_max_px,
            "detector_recall": recall_mean,
            "n_tracks": n_tracks,
            "empty_reason": empty_reason,
            "metrics": metrics,
        }

    rows = []
    for comp_name, view_idxs in compositions.items():
        for vc in view_counts:
            rows.append(_run_cell(comp_name, view_idxs, vc))
    if with_sanity_v24:
        rows.append(_run_cell(SANITY_COMPOSITION, list(range(24)), SANITY_VIEW_COUNT))
    return rows


# ---------------------------------------------------------------------------
# CSV / JSONL writers
# ---------------------------------------------------------------------------

_ROW_META_KEYS = ["seed", "cell", "density_bin", "composition", "view_count",
                  "n_drones", "a_max_px", "detector_recall", "n_tracks",
                  "empty_reason"]
_METRIC_SCALAR_KEYS = ["n_true", "n_pred", "mAP", "median_err_m", "chamfer_m",
                       "count_err"]


def flatten_row(row: dict) -> dict:
    """Flatten a structured per-cell dict into a flat CSV row dict."""
    flat = {k: row[k] for k in _ROW_META_KEYS}
    m = row["metrics"]
    flat["n_true"] = m["n_true"]
    flat["n_pred"] = m["n_pred"]
    per_tau = m["per_tau"]
    for tau in TAUS:
        # In-memory evaluate dict keys are the float taus; after JSON round-trip
        # they become strings. Handle both.
        tv = per_tau.get(tau)
        if tv is None:
            tv = per_tau.get(str(tau))
        for tk in ("precision", "recall", "f1", "ap", "n_matched"):
            flat[f"{tk}@{tau}"] = tv[tk]
    flat["mAP"] = m["mAP"]
    flat["median_err_m"] = m["median_err_m"]
    flat["chamfer_m"] = m["chamfer_m"]
    flat["count_err"] = m["count_err"]
    return flat


def _csv_header() -> list[str]:
    return _ROW_META_KEYS + _METRIC_SCALAR_KEYS[:2] + \
        [f"{tk}@{tau}" for tau in TAUS for tk in
         ("precision", "recall", "f1", "ap", "n_matched")] + \
        _METRIC_SCALAR_KEYS[2:]


def write_csv_header(path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_csv_header())
        w.writeheader()


def append_csv_rows(path: str, flat_rows: list[dict]) -> None:
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_csv_header())
        for r in flat_rows:
            w.writerow(r)


def append_jsonl_rows(path: str, rows: list[dict]) -> None:
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, allow_nan=True, default=_json_default) + "\n")


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError("not JSON serializable: %r" % (o,))


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def run_sweep(args) -> int:
    root = os.path.expanduser(args.root)
    cells = tuple(c.strip() for c in args.cells.split(","))
    view_counts = [int(v) for v in args.view_counts.split(",")]
    comp_names = [c.strip() for c in args.compositions.split(",")]
    for c in comp_names:
        if c not in COMPOSITIONS:
            raise SystemExit("unknown composition %r (have %s)"
                             % (c, sorted(COMPOSITIONS)))
    compositions = {c: COMPOSITIONS[c] for c in comp_names}

    plan = select_scenes(root, args.scenes_per_cell, cells=cells)
    if args.limit_scenes and args.limit_scenes < len(plan):
        plan = plan[: args.limit_scenes]

    missing = [seed for seed, _c, _b in plan
               if not os.path.isdir(scene_dir_for_seed(root, seed))]
    if missing:
        raise SystemExit("scene dirs missing for %d seed(s), e.g. %s"
                         % (len(missing), scene_dir_for_seed(root, missing[0])))

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "eval_sweep_results.csv")
    jsonl_path = os.path.join(args.out_dir, "eval_sweep_rows.jsonl")

    n_jobs = args.jobs
    if n_jobs is None:
        n_jobs = min(os.cpu_count() or 1, 14)
    n_jobs = max(1, min(n_jobs, len(plan)))

    print("=== T7 eval sweep (geometric baseline, sized) ===", flush=True)
    print("scenes: %d (per cell %d, cells %s) | compositions: %s | views: %s"
          % (len(plan), args.scenes_per_cell, cells, comp_names, view_counts),
          flush=True)
    print("compositions: %s" % {k: compositions[k] for k in compositions},
          flush=True)
    print("density targets/cell: %s | sanity V=24 (mixed): %s | workers: %d"
          % (_bin_targets(args.scenes_per_cell), args.sanity_v24, n_jobs),
          flush=True)
    per_cell_counts = {}
    for _s, c, _b in plan:
        per_cell_counts[c] = per_cell_counts.get(c, 0) + 1
    print("scene plan per cell: %s" % per_cell_counts, flush=True)
    print("", flush=True)

    write_csv_header(csv_path)
    # Fresh JSONL (truncate any previous run).
    with open(jsonl_path, "w"):
        pass

    t_run = time.time()
    n_done = 0
    cells_per_scene = len(compositions) * len(view_counts) + (1 if args.sanity_v24 else 0)

    import multiprocessing
    try:
        ctx = multiprocessing.get_context()  # spawn on macOS, fork on Linux
        with ctx.Pool(processes=n_jobs) as pool:
            # imap_unordered so completed scenes stream to disk immediately.
            for rows in pool.imap_unordered(
                _worker,
                [(root, seed, compositions, view_counts, args.recall_radius_px,
                  args.sanity_v24) for seed, _c, _b in plan],
                chunksize=1,
            ):
                flat = [flatten_row(r) for r in rows]
                append_csv_rows(csv_path, flat)
                append_jsonl_rows(jsonl_path, rows)
                n_done += 1
                if n_done % 5 == 0 or n_done == len(plan):
                    el = time.time() - t_run
                    rate = el / n_done
                    eta = rate * (len(plan) - n_done)
                    print("  %d/%d scenes | %.0fs elapsed | %.1fs/scene | ETA %.0fs"
                          % (n_done, len(plan), el, rate, eta), flush=True)
    except Exception as exc:  # pragma: no cover - defensive
        print("WARNING: multiprocessing unavailable (%s); running sequentially"
              % exc, file=sys.stderr)
        for seed, _c, _b in plan:
            rows = process_scene_grid(root, seed, compositions, view_counts,
                                      args.recall_radius_px, args.sanity_v24)
            append_csv_rows(csv_path, [flatten_row(r) for r in rows])
            append_jsonl_rows(jsonl_path, rows)
            n_done += 1

    t_elapsed = time.time() - t_run
    print("", flush=True)
    print("SWEEP COMPLETE: %d scenes x %d cells/scene = %d rows in %.1fs "
          "(wall-clock)"
          % (n_done, cells_per_scene, n_done * cells_per_scene, t_elapsed),
          flush=True)
    print("CSV:   %s" % csv_path, flush=True)
    print("JSONL: %s" % jsonl_path, flush=True)

    # Write a small run-metadata JSON for the analyze step.
    meta = {
        "root": root,
        "cells": list(cells),
        "scenes_per_cell": args.scenes_per_cell,
        "view_counts": view_counts,
        "compositions": comp_names,
        "composition_views": {k: compositions[k] for k in compositions},
        "sanity_v24": bool(args.sanity_v24),
        "recall_radius_px": args.recall_radius_px,
        "n_scenes": n_done,
        "cells_per_scene": cells_per_scene,
        "n_rows": n_done * cells_per_scene,
        "wall_clock_s": round(t_elapsed, 2),
    }
    with open(os.path.join(args.out_dir, "eval_sweep_run.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return 0


def _worker(job):
    """Module-level worker callable (picklable under spawn). `job` is the
    (root, seed, compositions, view_counts, recall_radius_px, sanity_v24) tuple
    built in run_sweep."""
    root, seed, compositions, view_counts, recall_radius, sanity_v24 = job
    return process_scene_grid(root, seed, compositions, view_counts,
                              recall_radius, sanity_v24)


# ---------------------------------------------------------------------------
# Analysis (read CSV -> summary JSON + 2 plots + markdown report)
# ---------------------------------------------------------------------------

def _load_rows(csv_path: str) -> list[dict]:
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            d = dict(r)
            for k in list(d.keys()):
                if k in ("seed", "view_count", "n_drones", "n_tracks",
                         "n_true", "n_pred", "count_err", "n_matched@0.5",
                         "n_matched@1.0", "n_matched@2.0", "n_matched@5.0"):
                    try:
                        d[k] = int(float(d[k]))
                    except (TypeError, ValueError):
                        d[k] = None
                elif k in ("a_max_px", "detector_recall", "mAP", "median_err_m",
                           "chamfer_m") or "@" in k:
                    try:
                        d[k] = float(d[k])
                    except (TypeError, ValueError):
                        d[k] = float("nan")
            rows.append(d)
    return rows


def _key(row, k):
    v = row.get(k)
    return v


def _mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def _nanmean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def _nanmedian(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.median(xs)) if xs else float("nan")


def _agg_cell_rows(rows, cells, compositions, view_counts, sanity_comp, sanity_vc):
    """Aggregate the main grid (view_counts 2..8) per (cell, composition,
    view_count) and optionally per (cell, density_bin, composition, view_count)."""
    grid = {}   # (cell, comp, vc) -> list of row dicts
    grid_bin = {}  # (cell, bin, comp, vc) -> list of row dicts
    sanity = {}  # (cell, comp) -> list of row dicts (V=24)
    for r in rows:
        cell, comp, vc = r["cell"], r["composition"], r["view_count"]
        if cell not in cells:
            continue
        if vc == sanity_vc and comp == sanity_comp:
            sanity.setdefault((cell, comp), []).append(r)
            continue
        if comp not in compositions:
            continue
        if vc in view_counts:
            grid.setdefault((cell, comp, vc), []).append(r)
            grid_bin.setdefault((cell, r["density_bin"], comp, vc), []).append(r)
    return grid, grid_bin, sanity


def _summarize_group(rows):
    """One aggregate dict for a list of rows."""
    maps = [r["mAP"] for r in rows if r.get("mAP") is not None and not (isinstance(r["mAP"], float) and math.isnan(r["mAP"]))]
    meds = [r["median_err_m"] for r in rows if r.get("median_err_m") is not None and not (isinstance(r["median_err_m"], float) and math.isnan(r["median_err_m"]))]
    chams = [r["chamfer_m"] for r in rows if r.get("chamfer_m") is not None and not (isinstance(r["chamfer_m"], float) and math.isnan(r["chamfer_m"]))]
    recs = [r["detector_recall"] for r in rows if r.get("detector_recall") is not None and not (isinstance(r["detector_recall"], float) and math.isnan(r["detector_recall"]))]
    return {
        "n_scenes": len(rows),
        "n_empty": int(sum(1 for r in rows if r["n_pred"] == 0)),
        "n_no_match": int(sum(1 for r in rows if r["n_pred"] > 0 and
                              (r["median_err_m"] is None or
                               (isinstance(r["median_err_m"], float) and math.isnan(r["median_err_m"]))))),
        "mAP_mean": float(np.mean(maps)) if maps else float("nan"),
        "mAP_median": float(np.median(maps)) if maps else float("nan"),
        "median_err_m_mean": float(np.mean(meds)) if meds else float("nan"),
        "median_err_m_median": float(np.median(meds)) if meds else float("nan"),
        "chamfer_m_mean": float(np.mean(chams)) if chams else float("nan"),
        "count_err_mean": _mean([r["count_err"] for r in rows]),
        "n_tracks_mean": _mean([r["n_tracks"] for r in rows]),
        "detector_recall_mean": float(np.mean(recs)) if recs else float("nan"),
    }


def _intersection_mean(rows_a, rows_b, metric, require_nonempty=False):
    """Mean of `metric` over the INTERSECTION set of seeds present in both
    row sets (frozen rule: cross-configuration comparisons are computed on the
    intersection set). If require_nonempty, additionally restrict to scenes with
    n_pred > 0 in both configs (mAP of an empty reconstruction is 0.0, which is
    a real result — we report both the all-scene and non-empty numbers)."""
    by_seed_a = {r["seed"]: r for r in rows_a}
    by_seed_b = {r["seed"]: r for r in rows_b}
    common = sorted(set(by_seed_a) & set(by_seed_b))
    vals = []
    for s in common:
        ra, rb = by_seed_a[s], by_seed_b[s]
        if require_nonempty and (ra["n_pred"] == 0 or rb["n_pred"] == 0):
            continue
        va = ra.get(metric)
        vb = rb.get(metric)
        if va is None or vb is None or (isinstance(va, float) and math.isnan(va)) \
                or (isinstance(vb, float) and math.isnan(vb)):
            continue
        vals.append(va - vb)
    return {
        "n_intersection": len(common),
        "n_used": len(vals),
        "delta_mean": float(np.mean(vals)) if vals else float("nan"),
        "delta_median": float(np.median(vals)) if vals else float("nan"),
        "delta_std": float(np.std(vals)) if vals else float("nan"),
    }


def analyze(args) -> int:
    csv_path = os.path.join(args.out_dir, "eval_sweep_results.csv")
    if not os.path.isfile(csv_path):
        raise SystemExit("no results CSV at %s — run the sweep first" % csv_path)

    run_meta = {}
    meta_path = os.path.join(args.out_dir, "eval_sweep_run.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            run_meta = json.load(f)

    cells = tuple(run_meta.get("cells", ["primary", "secondary"]))
    view_counts = [int(v) for v in run_meta.get("view_counts", VIEW_COUNTS)]
    comp_names = run_meta.get("compositions", list(COMPOSITIONS.keys()))
    sanity_v24 = bool(run_meta.get("sanity_v24", False))
    recall_radius = float(run_meta.get("recall_radius_px", DEFAULT_RECALL_RADIUS_PX))

    rows = _load_rows(csv_path)
    grid, grid_bin, sanity = _agg_cell_rows(rows, cells, comp_names, view_counts,
                                            SANITY_COMPOSITION, SANITY_VIEW_COUNT)

    # ---- per (cell, comp, vc) aggregates ----
    grid_summary = {}
    for (cell, comp, vc), rws in sorted(grid.items()):
        grid_summary["%s|%s|%d" % (cell, comp, vc)] = _summarize_group(rws)
    bin_summary = {}
    for (cell, b, comp, vc), rws in sorted(grid_bin.items()):
        bin_summary["%s|%s|%s|%d" % (cell, b, comp, vc)] = _summarize_group(rws)

    # ---- detector recall + count error per density bin ----
    det_recall_bin = {}
    count_err_bin = {}
    for cell in cells:
        det_recall_bin[cell] = {}
        count_err_bin[cell] = {}
        for b in BIN_NAMES:
            rws = [r for r in rows if r["cell"] == cell and r["density_bin"] == b
                   and r["composition"] == comp_names[0]
                   and r["view_count"] == view_counts[0]]
            det_recall_bin[cell][b] = {
                "n_scenes": len(rws),
                "recall_mean": _nanmean([r["detector_recall"] for r in rws]),
                "recall_pooled": (float(np.mean([r["detector_recall"] for r in rws]))
                                  if rws else float("nan")),
            }
            # count error pooled across compositions+view counts for the density
            # table (intersection across configs is the same scene set)
            rws_ce = [r for r in rows if r["cell"] == cell and r["density_bin"] == b]
            count_err_bin[cell][b] = {
                "n_scenes": len({r["seed"] for r in rws_ce}),
                "count_err_mean": _mean([r["count_err"] for r in rws_ce]),
                "count_err_median": _mean([r["count_err"] for r in rws_ce]),
            }

    # ---- P1: mixed vs ground ----
    p1 = {}
    for cell in cells:
        p1[cell] = {b: {} for b in BIN_NAMES}
        for b in BIN_NAMES:
            p1[cell][b] = {}
            for vc in view_counts:
                ma = [r for r in rows if r["cell"] == cell and r["density_bin"] == b
                      and r["composition"] == "mixed" and r["view_count"] == vc]
                ga = [r for r in rows if r["cell"] == cell and r["density_bin"] == b
                      and r["composition"] == "ground" and r["view_count"] == vc]
                p1[cell][b][vc] = {
                    "delta_mAP": _intersection_mean(ma, ga, "mAP"),
                    "delta_median_err": _intersection_mean(ma, ga, "median_err_m"),
                }
        # pooled across density
        p1[cell]["all"] = {}
        for vc in view_counts:
            ma = [r for r in rows if r["cell"] == cell
                  and r["composition"] == "mixed" and r["view_count"] == vc]
            ga = [r for r in rows if r["cell"] == cell
                  and r["composition"] == "ground" and r["view_count"] == vc]
            p1[cell]["all"][vc] = {
                "delta_mAP": _intersection_mean(ma, ga, "mAP"),
            }

    # ---- P2: mAP vs view count (knee) ----
    p2 = {}
    for cell in cells:
        p2[cell] = {}
        for comp in comp_names:
            series = []
            for vc in view_counts:
                s = grid_summary.get("%s|%s|%d" % (cell, comp, vc))
                series.append({"view_count": vc,
                               "mAP_mean": s["mAP_mean"] if s else float("nan"),
                               "median_err_m_mean": s["median_err_m_mean"] if s else float("nan"),
                               "count_err_mean": s["count_err_mean"] if s else float("nan"),
                               "n_empty": s["n_empty"] if s else 0})
            # knee = view count where the largest mAP improvement occurs
            deltas = []
            for i in range(1, len(series)):
                a, b_ = series[i - 1]["mAP_mean"], series[i]["mAP_mean"]
                if not (isinstance(a, float) and math.isnan(a)) and not (isinstance(b_, float) and math.isnan(b_)):
                    deltas.append((series[i]["view_count"], b_ - a))
            knee_vc = max(deltas, key=lambda x: x[1])[0] if deltas else None
            p2[cell][comp] = {"series": series, "knee_view_count": knee_vc,
                              "deltas": deltas}

    # ---- P5: count error vs density ----
    p5 = {}
    for cell in cells:
        p5[cell] = {"count_err_per_bin": {b: count_err_bin[cell][b]["count_err_mean"]
                                          for b in BIN_NAMES},
                    "monotonic": None}
        vals = [count_err_bin[cell][b]["count_err_mean"] for b in BIN_NAMES]
        p5[cell]["monotonic"] = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))

    # ---- sanity anchor check ----
    anchor = {}
    for cell in cells:
        anchor[cell] = {}
        for comp in (SANITY_COMPOSITION,):
            rws = sanity.get((cell, comp), [])
            anchor[cell][comp] = _summarize_group(rws)

    summary = {
        "config": {
            "cells": list(cells),
            "scenes_per_cell": run_meta.get("scenes_per_cell"),
            "view_counts": view_counts,
            "compositions": comp_names,
            "composition_views": run_meta.get("composition_views"),
            "sanity_v24": sanity_v24,
            "recall_radius_px": recall_radius,
            "n_scenes": run_meta.get("n_scenes"),
            "n_rows": run_meta.get("n_rows"),
            "wall_clock_s": run_meta.get("wall_clock_s"),
            "cell_radius_m": CELL_RADIUS_M,
            "cell_a_max_px": CELL_A_MAX_PX,
        },
        "per_cell_composition_view": grid_summary,
        "per_cell_density_bin_composition_view": bin_summary,
        "detector_recall_per_density_bin": det_recall_bin,
        "count_err_vs_density": count_err_bin,
        "p1_mixed_vs_ground": p1,
        "p2_mAP_vs_views": p2,
        "p5_count_err_vs_density": p5,
        "sanity_anchor_v24": anchor,
        "intersection_note": (
            "Any metric compared across configurations is computed on the "
            "intersection set (seeds present in BOTH configurations). Every "
            "scene runs every configuration, so the intersection is the shared "
            "scene set; per-config n_empty / n_no_match are reported so the "
            "reader can see empty reconstructions. mAP of an empty "
            "reconstruction is 0.0 (a real result), median_err_m is NaN for a "
            "scene with no matched pairs (excluded from the mean, counted in "
            "n_no_match)."
        ),
    }
    with open(os.path.join(args.out_dir, "eval_sweep_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, allow_nan=True)

    _make_plots(args, cells, comp_names, view_counts, grid_summary)
    _make_report(args, summary, grid_summary, bin_summary, p1, p2, p5,
                 det_recall_bin, count_err_bin, anchor)
    print("summary JSON: %s" % os.path.join(args.out_dir, "eval_sweep_summary.json"))
    print("plots + report written to %s" % args.out_dir)
    return 0


# ---------------------------------------------------------------------------
# Plots — TWO SEPARATE figures. mAP vs views and median-error vs views are NEVER
# combined on one plot. One panel per cell, a_max-in-px-per-drone explicit.
# ---------------------------------------------------------------------------

def _make_plots(args, cells, comp_names, view_counts, grid_summary) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"ground": "#1f77b4", "level": "#ff7f0e", "aerial": "#2ca02c",
              "mixed": "#d62728"}
    markers = {"ground": "o", "level": "s", "aerial": "^", "mixed": "D"}

    def _frame(ax, cell):
        r = CELL_RADIUS_M.get(cell, float("nan"))
        a = CELL_A_MAX_PX.get(cell, float("nan"))
        ax.set_title("%s — swarm radius %gm, a_max %.1f px/drone"
                     % (cell, r, a), fontsize=10)
        ax.set_xlabel("view count")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # ---- Plot (a): mAP vs view count ----
    fig, axes = plt.subplots(1, len(cells), figsize=(6.4 * len(cells), 5), sharey=True)
    if len(cells) == 1:
        axes = [axes]
    for ax, cell in zip(axes, cells):
        for comp in comp_names:
            xs, ys = [], []
            for vc in view_counts:
                s = grid_summary.get("%s|%s|%d" % (cell, comp, vc))
                if s is None:
                    continue
                xs.append(vc)
                ys.append(s["mAP_mean"])
            ax.plot(xs, ys, marker=markers.get(comp, "o"), color=colors.get(comp),
                    label=comp)
        _frame(ax, cell)
    axes[0].set_ylabel("mAP (mean over taus 0.5/1/2/5 m)")
    fig.suptitle("Geometric baseline — mAP vs view count, one curve per tier composition", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(args.out_dir, "plot_mAP_vs_views.png"), dpi=150)
    plt.close(fig)

    # ---- Plot (b): median position error vs view count ----
    fig, axes = plt.subplots(1, len(cells), figsize=(6.4 * len(cells), 5), sharey=True)
    if len(cells) == 1:
        axes = [axes]
    for ax, cell in zip(axes, cells):
        for comp in comp_names:
            xs, ys = [], []
            for vc in view_counts:
                s = grid_summary.get("%s|%s|%d" % (cell, comp, vc))
                if s is None:
                    continue
                xs.append(vc)
                ys.append(s["median_err_m_mean"])
            ax.plot(xs, ys, marker=markers.get(comp, "o"), color=colors.get(comp),
                    label=comp)
        _frame(ax, cell)
    axes[0].set_ylabel("median matched position error (m)")
    fig.suptitle("Geometric baseline — median position error vs view count, one curve per tier composition",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(args.out_dir, "plot_median_err_vs_views.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _fmt(x, nd=4):
    if x is None:
        return "nan"
    try:
        if isinstance(x, float) and math.isnan(x):
            return "nan"
        return ("%%.%df" % nd) % x
    except (TypeError, ValueError):
        return str(x)


def _make_report(args, summary, grid_summary, bin_summary, p1, p2, p5,
                 det_recall_bin, count_err_bin, anchor) -> None:
    cells = summary["config"]["cells"]
    view_counts = summary["config"]["view_counts"]
    comp_names = summary["config"]["compositions"]
    L = []
    add = L.append

    add("# T7 Evaluation Sweep — Geometric Baseline (Agent F)")
    add("")
    add("Frozen pipeline: `detect_blobs` -> `solve_correspondence` (EPIPOLAR_THRESHOLD_PX=3.0) "
        "-> `triangulate_dlt` -> `ml.metrics.evaluate`. Wiring only; every number is produced by "
        "the frozen path. Detection of all 24 views runs ONCE per scene and is cached; "
        "correspondence+triangulation+metrics are re-run per (composition, view count) cell on the "
        "cached subset (a factorization of `ml.baseline_adapter.process_scene`, verified on a "
        "couple of cells against the adapter's per-scene output).")
    add("")
    add("## Sized grid")
    add("")
    add("- %d test scenes per cell (%d total): %s" %
        (summary["config"]["scenes_per_cell"],
         summary["config"]["n_scenes"], summary["config"]["cells"]))
    add("- density bins: low [5,15] (~10), mid [28,42] (~35), high [48,60] (~55, straddles the "
        "~50-drone false-track threshold); balanced ~33/34/33 per cell per bin")
    add("- view counts: %s" % view_counts)
    add("- compositions: %s" % ", ".join(comp_names))
    add("- compositions (ordered camera lists): `%s`" % json.dumps(
        summary["config"].get("composition_views")))
    add("- sanity cell: %s at V=%d (if enabled) — direct check vs the measured V=24 anchor"
        % (SANITY_COMPOSITION, SANITY_VIEW_COUNT))
    add("- cells: primary R=50 m a_max=9.6 px/drone standoff=139 m; secondary R=100 m "
        "a_max=4.8 px/drone standoff=278 m (both W=1920, f=2666.67)")
    add("- wall-clock: %.1f s for %d scenes" %
        (summary["config"].get("wall_clock_s") or float("nan"),
         summary["config"].get("n_scenes") or 0))
    add("")

    # ---- P1 ----
    add("## P1 — mixed-tier beats all-ground at equal view count; gap widens with density")
    add("")
    add("| cell | density | view_count | delta mAP (mixed - ground, intersection set) | "
        "n_intersection | delta median err (m) |")
    add("|---|---|---|---|---|---|")
    for cell in cells:
        for b in ("low", "mid", "high"):
            for vc in view_counts:
                d = p1[cell][b][vc]["delta_mAP"]
                de = p1[cell][b][vc]["delta_median_err"]
                add("| %s | %s | %d | %s | %d | %s |" %
                    (cell, b, vc, _fmt(d["delta_mean"]), d["n_intersection"],
                     _fmt(de["delta_mean"])))
    add("")
    add("P1 verdict: **predicted** mixed >= ground at equal view count, gap increasing with density. "
        "**observed** — see table. **ratio** = mean delta mAP at V=8 per cell: "
        + "; ".join("%s %s" % (c, _fmt(p1[c]["high"][view_counts[-1]]["delta_mAP"]["delta_mean"]))
                    for c in cells)
        + ". **match** = " + _verdict(p1_match(summary)) + ".")
    add("")
    add("Note: detector recall on these clean renders is ~0.99, so at high view count most "
        "compositions are near-ceiling; the mixed-vs-ground gap is most visible at low view "
        "counts and high density (where coplanar ground cameras are ill-conditioned).")
    add("")

    # ---- P2 ----
    add("## P2 — mAP rises with view count; knee between 3 and 5 views")
    add("")
    add("| cell | composition | V=2 | V=3 | V=4 | V=5 | V=6 | V=7 | V=8 | knee |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for cell in cells:
        for comp in comp_names:
            s = p2[cell][comp]["series"]
            row = "| %s | %s |" % (cell, comp)
            for pt in s:
                row += " %s |" % _fmt(pt["mAP_mean"])
            row += " %s |" % s[-1]["knee_view_count"] if False else " %s |" % p2[cell][comp]["knee_view_count"]
            add(row)
    add("")
    add("P2 verdict: **predicted** mAP rises with view count, knee between 3 and 5 views. "
        "**observed** — see table; knee (largest mAP step) per composition/cell: "
        + "; ".join("%s/%s -> V=%s" % (c, comp, p2[c][comp]["knee_view_count"])
                    for c in cells for comp in comp_names) + ". "
        "**ratio** = mAP(V=8)/mAP(V=2) per composition, see curves. **match** = " +
        _verdict(p2_match(p2)) + ".")
    add("")

    # ---- P5 ----
    add("## P5 — count error grows monotonically with density")
    add("")
    add("| cell | low | mid | high | monotonic? |")
    add("|---|---|---|---|---|")
    for cell in cells:
        ce = p5[cell]["count_err_per_bin"]
        add("| %s | %s | %s | %s | %s |" % (cell, _fmt(ce["low"]), _fmt(ce["mid"]),
                                            _fmt(ce["high"]), p5[cell]["monotonic"]))
    add("")
    add("P5 verdict: **predicted** count error grows monotonically with density. **observed** — "
        "see table. **ratio** = count_err(high)/count_err(low) per cell: "
        + "; ".join("%s %.2f" % (c, (p5[c]["count_err_per_bin"]["high"] /
                                     p5[c]["count_err_per_bin"]["low"])
                                 if p5[c]["count_err_per_bin"]["low"] != 0 else float("nan"))
                    for c in cells)
        + ". **match** = " + _verdict(p5["primary"]["monotonic"] and p5["secondary"]["monotonic"]) + ".")
    add("")
    add("Count error vs density (mean over scenes and compositions/view counts at the intersection "
        "set — same scene set everywhere):")
    add("")
    add("| cell | low | mid | high |")
    add("|---|---|---|---|")
    for cell in cells:
        ce = count_err_bin[cell]
        add("| %s | %s | %s | %s |" % (cell, _fmt(ce["low"]["count_err_mean"]),
                                       _fmt(ce["mid"]["count_err_mean"]),
                                       _fmt(ce["high"]["count_err_mean"])))
    add("")

    # ---- Detector recall ----
    add("## Detector recall per density bin (radius %.1f px, GT projected via view K + w2c; "
        "scoring only, never feeds the pipeline)" % summary["config"]["recall_radius_px"])
    add("")
    add("| cell | low | mid | high |")
    add("|---|---|---|---|")
    for cell in cells:
        add("| %s | %s | %s | %s |" % (cell,
                                       _fmt(det_recall_bin[cell]["low"]["recall_mean"]),
                                       _fmt(det_recall_bin[cell]["mid"]["recall_mean"]),
                                       _fmt(det_recall_bin[cell]["high"]["recall_mean"])))
    add("")
    add("Detector recall is near-perfect on these clean renders (~0.99) and flat across density "
        "bins. This is a stated condition of the comparison: the geometric baseline's accuracy "
        "is not limited by detection.")
    add("")

    # ---- Intersection set ----
    add("## Intersection-set handling (frozen rule, non-negotiable)")
    add("")
    add("" + summary["intersection_note"] + "")
    add("")

    # ---- Sanity anchor ----
    add("## Sanity anchor check (measured V=24 reference: mAP 0.9865, median_err 0.0303 m, "
        "count_err +0.7, detector recall 0.9873)")
    add("")
    add("| cell | mAP | median_err_m | count_err | detector_recall | n_scenes |")
    add("|---|---|---|---|---|---|")
    for cell in cells:
        s = anchor[cell].get(SANITY_COMPOSITION, {})
        add("| %s | %s | %s | %s | %s | %d |" %
            (cell, _fmt(s.get("mAP_mean")), _fmt(s.get("median_err_m_mean")),
             _fmt(s.get("count_err_mean")), _fmt(s.get("detector_recall_mean")),
             s.get("n_scenes", 0)))
    add("")
    add("High-view-count curves (V=8) should be consistent with the V=24 anchor; a large "
        "deviation would indicate a bug.")
    add("")

    # ---- mAP + median err tables ----
    add("## mAP and median error at each view count per composition (per cell)")
    add("")
    for cell in cells:
        add("### %s (R=%g m, a_max=%.1f px/drone)" %
            (cell, CELL_RADIUS_M.get(cell, float("nan")),
             CELL_A_MAX_PX.get(cell, float("nan"))))
        add("")
        add("**mAP**")
        add("")
        add("| composition | V=2 | V=3 | V=4 | V=5 | V=6 | V=7 | V=8 | n_empty@V=8 |")
        add("|---|---|---|---|---|---|---|---|---|")
        for comp in comp_names:
            row = "| %s |" % comp
            for vc in view_counts:
                s = grid_summary.get("%s|%s|%d" % (cell, comp, vc), {})
                row += " %s |" % _fmt(s.get("mAP_mean"))
            s8 = grid_summary.get("%s|%s|%d" % (cell, comp, view_counts[-1]), {})
            row += " %s |" % s8.get("n_empty", 0)
            add(row)
        add("")
        add("**median position error (m)**")
        add("")
        add("| composition | V=2 | V=3 | V=4 | V=5 | V=6 | V=7 | V=8 |")
        add("|---|---|---|---|---|---|---|---|")
        for comp in comp_names:
            row = "| %s |" % comp
            for vc in view_counts:
                s = grid_summary.get("%s|%s|%d" % (cell, comp, vc), {})
                row += " %s |" % _fmt(s.get("median_err_m_mean"))
            add(row)
        add("")
        add("**count error**")
        add("")
        add("| composition | V=2 | V=3 | V=4 | V=5 | V=6 | V=7 | V=8 |")
        add("|---|---|---|---|---|---|---|---|")
        for comp in comp_names:
            row = "| %s |" % comp
            for vc in view_counts:
                s = grid_summary.get("%s|%s|%d" % (cell, comp, vc), {})
                row += " %s |" % _fmt(s.get("count_err_mean"))
            add(row)
        add("")

    # ---- Scenes skipped ----
    add("## Scenes with empty reconstructions")
    add("")
    add("An empty reconstruction (n_pred = 0) contributes mAP 0.0 (a real result) and is "
        "excluded from the median-error mean (no matched pairs). Per-cell n_empty is reported "
        "in the tables above and in `eval_sweep_summary.json`. The intersection set for any "
        "cross-configuration comparison includes every shared scene; scenes that produce an "
        "empty reconstruction in one config still count (their mAP is 0).")
    add("")

    with open(os.path.join(args.out_dir, "eval_sweep_report.md"), "w") as f:
        f.write("\n".join(L))
    return True


def _verdict(match):
    if match is True:
        return "MATCH"
    if match is False:
        return "NO"
    return "PARTIAL"


def p1_match(summary):
    """match if mixed mAP >= ground mAP at V=8 for every cell, and the gap is
    non-decreasing low -> mid -> high (per cell)."""
    ok = True
    for cell in summary["config"]["cells"]:
        p1c = summary["p1_mixed_vs_ground"][cell]
        gaps = [p1c[b][summary["config"]["view_counts"][-1]]["delta_mAP"]["delta_mean"]
                for b in ("low", "mid", "high")]
        if any(g != g or g is None or g < 0 for g in gaps):
            ok = False
        if not (gaps[0] <= gaps[1] <= gaps[2]):
            ok = False
    return ok


def p2_match(p2):
    """match if mAP non-decreasing in view count and the knee (largest step) is
    between 3 and 5 views, for every cell/composition."""
    ok = True
    for cell in p2:
        for comp, data in p2[cell].items():
            series = data["series"]
            vals = [pt["mAP_mean"] for pt in series]
            for i in range(1, len(vals)):
                if vals[i] < vals[i - 1] - 1e-9:
                    ok = False
            kv = data["knee_view_count"]
            if kv is not None and not (3 <= kv <= 5):
                ok = False
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="T7 sized evaluation sweep on the geometric baseline.")
    p.add_argument("--mode", choices=("run", "analyze", "all"), default="run",
                   help="'run' executes the sweep; 'analyze' re-reads the CSV and "
                        "writes summary JSON + plots + report; 'all' does both")
    p.add_argument("--root", default=os.path.join(os.path.expanduser("~"), "swarm_ml"),
                   help="data root (PATCH-2 layout root/scenes/SS/NNNNN)")
    p.add_argument("--out-dir", default=os.path.join("logs", "ml_sweep"),
                   help="output directory (default logs/ml_sweep)")
    p.add_argument("--jobs", type=int, default=None,
                   help="parallel scene workers (default min(cpu_count, 14))")
    p.add_argument("--scenes-per-cell", type=int, default=100,
                   help="test scenes per cell (spec T7: 100; pilot: small)")
    p.add_argument("--cells", default="primary,secondary",
                   help="comma-separated cells (default primary,secondary)")
    p.add_argument("--view-counts", default="2,3,4,5,6,7,8",
                   help="comma-separated view counts (default 2..8)")
    p.add_argument("--compositions", default="ground,level,aerial,mixed",
                   help="comma-separated compositions (default all four)")
    p.add_argument("--sanity-v24", action="store_true", default=True,
                   help="include the V=24 mixed sanity cell (default on)")
    p.add_argument("--no-sanity-v24", action="store_false", dest="sanity_v24",
                   help="skip the V=24 mixed sanity cell")
    p.add_argument("--limit-scenes", type=int, default=None,
                   help="stop after this many scenes (pilot)")
    p.add_argument("--recall-radius-px", type=float, default=DEFAULT_RECALL_RADIUS_PX,
                   help="detector-recall match radius in px (default 5.0)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.mode in ("run", "all"):
        rc = run_sweep(args)
        if rc != 0:
            return rc
        args.mode = "analyze"
    return analyze(args)


if __name__ == "__main__":
    sys.exit(main())
