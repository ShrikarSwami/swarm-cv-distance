#!/usr/bin/env python3
"""T5 — ml/baseline_adapter.py: frozen geometric baseline wired to ML scenes.

OWNED FILE: this is the ONLY file this adapter may modify. Every stage-1
module it touches (`stage1_geometry/**`), `ml/metrics.py`, `ml/scene_gen.py`,
`ml/splits.json` and the packed/model modules are FROZEN — read and call, never
edit.

What this adapter does
----------------------
Wires the frozen geometric control path to the freshly rendered ML scenes
(`~/swarm_ml/scenes/SS/NNNNN`, PATCH-2 layout):

    detect_blobs (frozen detector on the same PNGs the model sees)
      -> b3_correspondence.solve_correspondence  (frozen epipolar solver)
      -> b5_triangulation.triangulate_dlt        (frozen DLT triangulation)
      -> ml.metrics.evaluate                     (frozen metric dict)

This is WIRING, NOT IMPROVEMENT: the correspondence solver, count policy
(epipolar_threshold=3.0 px, min_views=2, max_reproj_error=10.0) and the
triangulator are exactly the frozen sweep_b control (`EPIPOLAR_THRESHOLD_PX`,
`solve_correspondence` defaults). Ground-truth 2D projections are used ONLY to
score the detector (mandatory recall report); they never enter the
reconstruction path.

Detector recall (mandatory report)
----------------------------------
For each of the 24 views, ground-truth 3D positions are projected into the
image via the view's K + w2c_R/w2c_t (`data_contract.project_point`), and a
view is counted as recalled when a `detect_blobs` centroid lands within
`--recall-radius-px` (default 5 px) of the projected truth. The radius is
justified by the render geometry: blob centroids of clean rendered drones
localise within ~1-2 px, and the minimum inter-drone spacing (3 m) at the
framing standoff (~139 m) projects to ~57 px separation, so 5 px cannot bridge
two different drones. Drones that project outside the image or behind the
camera are excluded from the denominator. This measurement uses ground truth
for scoring only.

CLI
---
    python3 -m ml.baseline_adapter --root ~/swarm_ml --scenes 20 \
        [--views all] [--seed-start 0] [--jobs 14] [--recall-radius-px 5.0]

    --views  'all' (default) or an int 2..24 = first N views (composition
              sweep; 1 is tolerated and yields an explicit empty/garbage note,
              never a crash).
    --jobs   parallel scene workers (default min(cpu_count, 14)); independent
              scenes run concurrently, per-scene results are bit-identical to
              sequential (frozen per-scene code path, deterministic solver).

Output (stdout)
---------------
    1. human-readable detector-recall block (overall mean, pooled, per-scene)
    2. one JSON line per scene: {"section":"scene", "seed":.., "metrics": {evaluate dict}, ...}
    3. one final JSON summary line: {"section":"summary", ...}

Constraints: numpy / scikit-image / scipy / PIL only (no torch, no bpy, no
Blender). Pure geometry — MPS not needed.
"""

from __future__ import annotations

import argparse
import json
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
    CameraRig,
    CONVENTION_TAG,
    IMAGE_SIZE,
    Detections,
    blender_c2w_to_opencv_w2c,
    project_point,
)
from detect_blobs import detect_blobs  # noqa: E402  frozen detector
from b3_correspondence import solve_correspondence  # noqa: E402  frozen solver
from b5_triangulation import triangulate_dlt  # noqa: E402  frozen triangulator

from ml.metrics import evaluate  # noqa: E402  frozen metrics (single truth)

# ---------------------------------------------------------------------------
# Frozen operating constants (mirror sweep_b / scene_gen — do not invent)
# ---------------------------------------------------------------------------
DRONE_SIZE_M = 0.5            # scene_gen.DRONE_SIZE_M
EPIPOLAR_THRESHOLD_PX = 3.0   # sweep_b.EPIPOLAR_THRESHOLD_PX
N_VIEWS_TOTAL = 24            # scene_gen.N_VIEWS (8 per tier x 3 tiers)
DEFAULT_RECALL_RADIUS_PX = 5.0
_IMAGE_W, _IMAGE_H = IMAGE_SIZE


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _read_rgb(path: str) -> np.ndarray:
    """Read a PNG as uint8 RGB (H, W, 3). PIL is the frozen control's primary
    reader (`sweep_b._read_image` tries PIL first); not an ML dependency."""
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def scene_dir_for_seed(root: str, seed: int) -> str:
    """PATCH-2 scene layout: root/scenes/SS/NNNNN with SS = seed // 100."""
    return os.path.join(root, "scenes", "%02d" % (seed // 100), "%05d" % seed)


def _build_rig(cam: dict) -> CameraRig:
    """Build the frozen CameraRig from cameras.json.

    Mirrors sweep_b.bundle_poses_to_rig: the Blender c2w matrices are converted
    to OpenCV w2c via `data_contract.blender_c2w_to_opencv_w2c` (the stored
    w2c_R/w2c_t are consistent with that conversion, verified)."""
    K_list, R_list, t_list, c2w_list = [], [], [], []
    for v in cam["views"]:
        K_list.append(np.asarray(v["K"], dtype=np.float64))
        c2w = np.asarray(v["c2w"], dtype=np.float64)
        c2w_list.append(c2w)
        R_w2c, t_w2c = blender_c2w_to_opencv_w2c(c2w)
        R_list.append(R_w2c)
        t_list.append(t_w2c)
    return CameraRig(
        K=np.stack(K_list),
        w2c_R=np.stack(R_list),
        w2c_t=np.stack(t_list),
        c2w=np.stack(c2w_list),
        focal_px=float(cam["focal_px"]),
        convention=CONVENTION_TAG,
        geometry_class="mixed",  # 8 ground / 8 level / 8 aerial = frozen "mixed"
    )


def _subset_rig(rig: CameraRig, view_idxs: list[int]) -> CameraRig:
    """Restrict a rig to a subset of views (composition sweep)."""
    return CameraRig(
        K=rig.K[view_idxs],
        w2c_R=rig.w2c_R[view_idxs],
        w2c_t=rig.w2c_t[view_idxs],
        c2w=rig.c2w[view_idxs],
        focal_px=rig.focal_px,
        convention=rig.convention,
        geometry_class=rig.geometry_class,
    )


def _detect_views(scene_dir: str, cam: dict, rig: CameraRig, standoff_m: float,
                  view_idxs: list[int]) -> Detections:
    """Run the FROZEN detector (detect_blobs) on the rendered PNGs.

    Same pixels the model sees; the exact call the frozen control makes
    (`detect_from_bundle_views`), with the scene's stored standoff (== the
    camera-to-swarm-centre standoff the render used)."""
    pts: list[np.ndarray] = []
    for v in view_idxs:
        rgb = _read_rgb(os.path.join(scene_dir, "angle_%02d.png" % v))
        dets = detect_blobs(
            rgb=rgb,
            drone_size_m=DRONE_SIZE_M,
            focal_px=rig.focal_px,
            standoff_m=standoff_m,
            image_width_px=_IMAGE_W,
        )
        pts.append(dets.points_per_view[0] if dets.points_per_view else np.empty((0, 2), dtype=np.float64))
    return Detections(points_per_view=pts, image_size=IMAGE_SIZE)


def measure_detector_recall(rig: CameraRig, dets: Detections, true_positions: np.ndarray,
                            view_idxs: list[int], radius_px: float) -> dict:
    """Score the FROZEN detector against projected ground truth.

    REPORT ONLY — never feeds the reconstruction pipeline. For each view,
    project every ground-truth position via the view's K + w2c; a view counts a
    drone as recalled when a detection centroid is within `radius_px` of the
    projection. Drones behind the camera or projecting outside the frame are
    excluded from the denominator (not visible, cannot be detected)."""
    per_view = []
    total_visible = 0
    total_matched = 0
    for v in view_idxs:
        K = rig.K[v]
        R = rig.w2c_R[v]
        t = rig.w2c_t[v]
        det_pts = dets.points_per_view[v]
        visible = 0
        matched = 0
        for p in true_positions:
            pix = project_point(p, K, R, t)
            if pix is None:  # behind camera
                continue
            u = float(pix[0])
            vv = float(pix[1])
            if not (0.0 <= u <= _IMAGE_W and 0.0 <= vv <= _IMAGE_H):  # outside frame
                continue
            visible += 1
            if len(det_pts) > 0:
                if float(np.linalg.norm(det_pts - np.array([u, vv], dtype=np.float64), axis=1).min()) <= radius_px:
                    matched += 1
        recall_view = (matched / visible) if visible > 0 else None
        per_view.append({
            "view": int(v),
            "visible": visible,
            "matched": matched,
            "recall": recall_view,
        })
        total_visible += visible
        total_matched += matched
    recalls = [pv["recall"] for pv in per_view if pv["recall"] is not None]
    return {
        "radius_px": float(radius_px),
        "overall_mean": float(np.mean(recalls)) if recalls else 0.0,
        "pooled": float(total_matched / total_visible) if total_visible else 0.0,
        "per_view": per_view,
    }


# ---------------------------------------------------------------------------
# Per-scene pipeline (module-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

def process_scene(root: str, seed: int, view_idxs: list[int],
                  recall_radius_px: float) -> dict:
    """Process one scene end-to-end with the frozen control.

    - builds the full 24-view rig + detections (recall measured on all 24 views)
    - reconstructs with the `view_idxs` subset (same solver / count policy /
      triangulator as the frozen control)
    - scores with the frozen `ml.metrics.evaluate`
    """
    t_start = time.time()
    scene_dir = scene_dir_for_seed(root, seed)
    with open(os.path.join(scene_dir, "cameras.json")) as f:
        cam = json.load(f)
    with open(os.path.join(scene_dir, "ground_truth.json")) as f:
        gt = json.load(f)

    true = np.asarray(gt["positions"], dtype=np.float64)
    standoff_m = float(cam["standoff_m"])

    full_rig = _build_rig(cam)
    all_view_idxs = list(range(len(cam["views"])))
    full_dets = _detect_views(scene_dir, cam, full_rig, standoff_m, all_view_idxs)

    # Mandatory detector-recall report (all 24 views) + subset (if different).
    recall_all = measure_detector_recall(full_rig, full_dets, true, all_view_idxs, recall_radius_px)
    recall_sel = None
    if len(view_idxs) < len(all_view_idxs):
        recall_sel = measure_detector_recall(full_rig, full_dets, true, view_idxs, recall_radius_px)

    # --- Frozen geometric reconstruction (subset of views) ---
    empty_reason = None
    n_tracks = 0
    if len(view_idxs) < 2:
        # A view count that cannot triangulate: explicit note, no crash.
        pred = np.empty((0, 3), dtype=np.float64)
        empty_reason = "fewer than 2 views cannot triangulate"
    else:
        if len(view_idxs) == len(all_view_idxs):
            rig, dets = full_rig, full_dets
        else:
            rig, dets = _subset_rig(full_rig, view_idxs), Detections(
                points_per_view=[full_dets.points_per_view[i] for i in view_idxs],
                image_size=IMAGE_SIZE,
            )
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

    # --- Frozen metrics ---
    metrics = evaluate(pred, true)

    return {
        "section": "scene",
        "seed": int(seed),
        "scene_dir": scene_dir,
        "cell": str(cam.get("cell", "?")),
        "n_drones": int(gt["n_drones"]),
        "views_used": len(view_idxs),
        "empty_reason": empty_reason,
        "n_tracks": n_tracks,
        "detector_recall": float(recall_all["overall_mean"]),
        "detector_recall_pooled": float(recall_all["pooled"]),
        "detector_recall_views": recall_all,
        "detector_recall_selected_views": recall_sel,
        "metrics": metrics,
        "wall_clock_s": round(time.time() - t_start, 2),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_views(s: str) -> list[int]:
    """Parse --views: 'all' -> 24 views, int 1..24 -> first N views."""
    if isinstance(s, str) and s.strip().lower() == "all":
        return list(range(N_VIEWS_TOTAL))
    try:
        n = int(s)
    except (TypeError, ValueError):
        raise SystemExit("--views must be 'all' or an int 1..%d, got %r" % (N_VIEWS_TOTAL, s))
    if n < 1 or n > N_VIEWS_TOTAL:
        raise SystemExit("--views must be an int 1..%d, got %d" % (N_VIEWS_TOTAL, n))
    return list(range(n))


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="T5 baseline: frozen geometric epipolar+DLT reconstruction "
                    "wired to rendered ML scenes (acceptance).")
    p.add_argument("--root", default=os.path.join(os.path.expanduser("~"), "swarm_ml"),
                   help="data root (PATCH-2 layout root/scenes/SS/NNNNN)")
    p.add_argument("--scenes", type=int, default=20,
                   help="number of scenes to process, starting at --seed-start (default 20)")
    p.add_argument("--views", default="all",
                   help="'all' (default) or an int 2..24 = first N views (composition sweep)")
    p.add_argument("--seed-start", type=int, default=0,
                   help="first seed (default 0; acceptance run uses seeds 0-19)")
    p.add_argument("--jobs", type=int, default=None,
                   help="parallel scene workers (default min(cpu_count, 14))")
    p.add_argument("--recall-radius-px", type=float, default=DEFAULT_RECALL_RADIUS_PX,
                   help="detector-recall match radius in px (default 5.0)")
    return p.parse_args(argv)


def _scene_recall_summary(results: list[dict]) -> dict:
    """Aggregate per-scene detector recall into the report block."""
    per_scene = [r["detector_recall"] for r in results]
    pooled_num = sum(int(r["detector_recall_views"]["per_view"][i]["matched"])
                     for r in results for i in range(len(r["detector_recall_views"]["per_view"])))
    pooled_den = sum(int(r["detector_recall_views"]["per_view"][i]["visible"])
                     for r in results for i in range(len(r["detector_recall_views"]["per_view"])))
    return {
        "overall_mean": float(np.mean(per_scene)) if per_scene else 0.0,
        "pooled": float(pooled_num / pooled_den) if pooled_den else 0.0,
        "per_scene_min": float(np.min(per_scene)) if per_scene else 0.0,
        "per_scene_max": float(np.max(per_scene)) if per_scene else 0.0,
        "per_scene": [round(float(x), 6) for x in per_scene],
    }


def _build_summary(results: list[dict], view_idxs: list[int], args: argparse.Namespace) -> dict:
    """Aggregate per-scene metric dicts into one summary dict."""
    taus = sorted({float(t) for r in results for t in r["metrics"]["per_tau"]})
    per_tau_agg = {}
    for tau in taus:
        pr = [r["metrics"]["per_tau"][str(tau)] if str(tau) in r["metrics"]["per_tau"]
              else r["metrics"]["per_tau"][tau] for r in results]
        per_tau_agg[str(tau)] = {
            "precision_mean": float(np.mean([x["precision"] for x in pr])),
            "recall_mean": float(np.mean([x["recall"] for x in pr])),
            "f1_mean": float(np.mean([x["f1"] for x in pr])),
            "ap_mean": float(np.mean([x["ap"] for x in pr])),
            "n_matched_total": int(sum(x["n_matched"] for x in pr)),
        }

    median_errs = np.array([r["metrics"]["median_err_m"] for r in results], dtype=np.float64)
    chamfers = np.array([r["metrics"]["chamfer_m"] for r in results], dtype=np.float64)
    maps = np.array([r["metrics"]["mAP"] for r in results], dtype=np.float64)

    return {
        "section": "summary",
        "n_scenes": len(results),
        "seeds": [results[0]["seed"], results[-1]["seed"]] if results else [],
        "views": "all" if len(view_idxs) == N_VIEWS_TOTAL else len(view_idxs),
        "recall_radius_px": args.recall_radius_px,
        "detector_recall": _scene_recall_summary(results),
        "metrics_mean": {
            "median_err_m": float(np.nanmean(median_errs)) if len(median_errs) else float("nan"),
            "median_err_m_median": float(np.nanmedian(median_errs)) if len(median_errs) else float("nan"),
            "chamfer_m": float(np.nanmean(chamfers)) if len(chamfers) else float("nan"),
            "mAP": float(np.nanmean(maps)) if len(maps) else float("nan"),
            "count_err": float(np.mean([r["metrics"]["count_err"] for r in results])),
            "n_pred_total": int(sum(r["metrics"]["n_pred"] for r in results)),
            "n_true_total": int(sum(r["metrics"]["n_true"] for r in results)),
        },
        "per_tau": per_tau_agg,
        "n_empty_scenes": int(sum(1 for r in results if r["metrics"]["n_pred"] == 0)),
    }


def run_scenes(seeds: list[int], view_idxs: list[int], args: argparse.Namespace) -> list[dict]:
    """Run scenes, parallelised across independent seeds when possible.

    Per-scene results are bit-identical to sequential: each worker executes the
    same frozen per-scene code path, and the frozen solver is deterministic
    (seeds its own RNG; set/dict iteration over ints is CPython-deterministic).
    Falls back to sequential on any multiprocessing failure."""
    n_jobs = args.jobs
    if n_jobs is None:
        n_jobs = min(os.cpu_count() or 1, 14)
    n_jobs = max(1, min(n_jobs, len(seeds)))

    if n_jobs <= 1:
        return [process_scene(args.root, s, view_idxs, args.recall_radius_px) for s in seeds]

    try:
        import multiprocessing
        # Use the platform default context (spawn on macOS, fork on Linux).
        # spawn is the safe choice on macOS: forking a process that has already
        # imported BLAS-backed numpy/scipy can inherit OpenBLAS thread pools.
        # Spawn startup cost (~2-3 s per worker, parallelised) is negligible
        # against the ~60 s/scene correspondence. Per-scene results are
        # bit-identical to sequential: each worker runs the same frozen code
        # path and the frozen solver is deterministic (seeds its own RNG; set /
        # dict iteration over ints is CPython-deterministic).
        ctx = multiprocessing.get_context()
        with ctx.Pool(processes=n_jobs) as pool:
            results = pool.starmap(
                process_scene,
                [(args.root, s, view_idxs, args.recall_radius_px) for s in seeds],
            )
        return results
    except Exception as exc:  # pragma: no cover - defensive
        print("WARNING: multiprocessing unavailable (%s); running sequentially"
              % exc, file=sys.stderr)
        return [process_scene(args.root, s, view_idxs, args.recall_radius_px) for s in seeds]


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.seed_start < 0:
        raise SystemExit("--seed-start must be >= 0, got %d" % args.seed_start)
    if args.scenes < 1:
        raise SystemExit("--scenes must be >= 1, got %d" % args.scenes)

    seeds = list(range(args.seed_start, args.seed_start + args.scenes))
    view_idxs = parse_views(args.views)
    n_jobs = args.jobs if args.jobs else min(os.cpu_count() or 1, 14)
    n_jobs = max(1, min(n_jobs, len(seeds)))

    root = os.path.expanduser(args.root)
    missing = [scene_dir_for_seed(root, s) for s in seeds
               if not os.path.isdir(scene_dir_for_seed(root, s))]
    if missing:
        raise SystemExit("scene dirs missing for %d seed(s), e.g. %s"
                         % (len(missing), missing[0]))

    print("=== T5 baseline adapter (frozen geometric control) ===")
    print("command: python3 -m ml.baseline_adapter --root %s --scenes %d "
          "--views %s --seed-start %d --jobs %d --recall-radius-px %.1f"
          % (root, args.scenes, args.views, args.seed_start, n_jobs, args.recall_radius_px))
    print("seeds: %d..%d (%d) | views: %s (%d) | recall radius: %.1f px | workers: %d"
          % (seeds[0], seeds[-1], len(seeds),
             "all" if len(view_idxs) == N_VIEWS_TOTAL else len(view_idxs),
             len(view_idxs), args.recall_radius_px, n_jobs))
    print("", flush=True)

    t_run = time.time()
    results = run_scenes(seeds, view_idxs, args)
    t_elapsed = time.time() - t_run

    # 1. Detector-recall block
    rec = _scene_recall_summary(results)
    print("DETECTOR RECALL (radius %.1f px, GT projected via view K + w2c):"
          % args.recall_radius_px)
    print("  overall mean (per-scene mean of per-view means): %.4f" % rec["overall_mean"])
    print("  pooled (matched / visible over all views):      %.4f" % rec["pooled"])
    print("  per-scene range: [%.4f, %.4f]" % (rec["per_scene_min"], rec["per_scene_max"]))
    for r in results:
        print("  seed=%4d cell=%-8s recall=%.4f  n_true=%d n_pred=%d median_err=%.4fm"
              % (r["seed"], r["cell"], r["detector_recall"], r["metrics"]["n_true"],
                 r["metrics"]["n_pred"], r["metrics"]["median_err_m"]))
    print("", flush=True)

    # 2. One JSON line per scene (raw evaluate dict under "metrics")
    for r in results:
        print(json.dumps(r, allow_nan=True, default=_json_default))
    print("", flush=True)

    # 3. Summary dict
    summary = _build_summary(results, view_idxs, args)
    summary["wall_clock_s"] = round(t_elapsed, 2)
    print(json.dumps(summary, allow_nan=True, default=_json_default))

    return 0


def _json_default(o):
    """Fallback encoder for any non-JSON types that slip through."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    raise TypeError("not JSON serializable: %r" % (o,))


if __name__ == "__main__":
    sys.exit(main())
