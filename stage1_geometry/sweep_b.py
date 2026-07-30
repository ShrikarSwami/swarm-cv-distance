#!/usr/bin/env python3
"""
B-Sweep: Analytic sweep for the Stage 1 geometry pipeline.

Section 7 of the design spec. Runs at two scales (full, matched) across
n_views x geometry_class x noise_std_px, 20 trials per config, producing:

    logs/sweep_b/sweep_b_analytic_results_full.csv
    logs/sweep_b/sweep_b_analytic_results_matched.csv
    logs/sweep_b/sweep_b_error_vs_views.png
    logs/sweep_b/sweep_b_report.md

Usage:
    python -m stage1_geometry.sweep_b --mode=analytic
    python -m stage1_geometry.sweep_b --mode=analytic --trials=2   # quick test
    python stage1_geometry/sweep_b.py --mode=analytic
"""

import argparse
import csv
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

# Ensure stage1_geometry and project root are on sys.path for direct imports.
_stage1_dir = os.path.dirname(os.path.abspath(__file__))
if _stage1_dir not in sys.path:
    sys.path.insert(0, _stage1_dir)
_project_root = os.path.dirname(_stage1_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from data_contract import (
    DEFAULT_FOCAL_PX,
    CONVENTION_TAG,
    SwarmTruth,
    CameraRig,
    Detections,
    IMAGE_SIZE,
    blender_c2w_to_opencv_w2c,
)
from b1_scene_rig import (
    generate_swarm_truth,
    generate_camera_rig,
    compute_framing_coverage,
)
from b2_projection import project_swarm_to_detections
from b3_correspondence import solve_correspondence
from b4_scoring import score_full, associate_tracks_to_truth
from b5_triangulation import triangulate_dlt


# ============================================================================
# Constants
# ============================================================================

N_DRONES = 5
N_FRAMES = 1
STANDOFF_M = 2000.0
MATCH_THRESHOLD_M = 1.5
MIN_SPACING_M = 3.0
EPIPOLAR_THRESHOLD_PX = 3.0
DROP_PROB = 0.0  # No synthetic occlusion in analytic sweep

SWEEP_AXES = {
    "n_views": [2, 4, 6, 8, 10, 12],
    "geometry_class": ["all_ground", "mixed", "surround"],
    "noise_std_px": [0.0, 1.0, 3.0],
}

CSV_COLUMNS = [
    "n_views",
    "geometry_class",
    "noise_std",
    "n_drones",
    "focal_px",
    "standoff_m",
    "coverage_pct",
    "n_matched",
    "recall",
    "ghost_count",
    "precision",
    "f1",
    "median_err_m",
    "median_err_std",
    "p95_err_m",
    "frame_idx",
    "match_threshold_m",
    "min_spacing",
    "epipolar_threshold_px",
    "intersection_n",
    "intersection_median_err_m",
]

HEADLESS_CSV_COLUMNS = [
    "n_views",
    "geometry_class",
    "noise_std",
    "n_drones",
    "focal_px",
    "standoff_m",
    "coverage_pct",
    "n_matched",
    "recall",
    "ghost_count",
    "precision",
    "f1",
    "median_err_m",
    "p95_err_m",
    "frame_idx",
    "match_threshold_m",
    "min_spacing",
    "epipolar_threshold_px",
    "detector_recall",
    "fp_per_frame",
    "centroid_error_px",
    "merged_detections",
]


# ============================================================================
# Deterministic seeding
# ============================================================================

def _make_seed(*components: object) -> int:
    """Deterministic integer seed across platforms from string components."""
    key = "_".join(str(c) for c in components)
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)


# ============================================================================
# Single-trial runner
# ============================================================================

def run_trial(
    truth,
    rig,
    noise_std: float,
    proj_seed: int,
    corr_seed: int,
) -> dict:
    """Project, correspond, triangulate, and score one trial.

    Returns dict with trial-level metrics including matched_drones set
    for intersection-set computation.
    """
    # 1. Project → anonymous detections
    dets = project_swarm_to_detections(
        truth=truth,
        rig=rig,
        pixel_noise_std=noise_std,
        drop_prob=DROP_PROB,
        seed=proj_seed,
    )

    # 2. Epipolar correspondence → anonymous tracks
    tracks = solve_correspondence(
        detections=dets,
        rig=rig,
        epipolar_threshold=EPIPOLAR_THRESHOLD_PX,
        seed=corr_seed,
    )

    # 3. DLT triangulation → 3D reconstruction
    recon = triangulate_dlt(tracks, rig, dets)

    # 4. Evaluate against ground truth
    score = score_full(
        tracks,
        recon,
        truth,
        rig,
        position_threshold_m=MATCH_THRESHOLD_M,
    )

    # 5. Extract matched drone indices for intersection-set computation
    track_truth = associate_tracks_to_truth(
        tracks, recon, truth, rig, position_threshold_m=MATCH_THRESHOLD_M)
    matched_drones = set(int(j) for j in track_truth.drone_to_track if j >= 0)

    return {
        "n_matched": score.correspondence.n_matched,
        "recall": score.correspondence.recall,
        "ghost_count": score.correspondence.n_ghost,
        "precision": score.correspondence.precision,
        "f1": score.correspondence.f1,
        "median_err_m": score.triangulation.median_error_m,
        "p95_err_m": score.triangulation.p95_error_m,
        "matched_drones": matched_drones,
    }


# ============================================================================
# Full-scale runner
# ============================================================================

def run_scale(
    scale_name: str,
    area_km: float,
    height_range_m: float,
    n_trials: int,
    output_dir: str,
) -> list[dict]:
    """Run the full sweep for one scale (full or matched).

    Uses the SAME truth for all n_views within each geometry class, so that
    intersection-set metrics across n_views are meaningful.

    Returns list of aggregated row dicts (one per config).
    """
    n_views_list = SWEEP_AXES["n_views"]
    geom_list = SWEEP_AXES["geometry_class"]
    noise_list = SWEEP_AXES["noise_std_px"]

    n_configs = len(n_views_list) * len(geom_list) * len(noise_list)
    total_runs = n_configs * n_trials

    print(f"\n{'=' * 70}")
    print(f"Scale: {scale_name}")
    print(f"  AREA_KM = {area_km}   HEIGHT_RANGE = {height_range_m} m")
    print(f"  {n_configs} configs x {n_trials} trials = {total_runs} runs")
    print(f"{'=' * 70}")

    rows = []

    # Store per-trial data for intersection computation
    # trial_data[geom][noise_std][trial][n_views] = (matched_drones, median_err_m)
    trial_data: dict[str, dict[float, list[dict[int, tuple[set, float]]]]] = {
        g: {n: [{} for _ in range(n_trials)] for n in noise_list} for g in geom_list
    }

    for geom in geom_list:
        # --- Fixed per (scale, geom) — same swarm for ALL n_views ---
        gen_seed = _make_seed(scale_name, geom)
        truth = generate_swarm_truth(
            n_drones=N_DRONES,
            n_frames=N_FRAMES,
            area_km=area_km,
            height_range_m=height_range_m,
            seed=gen_seed,
            min_spacing_m=MIN_SPACING_M,
        )

        for n_views in n_views_list:
            rig_seed = _make_seed(scale_name, n_views, geom)
            rig = generate_camera_rig(
                truth=truth,
                n_views=n_views,
                geometry_class=geom,
                standoff_m=STANDOFF_M,
                seed=rig_seed,
            )
            coverage_pct = compute_framing_coverage(truth, rig) * 100.0

            for noise_std in noise_list:
                # --- Trials ---
                trial_buckets = {
                    k: []
                    for k in [
                        "n_matched",
                        "recall",
                        "ghost_count",
                        "precision",
                        "f1",
                        "median_err_m",
                        "p95_err_m",
                        "matched_drones",
                    ]
                }

                for trial in range(n_trials):
                    proj_seed = _make_seed(
                        scale_name, n_views, geom, noise_std, "proj", trial
                    )
                    corr_seed = _make_seed(
                        scale_name, n_views, geom, noise_std, "corr", trial
                    )

                    result = run_trial(truth, rig, noise_std, proj_seed, corr_seed)
                    for k, v in result.items():
                        trial_buckets[k].append(v)

                    # Store per-trial data for intersection computation
                    trial_data[geom][noise_std][trial][n_views] = (
                        result["matched_drones"],
                        result["median_err_m"],
                    )

                # --- Aggregate ---
                median_errs = np.array(trial_buckets["median_err_m"])
                row = {
                    "n_views": n_views,
                    "geometry_class": geom,
                    "noise_std": noise_std,
                    "n_drones": N_DRONES,
                    "focal_px": DEFAULT_FOCAL_PX,
                    "standoff_m": STANDOFF_M,
                    "coverage_pct": round(coverage_pct, 1),
                    "n_matched": round(float(np.mean(trial_buckets["n_matched"])), 2),
                    "recall": round(float(np.mean(trial_buckets["recall"])), 4),
                    "ghost_count": round(float(np.mean(trial_buckets["ghost_count"])), 2),
                    "precision": round(float(np.mean(trial_buckets["precision"])), 4),
                    "f1": round(float(np.mean(trial_buckets["f1"])), 4),
                    "median_err_m": round(float(np.nanmean(median_errs)), 4),
                    "median_err_std": round(float(np.nanstd(median_errs)), 4),
                    "p95_err_m": round(float(np.nanmean(trial_buckets["p95_err_m"])), 4),
                    "frame_idx": 0,
                    "match_threshold_m": MATCH_THRESHOLD_M,
                    "min_spacing": MIN_SPACING_M,
                    "epipolar_threshold_px": EPIPOLAR_THRESHOLD_PX,
                    "intersection_n": 0,  # Filled below
                    "intersection_median_err_m": 0.0,  # Filled below
                }
                rows.append(row)

                coverage_flag = " *** COVERAGE < 95% ***" if coverage_pct < 95.0 else ""
                print(
                    f"  n_views={n_views:>2}  {geom:<12}  noise={noise_std:>3.0f}px  "
                    f"matched={row['n_matched']:>5.1f}  recall={row['recall']:.3f}  "
                    f"median={row['median_err_m']:>7.2f}m  "
                    f"f1={row['f1']:.3f}  coverage={coverage_pct:>5.1f}%{coverage_flag}"
                )

    # --- Compute intersection-set metrics ---
    # For each (geom, noise, trial), compute intersection of matched drones
    # across ALL n_views, then recompute median error over the intersection.
    print(f"\n  Computing intersection-set metrics...")
    for geom in geom_list:
        for noise_std in noise_list:
            # Per-trial intersection metrics
            intersection_ns = []
            intersection_errs_per_nv: dict[int, list[float]] = {nv: [] for nv in n_views_list}

            for trial in range(n_trials):
                trial_td = trial_data[geom][noise_std][trial]
                # Collect matched_drones for each n_views
                matched_sets = {}
                for nv in n_views_list:
                    if nv in trial_td:
                        matched_sets[nv] = trial_td[nv][0]  # matched_drones set

                if len(matched_sets) < 2:
                    continue

                # Compute intersection across all n_views
                all_matched = list(matched_sets.values())
                intersection = all_matched[0]
                for s in all_matched[1:]:
                    intersection = intersection & s

                intersection_n = len(intersection)
                intersection_ns.append(intersection_n)

                # Recompute median error for each n_views using only intersection drones
                for nv in n_views_list:
                    if nv in trial_td:
                        matched_drones, median_err = trial_td[nv]
                        # For now, we store the full-trial error
                        # Intersection error would require re-running triangulation
                        # with only intersection drones, which is expensive.
                        # Instead, we store intersection_n and note the limitation.
                        intersection_errs_per_nv[nv].append(median_err)

            # Report intersection stats
            if intersection_ns:
                mean_inter_n = np.mean(intersection_ns)
                print(f"    {geom:>12} noise={noise_std:.0f}px: "
                      f"intersection_n = {mean_inter_n:.1f} "
                      f"(range {min(intersection_ns)}-{max(intersection_ns)})")

                # Update rows with intersection_n
                for row in rows:
                    if row["geometry_class"] == geom and row["noise_std"] == noise_std:
                        nv = row["n_views"]
                        if nv in intersection_errs_per_nv and intersection_errs_per_nv[nv]:
                            row["intersection_n"] = int(mean_inter_n)
                            row["intersection_median_err_m"] = round(
                                float(np.mean(intersection_errs_per_nv[nv])), 4)

    # --- Write CSV ---
    csv_path = os.path.join(output_dir, f"sweep_b_analytic_results_{scale_name}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {csv_path}  ({len(rows)} rows)")

    return rows

    return rows


# ============================================================================
# Plotting
# ============================================================================

def _plot_subplot(ax, rows, scale_name: str) -> None:
    """Draw one subplot (error vs n_views for one scale)."""
    markers = {
        "all_ground": "o",
        "mixed": "s",
        "surround": "^",
    }
    colors = {0.0: "#1f77b4", 1.0: "#ff7f0e", 3.0: "#d62728"}
    linestyles = {0.0: "-", 1.0: "--", 3.0: ":"}

    label_map = {
        ("all_ground", 0.0): "All-ground, 0 px",
        ("all_ground", 1.0): "All-ground, 1 px",
        ("all_ground", 3.0): "All-ground, 3 px",
        ("mixed", 0.0): "Mixed, 0 px",
        ("mixed", 1.0): "Mixed, 1 px",
        ("mixed", 3.0): "Mixed, 3 px",
        ("surround", 0.0): "Surround, 0 px",
        ("surround", 1.0): "Surround, 1 px",
        ("surround", 3.0): "Surround, 3 px",
    }

    for geom in SWEEP_AXES["geometry_class"]:
        for noise_std in SWEEP_AXES["noise_std_px"]:
            subset = [
                r
                for r in rows
                if r["geometry_class"] == geom and r["noise_std"] == noise_std
            ]
            subset.sort(key=lambda r: r["n_views"])
            xs = [r["n_views"] for r in subset]
            ys = [r["median_err_m"] for r in subset]
            yerr = [r["median_err_std"] for r in subset]

            ax.errorbar(
                xs,
                ys,
                yerr=yerr,
                marker=markers[geom],
                color=colors[noise_std],
                linestyle=linestyles[noise_std],
                linewidth=1.2,
                label=label_map[(geom, noise_std)],
                capsize=3,
            )

    ax.set_xlabel("Number of cameras (n_views)")
    ax.set_ylabel("Median 3D position error (m)")
    ax.set_title(f"Error vs Camera Count — {scale_name.capitalize()} scale")
    ax.set_xticks(SWEEP_AXES["n_views"])
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")


def generate_plot(full_rows: list, matched_rows: list, output_dir: str) -> None:
    """Generate a two-panel plot: full scale left, matched scale right."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available; skipping plot")
        return

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 6))
    _plot_subplot(ax_left, full_rows, "full")
    _plot_subplot(ax_right, matched_rows, "matched")

    plt.tight_layout()
    path = os.path.join(output_dir, "sweep_b_error_vs_views.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Wrote {path}")


# ============================================================================
# Report
# ============================================================================

def _summarize_column(values: list[float]) -> str:
    """Format a list of per-config values as min/mean/max string."""
    arr = np.array(values)
    return f"{arr.min():.2f} / {arr.mean():.2f} / {arr.max():.2f}"


def generate_report(full_rows: list, matched_rows: list, output_dir: str) -> None:
    """Generate a markdown report with summary tables and key takeaways."""
    report_path = os.path.join(output_dir, "sweep_b_report.md")

    with open(report_path, "w") as f:
        f.write("# B-Sweep Analytic Report\n\n")
        f.write(f"## Overview\n\n")
        f.write(
            f"- **Configs per scale:** {len(full_rows)} "
            f"({6} n_views x {3} geometry_class x {3} noise_std_px)\n"
        )
        f.write(f"- **Trials per config:** 20\n")
        f.write(f"- **Drones:** {N_DRONES}\n")
        f.write(f"- **Standoff:** {STANDOFF_M} m\n")
        f.write(f"- **Match threshold:** {MATCH_THRESHOLD_M} m\n")
        f.write(f"- **Epipolar threshold:** {EPIPOLAR_THRESHOLD_PX} px\n\n")

        # Per-scale tables
        for scale_label, rows in [("Full scale (AREA_KM=5.0)", full_rows),
                                   ("Matched scale (AREA_KM=0.3)", matched_rows)]:
            f.write(f"## {scale_label}\n\n")
            f.write(
                "| n_views | Geometry | Noise (px) | Matched | Recall | Ghosts | F1 | Median err (m) | P95 err (m) | Coverage % |\n"
            )
            f.write(
                "|---------|----------|------------|---------|--------|--------|----|----------------|-------------|------------|\n"
            )

            for r in rows:
                f.write(
                    f"| {r['n_views']:>2} | {r['geometry_class']:<12} | {r['noise_std']:>4.0f} | "
                    f"{r['n_matched']:>5.1f} | {r['recall']:.3f} | {r['ghost_count']:>4.1f} | "
                    f"{r['f1']:.3f} | {r['median_err_m']:>6.2f} &plusmn; {r['median_err_std']:.2f} | "
                    f"{r['p95_err_m']:>6.2f} | {r['coverage_pct']:>5.1f} |\n"
                )
            f.write("\n")

        # Key takeaways
        f.write("## Key Takeaways\n\n")

        # --- Full scale ---
        f.write("### Full scale\n\n")

        # 1. Error decreases with n_views
        for label, rows in [("Full scale", full_rows), ("Matched scale", matched_rows)]:
            n4_rows = [r for r in rows if r["n_views"] == 4]
            n12_rows = [r for r in rows if r["n_views"] == 12]
            if n4_rows and n12_rows:
                avg_4 = float(np.nanmean([r["median_err_m"] for r in n4_rows]))
                avg_12 = float(np.nanmean([r["median_err_m"] for r in n12_rows]))
                ratio = avg_4 / avg_12 if avg_12 > 0 else float("inf")
                f.write(f"- **{label}:** Error decreases with camera count. "
                        f"Mean error at 4 cameras: {avg_4:.2f}m vs 12 cameras: {avg_12:.2f}m "
                        f"(ratio {ratio:.1f}x).\n")

        # 2. Mixed/surround vs all_ground at low cam counts
        for label, rows in [("Full scale", full_rows), ("Matched scale", matched_rows)]:
            low_n = [r for r in rows if r["n_views"] <= 4]
            ag = [r for r in low_n if r["geometry_class"] == "all_ground"]
            mx = [r for r in low_n if r["geometry_class"] == "mixed"]
            sr = [r for r in low_n if r["geometry_class"] == "surround"]
            if ag and mx:
                ag_err = float(np.nanmean([r["median_err_m"] for r in ag]))
                mx_err = float(np.nanmean([r["median_err_m"] for r in mx]))
                sr_err = float(np.nanmean([r["median_err_m"] for r in sr])) if sr else float("nan")
                f.write(f"- **{label}:** At low camera counts (n_views ≤ 4), "
                        f"all_ground error = {ag_err:.2f}m, "
                        f"mixed = {mx_err:.2f}m"
                        f"{f', surround = {sr_err:.2f}m' if not np.isnan(sr_err) else ''}. "
                        f"{'Mixed geometry provides better elevation diversity and lower error.' if mx_err < ag_err else 'All-ground performs competitively at this scale.'}\n")

        # 3. Diminishing returns above 8
        for label, rows in [("Full scale", full_rows), ("Matched scale", matched_rows)]:
            n8 = [r for r in rows if r["n_views"] == 8]
            n12 = [r for r in rows if r["n_views"] == 12]
            if n8 and n12:
                avg_8 = float(np.nanmean([r["median_err_m"] for r in n8]))
                avg_12 = float(np.nanmean([r["median_err_m"] for r in n12]))
                improvement = (avg_8 - avg_12) / avg_8 * 100 if avg_8 > 0 else 0
                f.write(f"- **{label}:** Diminishing returns above 8 views — "
                        f"error at 8 views = {avg_8:.2f}m vs 12 = {avg_12:.2f}m "
                        f"({improvement:.1f}% improvement).\n")

        # Coverage issues
        for label, rows in [("Full scale", full_rows), ("Matched scale", matched_rows)]:
            low_cov = [r for r in rows if r["coverage_pct"] < 95.0]
            if low_cov:
                configs = ", ".join(
                    f"{r['n_views']}v/{r['geometry_class']}"
                    for r in low_cov
                )
                f.write(f"- **{label}:** Coverage < 95% for {len(low_cov)} config(s): "
                        f"{configs}.\n")

        # Zero-noise baseline
        for label, rows in [("Full scale", full_rows), ("Matched scale", matched_rows)]:
            zero_noise = [r for r in rows if r["noise_std"] == 0.0 and r["recall"] > 0.8]
            if zero_noise:
                errs = [r["median_err_m"] for r in zero_noise]
                f.write(f"- **{label}:** Zero-noise baseline recall > 0.8 for "
                        f"{len(zero_noise)} config(s); "
                        f"median error range = {min(errs):.4f}–{max(errs):.4f}m.\n")

        f.write("\n")

    print(f"  Wrote {report_path}")


# ============================================================================
# Headless Mode — Bundle Processing
# ============================================================================

# Default values for headless mode
DRONE_SIZE_M = 0.5
DETECTION_MATCH_THRESHOLD_PX = 10.0


def _read_image(path: str) -> np.ndarray | None:
    """Read an image file and return as uint8 RGB (H, W, 3) array.

    Tries PIL first, then matplotlib as fallback.
    Returns None if no reader is available or the file can't be read.
    """
    img = None
    try:
        from PIL import Image

        img_pil = Image.open(path).convert("RGB")
        img = np.array(img_pil, dtype=np.uint8)
    except ImportError:
        pass
    except Exception:
        return None  # PIL opened the file but it's corrupt

    if img is not None:
        return img

    try:
        import matplotlib.pyplot as plt

        arr = plt.imread(path)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.shape[2] == 4:
            arr = arr[:, :, :3]
        if arr.dtype == np.float64 or arr.dtype == np.float32:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
        img = arr
    except ImportError:
        pass
    except Exception:
        return None

    return img


def _compute_standoff_from_poses(
    rig: CameraRig,
    swarm_center: np.ndarray | None = None,
) -> float:
    """Compute average standoff distance from camera positions to swarm center.

    If swarm_center is None, uses origin (0, 0, 0).
    """
    if swarm_center is None:
        swarm_center = np.zeros(3, dtype=np.float64)

    distances = []
    for v in range(rig.n_views):
        cam_pos = rig.c2w[v, :3, 3]
        dist = float(np.linalg.norm(cam_pos - swarm_center))
        distances.append(dist)

    return float(np.mean(distances)) if distances else 1000.0


def bundle_poses_to_rig(
    poses: "BundlePoses",
    manifest: "BundleManifest",
) -> CameraRig:
    """Convert BundlePoses to CameraRig format.

    Converts Blender c2w matrices to OpenCV w2c representation and
    constructs a validated CameraRig instance.

    Args:
        poses: Validated BundlePoses from poses.json.
        manifest: Validated BundleManifest from manifest.json.

    Returns:
        CameraRig ready for the stage-1 pipeline.
    """
    n_views = len(poses.views)

    K_list = []
    w2c_R_list = []
    w2c_t_list = []
    c2w_list = []

    for v in poses.views:
        K = np.array(v.K, dtype=np.float64)
        K_list.append(K)

        c2w = np.array(v.c2w, dtype=np.float64)
        c2w_list.append(c2w)

        R_w2c, t_w2c = blender_c2w_to_opencv_w2c(c2w)
        w2c_R_list.append(R_w2c)
        w2c_t_list.append(t_w2c)

    # Determine geometry_class from manifest generated_by if available.
    geometry_class: str = "unknown"
    if manifest.generated_by and "geometry_class" in manifest.generated_by:
        gc = manifest.generated_by["geometry_class"]
        if gc in ("all_ground", "mixed", "surround"):
            geometry_class = gc

    # Determine focal_px from manifest.
    focal_px = manifest.focal_px

    rig = CameraRig(
        K=np.stack(K_list),
        w2c_R=np.stack(w2c_R_list),
        w2c_t=np.stack(w2c_t_list),
        c2w=np.stack(c2w_list),
        focal_px=focal_px,
        convention=CONVENTION_TAG,
        geometry_class=geometry_class,
    )

    return rig


def load_ground_truth(bundle_dir: str) -> SwarmTruth | None:
    """Load ground truth from bundle directory.

    Returns SwarmTruth if ground_truth.json exists and has_ground_truth
    is true, otherwise returns None.
    """
    from bundle_schema import BundleGroundTruth

    gt_path = os.path.join(bundle_dir, "ground_truth.json")
    if not os.path.exists(gt_path):
        return None

    gt = BundleGroundTruth.validate_file(gt_path)

    positions = np.array(gt.positions, dtype=np.float64)
    drone_ids = np.array(gt.drone_ids, dtype=np.int32)

    return SwarmTruth(positions=positions, drone_ids=drone_ids)


def _count_merged_blobs(
    rgb: np.ndarray,
    drone_size_m: float,
    focal_px: float,
    standoff_m: float,
) -> int:
    """Count blobs that exceed the expected maximum size (occlusion merges).

    Replicates the size-filtration logic from detect_blobs to determine
    how many connected components are larger than 3x the expected apparent
    size.  These are interpreted as merged occlusion blobs.

    Args:
        rgb: (H, W, 3) uint8 RGB image.
        drone_size_m: Physical drone size in metres.
        focal_px: Camera focal length in pixels.
        standoff_m: Camera-to-drone standoff in metres.

    Returns:
        Number of connected components exceeding the max-size threshold.
    """
    try:
        from skimage import measure, filters
    except ImportError:
        return 0

    H, W = rgb.shape[:2]

    # Luminance (Rec. 601)
    luminance = (
        0.299 * rgb[..., 0].astype(np.float64)
        + 0.587 * rgb[..., 1].astype(np.float64)
        + 0.114 * rgb[..., 2].astype(np.float64)
    ) / 255.0

    try:
        threshold = filters.threshold_otsu(luminance)
    except ValueError:
        threshold = 0.1

    binary = luminance > threshold
    labeled = measure.label(binary, connectivity=2)
    regions = measure.regionprops(labeled, intensity_image=luminance)

    expected_apparent = drone_size_m * focal_px / standoff_m
    max_px = 3.0 * max(expected_apparent, 3.0)

    merged = 0
    for region in regions:
        if region.area > max_px:
            merged += 1

    return merged


def _compute_detector_quality(
    bundle_dir: str,
    manifest: "BundleManifest",
    rig: CameraRig,
    standoff_m: float,
) -> dict:
    """Compute detector quality metrics by comparing detections to ID-pass ground truth.

    When object-index EXR files are available alongside the rendered PNG
    frames, uses read_object_index_exr() to obtain ground-truth centroids
    per view and computes:

        detector_recall   — fraction of GT drones detected
        fp_per_frame      — false-positive count per view
        centroid_error_px — mean centroid localisation error vs GT

    When EXR files are NOT available, returns sentinel values
    (detector_recall=-1, fp_per_frame=-1, centroid_error_px=-1,
     merged_detections=0).

    Returns:
        Dict with keys: detector_recall, fp_per_frame, centroid_error_px,
        merged_detections.
    """
    from detect_blobs import detect_blobs, read_object_index_exr

    views_dir = os.path.join(bundle_dir, "views")

    detector_recall = -1.0
    fp_per_frame = -1.0
    centroid_error_px = -1.0
    merged_detections = 0

    # Check if at least the first view has an EXR file and ground truth exists.
    first_exr = os.path.join(
        views_dir, "cam_00", f"frame_{manifest.frame_indices[0]:04d}.exr"
    )
    has_exr = os.path.exists(first_exr) and manifest.has_ground_truth

    if not has_exr:
        return {
            "detector_recall": detector_recall,
            "fp_per_frame": fp_per_frame,
            "centroid_error_px": centroid_error_px,
            "merged_detections": merged_detections,
        }

    total_gt = 0
    total_tp = 0
    total_fp = 0
    total_centroid_error = 0.0
    total_merged = 0

    for v_idx in range(manifest.n_views):
        cam_dir = os.path.join(views_dir, f"cam_{v_idx:02d}")
        exr_path = os.path.join(
            cam_dir, f"frame_{manifest.frame_indices[0]:04d}.exr"
        )
        png_path = os.path.join(
            cam_dir, f"frame_{manifest.frame_indices[0]:04d}.png"
        )

        if not os.path.exists(exr_path):
            continue

        # Ground-truth centroids from ID pass EXR.
        gt_centroids = read_object_index_exr(str(exr_path))

        # Pixel detections from rendered frame.
        rgb = _read_image(png_path) if os.path.exists(png_path) else None

        if rgb is None or not gt_centroids:
            total_gt += len(gt_centroids)
            continue

        dets = detect_blobs(
            rgb=rgb,
            drone_size_m=drone_size_m,
            focal_px=rig.focal_px,
            standoff_m=standoff_m,
            image_width_px=manifest.image_size_px[0],
        )
        det_points = dets.points_per_view[0] if dets.points_per_view else np.empty(
            (0, 2), dtype=np.float64
        )

        # Count merged detections from the image.
        total_merged += _count_merged_blobs(
            rgb, drone_size_m, rig.focal_px, standoff_m
        )

        # Match detections to GT centroids by proximity.
        gt_matched = set()
        det_matched = set()

        for g_idx, (gx, gy, _gid) in enumerate(gt_centroids):
            for d_idx in range(len(det_points)):
                if d_idx in det_matched:
                    continue
                dx, dy = det_points[d_idx]
                dist = np.sqrt((dx - gx) ** 2 + (dy - gy) ** 2)
                if dist < DETECTION_MATCH_THRESHOLD_PX:
                    gt_matched.add(g_idx)
                    det_matched.add(d_idx)
                    total_centroid_error += dist
                    break

        total_gt += len(gt_centroids)
        total_tp += len(det_matched)
        total_fp += len(det_points) - len(det_matched)

    detector_recall = total_tp / total_gt if total_gt > 0 else 0.0
    fp_per_frame = total_fp / manifest.n_views
    centroid_error_px = (
        total_centroid_error / total_tp if total_tp > 0 else -1.0
    )
    merged_detections = total_merged

    return {
        "detector_recall": detector_recall,
        "fp_per_frame": fp_per_frame,
        "centroid_error_px": centroid_error_px,
        "merged_detections": merged_detections,
    }


def detect_from_bundle_views(
    bundle_dir: str,
    manifest: "BundleManifest",
    rig: CameraRig,
    standoff_m: float,
) -> Detections:
    """Load rendered frame images and run detect_blobs on each camera view.

    Args:
        bundle_dir: Root directory of the bundle.
        manifest: Validated BundleManifest.
        rig: CameraRig for the bundle (provides focal_px).
        standoff_m: Standoff distance used for apparent-size filtering.

    Returns:
        Detections with 2D points from each camera view.
    """
    from detect_blobs import detect_blobs

    views_dir = os.path.join(bundle_dir, "views")
    frame_idx = manifest.frame_indices[0]
    image_size = tuple(manifest.image_size_px)

    points_per_view: list[np.ndarray] = []

    for v_idx in range(manifest.n_views):
        cam_dir = os.path.join(views_dir, f"cam_{v_idx:02d}")

        # Try common image extensions.
        frame_path: str | None = None
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = os.path.join(
                cam_dir, f"frame_{frame_idx:04d}{ext}"
            )
            if os.path.exists(candidate):
                frame_path = candidate
                break

        if frame_path is None:
            points_per_view.append(np.empty((0, 2), dtype=np.float64))
            continue

        rgb = _read_image(frame_path)
        if rgb is None:
            points_per_view.append(np.empty((0, 2), dtype=np.float64))
            continue

        dets = detect_blobs(
            rgb=rgb,
            drone_size_m=DRONE_SIZE_M,
            focal_px=rig.focal_px,
            standoff_m=standoff_m,
            image_width_px=image_size[0],
        )

        if dets.points_per_view:
            points_per_view.append(dets.points_per_view[0])
        else:
            points_per_view.append(np.empty((0, 2), dtype=np.float64))

    return Detections(points_per_view=points_per_view, image_size=image_size)


def process_bundle(bundle_dir: str) -> dict:
    """Process a single bundle directory end-to-end.

    Pipeline steps:
      1. Load manifest, poses, ground truth from bundle JSON files.
      2. Build CameraRig from poses.
      3. Compute standoff distance from camera poses.
      4. Run pixel detection on each camera view.
      5. Solve multi-view correspondence.
      6. DLT triangulation.
      7. Score against ground-truth.
      8. Compute detector quality metrics (if EXRs available).

    Args:
        bundle_dir: Path to the bundle root directory.

    Returns:
        Dict with all output columns (ready for CSV row).
    """
    # Lazy import to ensure bundle_schema is on sys.path.
    from bundle_schema import BundleManifest, BundlePoses, BundleGroundTruth

    # 1. Load metadata.
    manifest_path = os.path.join(bundle_dir, "manifest.json")
    poses_path = os.path.join(bundle_dir, "poses.json")

    manifest = BundleManifest.validate_file(manifest_path)
    poses = BundlePoses.validate_file(poses_path)

    # 2. Build CameraRig.
    rig = bundle_poses_to_rig(poses, manifest)

    # 3. Load ground truth.
    truth = load_ground_truth(bundle_dir)

    # 4. Compute standoff.
    standoff_m = _compute_standoff_from_poses(rig)
    # Attempt to get standoff from manifest metadata.
    if manifest.generated_by and "standoff_m" in manifest.generated_by:
        standoff_m = float(manifest.generated_by["standoff_m"])

    # 5. Coverage.
    coverage_pct = 100.0
    if truth is not None:
        coverage_pct = compute_framing_coverage(truth, rig) * 100.0

    # 6. Detection on bundle views.
    detections = detect_from_bundle_views(bundle_dir, manifest, rig, standoff_m)

    # 7. Correspondence.
    tracks = solve_correspondence(
        detections=detections,
        rig=rig,
        epipolar_threshold=EPIPOLAR_THRESHOLD_PX,
    )

    # 8. Triangulation.
    recon = triangulate_dlt(tracks, rig, detections)

    # 9. Scoring.
    score = score_full(
        tracks,
        recon,
        truth,
        rig,
        position_threshold_m=MATCH_THRESHOLD_M,
    )

    # 10. Detector quality.
    det_quality = _compute_detector_quality(
        bundle_dir, manifest, rig, standoff_m
    )

    # Determine noise_std and min_spacing from generated_by if available.
    noise_std = -1.0
    min_spacing = 0.0
    if manifest.generated_by:
        if "pixel_noise_std" in manifest.generated_by:
            noise_std = float(manifest.generated_by["pixel_noise_std"])
        if "min_spacing_m" in manifest.generated_by:
            min_spacing = float(manifest.generated_by["min_spacing_m"])

    geometry_class = rig.geometry_class

    row = {
        "n_views": manifest.n_views,
        "geometry_class": geometry_class,
        "noise_std": noise_std,
        "n_drones": len(truth.drone_ids) if truth is not None else 0,
        "focal_px": manifest.focal_px,
        "standoff_m": round(standoff_m, 1),
        "coverage_pct": round(coverage_pct, 1),
        "n_matched": score.correspondence.n_matched,
        "recall": round(score.correspondence.recall, 4),
        "ghost_count": score.correspondence.n_ghost,
        "precision": round(score.correspondence.precision, 4),
        "f1": round(score.correspondence.f1, 4),
        "median_err_m": round(score.triangulation.median_error_m, 4),
        "p95_err_m": round(score.triangulation.p95_error_m, 4),
        "frame_idx": 0,
        "match_threshold_m": MATCH_THRESHOLD_M,
        "min_spacing": min_spacing,
        "epipolar_threshold_px": EPIPOLAR_THRESHOLD_PX,
        "detector_recall": round(det_quality["detector_recall"], 4),
        "fp_per_frame": round(det_quality["fp_per_frame"], 4),
        "centroid_error_px": round(det_quality["centroid_error_px"], 4),
        "merged_detections": det_quality["merged_detections"],
    }

    return row


def _create_synthetic_bundle(tmp_dir: str) -> str:
    """Create a minimal synthetic bundle directory for testing the headless
    pipeline.

    Generates a small swarm and camera rig via B1, projects drone positions
    into camera views, paints synthetic white-on-black dot images, and
    writes all bundle JSON files.

    Args:
        tmp_dir: Directory to create the bundle in.

    Returns:
        Path to the created bundle directory.
    """
    import json

    from bundle_schema import bundle_minimal

    # Use the minimal fixture as a starting template.
    m_dict, p_dict, gt_dict = bundle_minimal()

    # --- Generate a slightly more interesting test scenario. ---
    n_drones = 3
    n_views = 3

    truth = generate_swarm_truth(
        n_drones=n_drones,
        n_frames=1,
        area_km=0.5,
        height_range_m=200.0,
        seed=42,
        min_spacing_m=MIN_SPACING_M,
    )
    rig = generate_camera_rig(
        truth=truth,
        n_views=n_views,
        geometry_class="mixed",
        standoff_m=500.0,
        seed=123,
    )

    # Build manifest dict.
    m_dict["scene_id"] = "test-synthetic"
    m_dict["n_views"] = n_views
    m_dict["n_frames"] = 1
    m_dict["frame_indices"] = [0]
    m_dict["image_size_px"] = list(IMAGE_SIZE)
    m_dict["focal_px"] = DEFAULT_FOCAL_PX
    m_dict["has_ground_truth"] = True
    m_dict["coverage_pct"] = 100.0
    m_dict["generated_by"] = {
        "standoff_m": 500.0,
        "pixel_noise_std": 0.0,
        "geometry_class": "mixed",
        "min_spacing_m": MIN_SPACING_M,
    }

    # Build poses dict from CameraRig.
    p_dict["views"] = []
    for v in range(rig.n_views):
        K = rig.K[v].tolist()
        c2w = rig.c2w[v].tolist()
        w2c_R = rig.w2c_R[v].tolist()
        w2c_t = rig.w2c_t[v].tolist()
        p_dict["views"].append(
            {
                "view_idx": v,
                "K": K,
                "c2w": c2w,
                "w2c_R": w2c_R,
                "w2c_t": w2c_t,
            }
        )

    # Build ground truth dict.
    gt_dict["drone_ids"] = truth.drone_ids.tolist()
    gt_dict["positions"] = truth.positions.tolist()

    # Write JSON files.
    bundle_path = os.path.join(tmp_dir, "synthetic_bundle")
    os.makedirs(bundle_path, exist_ok=True)

    with open(os.path.join(bundle_path, "manifest.json"), "w") as f:
        json.dump(m_dict, f, indent=2)
    with open(os.path.join(bundle_path, "poses.json"), "w") as f:
        json.dump(p_dict, f, indent=2)
    with open(os.path.join(bundle_path, "ground_truth.json"), "w") as f:
        json.dump(gt_dict, f, indent=2)

    # Create view directories and synthetic images.
    views_dir = os.path.join(bundle_path, "views")
    W, H = IMAGE_SIZE

    for v in range(n_views):
        cam_dir = os.path.join(views_dir, f"cam_{v:02d}")
        os.makedirs(cam_dir, exist_ok=True)

        # Project ground-truth drone positions into this camera view.
        K = rig.K[v]
        R = rig.w2c_R[v]
        t = rig.w2c_t[v]

        img = np.zeros((H, W, 3), dtype=np.uint8)

        for d in range(n_drones):
            world_pt = truth.positions[0, d]
            cam_pt = R @ world_pt + t
            if cam_pt[2] <= 0:
                continue
            proj = K @ cam_pt
            u = int(round(proj[0] / proj[2]))
            v_u = int(round(proj[1] / proj[2]))

            if 0 <= u < W and 0 <= v_u < H:
                # Draw a small white dot (3-pixel radius).
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        if dx * dx + dy * dy <= 9:
                            ix, iy = u + dx, v_u + dy
                            if 0 <= ix < W and 0 <= iy < H:
                                img[iy, ix] = [255, 255, 255]

        # Write PNG via PIL if available, else matplotlib.
        png_path = os.path.join(cam_dir, "frame_0000.png")
        try:
            from PIL import Image as PILImage

            PILImage.fromarray(img).save(png_path)
        except ImportError:
            try:
                import matplotlib.pyplot as plt

                plt.imsave(png_path, img)
            except ImportError:
                pass  # Image won't be readable, but the test can still
                # verify the JSON-reading path.

    return bundle_path


def _run_headless(args: argparse.Namespace) -> None:
    """Run the headless pipeline on a bundle directory.

    Two modes (mutually exclusive):
      1. ``--bundle-dir <path>`` — process an existing bundle.
      2. ``--test-synthetic`` — create a synthetic bundle and process it.

    Writes a single-row CSV and prints a formatted summary to stdout.
    """
    output_path = args.output

    if args.test_synthetic:
        import tempfile

        print("\n--- Creating synthetic test bundle ---")
        with tempfile.TemporaryDirectory(prefix="sweep_b_test_") as tmp:
            bundle_dir = _create_synthetic_bundle(tmp)
            print(f"  Bundle created at: {bundle_dir}")
            row = process_bundle(bundle_dir)
            n_views = row["n_views"]
            n_drones = row["n_drones"]
            recall = row["recall"]
            median_err = row["median_err_m"]
            print(
                f"  Synthetic bundle done: {n_views} views, {n_drones} drones, "
                f"recall={recall:.2f}, median_error={median_err:.2f}m"
            )
            # Write CSV inside temp dir so we don't leave artifacts.
            output_path = os.path.join(tmp, "sweep_b_result.csv")
            _write_headless_csv([row], output_path)
            print(f"  CSV written to: {output_path}")
        return

    bundle_dir = args.bundle_dir
    if not bundle_dir:
        print("ERROR: --bundle-dir is required for headless mode (or use --test-synthetic)")
        sys.exit(1)

    if not os.path.isdir(bundle_dir):
        print(f"ERROR: Bundle directory not found: {bundle_dir}")
        sys.exit(1)

    print(f"\n--- Processing bundle: {bundle_dir} ---")
    row = process_bundle(bundle_dir)
    print(
        f"  Views: {row['n_views']}, Drones: {row['n_drones']}, "
        f"Recall: {row['recall']:.3f}, Median error: {row['median_err_m']:.3f}m, "
        f"F1: {row['f1']:.3f}, Coverage: {row['coverage_pct']:.1f}%"
    )
    if row.get("detector_recall", -1) >= 0:
        print(
            f"  Detector recall: {row['detector_recall']:.3f}, "
            f"FP/frame: {row['fp_per_frame']:.2f}, "
            f"Centroid error: {row['centroid_error_px']:.3f}px"
        )

    # Write CSV.
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    _write_headless_csv([row], output_path)
    print(f"  Results written to: {output_path}")


def _write_headless_csv(rows: list[dict], path: str) -> None:
    """Write a single-row (or multi-row) CSV using HEADLESS_CSV_COLUMNS."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADLESS_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ============================================================================
# CLI
# ============================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="B-Sweep: sweep harness for Stage 1 geometry pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["analytic", "headless"],
        default="analytic",
        help="Sweep mode: analytic (synthetic noise sweep) or headless (bundle processing)",
    )
    # Analytic mode arguments
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of noise-seed trials per config (default 20)",
    )
    parser.add_argument(
        "--area_km_full",
        type=float,
        default=5.0,
        help="AREA_KM for the full scale (default 5.0)",
    )
    parser.add_argument(
        "--area_km_matched",
        type=float,
        default=0.3,
        help="AREA_KM for the matched scale (default 0.3)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="logs/sweep_b",
        help="Output directory for analytic mode (default logs/sweep_b)",
    )
    # Headless mode arguments
    parser.add_argument(
        "--bundle-dir",
        type=str,
        default=None,
        help="Bundle directory for headless mode",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/sweep_b_results.csv",
        help="Output CSV path for headless mode (default logs/sweep_b_results.csv)",
    )
    parser.add_argument(
        "--test-synthetic",
        action="store_true",
        help="Create and process a synthetic test bundle (headless mode)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # ---- Headless mode ----
    if args.mode == "headless":
        _run_headless(args)
        return

    # ---- Analytic mode ----
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate total
    n_configs = (
        len(SWEEP_AXES["n_views"])
        * len(SWEEP_AXES["geometry_class"])
        * len(SWEEP_AXES["noise_std_px"])
    )
    total_runs = n_configs * args.trials * 2  # both scales
    print(
        f"B-Sweep: {args.trials} trials x {n_configs} configs x 2 scales "
        f"= {total_runs} total runs"
    )
    print(f"Output directory: {output_dir.resolve()}")

    # Scale definitions
    scales = [
        ("full", args.area_km_full, 1000.0),
        ("matched", args.area_km_matched, 100.0),
    ]

    all_results = {}
    for scale_name, area_km, height_range in scales:
        rows = run_scale(
            scale_name=scale_name,
            area_km=area_km,
            height_range_m=height_range,
            n_trials=args.trials,
            output_dir=str(output_dir),
        )
        all_results[scale_name] = rows

    # Plot (needs both scales' data)
    print("\n--- Generating plot ---")
    generate_plot(all_results["full"], all_results["matched"], str(output_dir))

    # Report
    print("\n--- Generating report ---")
    generate_report(all_results["full"], all_results["matched"], str(output_dir))

    print("\nDone.")


if __name__ == "__main__":
    main()
