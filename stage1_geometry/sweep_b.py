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

# Ensure stage1_geometry is on sys.path for direct imports.
_stage1_dir = os.path.dirname(os.path.abspath(__file__))
if _stage1_dir not in sys.path:
    sys.path.insert(0, _stage1_dir)

from data_contract import DEFAULT_FOCAL_PX
from b1_scene_rig import (
    generate_swarm_truth,
    generate_camera_rig,
    compute_framing_coverage,
)
from b2_projection import project_swarm_to_detections
from b3_correspondence import solve_correspondence
from b4_scoring import score_full
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

    Returns dict with trial-level metrics.
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

    return {
        "n_matched": score.correspondence.n_matched,
        "recall": score.correspondence.recall,
        "ghost_count": score.correspondence.n_ghost,
        "precision": score.correspondence.precision,
        "f1": score.correspondence.f1,
        "median_err_m": score.triangulation.median_error_m,
        "p95_err_m": score.triangulation.p95_error_m,
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

    for n_views in n_views_list:
        for geom in geom_list:
            # --- Fixed per (scale, n_views, geom) — same swarm & cameras
            #     regardless of noise_std, so noise comparisons are clean. ---
            scene_seed = _make_seed(scale_name, n_views, geom)
            gen_seed = scene_seed + 1
            rig_seed = scene_seed + 2

            truth = generate_swarm_truth(
                n_drones=N_DRONES,
                n_frames=N_FRAMES,
                area_km=area_km,
                height_range_m=height_range_m,
                seed=gen_seed,
                min_spacing_m=MIN_SPACING_M,
            )
            rig = generate_camera_rig(
                truth=truth,
                n_views=n_views,
                geometry_class=geom,
                standoff_m=STANDOFF_M,
                seed=rig_seed,
            )
            coverage_pct = compute_framing_coverage(truth, rig) * 100.0

            for noise_std in noise_list:
                # --- Trials (only noise / correspondence seeds vary with
                #     noise_std and trial; swarm + rig stay fixed) ---
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
                }
                rows.append(row)

                coverage_flag = " *** COVERAGE < 95% ***" if coverage_pct < 95.0 else ""
                print(
                    f"  n_views={n_views:>2}  {geom:<12}  noise={noise_std:>3.0f}px  "
                    f"matched={row['n_matched']:>5.1f}  recall={row['recall']:.3f}  "
                    f"median={row['median_err_m']:>7.2f}m  "
                    f"f1={row['f1']:.3f}  coverage={coverage_pct:>5.1f}%{coverage_flag}"
                )

    # --- Write CSV ---
    csv_path = os.path.join(output_dir, f"sweep_b_analytic_results_{scale_name}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {csv_path}  ({len(rows)} rows)")

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
# CLI
# ============================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="B-Sweep: analytic sweep for Stage 1 geometry pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["analytic"],
        default="analytic",
        help="Sweep mode (only analytic for now)",
    )
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
        help="Output directory (default logs/sweep_b)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

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
