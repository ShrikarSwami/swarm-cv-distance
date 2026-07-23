"""
Cross-reference the M1 optics sweep (apparent pixel sizes) with the
coverage simulation (actual ≥2-view camera counts from dome placement).

The optics sweep's 'cameras_for_coverage' is a 1D linear tiling —
optimistic. The coverage sim is 2D dome placement — realistic.
This script finds configs where BOTH constraints are satisfied simultaneously
at TRUE SCALE (no display inflation).

Also extends the sweep to closer standoffs (100-500m) to cover the
under-explored regime.
"""

import json
import math
import numpy as np
from optics_sweep import (
    angular_resolution_um_rad, apparent_pixels, horizontal_fov_deg,
    DRONE_SIZE_M, SWARM_WIDTH_M, DETECTOR_BANDS, SENSORS, FOCAL_LENGTHS,
)
from coverage_sim import (
    make_swarm, horizontal_fov_rad, vertical_fov_rad,
    find_min_cams, camera_fov_mask,
)

# ---------------------------------------------------------------------------
# Extended standoff range (closer than M1's 500m floor)
# ---------------------------------------------------------------------------
EXTENDED_STANDOFFS = [100, 150, 200, 250, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 5000, 7500, 10000]

# Configs that could plausibly satisfy both constraints
# (wide lenses for coverage, moderate standoffs for pixels)
CONFIGS_TO_CHECK = [
    # (label, sensor_mm, focal_mm, h_px, v_px, max_cameras)
    # Ground post tier
    ("24mm FF 1920",     36.0,  24, 1920, 1080, 12),
    ("24mm FF 6000",     36.0,  24, 6000, 4000, 12),
    ("50mm FF 1920",     36.0,  50, 1920, 1080, 12),
    ("50mm FF 6000",     36.0,  50, 6000, 4000, 12),
    # Airborne tier
    ("24mm APS-C 1920",  23.5,  24, 1920, 1080, 6),
    ("24mm APS-C 6000",  23.5,  24, 6000, 4000, 6),
    ("50mm APS-C 1920",  23.5,  50, 1920, 1080, 6),
    ("50mm APS-C 6000",  23.5,  50, 6000, 4000, 6),
    ("24mm 1-inch 1920", 13.2,  24, 1920, 1080, 6),
    # Hybrid / close range
    ("24mm FF 8192",     36.0,  24, 8192, 5464, 12),
    ("50mm FF 8192",     36.0,  50, 8192, 5464, 12),
    ("50mm APS-C 6000h", 23.5,  50, 6000, 4000, 6),
]


def main():
    swarm = make_swarm(20, seed=42)

    print("=" * 120)
    print("CROSS-REFERENCE: coverage simulation + apparent pixel size (TRUE SCALE, no display inflation)")
    print("=" * 120)
    print()

    all_results = []

    for label, sensor_mm, focal_mm, h_px, v_px, max_cams in CONFIGS_TO_CHECK:
        theta = angular_resolution_um_rad(sensor_mm, focal_mm, h_px)
        h_fov = horizontal_fov_rad(sensor_mm, focal_mm)
        v_fov = vertical_fov_rad(sensor_mm, focal_mm, h_px, v_px)
        h_fov_deg = math.degrees(h_fov)

        for standoff in EXTENDED_STANDOFFS:
            n_cams, view_counts = find_min_cams(swarm, standoff, h_fov, v_fov,
                                                 target_coverage=2, max_cams=30)
            min_v = view_counts.min()
            px_size = apparent_pixels(DRONE_SIZE_M, standoff, theta)
            passes_coverage = (min_v >= 2) and (n_cams <= max_cams)

            # Check all detector bands
            viable_bands = []
            for band_name, band in DETECTOR_BANDS.items():
                if px_size >= band["min_px"]:
                    viable_bands.append(band_name)

            r = {
                "config": label,
                "standoff_m": standoff,
                "h_fov_deg": round(h_fov_deg, 2),
                "n_cams_for_2view": n_cams,
                "min_views": min_v,
                "passes_coverage": passes_coverage,
                "apparent_px": round(px_size, 2),
                "viable_bands": viable_bands,
                "max_cams_allowed": max_cams,
            }
            all_results.append(r)

    # Filter to configs that pass BOTH constraints
    print("\n" + "=" * 120)
    print("CONFIGS PASSING ≥2-VIEW COVERAGE (within camera budget)")
    print("=" * 120)
    print(f"\n{'Config':24s} {'Standoff':>8s} {'hFOV':>6s} {'Cams':>4s} {'MinV':>4s} {'Px':>7s} {'Bands'}")
    print("-" * 100)

    passing = [r for r in all_results if r["passes_coverage"]]
    for r in passing:
        bands_str = ", ".join(r["viable_bands"]) if r["viable_bands"] else "none"
        print(f"{r['config']:24s} {r['standoff_m']:>7,}m {r['h_fov_deg']:>5.1f}° "
              f"{r['n_cams_for_2view']:>4d} {r['min_views']:>4d} {r['apparent_px']:>6.2f}px {bands_str}")

    # Now the critical question: which of those ALSO have detectable pixels?
    print("\n" + "=" * 120)
    print("CONFIGS PASSING BOTH: ≥2-view coverage AND ≥8px (bbox detector)")
    print("=" * 120)
    bbox_pass = [r for r in passing if "bbox_detector" in r["viable_bands"]]
    if bbox_pass:
        for r in bbox_pass:
            print(f"  ✓ {r['config']} @ {r['standoff_m']:,}m: {r['apparent_px']:.2f}px, "
                  f"{r['n_cams_for_2view']} cams, min_views={r['min_views']}")
    else:
        print("  NONE FOUND.")

    print("\n" + "=" * 120)
    print("CONFIGS PASSING BOTH: ≥2-view coverage AND ≥3px (centroid detector)")
    print("=" * 120)
    centroid_pass = [r for r in passing if "centroid_known_size" in r["viable_bands"]]
    if centroid_pass:
        for r in centroid_pass:
            print(f"  ✓ {r['config']} @ {r['standoff_m']:,}m: {r['apparent_px']:.2f}px, "
                  f"{r['n_cams_for_2view']} cams, min_views={r['min_views']}")
    else:
        print("  NONE FOUND.")

    print("\n" + "=" * 120)
    print("CONFIGS PASSING BOTH: ≥2-view coverage AND ≥1px (subpixel/temporal)")
    print("=" * 120)
    sub_pass = [r for r in passing if "subpixel_temporal" in r["viable_bands"]]
    if sub_pass:
        for r in sub_pass:
            print(f"  ✓ {r['config']} @ {r['standoff_m']:,}m: {r['apparent_px']:.2f}px, "
                  f"{r['n_cams_for_2view']} cams, min_views={r['min_views']}")
    else:
        print("  NONE FOUND.")

    # Summary: what's the best we can do?
    print("\n" + "=" * 120)
    print("SUMMARY: Best true-scale configs by detector class")
    print("=" * 120)

    for band_name, band_info in DETECTOR_BANDS.items():
        min_px = band_info["min_px"]
        candidates = [r for r in passing if r["apparent_px"] >= min_px]
        candidates.sort(key=lambda r: -r["apparent_px"])
        print(f"\n  {band_name} (≥{min_px}px):")
        if candidates:
            for r in candidates[:5]:
                print(f"    {r['config']:24s} @ {r['standoff_m']:>6,}m → "
                      f"{r['apparent_px']:.2f}px, {r['n_cams_for_2view']} cams, "
                      f"min_views={r['min_views']}")
        else:
            print("    No configs satisfy both coverage and resolution.")

    # Also show: what pixel size does the BEST coverage config give?
    print("\n" + "=" * 120)
    print("ANALYSIS: Best coverage configs and their true-scale pixel sizes")
    print("=" * 120)

    # Group by config, find minimum standoff that achieves coverage
    configs_seen = set()
    for r in passing:
        cfg = r["config"]
        if cfg in configs_seen:
            continue
        configs_seen.add(cfg)
        print(f"\n  {cfg}:")
        cfg_results = [x for x in passing if x["config"] == cfg]
        cfg_results.sort(key=lambda x: x["standoff_m"])
        for x in cfg_results:
            print(f"    {x['standoff_m']:>6,}m → {x['apparent_px']:.2f}px "
                  f"({x['n_cams_for_2view']} cams, min_views={x['min_views']})")

    # Save full results
    with open("logs/m1_sweep/cross_reference.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to logs/m1_sweep/cross_reference.json")


if __name__ == "__main__":
    main()
