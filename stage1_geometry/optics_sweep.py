"""
M1 Optics/standoff trade study — analytical sweep.

Computes angular resolution, apparent pixel size on a 0.5m drone,
FOV coverage, and minimum camera count for all configs across three
platform tiers (ground post, airborne UAS, cheap commodity).

Run: python stage1_geometry/optics_sweep.py
"""

import json
import math
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DRONE_SIZE_M = 0.5
SWARM_WIDTH_M = 5000.0  # 5km scene
D_MAX_M = 3949.0

# Detector-class thresholds (rules of thumb, NOT hard physics — see spec)
DETECTOR_BANDS = {
    "bbox_detector": {"min_px": 8.0, "note": "COCO-scale training heuristic; detector-specific"},
    "centroid_known_size": {"min_px": 3.0, "note": "Requires known dimensions + clean background; degrades with terrain clutter"},
    "subpixel_temporal": {"min_px": 1.0, "note": "Requires known target, clean background (sky), or temporal consistency. Not unconditional."},
}

# ---------------------------------------------------------------------------
# Sensor classes (real specs from manufacturer datasheets)
# ---------------------------------------------------------------------------

@dataclass
class Sensor:
    name: str
    width_mm: float
    resolutions: dict  # label -> (h_px, v_px)

FULL_FRAME = Sensor("full-frame", 36.0, {
    "R0": (1280, 720), "R1": (1920, 1080),
    "R2": (6000, 4000), "R3": (8192, 5464),
})
APS_C = Sensor("APS-C", 23.5, {
    "R0": (1280, 720), "R1": (1920, 1080),
    "R2": (6000, 4000), "R3": (8192, 5464),
})
ONE_INCH = Sensor("1-inch", 13.2, {
    "R0": (1280, 720), "R1": (1920, 1080),
    "R2": (5472, 3648),
})
ONE_TWO3 = Sensor("1/2.3-inch", 6.17, {
    "R0": (1280, 720), "R1": (1920, 1080),
    "R2": (4608, 3456),
})

# ---------------------------------------------------------------------------
# Focal lengths per sensor class (actual, not equivalent)
# ---------------------------------------------------------------------------

FOCAL_LENGTHS = {
    "full-frame": [24, 50, 100, 200, 400, 800, 1200],
    "APS-C":      [24, 50, 100, 200, 400, 800],
    "1-inch":     [24, 50, 100, 200, 400],
    "1/2.3-inch": [24, 50, 100, 200, 400, 539],  # 539mm = P1000 max zoom actual
}

# ---------------------------------------------------------------------------
# Standoff distances
# ---------------------------------------------------------------------------

STANDOFFS = [500, 750, 1000, 2000, 3000, 5000, 7500, 10000]

# ---------------------------------------------------------------------------
# Platform tiers
# ---------------------------------------------------------------------------

PLATFORM_TIERS = {
    "A_ground_post": {
        "sensor_classes": ["full-frame", "APS-C"],
        "max_focal_mm": 1200,
        "max_standoff_m": 10000,
        "resolution_labels": ["R0", "R1", "R2", "R3"],
        "max_cameras": 12,
        "description": "Unconstrained optics, mains power, tripod-stable. Max 12 observation posts.",
    },
    "B_airborne_uas": {
        "sensor_classes": ["APS-C", "1-inch"],
        "max_focal_mm": 200,
        "max_standoff_m": 5000,
        "resolution_labels": ["R0", "R1", "R2"],
        "max_cameras": 6,
        "description": "Weight/power/stability constrained. Max 6 UAS observers.",
    },
    "C_cheap_commodity": {
        "sensor_classes": ["1/2.3-inch", "1-inch"],
        "max_focal_mm": 539,
        "max_standoff_m": 10000,
        "resolution_labels": ["R0", "R1"],
        "max_cameras": 8,
        "description": "Floor-tier baseline, both regimes. Max 8 cheap units.",
    },
}

SENSORS = {"full-frame": FULL_FRAME, "APS-C": APS_C, "1-inch": ONE_INCH, "1/2.3-inch": ONE_TWO3}

# ---------------------------------------------------------------------------
# Core optics computations
# ---------------------------------------------------------------------------

def angular_resolution_um_rad(sensor_width_mm: float, focal_mm: float, h_pixels: int) -> float:
    """Angular resolution in microradians per pixel."""
    return (sensor_width_mm * 1e-3) / (focal_mm * 1e-3 * h_pixels) * 1e6

def horizontal_fov_deg(sensor_width_mm: float, focal_mm: float) -> float:
    """Horizontal field of view in degrees."""
    return 2.0 * math.degrees(math.atan(sensor_width_mm / (2.0 * focal_mm)))

def apparent_pixels(drone_size_m: float, standoff_m: float, theta_um_rad: float) -> float:
    """Apparent pixel extent on a target of given size at given standoff."""
    theta_rad = theta_um_rad * 1e-6
    return drone_size_m / (standoff_m * theta_rad)

def cameras_for_full_coverage(h_fov_deg: float, swarm_width_m: float, standoff_m: float) -> int:
    """Minimum cameras for full swarm coverage (linear tiling, 1D)."""
    angular_width = 2.0 * math.degrees(math.atan(swarm_width_m / (2.0 * standoff_m)))
    if angular_width <= h_fov_deg:
        return 1
    return math.ceil(angular_width / h_fov_deg)

# ---------------------------------------------------------------------------
# Sanity check: same angular resolution → same pixel size
# ---------------------------------------------------------------------------

def run_sanity_check():
    """Verify that two configs with identical angular resolution produce
    identical apparent pixel sizes. Returns (passed, details)."""
    # Config A: full-frame 36mm, 800mm, 8192px
    theta_a = angular_resolution_um_rad(36.0, 800, 8192)
    px_a = apparent_pixels(DRONE_SIZE_M, 5000, theta_a)

    # Config B: APS-C 23.5mm, 527.78mm, 5400px (chosen to match theta_a)
    # theta_a = 36e-3 / (800e-3 * 8192) = 5.493e-6 rad = 5.493 um/px
    # For B: 23.5e-3 / (f * 5400) = 5.493e-6 => f = 23.5e-3 / (5.493e-6 * 5400) = 0.7895 m = 789.5 mm
    f_b = 23.5e-3 / (5.493e-6 * 5400) * 1e3  # in mm
    theta_b = angular_resolution_um_rad(23.5, f_b, 5400)
    px_b = apparent_pixels(DRONE_SIZE_M, 5000, theta_b)

    passed = abs(theta_a - theta_b) < 0.01 and abs(px_a - px_b) < 0.01
    details = {
        "config_a": {"sensor_mm": 36.0, "focal_mm": 800, "h_px": 8192,
                      "theta_um_rad": round(theta_a, 4), "px_at_5km": round(px_a, 2)},
        "config_b": {"sensor_mm": 23.5, "focal_mm": round(f_b, 2), "h_px": 5400,
                      "theta_um_rad": round(theta_b, 4), "px_at_5km": round(px_b, 2)},
        "theta_match": abs(theta_a - theta_b) < 0.01,
        "px_match": abs(px_a - px_b) < 0.01,
    }
    return passed, details

# ---------------------------------------------------------------------------
# Full analytical sweep
# ---------------------------------------------------------------------------

def sweep_all():
    """Run the full analytical sweep across all platform tiers.
    Returns list of config dicts with all computed metrics."""
    results = []

    for tier_name, tier in PLATFORM_TIERS.items():
        for sensor_label in tier["sensor_classes"]:
            sensor = SENSORS[sensor_label]
            for focal_mm in FOCAL_LENGTHS[sensor_label]:
                if focal_mm > tier["max_focal_mm"]:
                    continue
                for res_label in tier["resolution_labels"]:
                    if res_label not in sensor.resolutions:
                        continue
                    h_px, v_px = sensor.resolutions[res_label]
                    theta = angular_resolution_um_rad(sensor.width_mm, focal_mm, h_px)
                    h_fov = horizontal_fov_deg(sensor.width_mm, focal_mm)

                    for standoff_m in STANDOFFS:
                        if standoff_m > tier["max_standoff_m"]:
                            continue

                        px_size = apparent_pixels(DRONE_SIZE_M, standoff_m, theta)
                        n_cams = cameras_for_full_coverage(h_fov, SWARM_WIDTH_M, standoff_m)

                        # Determine which detector bands this config supports
                        viable_bands = []
                        for band_name, band in DETECTOR_BANDS.items():
                            if px_size >= band["min_px"]:
                                viable_bands.append(band_name)

                        # Check if this config exceeds UAS payload (Tier B)
                        exceeds_uas = (tier_name == "B_airborne_uas" and focal_mm > 200)
                        unlikely_range = (tier_name == "B_airborne_uas" and standoff_m > 5000)

                        results.append({
                            "tier": tier_name,
                            "sensor": sensor_label,
                            "sensor_width_mm": sensor.width_mm,
                            "focal_mm": focal_mm,
                            "resolution": res_label,
                            "h_px": h_px,
                            "v_px": v_px,
                            "standoff_m": standoff_m,
                            "theta_um_rad": round(theta, 4),
                            "h_fov_deg": round(h_fov, 2),
                            "apparent_px": round(px_size, 2),
                            "cameras_for_coverage": n_cams,
                            "viable_bands": viable_bands,
                            "exceeds_uas_payload": exceeds_uas,
                            "unlikely_airborne_range": unlikely_range,
                        })

    return results

# ---------------------------------------------------------------------------
# Decision boundary: max standoff per tier per pixel threshold
# ---------------------------------------------------------------------------

def compute_decision_boundary(results):
    """For each tier and detector band, find the max standoff where apparent
    pixel size meets the threshold AND camera count is practical.
    Returns dict: tier -> band -> {max_standoff_m, config, camera_limit}."""
    bands = list(DETECTOR_BANDS.keys())
    boundaries = {}

    for tier_name, tier_info in PLATFORM_TIERS.items():
        boundaries[tier_name] = {}
        tier_results = [r for r in results if r["tier"] == tier_name]
        max_cams = tier_info["max_cameras"]

        for band_name in bands:
            min_px = DETECTOR_BANDS[band_name]["min_px"]
            # Find max standoff where config meets threshold AND camera count is within limit
            max_standoff = 0
            best_config = None
            for r in tier_results:
                if (r["apparent_px"] >= min_px
                        and r["cameras_for_coverage"] <= max_cams
                        and r["standoff_m"] > max_standoff):
                    max_standoff = r["standoff_m"]
                    best_config = r
            boundaries[tier_name][band_name] = {
                "max_standoff_m": max_standoff,
                "config": best_config,
                "camera_limit": max_cams,
            }

    return boundaries

# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(results, boundaries):
    """Generate human-readable decision report."""
    lines = []
    lines.append("# M1 Optics/Standoff Trade Study — Decision Report")
    lines.append("")
    lines.append(f"Scene: {SWARM_WIDTH_M/1000:.0f}km x {SWARM_WIDTH_M/1000:.0f}km x 1km, drone size {DRONE_SIZE_M}m")
    lines.append(f"D_MAX: {D_MAX_M:.0f}m (provisional, 85% target)")
    lines.append("")

    # Detector bands
    lines.append("## Detector-class thresholds (rules of thumb)")
    lines.append("")
    for band_name, band in DETECTOR_BANDS.items():
        lines.append(f"- **{band_name}**: >= {band['min_px']:.0f} px — {band['note']}")
    lines.append("")

    # Decision boundary
    lines.append("## Decision boundary: max standoff per tier per threshold")
    lines.append("")
    lines.append("All entries respect practical camera-count limits per tier.")
    lines.append("")
    lines.append("| Tier | Max cameras | Bounding-box (≥8px) | Centroid (≥3px) | Sub-pixel (≥1px) |")
    lines.append("|---|---|---|---|---|")
    for tier_name in ["A_ground_post", "B_airborne_uas", "C_cheap_commodity"]:
        max_cams = PLATFORM_TIERS[tier_name]["max_cameras"]
        row = [tier_name, str(max_cams)]
        for band_name in ["bbox_detector", "centroid_known_size", "subpixel_temporal"]:
            info = boundaries[tier_name][band_name]
            s = info["max_standoff_m"]
            if s > 0:
                cfg = info["config"]
                row.append(f"{s:,}m ({cfg['sensor']} {cfg['focal_mm']}mm, {cfg['cameras_for_coverage']} cams)")
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Per-tier summary
    for tier_name, tier_info in PLATFORM_TIERS.items():
        lines.append(f"## Tier: {tier_name} ({tier_info['description']})")
        lines.append("")
        tier_results = [r for r in results if r["tier"] == tier_name]

        # Best configs at key standoffs
        lines.append("### Best configurations at key standoffs (within camera limit)")
        lines.append("")
        lines.append("| Standoff | Best config | θ (µrad/px) | Apparent px | Cameras | Viable bands |")
        lines.append("|---|---|---|---|---|---|")
        max_cams = tier_info["max_cameras"]
        for s in STANDOFFS:
            if s > tier_info["max_standoff_m"]:
                continue
            s_results = [r for r in tier_results if r["standoff_m"] == s
                         and not r["exceeds_uas_payload"]
                         and r["cameras_for_coverage"] <= max_cams]
            if not s_results:
                continue
            # Pick config with most cameras (highest resolution, narrowest FOV)
            best = max(s_results, key=lambda r: r["cameras_for_coverage"])
            bands_str = ", ".join(best["viable_bands"]) if best["viable_bands"] else "none"
            lines.append(
                f"| {s:,}m | {best['sensor']} {best['focal_mm']}mm {best['resolution']} "
                f"| {best['theta_um_rad']:.2f} | {best['apparent_px']:.1f}px "
                f"| {best['cameras_for_coverage']} | {bands_str} |"
            )
        lines.append("")

        # Configs that exceed constraints
        flagged = [r for r in tier_results if r["exceeds_uas_payload"] or r["unlikely_airborne_range"]]
        if flagged:
            lines.append("### Flagged configs")
            lines.append("")
            for r in flagged:
                reason = []
                if r["exceeds_uas_payload"]:
                    reason.append(f"{r['focal_mm']}mm exceeds UAS payload")
                if r["unlikely_airborne_range"]:
                    reason.append(f"{r['standoff_m']:,}m unlikely airborne range")
                lines.append(f"- {r['sensor']} {r['focal_mm']}mm @ {r['standoff_m']:,}m: {', '.join(reason)}")
            lines.append("")

    # Temporal integration note
    lines.append("## Temporal integration caveat")
    lines.append("")
    lines.append("A moving drone across multiple frames is detectable at smaller per-frame")
    lines.append("apparent size than a single-frame detector threshold implies. Estimated")
    lines.append("extension: ~1.5–2× (order of magnitude, not measured). This could shift")
    lines.append("the decision boundary outward. Revisit once dataset temporal characteristics")
    lines.append("are known.")
    lines.append("")

    # Coverage vs resolution tradeoff
    lines.append("## Coverage/resolution tradeoff")
    lines.append("")
    lines.append("Narrower FOV (longer focal length) gives more pixels on target but")
    lines.append("requires more cameras for full swarm coverage. The table above shows")
    lines.append("cameras needed at each config. Key tension:")
    lines.append("")
    lines.append("- Wide-angle (24mm): 1 camera covers the swarm, but drones are subpixel")
    lines.append("  at all but the closest standoffs")
    lines.append("- Telephoto (200mm+): good pixel density, but 3–15 cameras needed")
    lines.append("- The 'sweet spot' depends on how many observation platforms you can field")
    lines.append("")

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Sanity check first
    passed, details = run_sanity_check()
    print("=== SANITY CHECK ===")
    print(f"Config A: {details['config_a']}")
    print(f"Config B: {details['config_b']}")
    print(f"θ match: {details['theta_match']}, px match: {details['px_match']}")
    if not passed:
        print("FAILED — angular resolution math is wrong. Fix before proceeding.")
        sys.exit(1)
    print("PASSED — same angular resolution → same pixel size ✓")
    print()

    # Full sweep
    results = sweep_all()
    print(f"=== SWEEP: {len(results)} configs computed ===")

    # Decision boundary
    boundaries = compute_decision_boundary(results)

    # Generate report
    report = generate_report(results, boundaries)
    print(report)

    # Save results
    out_dir = Path(__file__).parent.parent / "logs" / "m1_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(out_dir / "decision_boundary.json", "w") as f:
        json.dump(boundaries, f, indent=2, default=str)

    with open(out_dir / "decision_report.md", "w") as f:
        f.write(report)

    print(f"\nResults saved to {out_dir}")

if __name__ == "__main__":
    main()
