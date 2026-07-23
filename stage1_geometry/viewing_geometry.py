"""
Viewing geometry analysis: sky vs terrain background fraction.

The temporal detection finding (frame differencing works on both backgrounds)
reframes the design constraint. Even though frame diff cancels static texture,
cameras looking UP at the swarm see sky behind targets (low clutter),
while cameras looking DOWN see terrain (high single-frame clutter).

This script computes, for candidate camera arrangements:
1. What fraction of drone-camera sightlines point toward sky vs terrain
2. The elevation angle distribution (key driver of background type)
3. Whether ≥2-view coverage is achievable with sky-dominated sightlines

The "sky fraction" for a sightline is determined by the look angle:
- Camera below the drone looking up → sightline terminates at sky
- Camera above the drone looking down → sightline terminates at terrain
- Near-horizontal → mixed, depends on actual terrain/sky at that azimuth

Threshold: sightline is "sky-dominated" if the camera is below the drone's
altitude (looking up) or if the elevation angle is <10° above horizontal
(pointing mostly at sky near horizon).
"""

import json
import math
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

SWARM_EXTENT_M = 5000.0
SWARM_HEIGHT_M = 1000.0
N_DRONES = 20
GROUND_Z = 0.0  # terrain at z=0

# Candidate camera arrangements
ARRANGEMENTS = {
    "ground_low_elev": {
        "description": "Ground posts, low elevation (10-30°) — historical edge-on problem",
        "camera_altitude_m": 10.0,  # ground-level tripods
        "elevation_range_deg": (10, 30),
        "max_cameras": 12,
    },
    "ground_mid_elev": {
        "description": "Ground posts, mid elevation (30-50°) — M3 rig style",
        "camera_altitude_m": 10.0,
        "elevation_range_deg": (30, 50),
        "max_cameras": 12,
    },
    "ground_high_elev": {
        "description": "Ground posts, high elevation (50-70°) — steep look-up angle",
        "camera_altitude_m": 10.0,
        "elevation_range_deg": (50, 70),
        "max_cameras": 12,
    },
    "tower_50m": {
        "description": "50m tower posts, moderate elevation",
        "camera_altitude_m": 50.0,
        "elevation_range_deg": (30, 50),
        "max_cameras": 12,
    },
    "tower_100m": {
        "description": "100m tower posts",
        "camera_altitude_m": 100.0,
        "elevation_range_deg": (30, 50),
        "max_cameras": 12,
    },
    "airborne_500m": {
        "description": "Airborne observers at 500m — looking down at swarm",
        "camera_altitude_m": 500.0,
        "elevation_range_deg": (-30, -10),  # negative = looking down
        "max_cameras": 6,
    },
    "airborne_1500m": {
        "description": "Airborne observers at 1500m — well above swarm",
        "camera_altitude_m": 1500.0,
        "elevation_range_deg": (-45, -20),
        "max_cameras": 6,
    },
    "hybrid_low_high": {
        "description": "Hybrid: 4 ground (10-30°) + 4 high tower (50-70°)",
        "camera_altitude_m": None,  # multi-altitude, handled specially
        "elevation_range_deg": None,
        "max_cameras": 8,
        "sub_configs": [
            {"altitude_m": 10.0, "elevation_range_deg": (10, 30), "n_cams": 4},
            {"altitude_m": 10.0, "elevation_range_deg": (50, 70), "n_cams": 4},
        ],
    },
}


def make_swarm(n_drones=N_DRONES, seed=42):
    """Generate random drone positions."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-SWARM_EXTENT_M / 2, SWARM_EXTENT_M / 2, size=(n_drones, 2))
    z = rng.uniform(0, SWARM_HEIGHT_M, size=(n_drones, 1))
    return np.hstack([xy, z])


def horizontal_fov_deg(sensor_mm, focal_mm):
    return math.degrees(2 * math.atan(sensor_mm / (2 * focal_mm)))


def vertical_fov_deg(sensor_mm, focal_mm, h_px, v_px):
    v_sensor = sensor_mm * v_px / h_px
    return math.degrees(2 * math.atan(v_sensor / (2 * focal_mm)))


def camera_in_fov(drone_pos, cam_pos, h_fov_rad, v_fov_rad):
    """Check if a drone is within a camera's FOV."""
    forward = (drone_pos.mean(axis=0) - cam_pos)
    forward = forward / np.linalg.norm(forward)

    up_hint = np.array([0, 0, 1.0])
    right = np.cross(forward, up_hint)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1, 0, 0])
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    diff = drone_pos - cam_pos
    dist = np.linalg.norm(diff, axis=1)
    cos_angle = np.sum(diff * forward[np.newaxis, :], axis=1) / np.maximum(dist, 1e-6)

    proj_right = np.sum(diff * right[np.newaxis, :], axis=1)
    proj_up = np.sum(diff * up[np.newaxis, :], axis=1)

    h_angle = np.arctan2(np.abs(proj_right), np.maximum(cos_angle * dist, 1e-6))
    v_angle = np.arctan2(np.abs(proj_up), np.maximum(cos_angle * dist, 1e-6))

    return (h_angle <= h_fov_rad / 2) & (v_angle <= v_fov_rad / 2) & (cos_angle > 0)


def compute_sky_fraction(drone_pos, cam_pos):
    """Compute what fraction of the sightline behind the drone points at sky.

    Returns 1.0 if the drone is above the camera (looking up = sky behind).
    Returns 0.0 if the drone is below the camera (looking down = terrain).
    Returns intermediate values for near-horizontal sightlines.
    """
    dz = drone_pos[:, 2] - cam_pos[2]  # positive = drone above camera
    # Sightline goes from camera through drone and continues to background
    # If drone is above camera → sightline goes up → sky
    # If drone is below camera → sightline goes down → terrain
    sky = (dz > 0).astype(float)

    # For near-horizontal sightlines (|dz| < 50m), interpolate
    # based on elevation angle
    horiz_dist = np.sqrt((drone_pos[:, 0] - cam_pos[0])**2 +
                         (drone_pos[:, 1] - cam_pos[1])**2)
    elev_angle = np.arctan2(dz, horiz_dist)

    # Transition zone: ±10° around horizontal
    transition_mask = np.abs(elev_angle) < math.radians(10)
    sky[transition_mask] = 0.5 + 0.5 * np.sin(elev_angle[transition_mask] / math.radians(10) * math.pi / 2)

    return sky


def analyze_arrangement(name, config, swarm):
    """Analyze one camera arrangement."""
    h_fov = math.radians(horizontal_fov_deg(36.0, 24.0))
    v_fov = math.radians(vertical_fov_deg(36.0, 24.0, 1920, 1080))
    center = swarm.mean(axis=0)

    def place_and_analyze(n_cams, alt, elev_range, seed=42):
        elev_min, elev_max = elev_range
        rng = np.random.default_rng(seed)
        cam_positions = []
        view_counts = np.zeros(len(swarm), dtype=int)
        sky_fractions = []
        elevations = []

        for i in range(n_cams):
            elev_deg = rng.uniform(elev_min, elev_max)
            elev = math.radians(elev_deg)
            az = 2 * math.pi * i / n_cams + rng.uniform(-0.1, 0.1)

            # Place camera at specified altitude, pointing toward swarm center
            # Compute horizontal distance needed for desired elevation angle
            # tan(elev) = (center_z - alt) / horizontal_dist
            dz = center[2] - alt
            if abs(math.cos(elev)) < 0.01:
                h_dist = 5000  # nearly vertical
            else:
                h_dist = abs(dz / math.tan(elev)) if abs(math.tan(elev)) > 0.01 else 3000
            h_dist = max(h_dist, 1000)  # minimum horizontal distance

            cam_pos = center + np.array([
                h_dist * math.cos(az),
                h_dist * math.sin(az),
                alt,
            ])
            cam_positions.append(cam_pos)
            in_fov = camera_in_fov(swarm, cam_pos, h_fov, v_fov)
            view_counts += in_fov.astype(int)
            sky_frac = compute_sky_fraction(swarm, cam_pos)
            sky_fractions.extend(sky_frac[in_fov].tolist())
            # Actual elevation angle from camera to swarm center
            actual_elev = math.degrees(math.atan2(dz, h_dist))
            elevations.append(actual_elev)

        return cam_positions, view_counts, sky_fractions, elevations

    if config.get("sub_configs"):
        all_cam_positions = []
        all_view_counts = np.zeros(len(swarm), dtype=int)
        all_sky_fractions = []
        all_elevations = []

        for j, sub in enumerate(config["sub_configs"]):
            cams, vc, sf, el = place_and_analyze(
                sub["n_cams"], sub["altitude_m"], sub["elevation_range_deg"],
                seed=42 + j * 10)
            all_cam_positions.extend(cams)
            all_view_counts += vc
            all_sky_fractions.extend(sf)
            all_elevations.extend(el)
    else:
        all_cam_positions, all_view_counts, all_sky_fractions, all_elevations = \
            place_and_analyze(config["max_cameras"], config["camera_altitude_m"],
                              config["elevation_range_deg"])

    min_views = int(all_view_counts.min())
    drones_2view = int((all_view_counts >= 2).sum())
    coverage_pct = drones_2view / len(swarm) * 100

    sky_arr = np.array(all_sky_fractions) if all_sky_fractions else np.array([0.5])
    mean_sky = float(sky_arr.mean())
    frac_sky_dominant = float((sky_arr > 0.7).mean() * 100)

    el_arr = np.array(all_elevations) if all_elevations else np.array([0.0])

    return {
        "name": name,
        "description": config["description"],
        "n_cameras": len(all_cam_positions),
        "min_views": min_views,
        "drones_2view": drones_2view,
        "coverage_pct": round(coverage_pct, 1),
        "full_coverage": min_views >= 2,
        "mean_sky_fraction": round(mean_sky, 3),
        "pct_sky_dominant_views": round(frac_sky_dominant, 1),
        "elevation_stats": {
            "mean": round(float(el_arr.mean()), 1),
            "min": round(float(el_arr.min()), 1),
            "max": round(float(el_arr.max()), 1),
        },
    }


def main():
    swarm = make_swarm()
    print("=" * 80)
    print("VIEWING GEOMETRY: sky vs terrain background fraction")
    print("=" * 80)
    print(f"Swarm: {N_DRONES} drones, {SWARM_EXTENT_M/1000:.0f}km×{SWARM_EXTENT_M/1000:.0f}km×{SWARM_HEIGHT_M:.0f}m")
    print(f"Camera: 24mm full-frame (73.7° hFOV)")
    print()

    results = []
    for name, config in ARRANGEMENTS.items():
        r = analyze_arrangement(name, config, swarm)
        results.append(r)

        status = "✓" if r["full_coverage"] else "✗"
        sky_indicator = "sky" if r["mean_sky_fraction"] > 0.7 else (
            "mixed" if r["mean_sky_fraction"] > 0.3 else "terrain")

        print(f"  {r['description']}")
        print(f"    Coverage: {status} {r['drones_2view']}/{N_DRONES} drones ≥2-view "
              f"(min={r['min_views']})")
        print(f"    Background: {sky_indicator} "
              f"(mean sky fraction={r['mean_sky_fraction']:.2f}, "
              f"{r['pct_sky_dominant_views']:.0f}% sky-dominated views)")
        print(f"    Elevation: {r['elevation_stats']}")
        print()

    # Summary: find configs that achieve BOTH constraints
    print("=" * 80)
    print("CONFIGS ACHIEVING BOTH ≥2-VIEW COVERAGE AND SKY-DOMINATED BACKGROUNDS")
    print("=" * 80)

    both = [r for r in results if r["full_coverage"] and r["mean_sky_fraction"] > 0.7]
    if both:
        for r in both:
            print(f"  ✓ {r['name']}: {r['description']}")
            print(f"    Coverage: {r['drones_2view']}/{N_DRONES}, "
                  f"Sky fraction: {r['mean_sky_fraction']:.2f}")
    else:
        print("  No arrangement achieves BOTH full ≥2-view coverage AND sky-dominated")
        print("  backgrounds for the full 5km×5km×1km swarm.")
        print()
        print("  Closest candidates:")
        # Sort by a combined metric: coverage × sky fraction
        for r in sorted(results, key=lambda x: (x["coverage_pct"] * x["mean_sky_fraction"]), reverse=True)[:3]:
            print(f"  → {r['name']}: coverage={r['coverage_pct']:.0f}%, "
                  f"sky={r['mean_sky_fraction']:.2f}")

    # The boundary
    print()
    print("=" * 80)
    print("THE BOUNDARY")
    print("=" * 80)
    print("""
  Ground cameras looking UP: sky behind targets, but coverage is hard
  (cameras close to ground see the 5km volume edge-on at low elevation).
  Coverage improves with elevation angle, but higher angles look more
  downward toward terrain.

  Airborne cameras looking DOWN: easy coverage (wide view of swarm from
  above), but terrain behind targets.

  The constraint is geometric: to see sky behind a drone, the camera must
  be below the drone. To see the full 5km volume, cameras must be far
  enough away that the volume fits in the FOV — but distant cameras at
  low elevation can't see drones near the far edge of the swarm because
  those drones are near the horizon (sky) while near drones are against
  terrain.

  Quantitative boundary: the sky fraction is determined by the camera's
  altitude relative to the drones it observes. Cameras at ground level
  looking at drones at 0-1km altitude see sky (good). Cameras at 500m+
  looking down see terrain (bad). The transition is at camera altitude
  ≈ drone altitude, which for a 0-1km swarm means cameras at 0-500m.
""")

    # Save
    out_dir = Path(__file__).parent.parent / "logs" / "temporal_detection"
    out_path = out_dir / "viewing_geometry.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
