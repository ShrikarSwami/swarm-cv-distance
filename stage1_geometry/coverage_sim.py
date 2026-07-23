"""
Simulate multi-camera coverage to find minimum cameras for ≥2-view overlap.

Places N cameras in a dome around the swarm, checks which drones each camera
seeps via FOV, finds minimum N for ≥2-view coverage of all drones.

Run: python stage1_geometry/coverage_sim.py
"""

import json
import math
import numpy as np


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

DRONE_SIZE_M = 0.5
AREA_M = 5000.0
HEIGHT_RANGE_M = 1000.0
N_DRONES = 20


def make_swarm(n_drones=20, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-AREA_M / 2, AREA_M / 2, size=(n_drones, 2))
    z = rng.uniform(0, HEIGHT_RANGE_M, size=(n_drones, 1))
    return np.hstack([xy, z])


# ---------------------------------------------------------------------------
# Camera FOV check
# ---------------------------------------------------------------------------

def camera_fov_mask(drone_positions, cam_pos, h_fov_rad, v_fov_rad):
    """Returns boolean mask: which drones are within the camera's FOV."""
    swarm_center = drone_positions.mean(axis=0)
    forward = (swarm_center - cam_pos)
    forward = forward / np.linalg.norm(forward)

    # Right and up vectors
    up_hint = np.array([0, 0, 1.0])
    right = np.cross(forward, up_hint)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1, 0, 0.0])
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    # Project drones into camera space
    diff = drone_positions - cam_pos  # (N, 3)
    dist = np.linalg.norm(diff, axis=1)  # (N,)

    # Angle from forward axis
    cos_angle = np.sum(diff * forward[np.newaxis, :], axis=1) / np.maximum(dist, 1e-6)

    # Horizontal and vertical angles
    proj_right = np.sum(diff * right[np.newaxis, :], axis=1)
    proj_up = np.sum(diff * up[np.newaxis, :], axis=1)

    h_angle = np.arctan2(np.abs(proj_right), np.maximum(cos_angle * dist, 1e-6))
    v_angle = np.arctan2(np.abs(proj_up), np.maximum(cos_angle * dist, 1e-6))

    # In FOV if within half-FOV angles and in front of camera
    in_hfov = h_angle <= h_fov_rad / 2
    in_vfov = v_angle <= v_fov_rad / 2
    in_front = cos_angle > 0

    return in_hfov & in_vfov & in_front


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def horizontal_fov_rad(sensor_width_mm, focal_mm):
    return 2 * math.atan(sensor_width_mm / (2 * focal_mm))


def vertical_fov_rad(sensor_width_mm, focal_mm, h_px, v_px):
    v_sensor = sensor_width_mm * v_px / h_px
    return 2 * math.atan(v_sensor / (2 * focal_mm))


def simulate_coverage(swarm, n_cams, standoff_m, h_fov_rad, v_fov_rad, seed=0):
    """Place n_cams in dome, return per-drone view count."""
    rng = np.random.default_rng(seed)
    center = swarm.mean(axis=0)
    n_drones = len(swarm)
    view_counts = np.zeros(n_drones, dtype=int)

    for i in range(n_cams):
        elev = math.radians(rng.uniform(20, 50))
        az = 2 * math.pi * i / n_cams + rng.uniform(-0.1, 0.1)
        cam_pos = center + np.array([
            standoff_m * math.cos(elev) * math.cos(az),
            standoff_m * math.cos(elev) * math.sin(az),
            standoff_m * math.sin(elev),
        ])
        mask = camera_fov_mask(swarm, cam_pos, h_fov_rad, v_fov_rad)
        view_counts += mask.astype(int)

    return view_counts


def find_min_cams(swarm, standoff_m, h_fov_rad, v_fov_rad, target_coverage=2, max_cams=30):
    """Find minimum cameras for target multi-view coverage."""
    for n in range(2, max_cams + 1):
        view_counts = simulate_coverage(swarm, n, standoff_m, h_fov_rad, v_fov_rad)
        min_views = view_counts.min()
        if min_views >= target_coverage:
            return n, view_counts
    return max_cams, view_counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    swarm = make_swarm(20, seed=42)
    center = swarm.mean(axis=0)
    print(f"Swarm center: {center}")
    print(f"Swarm extent: x=[{swarm[:,0].min():.0f}, {swarm[:,0].max():.0f}], "
          f"y=[{swarm[:,1].min():.0f}, {swarm[:,1].max():.0f}], "
          f"z=[{swarm[:,2].min():.0f}, {swarm[:,2].max():.0f}]")
    print()

    # Configs to test: sensor_w, focal_mm, h_px, v_px
    configs = [
        ("24mm FF",     36.0,  24, 1920, 1080),
        ("50mm FF",     36.0,  50, 1920, 1080),
        ("100mm FF",    36.0, 100, 1920, 1080),
        ("200mm FF",    36.0, 200, 1920, 1080),
        ("400mm FF",    36.0, 400, 1920, 1080),
        ("800mm FF",    36.0, 800, 1920, 1080),
        ("1200mm FF",   36.0,1200, 1920, 1080),
        ("24mm APS-C",  23.5,  24, 1920, 1080),
        ("100mm APS-C", 23.5, 100, 1920, 1080),
        ("539mm 1/2.3",  6.17,539, 1920, 1080),
    ]

    standoffs = [500, 1000, 2000, 3000, 5000, 7500, 10000]

    print("=== ≥2-CAMERA COVERAGE (minimum cameras for every drone seen by ≥2 views) ===\n")
    print(f"{'Config':16s}", end="")
    for s in standoffs:
        print(f" {s//1000}km" if s >= 1000 else f" {s}m", end="")
    print()
    print("-" * 90)

    results = []
    for name, sens_w, focal, h_px, v_px in configs:
        h_fov = horizontal_fov_rad(sens_w, focal)
        v_fov = vertical_fov_rad(sens_w, focal, h_px, v_px)
        h_fov_deg = math.degrees(h_fov)
        v_fov_deg = math.degrees(v_fov)

        print(f"{name:16s}", end="")
        for s in standoffs:
            n_cams, view_counts = find_min_cams(swarm, s, h_fov, v_fov, target_coverage=2)
            min_v = view_counts.min()
            avg_v = view_counts.mean()
            marker = "✓" if min_v >= 2 else "✗"
            print(f" {n_cams:2d}{marker}", end="")

            results.append({
                "config": name,
                "sensor_mm": sens_w,
                "focal_mm": focal,
                "standoff_m": s,
                "h_fov_deg": round(h_fov_deg, 2),
                "v_fov_deg": round(v_fov_deg, 2),
                "min_cams_2view": n_cams,
                "min_views": int(min_v),
                "avg_views": round(float(avg_v), 1),
            })
        print()

    print(f"\n✓ = all 20 drones seen by ≥2 cameras")
    print(f"✗ = some drones seen by <2 cameras (insufficient for triangulation)")

    # Save results
    out_dir = "logs/m1_sweep"
    import os
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "coverage_2view.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_dir}/coverage_2view.json")


if __name__ == "__main__":
    main()
