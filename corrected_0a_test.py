#!/usr/bin/env python3
"""
Corrected Gate 0a test: focal length comparison with proper framing.

The fix: use a swarm small enough to be fully framed at the narrower FOV,
keeping standoff FIXED. This ensures identical matched sets across focal lengths.

Requirements (from root cause analysis):
- 3D position error in meters (not reprojection error)
- Swarm fully framed at the narrower FOV (verify coverage = 100% in both)
- FIXED standoff (do NOT adjust for framing in 0a — focal is the studied variable)
- Same seeds throughout
- Report: focal, K[0,0], coverage%, n_matched, median 3D error (m), p95, ratio
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stage1_geometry'))

import numpy as np
from data_contract import (
    SwarmTruth,
    CameraRig,
    Detections,
    DEFAULT_FOCAL_PX,
    IMAGE_SIZE,
    project_points_batch,
)

from b1_scene_rig import (
    generate_swarm_truth,
    generate_camera_rig,
    compute_framing_coverage,
    compute_standoff_for_framing,
)
from b2_projection import project_swarm_to_detections
from b3_correspondence import solve_correspondence
from b5_triangulation import triangulate_dlt

def compute_3d_position_errors(recon, truth, rig):
    """Compute 3D position errors by matching reconstructed positions to nearest drone."""
    n_drones = truth.n_drones
    n_recon = len(recon.positions_3d)

    if n_recon == 0:
        return np.array([]), 0, 0.0

    gt_positions = truth.positions[0]  # (N, 3)

    matched_gt_indices = set()
    errors_3d = []

    for recon_pos in recon.positions_3d:
        distances = np.linalg.norm(gt_positions - recon_pos, axis=1)
        best_idx = np.argmin(distances)
        best_dist = distances[best_idx]

        if best_dist < 50.0:
            if best_idx not in matched_gt_indices:
                matched_gt_indices.add(best_idx)
                errors_3d.append(best_dist)

    return np.array(errors_3d), len(matched_gt_indices), len(matched_gt_indices) / n_drones

print("=" * 70)
print("CORRECTED GATE 0a TEST: Focal Length Comparison")
print("  Fixed standoff (1000m), tiny swarm (fully framed at both focal lengths)")
print("=" * 70)

# Phase 1: Configuration
print("\n[Phase 1] Configuration")
print("-" * 60)

# Use a very small swarm that's fully framed even at the narrowest FOV at 1000m
# At f=2667 (narrowest), h_fov ≈ 39.6°, at 1000m: ~720m visible width
# So swarm extent < 360m from center is safe
# Using area_km=0.3 gives max extent ~150m from center - should be fully framed
swarm_config = dict(n_drones=5, n_frames=1, area_km=0.3, height_range_m=100.0, seed=42)
truth = generate_swarm_truth(**swarm_config)

standoff_m = 1000.0  # FIXED standoff
noise_std = 1.0
n_views = 8
geometry = "mixed"

print(f"  Swarm: {truth.n_drones} drones, 1 frame")
print(f"  Area: {swarm_config['area_km']}km x {swarm_config['area_km']}km")
print(f"  Standoff: {standoff_m}m (fixed)")
print(f"  Noise: {noise_std}px std dev")
print(f"  Views: {n_views}, Geometry: {geometry}")

# Phase 2: Test focal=1400
print("\n[Phase 2] Test A: focal_px = 1400.0 (wide FOV)")
print("-" * 60)

rig_a = generate_camera_rig(
    truth, n_views=n_views,
    geometry_class=geometry,
    standoff_m=standoff_m,
    focal_px=1400.0,
    seed=123
)

coverage_a = compute_framing_coverage(truth, rig_a)
h_fov_a = 2 * np.degrees(np.arctan(IMAGE_SIZE[0] / (2 * 1400.0)))

print(f"  K[0,0,0] = {rig_a.K[0,0,0]:.2f}")
print(f"  Standoff: {standoff_m}m (fixed)")
print(f"  H-FOV: {h_fov_a:.1f}°")
print(f"  Coverage: {coverage_a:.1%}")

detections_a = project_swarm_to_detections(
    truth, rig_a, pixel_noise_std=noise_std, drop_prob=0.0, seed=1
)
tracks_a = solve_correspondence(
    detections_a, rig_a,
    epipolar_threshold=3.0, min_views=2, max_reproj_error=5.0, seed=42
)
recon_a = triangulate_dlt(tracks_a, rig_a, detections_a)
errors_3d_a, n_matched_a, recall_a = compute_3d_position_errors(recon_a, truth, rig_a)

print(f"  N matched: {n_matched_a}/{truth.n_drones}  (recall: {recall_a:.1%})")
if len(errors_3d_a) > 0:
    print(f"  3D errors (m): median={np.median(errors_3d_a):.4f}, p95={np.percentile(errors_3d_a, 95):.4f}")

# Phase 3: Test focal=2667
print("\n[Phase 3] Test B: focal_px = 2666.67 (narrow FOV)")
print("-" * 60)

rig_b = generate_camera_rig(
    truth, n_views=n_views,
    geometry_class=geometry,
    standoff_m=standoff_m,
    focal_px=2666.67,
    seed=123
)

coverage_b = compute_framing_coverage(truth, rig_b)
h_fov_b = 2 * np.degrees(np.arctan(IMAGE_SIZE[0] / (2 * 2666.67)))

print(f"  K[0,0,0] = {rig_b.K[0,0,0]:.2f}")
print(f"  Standoff: {standoff_m}m (fixed)")
print(f"  H-FOV: {h_fov_b:.1f}°")
print(f"  Coverage: {coverage_b:.1%}")

detections_b = project_swarm_to_detections(
    truth, rig_b, pixel_noise_std=noise_std, drop_prob=0.0, seed=1
)
tracks_b = solve_correspondence(
    detections_b, rig_b,
    epipolar_threshold=3.0, min_views=2, max_reproj_error=5.0, seed=42
)
recon_b = triangulate_dlt(tracks_b, rig_b, detections_b)
errors_3d_b, n_matched_b, recall_b = compute_3d_position_errors(recon_b, truth, rig_b)

print(f"  N matched: {n_matched_b}/{truth.n_drones}  (recall: {recall_b:.1%})")
if len(errors_3d_b) > 0:
    print(f"  3D errors (m): median={np.median(errors_3d_b):.4f}, p95={np.percentile(errors_3d_b, 95):.4f}")

# Phase 4: Final comparison
print("\n[Phase 4] FINAL COMPARISON")
print("=" * 70)

if coverage_a < 1.0 or coverage_b < 1.0:
    print("⚠ COVERAGE WARNING: Not 100% in both runs")
    print(f"  Coverage A: {coverage_a:.1%}, Coverage B: {coverage_b:.1%}")

if n_matched_a != n_matched_b:
    print(f"⚠ N_MATCHED DIFFERS: A={n_matched_a}, B={n_matched_b}")

print(f"\n  {'Metric':<30} {'f=1400 (A)':>16} {'f=2667 (B)':>16} {'Ratio':>10}")
print(f"  {'-'*30} {'-'*16} {'-'*16} {'-'*10}")

focal_ratio = rig_b.focal_px / rig_a.focal_px
print(f"  {'Focal ratio (B/A)':<30} {'':>16} {'':>16} {focal_ratio:>10.3f}")

if len(errors_3d_a) > 0:
    median_a = np.median(errors_3d_a)
    p95_a = np.percentile(errors_3d_a, 95)
    print(f"  {'Median 3D error (m)':<30} {median_a:>16.4f} {'':>16} {'':>10}")

if len(errors_3d_b) > 0:
    median_b = np.median(errors_3d_b)
    p95_b = np.percentile(errors_3d_b, 95)
    print(f"  {'Median 3D error (m)':<30} {'':>16} {median_b:>16.4f} {'':>10}")

if len(errors_3d_a) > 0 and len(errors_3d_b) > 0:
    error_ratio = median_b / max(median_a, 1e-10)
    print(f"  {'Error ratio (B/A)':<30} {'':>16} {'':>16} {error_ratio:>10.3f}")
    expected_ratio = rig_a.focal_px / rig_b.focal_px  # 1/focal ratio
    print(f"  {'Expected (1/focal)':<30} {'':>16} {'':>16} {expected_ratio:>10.3f}")
    print(f"  {'Coverage %':<30} {coverage_a*100:>15.0f}% {coverage_b*100:>15.0f}% {'':>10}")
    print(f"  {'N matched':<30} {n_matched_a:>16d} {n_matched_b:>16d} {'':>10}")

    print(f"\n  DIAGNOSTIC CHECK:")
    print(f"  Error ratio vs expected 1/focal ratio:")
    if abs(error_ratio - expected_ratio) < 0.1:
        print(f"    ✓ {error_ratio:.3f} ≈ {expected_ratio:.3f} — ERRORS SCALE AS 1/focal")
        print(f"    Pipeline correct. Gate 0a closes.")
    elif abs(error_ratio - 1.0) < 0.01:
        print(f"    ✗ {error_ratio:.3f} ≠ {expected_ratio:.3f} — Errors invariant!")
        print(f"    Bug persists: focal reaches K but cancels downstream.")
        print(f"    Check: noise in pixel space, DLT normalization, triangulation.")
    else:
        print(f"    ? {error_ratio:.3f} vs {expected_ratio:.3f} — indefinite")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
