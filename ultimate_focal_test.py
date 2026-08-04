#!/usr/bin/env python3
"""Ultimate focal length test - exact specification from task"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stage1_geometry'))

import numpy as np
from data_contract import (
    SwarmTruth,
    CameraRig,
    Detections,
    make_K,
    CONVENTION_TAG,
    DEFAULT_FOCAL_PX
)
from b1_scene_rig import generate_swarm_truth, generate_camera_rig
from b2_projection import project_swarm_to_detections
from b3_correspondence import solve_correspondence
from b5_triangulation import triangulate_dlt

print("=== Ultimate Focal Length Test ===\n")
print("Testing with: same swarm seed, same camera positions, same noise seed,")
print("             same standoff, same drone count, 8 cameras, 1px noise, 1000m.\n")

# Generate same swarm once
transcript = generate_swarm_truth(n_drones=5, n_frames=1, area_km=2.0, height_range_m=500.0, seed=42)

print("=== Run A: focal_px = 1400.0 ===")
print("=" * 60)
rig_a = generate_camera_rig(
    transcript,
    n_views=8,
    geometry_class="mixed",
    standoff_m=1000.0,
    focal_px=1400.0,
    seed=123
)

print(f"  rig_a.focal_px = {rig_a.focal_px:.2f}")
print(f"  rig_a.K[0,0,0] = {rig_a.K[0,0,0]:.2f}")

detections_a = project_swarm_to_detections(
    transcript, rig_a, pixel_noise_std=1.0, drop_prob=0.0, seed=1
)

tracks_a = solve_correspondence(
    detections_a, rig_a,
    epipolar_threshold=3.0,
    min_views=2,
    max_reproj_error=5.0,
    seed=42
)

recon_a = triangulate_dlt(tracks_a, rig_a, detections_a)

print(f"  triangulated points: {len(recon_a.positions_3d)}")
print(f"  median error: {np.median(recon_a.reprojection_errors):.6f}px")
print(f"  mean error: {np.mean(recon_a.reprojection_errors):.6f}px")
print(f"  p95 error: {np.percentile(recon_a.reprojection_errors, 95):.6f}px")
print(f"  n_matched: {len(recon_a.positions_3d)}")
print(f"  recall: {len(recon_a.positions_3d)}/5 = {len(recon_a.positions_3d)/5:.1%}")

print("\n=== Run B: focal_px = 2666.67 ===")
print("=" * 60)
rig_b = generate_camera_rig(
    transcript,
    n_views=8,
    geometry_class="mixed",
    standoff_m=1000.0,
    focal_px=2666.67,
    seed=123
)

print(f"  rig_b.focal_px = {rig_b.focal_px:.2f}")
print(f"  rig_b.K[0,0,0] = {rig_b.K[0,0,0]:.2f}")

detections_b = project_swarm_to_detections(
    transcript, rig_b, pixel_noise_std=1.0, drop_prob=0.0, seed=1
)

tracks_b = solve_correspondence(
    detections_b, rig_b,
    epipolar_threshold=3.0,
    min_views=2,
    max_reproj_error=5.0,
    seed=42
)

recon_b = triangulate_dlt(tracks_b, rig_b, detections_b)

print(f"  triangulated points: {len(recon_b.positions_3d)}")
print(f"  median error: {np.median(recon_b.reprojection_errors):.6f}px")
print(f"  mean error: {np.mean(recon_b.reprojection_errors):.6f}px")
print(f"  p95 error: {np.percentile(recon_b.reprojection_errors, 95):.6f}px")
print(f"  n_matched: {len(recon_b.positions_3d)}")
print(f"  recall: {len(recon_b.positions_3d)}/5 = {len(recon_b.positions_3d)/5:.1%}")

print("\n=== COMPARISON ===")
print(f"Focus A: {rig_a.focal_px:.1f}, Focus B: {rig_b.focal_px:.1f}")
print(f"Ratio B/A: {rig_b.focal_px/rig_a.focal_px:.3f}")

if len(recon_a.positions_3d) > 0 and len(recon_b.positions_3d) > 0:
    median_error_ratio = np.median(recon_b.reprojection_errors) / np.median(recon_a.reprojection_errors)
    mean_error_ratio = np.mean(recon_b.reprojection_errors) / np.mean(recon_a.reprojection_errors)

    print(f"\nMedian error ratio (B/A): {median_error_ratio:.3f}")
    print(f"Expected by 1/focal ratio: {rig_a.focal_px/rig_b.focal_px:.3f}")
    print(f"Mean error ratio (B/A): {mean_error_ratio:.3f}")
    print(f"Expected by 1/focal ratio: {rig_a.focal_px/rig_b.focal_px:.3f}")

    # Report outcomes
    print("\n" + "="*60)
    print("OUTCOME:")

    if abs(median_error_ratio - rig_a.focal_px/rig_b.focal_px) < 0.1:
        print("✓ Ratio near expected (1.9) - Pipeline is CORRECT")
        print("  Focal length wires through to affect reconstruction errors as expected")
    elif abs(median_error_ratio - 1.0) < 0.01:
        print("✗ Ratio near 1.0 - BUG found")
        print("  Focal length reaches K but doesn't affect errors downstream")
    else:
        print(f"? Ratio is {median_error_ratio:.3f} (expected ~{rig_a.focal_px/rig_b.focal_px:.3f})")
        if median_error_ratio < 0.1 and rig_a.focal_px/rig_b.focal_px > 0.5:
            print("  Issue: Error scaling too strong (possible noise too low?)")
        elif median_error_ratio > 0.5:
            print("  Issue: Error scaling too weak")
else:
    print("\nNote: Different number of triangulated points - may affect comparison")