#!/usr/bin/env python3
"""Final verification test for focal length effect"""

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

print("=== Final Focal Length Verification Test ===\n")

# Generate identical setup
transcript = generate_swarm_truth(n_drones=5, n_frames=1, area_km=2.0, height_range_m=500.0, seed=42)

# Test 1: focal_px = 1400.0 (low focal - wide field of view)
print("Test 1: focal_px = 1400.0 (low focal)")
print("=" * 60)

rig1 = generate_camera_rig(
    transcript,
    n_views=8,
    geometry_class="mixed",
    standoff_m=1000.0,
    focal_px=1400.0,
    seed=123
)

print(f"  rig1.focal_px = {rig1.focal_px:.2f}")
print(f"  rig1.K[0,0,0] = {rig1.K[0,0,0]:.2f}")

detections1 = project_swarm_to_detections(
    transcript, rig1, pixel_noise_std=0.5, drop_prob=0.0, seed=1
)

tracks1 = solve_correspondence(
    detections1, rig1,
    epipolar_threshold=3.0,
    min_views=2,
    max_reproj_error=5.0,
    seed=42
)

recon1 = triangulate_dlt(tracks1, rig1, detections1)

print(f"  triangulated points: {len(recon1.positions_3d)}")
print(f"  median repro error: {np.median(recon1.reprojection_errors):.6f}px")
print(f"  mean repro error: {np.mean(recon1.reprojection_errors):.6f}px")
print(f"  max repro error: {np.max(recon1.reprojection_errors):.6f}px")

# Test 2: focal_px = 2666.67 (high focal - narrow field of view)
print("\n\nTest 2: focal_px = 2666.67 (high focal)")
print("=" * 60)

rig2 = generate_camera_rig(
    transcript,
    n_views=8,
    geometry_class="mixed",
    standoff_m=1000.0,
    focal_px=2666.67,
    seed=123
)

print(f"  rig2.focal_px = {rig2.focal_px:.2f}")
print(f"  rig2.K[0,0,0] = {rig2.K[0,0,0]:.2f}")

detections2 = project_swarm_to_detections(
    transcript, rig2, pixel_noise_std=0.5, drop_prob=0.0, seed=1
)

tracks2 = solve_correspondence(
    detections2, rig2,
    epipolar_threshold=3.0,
    min_views=2,
    max_reproj_error=5.0,
    seed=42
)

recon2 = triangulate_dlt(tracks2, rig2, detections2)

print(f"  triangulated points: {len(recon2.positions_3d)}")
print(f"  median repro error: {np.median(recon2.reprojection_errors):.6f}px")
print(f"  mean repro error: {np.mean(recon2.reprojection_errors):.6f}px")
print(f"  max repro error: {np.max(recon2.reprojection_errors):.6f}px")

# ANALYSIS
print("\n\n=== ANALYSIS ===")
print(f"Focal length ratio (rig2/rig1): {rig2.focal_px/rig1.focal_px:.3f}")
print(f"Expected error ratio (1/focal): {rig1.focal_px/rig2.focal_px:.3f}")

if len(recon1.positions_3d) > 0 and len(recon2.positions_3d) > 0:
    error_ratio = np.median(recon2.reprojection_errors) / np.median(recon1.reprojection_errors)
    print(f"Actual error ratio (rig2/rig1): {error_ratio:.3f}")
    print(f"Difference from expected: {abs(error_ratio - rig1.focal_px/rig2.focal_px):.3f}")

    if abs(error_ratio - rig1.focal_px/rig2.focal_px) < 0.01:
        print("\n✓ CONFIRMED: Errors scale as 1/focal length!")
        print("  Higher focal = lower error (narrower field of view)")
        print("  The focal length IS affecting the reconstruction pipeline")
    elif abs(error_ratio - 1.0) < 0.01:
        print("\n✗ BUG DETECTED: Errors are IDENTICAL despite different focal lengths!")
        print("  Focal length reaches K matrices but has no effect downstream")
    else:
        print(f"\n? PARTIAL EFFECT: Errors differ but not by expected ratio")
else:
    print("\nNote: Different number of triangulated points in each run")