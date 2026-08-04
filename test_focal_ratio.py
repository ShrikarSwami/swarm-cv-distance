#!/usr/bin/env python3
"""Test focal length affects reconstruction errors - definitive test"""

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

print("=== Focal Length Ratio Test ===\n")

# Generate identical setup
transcript = generate_swarm_truth(n_drones=5, n_frames=1, area_km=2.0, height_range_m=500.0, seed=42)

# Test 1: focal_px = 1400.0
print("Test 1: focal_px = 1400.0")
rig1 = generate_camera_rig(
    transcript,
    n_views=8,
    geometry_class="mixed",
    standoff_m=1000.0,
    focal_px=1400.0,
    seed=123
)

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

print(f"  rig1.K[0,0,0] = {rig1.K[0,0,0]:.2f}")
print(f"  triangulated points: {len(recon1.positions_3d)}")
print(f"  median repro error: {np.median(recon1.reprojection_errors):.4f}px")
print(f"  mean repro error: {np.mean(recon1.reprojection_errors):.4f}px")

# Test 2: focal_px = 2666.67 (≈DEFAULT_FOCAL_PX)
print("\nTest 2: focal_px = 2666.67")
rig2 = generate_camera_rig(
    transcript,
    n_views=8,
    geometry_class="mixed",
    standoff_m=1000.0,
    focal_px=2666.67,
    seed=123
)

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

print(f"  rig2.K[0,0,0] = {rig2.K[0,0,0]:.2f}")
print(f"  triangulated points: {len(recon2.positions_3d)}")
print(f"  median repro error: {np.median(recon2.reprojection_errors):.4f}px")
print(f"  mean repro error: {np.mean(recon2.reprojection_errors):.4f}px")

# ANALYSIS
print(f"\n=== ANALYSIS ===")
print(f"Focal ratio rig2/rig1: {rig2.focal_px/rig1.focal_px:.3f}")

if len(recon1.positions_3d) > 0 and len(recon2.positions_3d) > 0:
    error_ratio = np.median(recon2.reprojection_errors) / np.median(recon1.reprojection_errors)
    print(f"Error ratio (rig2/rig1): {error_ratio:.3f}")
    print(f"Expected by focal ratio (1/focal): {rig1.focal_px/rig2.focal_px:.3f}")

    if abs(error_ratio - rig1.focal_px/rig2.focal_px) < 0.1:
        print("\n✓ RESULT: Errors scale as expected with focal length!")
        print("  FOCAL LENGTH IS WORKING correctly")
    elif abs(error_ratio - 1.0) < 0.01:
        print("\n✗ RESULT: Errors are IDENTICAL despite different focal lengths!")
        print("  BUG: Focal length reaches K matrices but cancels downstream")
    else:
        print(f"\n? RESULT: Errors differ but not by expected ratio")
        print(f"  Ratio: {error_ratio:.3f}, Expected: {rig1.focal_px/rig2.focal_px:.3f}")
else:
    print("\nNote: Different number of triangulated points due to correspondence")