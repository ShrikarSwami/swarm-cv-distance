#!/usr/bin/env python3
"""Debug the focal length issue in practice"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stage1_geometry'))

import numpy as np

from data_contract import (
    SwarmTruth,
    CameraRig,
    Detections,
    Tracks,
    make_K,
    CONVENTION_TAG,
    DEFAULT_FOCAL_PX
)
from b1_scene_rig import generate_swarm_truth, generate_camera_rig
from b2_projection import project_swarm_to_detections
from b3_correspondence import solve_correspondence
from b5_triangulation import triangulate_dlt

print("=== Practical Focal Length Test ===\n")

# Generate test data
truth = generate_swarm_truth(n_drones=5, n_frames=1, area_km=2.0, height_range_m=500.0, seed=42)

# Test 1: DEFAULT_FOCAL_PX
print(f"1. Testing with DEFAULT_FOCAL_PX = {DEFAULT_FOCAL_PX:.2f}")
rig1 = generate_camera_rig(
    truth,
    n_views=8,
    geometry_class="mixed",
    standoff_m=1000.0,
    focal_px=DEFAULT_FOCAL_PX,
    seed=123
)

detections1 = project_swarm_to_detections(
    truth, rig1, pixel_noise_std=0.5, drop_prob=0.0, seed=1
)

tracks1 = solve_correspondence(
    detections1, rig1,
    epipolar_threshold=3.0,
    min_views=2,
    max_reproj_error=5.0,
    seed=42
)

# Check K matrix
print(f"   rig1.K[0]: {rig1.K[0]}")
print(f"   rig1.focal_px: {rig1.focal_px}")
print(f"   K[0,0] == focal_px: {rig1.K[0,0] == rig1.focal_px}")

# Test 2: Lower focal length
print(f"\n2. Testing with focal_px = 1400px")
rig2 = generate_camera_rig(
    truth,
    n_views=8,
    geometry_class="mixed",
    standoff_m=1000.0,
    focal_px=1400.0,
    seed=456
)

detections2 = project_swarm_to_detections(
    truth, rig2, pixel_noise_std=0.5, drop_prob=0.0, seed=2
)

tracks2 = solve_correspondence(
    detections2, rig2,
    epipolar_threshold=3.0,
    min_views=2,
    max_reproj_error=5.0,
    seed=42
)

# Check K matrix
print(f"   rig2.K[0]: {rig2.K[0]}")
print(f"   rig2.focal_px: {rig2.focal_px}")
print(f"   K[0,0] == focal_px: {rig2.K[0,0] == rig2.focal_px}")

# Test 3: Compare triangulated positions
print(f"\n3. Comparing triangulated positions:")
print(f"   Rig 1 focal: {rig1.focal_px:.2f}")
print(f"   Rig 2 focal: {rig2.focal_px:.2f}")
print(f"   Focal ratio: {rig2.focal_px/rig1.focal_px:.3f}")

# Reconstruct with both rigs
recon1 = triangulate_dlt(tracks1, rig1, detections1)
recon2 = triangulate_dlt(tracks2, rig2, detections2)

print(f"   Recon 1 positions: {recon1.positions_3d}")
print(f"   Recon 2 positions: {recon2.positions_3d}")

# Calculate distances from origin
dist1 = np.linalg.norm(recon1.positions_3d, axis=1)
dist2 = np.linalg.norm(recon2.positions_3d, axis=1)

print(f"   Distances from origin (rig1): {dist1}")
print(f"   Distances from origin (rig2): {dist2}")

# Expected relationship: positions should scale inversely with focal length
print(f"\n   Expected position scaling: positions should be ~{rig2.focal_px/rig1.focal_px:.3f}x larger with rig2 (lower focal)")
print(f"   Actual position ratio: {dist2[0]/dist1[0]:.3f} (if first drone valid)")

# For same 3D point at distance Z from camera:
# Pixel coordinates: u = f * X/Z, v = f * Y/Z
# So X/Z = u/f, Y/Z = v/f
# Expected X = u * Z / f, Y = v * Z / f
# Therefore positions should scale ~1/focal

# Find matching drones between reconstructions
print(f"\n4. Matched reconstruction analysis:")
for i, (pos1, pos2) in enumerate(zip(recon1.positions_3d, recon2.positions_3d)):
    dist1_i = np.linalg.norm(pos1)
    dist2_i = np.linalg.norm(pos2)
    if dist1_i > 0 and dist2_i > 0:
        focal_ratio = rig2.focal_px / rig1.focal_px
        expected_ratio = 1.0 / focal_ratio  # lower focal should give larger positions
        actual_ratio = dist2_i / dist1_i
        print(f"   Drone {i}: dist1={dist1_i:.1f}, dist2={dist2_i:.1f}, expected ratio={expected_ratio:.3f}, actual ratio={actual_ratio:.3f}, match={abs(actual_ratio - expected_ratio) < 0.1}")

print(f"\n=== Analysis Summary ===")
print(f"If positions do NOT change significantly between rigs (different focal lengths),")
print(f"then the triangulator is NOT using the rig's K matrix for projection.")

print(f"\nChecking if triangulator uses rig.K...")
print(f"rig1.K[0,0]: {rig1.K[0,0]}")
print(f"rig2.K[0,0]: {rig2.K[0,0]}")
print(f"rig1.focal_px: {rig1.focal_px}")
print(f"rig2.focal_px: {rig2.focal_px}")

print(f"\nIf rig1.K[0,0] == rig1.focal_px == 2666.67 and rig2.K[0,0] == rig2.focal_px == 1400,")
print(f"but reconstructed positions are similar, then there's a bug in the triangulation logic.")