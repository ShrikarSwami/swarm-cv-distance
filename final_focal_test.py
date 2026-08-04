#!/usr/bin/env python3
"""Final focal length test - two minutes as requested"""

import sys
import os
sys.path.insert(0, '.')
sys.path.insert(0, 'stage1_geometry')

import numpy as np
import time

from data_contract import SwarmTruth, CameraRig, Detections, Tracks
from b1_scene_rig import generate_swarm_truth, generate_camera_rig
from b2_projection import project_swarm_to_detections
from b3_correspondence import solve_correspondence
from b5_triangulation import triangulate_dlt

print("=== FINAL FOCL LENGTH TEST - TWO MINUTES ===")

# Generate identical setup for both runs
print("\n1. Generating identical setup (seed=42):")
truth = generate_swarm_truth(n_drones=5, n_frames=1, area_km=2.0, height_range_m=500.0, seed=42)
print(f"   Truth: {truth.n_drones} drones")

# Run A: f=1400
print("\n2. Running with focal_px = 1400.0")
start_a = time.time()

rig_a = generate_camera_rig(
    truth,
    n_views=8,
    geometry_class="mixed",
    standoff_m=1000.0,
    focal_px=1400.0,
    seed=123
)

detections_a = project_swarm_to_detections(
    truth, rig_a, pixel_noise_std=0.5, drop_prob=0.0, seed=1
)

tracks_a = solve_correspondence(
    detections_a, rig_a,
    epipolar_threshold=3.0,
    min_views=2,
    max_reproj_error=5.0,
    seed=42
)

# PRINT K MATRICES FROM TRIANGULATION
print("\n   K matrices used in triangulation:")
for v in range(min(3, rig_a.n_views)):
    print(f"   View {v}: K[{v},0] = {rig_a.K[v,0]}")

recon_a = triangulate_dlt(tracks_a, rig_a, detections_a)
end_a = time.time()

print(f"   Time: {end_a - start_a:.2f}s")
print(f"   Triangulated: {len(recon_a.positions_3d)} points")
print(f"   Median error: {np.median(recon_a.reprojection_errors):.4f}px")

# Run B: f=2666.67
print("\n3. Running with focal_px = 2666.67")
start_b = time.time()

rig_b = generate_camera_rig(
    truth,
    n_views=8,
    geometry_class="mixed",
    standoff_m=1000.0,
    focal_px=2666.67,
    seed=123
)

detections_b = project_swarm_to_detections(
    truth, rig_b, pixel_noise_std=0.5, drop_prob=0.0, seed=1
)

tracks_b = solve_correspondence(
    detections_b, rig_b,
    epipolar_threshold=3.0,
    min_views=2,
    max_reproj_error=5.0,
    seed=42
)

# PRINT K MATRICES FROM TRIANGULATION
print("\n   K matrices used in triangulation:")
for v in range(min(3, rig_b.n_views)):
    print(f"   View {v}: K[{v},0] = {rig_b.K[v,0]}")

recon_b = triangulate_dlt(tracks_b, rig_b, detections_b)
end_b = time.time()

print(f"   Time: {end_b - start_b:.2f}s")
print(f"   Triangulated: {len(recon_b.positions_3d)} points")
print(f"   Median error: {np.median(recon_b.reprojection_errors):.4f}px")

# ANALYSIS
print("\n\n=== ANALYSIS ===")
print(f"FocusA: {rig_a.focal_px:.1f}, FocusB: {rig_b.focal_px:.1f}")
print(f"Ratio B/A: {rig_b.focal_px/rig_a.focal_px:.3f}")

# Check if K matrices are different
print(f"\nK[0,0] different? {abs(rig_a.K[0,0] - rig_b.K[0,0]) > 1e-6}")
print(f"rig_a.K[0,0] = {rig_a.K[0,0]:.2f}")
print(f"rig_b.K[0,0] = {rig_b.K[0,0]:.2f}")

if abs(rig_a.K[0,0] - rig_b.K[0,0]) < 1e-6:
    print("\n*** K MATRICES ARE IDENTICAL ***")
    print("Focal length is NOT propagating to K matrices!")
elif abs(np.median(recon_a.reprojection_errors) - np.median(recon_b.reprojection_errors)) < 0.01:
    print("\n*** MEDIAN ERRORS ARE IDENTICAL ***")
    print("Focal length reaches K but error does NOT change with focal!")
else:
    error_ratio = np.median(recon_b.reprojection_errors) / np.median(recon_a.reprojection_errors)
    print(f"\nMedian errors differ by ratio: {error_ratio:.3f}")
    print(f"Expected by 1/focal ratio: {rig_a.focal_px/rig_b.focal_px:.3f}")
    if abs(error_ratio - rig_a.focal_px/rig_b.focal_px) < 0.1:
        print("*** ERROR SCALES AS EXPECTED WITH FOCUS ***")
    else:
        print("*** ERROR DOES NOT SCALE AS EXPECTED WITH FOCUS ***")

print("\n" + "="*50)
print("DECISION:")
if abs(rig_a.K[0,0] - rig_b.K[0,0]) < 1e-6:
    print("RESULT: K matrices identical - focal length wiring BROKEN")
elif abs(np.median(recon_a.reprojection_errors) - np.median(recon_b.reprojection_errors)) < 0.01:
    print("RESULT: Errors identical - focal length reaches K but cancels downstream")
else:
    print("RESULT: Errors differ - focal length working")
    print("  Close this item - focal wiring is correct")