#!/usr/bin/env python3
"""Analyze the focal length relationship in detail"""

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

print("=== Detailed Focal Length Analysis ===\n")

# Generate identical setup
transcript = generate_swarm_truth(n_drones=5, n_frames=1, area_km=2.0, height_range_m=500.0, seed=42)

print("A. Test WITHOUT noise (zero noise)")
print("=" * 50)

# Low focal (wide FOV)
rig_a = generate_camera_rig(
    transcript,
    n_views=8,
    geometry_class="mixed",
    standoff_m=1000.0,
    focal_px=1400.0,
    seed=123
)

print(f"Rig A: focal = {rig_a.focal_px:.2f}, K[0,0,0] = {rig_a.K[0,0,0]:.2f}")

# Project with zero noise
detections_a = project_swarm_to_detections(
    transcript, rig_a, pixel_noise_std=0.0, drop_prob=0.0, seed=1
)

# Show pixel positions of drones
drone_positions_a = []
for f in range(1):  # frame 0
    for d in range(5):  # drone 0
        px = detections_a.points_per_view[0][d]
        drone_positions_a.append(px)
        print(f"  Drone {d}: px = {px}")

# Triangulate
print("\nTriangulating...")
tracks_a = solve_correspondence(
    detections_a, rig_a,
    epipolar_threshold=3.0,
    min_views=2,
    max_reproj_error=5.0,
    seed=42
)

recon_a = triangulate_dlt(tracks_a, rig_a, detections_a)
print(f"  Reconstructed: {len(recon_a.positions_3d)} points")
if len(recon_a.positions_3d) > 0:
    print(f"  Pos: {recon_a.positions_3d}")
    print(f"  Reproj error: {recon_a.reprojection_errors}")

# High focal (narrow FOV)
rig_b = generate_camera_rig(
    transcript,
    n_views=8,
    geometry_class="mixed",
    standoff_m=1000.0,
    focal_px=2666.67,
    seed=123
)

print(f"\nRig B: focal = {rig_b.focal_px:.2f}, K[0,0,0] = {rig_b.K[0,0,0]:.2f}")

detections_b = project_swarm_to_detections(
    transcript, rig_b, pixel_noise_std=0.0, drop_prob=0.0, seed=1
)

print("\nShowing pixel positions...")
drone_positions_b = []
for f in range(1):  # frame 0
    for d in range(5):  # drone 0
        px = detections_b.points_per_view[0][d]
        drone_positions_b.append(px)
        print(f"  Drone {d}: px = {px}")

# Triangulate
tracks_b = solve_correspondence(
    detections_b, rig_b,
    epipolar_threshold=3.0,
    min_views=2,
    max_reproj_error=5.0,
    seed=42
)

recon_b = triangulate_dlt(tracks_b, rig_b, detections_b)
print(f"\n  Reconstructed: {len(recon_b.positions_3d)} points")
if len(recon_b.positions_3d) > 0:
    print(f"  Pos: {recon_b.positions_3d}")
    print(f"  Reproj error: {recon_b.reprojection_errors}")

print("\n\nB. Test WITH noise (1px std dev)")
print("=" * 50)

# Low focal with noise
detections_a_noise = project_swarm_to_detections(
    transcript, rig_a, pixel_noise_std=1.0, drop_prob=0.0, seed=1
)

tracks_a_noise = solve_correspondence(
    detections_a_noise, rig_a,
    epipolar_threshold=3.0,
    min_views=2,
    max_reproj_error=5.0,
    seed=42
)

recon_a_noise = triangulate_dlt(tracks_a_noise, rig_a, detections_a_noise)

print(f"Rig A (f=1400) with noise: {len(recon_a_noise.positions_3d)} points, median error: {np.median(recon_a_noise.reprojection_errors):.6f}px")

# High focal with noise
detections_b_noise = project_swarm_to_detections(
    transcript, rig_b, pixel_noise_std=1.0, drop_prob=0.0, seed=1
)

tracks_b_noise = solve_correspondence(
    detections_b_noise, rig_b,
    epipolar_threshold=3.0,
    min_views=2,
    max_reproj_error=5.0,
    seed=42
)

recon_b_noise = triangulate_dlt(tracks_b_noise, rig_b, detections_b_noise)

print(f"Rig B (f=2667) with noise: {len(recon_b_noise.positions_3d)} points, median error: {np.median(recon_b_noise.reprojection_errors):.6f}px")

print("\n\nC. Analysis of the relationship")
print("=" * 50)

print("Without noise:")
if len(recon_a.positions_3d) > 0 and len(recon_b.positions_3d) > 0:
    print(f"  Rig A pos: {recon_a.positions_3d[0]}")
    print(f"  Rig B pos: {recon_b.positions_3d[0]}")
    print(f"  Pos ratio (B/A): {recon_b.positions_3d[0][0]/recon_a.positions_3d[0][0] if recon_a.positions_3d[0][0] != 0 else 'N/A'}")

if len(recon_a_noise.positions_3d) > 0 and len(recon_b_noise.positions_3d) > 0:
    error_ratio = np.median(recon_b_noise.reprojection_errors) / np.median(recon_a_noise.reprojection_errors)
    focal_ratio = rig_b.focal_px / rig_a.focal_px
    expected_error_ratio = rig_a.focal_px / rig_b.focal_px

    print(f"  Error ratio (B/A): {error_ratio:.3f}")
    print(f"  Focal ratio (B/A): {focal_ratio:.3f}")
    print(f"  Expected error ratio: {expected_error_ratio:.3f}")
    print(f"  Distance from expected: {abs(error_ratio - expected_error_ratio):.3f}")

print("\nCONCLUSION:")
if len(recon_a.positions_3d) > 0 and len(recon_b.positions_3d) > 0:
    pos1 = recon_a.positions_3d[0]
    pos2 = recon_b.positions_3d[0]

    # If positions are the same despite different focal length, then focal is canceling downstream
    pos_diff = np.linalg.norm(pos1 - pos2)
    print(f"  Without noise, positions differ by {pos_diff:.6f}m")
    if pos_diff < 0.001:
        print("  -> POSITION CANCELLATION: Focal length IS canceling downstream")
    else:
        print("  -> POSITIONS DIFFERENT: Focal length IS affecting reconstruction")
else:
    print("  Could not compare positions due to missing reconstructions")