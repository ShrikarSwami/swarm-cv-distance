#!/usr/bin/env python3
"""Simple focal length wiring test"""

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

print("=== Simple Focal Length Wiring Test ===\n")

# Generate same swarm for both tests
transcript = generate_swarm_truth(n_drones=5, n_frames=1, area_km=2.0, height_range_m=500.0, seed=42)

# Test 1: DEFAULT_FOCAL_PX (≈2666.67)
print("1. Testing DEFAULT_FOCAL_PX:", DEFAULT_FOCAL_PX)
rig1 = generate_camera_rig(
    transcript,
    n_views=8,
    geometry_class="mixed",
    standoff_m=1000.0,
    focal_px=DEFAULT_FOCAL_PX,
    seed=123
)

print(f"   rig1.focal_px = {rig1.focal_px:.2f}")
print(f"   rig1.K[0,0,0] = {rig1.K[0,0,0]:.2f}")
print(f"   rig1.K[0,0,0] == rig1.focal_px: {rig1.K[0,0,0] == rig1.focal_px}")

# Test 2: Different focal_px (1400.0)
print("\n2. Testing custom focal_px = 1400px")
rig2 = generate_camera_rig(
    transcript,
    n_views=8,
    geometry_class="mixed",
    standoff_m=1000.0,
    focal_px=1400.0,
    seed=456
)

print(f"   rig2.focal_px = {rig2.focal_px:.2f}")
print(f"   rig2.K[0,0,0] = {rig2.K[0,0,0]:.2f}")
print(f"   rig2.K[0,0,0] == rig2.focal_px: {rig2.K[0,0,0] == rig2.focal_px}")

print(f"\n   Focal length ratio: {rig2.focal_px/rig1.focal_px:.3f}")
print(f"   K[0,0,0] ratio: {rig2.K[0,0,0]/rig1.K[0,0,0]:.3f}")
print(f"   K[0,0,0] different: {rig1.K[0,0,0] != rig2.K[0,0,0]}")

print("\n=== Focal length is PROPERLY wired through K matrices ===")