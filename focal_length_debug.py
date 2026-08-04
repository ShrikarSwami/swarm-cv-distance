#!/usr/bin/env python3
"""Debug the exact focal length wiring issue"""

import sys
import os
sys.path.insert(0, '.')
sys.path.insert(0, 'stage1_geometry')

import numpy as np

from data_contract import SwarmTruth, CameraRig, Detections
from b1_scene_rig import generate_swarm_truth, generate_camera_rig
from b2_projection import project_swarm_to_detections
from b5_triangulation import triangulate_dlt

print("=== DEBUGGING: Where does focal length stop propagating? ===")

# Generate identical setup
truth = generate_swarm_truth(n_drones=3, n_frames=1, area_km=2.0, height_range_m=500.0, seed=42)

print("\n1. Focal = 1400px")
rig1 = generate_camera_rig(truth, n_views=3, geometry_class="mixed", standoff_m=1000.0, focal_px=1400.0, seed=123)
print(f"   rig1.focal_px = {rig1.focal_px}")
print(f"   rig1.K[0] = {rig1.K[0]}")

print("\n2. Focal = 2667px")
rig2 = generate_camera_rig(truth, n_views=3, geometry_class="mixed", standoff_m=1000.0, focal_px=2666.67, seed=123)
print(f"   rig2.focal_px = {rig2.focal_px}")
print(f"   rig2.K[0] = {rig2.K[0]}")

# Check K matrix construction
print("\n\nK MATRIX CONSTRUCTION CHECK:")
print("   make_K(1400) = ", rig1.K[0])
print("   Expected: [[1400, 0, 960], [0, 1400, 540], [0, 0, 1]]")

from data_contract import make_K
K_expected_1400 = make_K(1400.0)
K_expected_2667 = make_K(2666.67)

print("   Actual rig1.K[0] matches: ", np.allclose(rig1.K[0], K_expected_1400))
print("   Actual rig2.K[0] matches: ", np.allclose(rig2.K[0], K_expected_2667))

# Now let's examine the actual triangulation code in detail
print("\n\nREPROJECTION ERROR CALCULATION:")
print("   For a 3D point, triangulation computes:")
print("   1. cam_pt = R @ pos_3d + t")
print("   2. proj = K @ cam_pt")
print("   3. proj_pixels = proj[:2] / proj[2]")
print("   4. error = norm(proj_pixels - measured_pixel)")
print("   ")
print("   If measured_pixel = true_pixel + noise, then:")
print("   - With different focal: true_pixel changes (since projection depends on focal)")
print("   - noise is in pixels, so error scales with focal!")

# Look at b2_projection.py to see how noise is applied
print("\n\nNOISE APPLICATION IN b2_projection.py:")
print("   Line 74-77 in b2_projection.py:")
print("   px = pixels[di, :2].copy()        # This is the projected pixel")
print("   noise = rng.normal(0, pixel_noise_std, size=2)")
print("   px += noise                       # Adds noise in pixel space")
print("   ")
print("   So noise is in pixel coordinates, NOT normalized coordinates!")

# The issue: in triangulation, proj = K @ cam_pt, so:
# proj_pixels = (K @ cam_pt)[:2] / (K @ cam_pt)[2]
# If we add noise to pixels, and K contains focal, then noise effect depends on focal
# Wait, let me think about this more carefully...

print("\n\nLET'S CALCULATE SYMPLECTICALLY:")
print("   For a camera at distance Z along optical axis:")
print("   Projected pixel: X_proj = f * X/Z, Y_proj = f * Y/Z")
print("   Add noise in pixels: (X_proj + nX, Y_proj + nY)")
print("   During triangulation, error = norm((X_proj + nX) - X_reproj)")
print("   Since X_reproj = f * X/Z, and X_proj = f * X/Z (perfect case):")
print("   error = norm((nX, nY))")
print("   The focal length f CANCELS OUT!")
print("   ")
print("   BUT: the measured_pixel WE MEASURE is what we see in images,")
print("   which includes the projection with focal length!")
print("   So if we have different focal lengths, we're measuring different")
print("   things, and the error calculation uses the SAME focal length.")
print("   ")
print("   So the triangulation error IS focal-dependent!")

# Let's do a simple numerical test
print("\n\nSIMPLE NUMERICAL TEST:")
print("   Let X = 0.3m, Y = 0.4m, Z = 1000m")
print("   f1 = 1400px")
print(f"     X_proj1 = {1400.0 * 0.3 / 1000.0:.4f}px")
print(f"     Y_proj1 = {1400.0 * 0.4 / 1000.0:.4f}px")
print(f"     pixel1 = ({1400.0 * 0.3 / 1000.0:.4f}, {1400.0 * 0.4 / 1000.0:.4f})")
print(f"   f2 = 2667px")
print(f"     X_proj2 = {2666.67 * 0.3 / 1000.0:.4f}px")
print(f"     Y_proj2 = {2666.67 * 0.4 / 1000.0:.4f}px")
print(f"     pixel2 = ({2666.67 * 0.3 / 1000.0:.4f}, {2666.67 * 0.4 / 1000.0:.4f})")

print("\n\nCONCLUSION:")
print("   The true projection (without noise) IS focal-dependent.")
print("   So if we have a perfect match during triangulation, the")
print("   reconstructed position should ALSO depend on focal length!")
print("   But in our tests, the RECONSTRUCTED POSITIONS and ERROR")
print("   are invariant to focal length.")
print("   ")
print("   This suggests the BUG is SOMEWHERE else...")
print("   ")
print("   Where can the bug be?")
print("   1. In the data structure (CameraRig.focal_px not being used)")
print("   2. In the triangulation math")
print("   3. In some other part of the pipeline")
print("   ")
print("   Let me trace through the triangulation step-by-step...")

# Let me look at what the triangulation actually does
print("\n\nTRILIATION STEP-BY-STEP:")
print("   1. Get 2D points from detections (these are in PIXEL COORDINATES)")
print("   2. For each 2D point, form the homogeneous pixel vector px = [u, v, 1]")
print("   3. Form projection matrix P = K @ [R|t]")
print("   4. Set up DLT equations: px x (P * X) = 0")
print("   5. Solve for X (3D position)")
print("   ")
print("   Note: The DLT solver uses the pixel coordinates (u, v),")
print("   which depend on focal length. So the triangulation SHOULD be")
print("   focal-dependent!")

print("\n\nI think I need to actually trace through a triangulation")
print("with two different focal lengths to see where the difference")
print("disappears...")