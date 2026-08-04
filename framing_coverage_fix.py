#!/usr/bin/env python3
"""Test framing coverage diagnostics and standoff adjustment"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stage1_geometry'))

import numpy as np
from data_contract import (
    SwarmTruth,
    CameraRig,
    DEFAULT_FOCAL_PX,
    IMAGE_SIZE,
)

from b1_scene_rig import (
    generate_swarm_truth,
    generate_camera_rig,
    compute_framing_coverage,
    compute_standoff_for_framing,
)

print("=== Framing Coverage Fix Test ===\n")

# Generate a small swarm that can be fully framed
# Use a smaller area to ensure drones stay in frame
transcript = generate_swarm_truth(n_drones=5, n_frames=1, area_km=0.2, height_range_m=50.0, seed=42)
print(f"Swarm: {transcript.n_drones} drones, {transcript.n_frames} frames")

# Compute swarm center and extent
swarm_center = transcript.positions.mean(axis=(0, 1))
swarm_std = transcript.positions.std(axis=(0, 1))
print(f"Swarm center: {swarm_center}")
print(f"Swarm extent (std): {swarm_std}")
print(f"Max extent from center: {np.max(np.linalg.norm(transcript.positions[0] - swarm_center, axis=1)):.1f}m")

# Reference config for comparison
ref_focal = 2666.67
ref_standoff = 1000.0

def print_coverage_stats(transcript, rig, label=""):
    """Print coverage diagnostics for a camera rig."""
    coverage = compute_framing_coverage(transcript, rig)
    h_fov = 2 * np.degrees(np.arctan(IMAGE_SIZE[0] / (2 * rig.focal_px)))
    print(f"  {label} focal={rig.focal_px:.0f}px  standoff={np.nan:>6s}  coverage={coverage:.1%}  h_fov={h_fov:.1f}deg")

# Test 1: Coverage WITHOUT standoff adjustment (current behavior)
print("\n1. Coverage WITHOUT standoff adjustment (current behavior):")
print("-" * 60)

for focal in [1400.0, 2666.67]:
    rig = generate_camera_rig(
        transcript,
        n_views=8,
        geometry_class="mixed",
        standoff_m=1000.0,
        focal_px=focal,
        seed=123
    )
    coverage = compute_framing_coverage(transcript, rig)
    h_fov = 2 * np.degrees(np.arctan(IMAGE_SIZE[0] / (2 * focal)))
    print(f"  focal={focal:.0f}px  standoff=1000m  coverage={coverage:.1%}  h_fov={h_fov:.1f}deg")

# Test 2: Coverage WITH standoff adjustment (fix)
print("\n2. Coverage WITH standoff adjustment (fix):")
print("-" * 60)

for focal in [1400.0, 2666.67]:
    adjusted_standoff = compute_standoff_for_framing(
        transcript,
        focal_px=focal,
        reference_standoff_m=ref_standoff,
        reference_focal_px=ref_focal
    )
    rig = generate_camera_rig(
        transcript,
        n_views=8,
        geometry_class="mixed",
        standoff_m=adjusted_standoff,
        focal_px=focal,
        seed=123
    )
    coverage = compute_framing_coverage(transcript, rig)
    h_fov = 2 * np.degrees(np.arctan(IMAGE_SIZE[0] / (2 * focal)))
    print(f"  focal={focal:.0f}px  standoff={adjusted_standoff:.0f}m  coverage={coverage:.1%}  h_fov={h_fov:.1f}deg")

# Test 3: Standoff ratio comparison
print("\n3. Standoff adjustment across focal range:")
print("-" * 60)
for focal in [1400.0, 2000.0, 2666.67, 4000.0, 6000.0]:
    adj = compute_standoff_for_framing(transcript, focal, ref_standoff, ref_focal)
    h_fov = 2 * np.degrees(np.arctan(IMAGE_SIZE[0] / (2 * focal)))
    print(f"  focal={focal:.0f}px  -> standoff={adj:.0f}m  (ratio: {adj/ref_standoff:.3f})  h_fov={h_fov:.1f}deg")

# Test 4: Integrated verification across focal range
print("\n4. Integrated verification (all focal lengths > 90% coverage):")
print("-" * 60)

all_covered = True
for focal in [1400.0, 2000.0, 2666.67, 4000.0]:
    adjusted_standoff = compute_standoff_for_framing(
        transcript, focal, ref_standoff, ref_focal
    )
    rig = generate_camera_rig(
        transcript,
        n_views=8,
        geometry_class="mixed",
        standoff_m=adjusted_standoff,
        focal_px=focal,
        seed=123
    )
    coverage = compute_framing_coverage(transcript, rig)
    status = "OK" if coverage > 0.9 else "LOW"
    if coverage <= 0.9:
        all_covered = False
    h_fov = 2 * np.degrees(np.arctan(IMAGE_SIZE[0] / (2 * focal)))
    print(f"  focal={focal:.0f}px  standoff={adjusted_standoff:.0f}m  coverage={coverage:.1%}  h_fov={h_fov:.1f}deg  [{status}]")

if all_covered:
    print("\n✓ ALL focal lengths achieve >90% coverage with adjusted standoff")
else:
    print("\n✗ Some focal lengths still have low coverage - need larger swarm margin")

print("\n=== Test Complete ===")