"""
B1: Scene & Rig Generator — builds SwarmTruth and CameraRig against frozen data contract.

All types imported from data_contract. No imports from other Stage B modules.
"""

from dataclasses import dataclass
from typing import Literal
import numpy as np
from numpy.typing import NDArray

from data_contract import (
    SwarmTruth,
    CameraRig,
    DEFAULT_FOCAL_PX,
    IMAGE_SIZE,
    PRINCIPAL_POINT,
    CONVENTION_TAG,
    make_K,
    blender_c2w_to_opencv_w2c,
    opencv_w2c_to_blender_c2w,
    validate_swarm_truth,
    validate_camera_rig,
)


# ============================================================================
# Swarm Generator
# ============================================================================

def generate_swarm_truth(
    n_drones: int,
    n_frames: int,
    area_km: float,
    height_range_m: float,
    seed: int,
) -> SwarmTruth:
    """
    Generate a swarm of drones in a bounded 3D volume.

    Args:
        n_drones: Number of drones (N)
        n_frames: Number of temporal frames (F)
        area_km: Horizontal span in km (X/Y in [-area_km*500, area_km*500])
        height_range_m: Vertical span in meters (Z in [0, height_range_m])
        seed: RNG seed for reproducibility

    Returns:
        Validated SwarmTruth with drone_ids = 0..N-1 stable across frames
    """
    rng = np.random.default_rng(seed)

    # Initial positions: uniform in volume
    xy = rng.uniform(-area_km * 500, area_km * 500, size=(n_drones, 2))
    z = rng.uniform(0.0, height_range_m, size=(n_drones, 1))
    positions_0 = np.hstack([xy, z]).astype(np.float64)  # (N, 3)

    # Add small per-frame drift (not full boids sim — just temporal variation)
    positions = np.zeros((n_frames, n_drones, 3), dtype=np.float64)
    positions[0] = positions_0
    for f in range(1, n_frames):
        # Small random walk: ~5m/std per frame per axis
        drift = rng.normal(0, 5.0, size=(n_drones, 3))
        positions[f] = positions[f - 1] + drift
        # Clamp to volume bounds
        positions[f, :, 0] = np.clip(positions[f, :, 0], -area_km * 500, area_km * 500)
        positions[f, :, 1] = np.clip(positions[f, :, 1], -area_km * 500, area_km * 500)
        positions[f, :, 2] = np.clip(positions[f, :, 2], 0, height_range_m)

    drone_ids = np.arange(n_drones, dtype=np.int32)

    truth = SwarmTruth(positions=positions, drone_ids=drone_ids)
    assert validate_swarm_truth(truth), "Generated SwarmTruth failed validation"
    return truth


# ============================================================================
# Camera Rig Generator
# ============================================================================

def _look_at_opencv(eye: NDArray[np.float64], target: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Build OpenCV convention world-to-camera (R, t) from eye and target.

    OpenCV camera: +X right, +Y down, +Z forward (looking down +Z)
    R rows = [right, -up, forward]  (world-to-camera)
    t = -R @ eye
    """
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])  # ENU world up
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    # OpenCV: R rows = [right, -up, forward]
    R = np.vstack([right, -up, forward]).astype(np.float64)
    t = (-R @ eye).astype(np.float64)
    return R, t


def _dome_camera_positions(
    swarm_center: NDArray[np.float64],
    n_views: int,
    standoff_m: float,
    elev_min_deg: float = 20.0,
    elev_max_deg: float = 50.0,
    seed: int = 0,
) -> list[NDArray[np.float64]]:
    """Place cameras on a dome around swarm center (same slant range)."""
    rng = np.random.default_rng(seed)
    positions = []
    for i in range(n_views):
        azimuth = 2 * np.pi * i / n_views + rng.uniform(-0.15, 0.15)
        elev_deg = elev_min_deg + (elev_max_deg - elev_min_deg) * i / max(n_views - 1, 1)
        elev = np.radians(elev_deg)
        horiz = standoff_m * np.cos(elev)
        height_offset = standoff_m * np.sin(elev)
        pos = swarm_center + np.array([
            horiz * np.cos(azimuth),
            horiz * np.sin(azimuth),
            height_offset,
        ])
        positions.append(pos.astype(np.float64))
    return positions


def _ground_ring_camera_positions(
    swarm_center: NDArray[np.float64],
    n_views: int,
    standoff_m: float,
    elev_deg: float = 10.0,
    seed: int = 0,
) -> list[NDArray[np.float64]]:
    """All cameras at ground level (Z=0), looking up at shallow elevation."""
    rng = np.random.default_rng(seed)
    positions = []
    elev = np.radians(elev_deg)
    horiz = standoff_m * np.cos(elev)
    height_offset = standoff_m * np.sin(elev)
    for i in range(n_views):
        azimuth = 2 * np.pi * i / n_views + rng.uniform(-0.15, 0.15)
        pos = swarm_center + np.array([
            horiz * np.cos(azimuth),
            horiz * np.sin(azimuth),
            height_offset,
        ])
        positions.append(pos.astype(np.float64))
    return positions


def generate_camera_rig(
    truth: SwarmTruth,
    n_views: int,
    geometry_class: Literal["all_ground", "mixed", "surround"],
    standoff_m: float,
    focal_px: float = DEFAULT_FOCAL_PX,
    seed: int = 0,
) -> CameraRig:
    """
    Generate a multi-camera rig looking at the swarm centroid.

    Args:
        truth: SwarmTruth (used for swarm centroid)
        n_views: Number of cameras
        geometry_class: "all_ground", "mixed", or "surround"
        standoff_m: Slant range from swarm center to cameras
        focal_px: Focal length in pixels (same for all views)
        seed: RNG seed

    Returns:
        Validated CameraRig with consistent w2c and c2w poses
    """
    swarm_center = truth.positions.mean(axis=(0, 1)).astype(np.float64)  # (3,)

    # Choose camera positions based on geometry class
    if geometry_class == "all_ground":
        cam_positions = _ground_ring_camera_positions(swarm_center, n_views, standoff_m, seed=seed)
    elif geometry_class == "mixed":
        cam_positions = _dome_camera_positions(swarm_center, n_views, standoff_m, seed=seed)
    elif geometry_class == "surround":
        # Same as mixed for now; full sphere is future work
        cam_positions = _dome_camera_positions(swarm_center, n_views, standoff_m, elev_min_deg=10, elev_max_deg=70, seed=seed)
    else:
        raise ValueError(f"Unknown geometry_class: {geometry_class}")

    # Build K matrix (same for all views)
    K = make_K(focal_px)

    # Compute w2c (OpenCV) and c2w (Blender) for each camera
    w2c_R_list = []
    w2c_t_list = []
    c2w_list = []

    for pos in cam_positions:
        R_w2c, t_w2c = _look_at_opencv(pos, swarm_center)
        w2c_R_list.append(R_w2c)
        w2c_t_list.append(t_w2c)
        # Convert to Blender camera-to-world
        c2w = opencv_w2c_to_blender_c2w(R_w2c, t_w2c)
        c2w_list.append(c2w)

    rig = CameraRig(
        K=np.stack([K] * n_views),              # (V, 3, 3)
        w2c_R=np.stack(w2c_R_list),             # (V, 3, 3)
        w2c_t=np.stack(w2c_t_list),             # (V, 3)
        c2w=np.stack(c2w_list),                 # (V, 4, 4)
        focal_px=focal_px,
        convention=CONVENTION_TAG,
        geometry_class=geometry_class,
    )
    assert validate_camera_rig(rig), "Generated CameraRig failed validation"
    return rig


def generate_degenerate_rig(
    truth: SwarmTruth,
    n_views: int,
    degeneracy: Literal["coplanar", "collinear", "single_view"],
    standoff_m: float = 1000.0,
    focal_px: float = DEFAULT_FOCAL_PX,
    seed: int = 0,
) -> CameraRig:
    """
    Generate a deliberately degenerate camera rig for testing robustness.

    - coplanar: all cameras in a plane (e.g., all at Z=0)
    - collinear: all cameras on a line
    - single_view: only 1 camera (n_views forced to 1)
    """
    swarm_center = truth.positions.mean(axis=(0, 1)).astype(np.float64)
    K = make_K(focal_px)

    if degeneracy == "single_view":
        n_views = 1

    if degeneracy == "coplanar":
        # All cameras at Z=0, ring around center
        cam_positions = _ground_ring_camera_positions(
            swarm_center, n_views, standoff_m, elev_deg=0.0, seed=seed
        )
    elif degeneracy == "collinear":
        # All cameras on X-axis
        rng = np.random.default_rng(seed)
        cam_positions = []
        for i in range(n_views):
            x = rng.uniform(-standoff_m, standoff_m)
            cam_positions.append(np.array([swarm_center[0] + x, swarm_center[1], 0.0]))
    elif degeneracy == "single_view":
        cam_positions = [swarm_center + np.array([0.0, -standoff_m, 100.0])]
    else:
        raise ValueError(f"Unknown degeneracy: {degeneracy}")

    w2c_R_list = []
    w2c_t_list = []
    c2w_list = []
    for pos in cam_positions:
        R_w2c, t_w2c = _look_at_opencv(pos, swarm_center)
        w2c_R_list.append(R_w2c)
        w2c_t_list.append(t_w2c)
        c2w_list.append(opencv_w2c_to_blender_c2w(R_w2c, t_w2c))

    rig = CameraRig(
        K=np.stack([K] * len(cam_positions)),
        w2c_R=np.stack(w2c_R_list),
        w2c_t=np.stack(w2c_t_list),
        c2w=np.stack(c2w_list),
        focal_px=focal_px,
        convention=CONVENTION_TAG,
        geometry_class=degeneracy,
    )
    assert validate_camera_rig(rig), "Degenerate CameraRig failed validation"
    return rig


# ============================================================================
# Self-Test
# ============================================================================

if __name__ == "__main__":
    print("=== B1 Scene & Rig Generator Self-Test ===")

    # Test 1: Basic swarm + mixed rig
    truth = generate_swarm_truth(n_drones=5, n_frames=1, area_km=5.0, height_range_m=1000.0, seed=42)
    print(f"SwarmTruth: {truth.n_frames} frames, {truth.n_drones} drones, IDs {truth.drone_ids}")

    rig = generate_camera_rig(truth, n_views=6, geometry_class="mixed", standoff_m=2000.0, seed=123)
    print(f"CameraRig: {rig.n_views} views, focal_px={rig.focal_px}, geometry={rig.geometry_class}")

    # Test 2: All-ground rig
    rig2 = generate_camera_rig(truth, n_views=4, geometry_class="all_ground", standoff_m=2000.0, seed=456)
    print(f"All-ground rig: {rig2.n_views} views")

    # Test 3: Degenerate coplanar
    rig3 = generate_degenerate_rig(truth, n_views=4, degeneracy="coplanar", standoff_m=2000.0, seed=789)
    print(f"Coplanar degenerate rig: {rig3.n_views} views, class={rig3.geometry_class}")

    # Test 4: Degenerate single view
    rig4 = generate_degenerate_rig(truth, n_views=1, degeneracy="single_view", standoff_m=2000.0, seed=999)
    print(f"Single-view rig: {rig4.n_views} views")

    print("\n=== B1: ALL TESTS PASSED ===")