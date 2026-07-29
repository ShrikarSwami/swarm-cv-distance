"""
B2: Projection + Noise — projects SwarmTruth through CameraRig to produce anonymous Detections.

All types imported from data_contract. No imports from other Stage B modules.
"""

import numpy as np
from numpy.typing import NDArray

from data_contract import (
    SwarmTruth,
    CameraRig,
    Detections,
    IMAGE_SIZE,
    project_points_batch,
    validate_detections,
)


def project_swarm_to_detections(
    truth: SwarmTruth,
    rig: CameraRig,
    pixel_noise_std: float = 2.0,
    drop_prob: float = 0.05,
    seed: int = 0,
) -> Detections:
    """
    Project all drones in all frames through all cameras to produce anonymous detections.

    Args:
        truth: SwarmTruth with (n_frames, n_drones, 3) world positions
        rig: CameraRig with K, w2c_R, w2c_t for each view
        pixel_noise_std: Standard deviation of Gaussian pixel noise (px)
        drop_prob: Probability each detection is dropped (occlusion/miss)
        seed: RNG seed for reproducibility

    Returns:
        Detections: list of length n_views, each element is (n_dets_v, 2) array
    """
    rng = np.random.default_rng(seed)

    n_frames = truth.n_frames
    n_drones = truth.n_drones
    n_views = rig.n_views
    width, height = IMAGE_SIZE

    points_per_view = []

    for v in range(n_views):
        K = rig.K[v]
        R_w2c = rig.w2c_R[v]
        t_w2c = rig.w2c_t[v]

        view_detections = []

        for f in range(n_frames):
            # Project all drones in this frame
            world_pts = truth.positions[f]  # (N, 3)
            pixels = project_points_batch(world_pts, K, R_w2c, t_w2c)  # (N, 3) with depth in col 2

            # Filter valid projections (in front of camera, in frame)
            valid = (pixels[:, 2] > 0) & \
                    (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & \
                    (pixels[:, 1] >= 0) & (pixels[:, 1] < height)

            valid_indices = np.where(valid)[0]

            for di in valid_indices:
                # Apply drop probability (simulate occlusion/missed detection)
                if rng.random() < drop_prob:
                    continue

                # Add Gaussian pixel noise
                px = pixels[di, :2].copy()
                noise = rng.normal(0, pixel_noise_std, size=2)
                px += noise
                view_detections.append(px)

        # Convert to array (M_v, 2)
        if view_detections:
            points_per_view.append(np.array(view_detections, dtype=np.float64))
        else:
            points_per_view.append(np.empty((0, 2), dtype=np.float64))

    detections = Detections(points_per_view=points_per_view)
    assert validate_detections(detections), "Generated Detections failed validation"
    return detections


def project_frame_to_detections(
    frame_positions: NDArray[np.float64],  # (N, 3)
    rig: CameraRig,
    pixel_noise_std: float = 2.0,
    drop_prob: float = 0.05,
    seed: int = 0,
) -> Detections:
    """
    Project a single frame of drone positions to detections.

    Args:
        frame_positions: (n_drones, 3) world positions for one frame
        rig: CameraRig
        pixel_noise_std: Gaussian pixel noise std
        drop_prob: Drop probability per detection
        seed: RNG seed

    Returns:
        Detections for this single frame
    """
    rng = np.random.default_rng(seed)

    n_drones = frame_positions.shape[0]
    n_views = rig.n_views
    width, height = IMAGE_SIZE

    points_per_view = []

    for v in range(n_views):
        K = rig.K[v]
        R_w2c = rig.w2c_R[v]
        t_w2c = rig.w2c_t[v]

        view_detections = []

        # Project all drones
        pixels = project_points_batch(frame_positions, K, R_w2c, t_w2c)  # (N, 3)

        # Filter valid
        valid = (pixels[:, 2] > 0) & \
                (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & \
                (pixels[:, 1] >= 0) & (pixels[:, 1] < height)

        valid_indices = np.where(valid)[0]

        for di in valid_indices:
            if rng.random() < drop_prob:
                continue
            px = pixels[di, :2].copy()
            px += rng.normal(0, pixel_noise_std, size=2)
            view_detections.append(px)

        if view_detections:
            points_per_view.append(np.array(view_detections, dtype=np.float64))
        else:
            points_per_view.append(np.empty((0, 2), dtype=np.float64))

    detections = Detections(points_per_view=points_per_view)
    assert validate_detections(detections), "Generated Detections failed validation"
    return detections


# ============================================================================
# Self-Test
# ============================================================================

if __name__ == "__main__":
    print("=== B2 Projection + Noise Self-Test ===")

    # Import B1 to generate test data
    from b1_scene_rig import generate_swarm_truth, generate_camera_rig

    # Test 1: Multi-frame projection
    print("\nTest 1: Multi-frame projection")
    truth = generate_swarm_truth(n_drones=10, n_frames=3, area_km=5.0, height_range_m=1000.0, seed=42)
    rig = generate_camera_rig(truth, n_views=6, geometry_class="mixed", standoff_m=2000.0, seed=123)

    detections = project_swarm_to_detections(truth, rig, pixel_noise_std=2.0, drop_prob=0.05, seed=0)
    print(f"  Frames: {truth.n_frames}, Drones: {truth.n_drones}, Views: {rig.n_views}")
    print(f"  Detections per view: {[len(d) for d in detections.points_per_view]}")
    print(f"  Total detections: {sum(len(d) for d in detections.points_per_view)}")

    # Test 2: Single-frame projection
    print("\nTest 2: Single-frame projection")
    single_frame = truth.positions[0]
    n_drones = single_frame.shape[0]
    detections_1f = project_frame_to_detections(single_frame, rig, pixel_noise_std=1.0, drop_prob=0.1, seed=1)
    print(f"  Drones: {n_drones}, Views: {rig.n_views}")
    print(f"  Detections per view: {[len(d) for d in detections_1f.points_per_view]}")

    # Test 3: Zero noise (deterministic projection)
    print("\nTest 3: Zero noise (deterministic)")
    detections_0 = project_swarm_to_detections(truth, rig, pixel_noise_std=0.0, drop_prob=0.0, seed=42)
    total_dets = sum(len(d) for d in detections_0.points_per_view)
    print(f"  Total detections: {total_dets}")
    print(f"  Max possible (all drones in all views): {truth.n_frames * truth.n_drones * rig.n_views}")
    print(f"  (Low count is expected - cameras only see subset of drones at this standoff)")

    # Test 4: Verify Detections structure
    print("\nTest 4: Detections validation")
    for v, pts in enumerate(detections_0.points_per_view):
        print(f"  View {v}: {pts.shape[0]} detections, shape={pts.shape}")
        assert pts.shape[1] == 2, "Each detection must be (x, y)"
        # Check bounds
        w, h = IMAGE_SIZE
        assert np.all(pts[:, 0] >= 0) and np.all(pts[:, 0] <= w), f"View {v}: x out of bounds"
        assert np.all(pts[:, 1] >= 0) and np.all(pts[:, 1] <= h), f"View {v}: y out of bounds"

    print("\n=== B2: ALL TESTS PASSED ===")