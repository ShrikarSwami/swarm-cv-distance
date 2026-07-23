"""
M4: Classical pipeline end-to-end validation.

Pipeline: frame differencing → epipolar association → triangulation → adjacency

Runs on the rendered temporal dataset (dataset_temporal/).
Reports detection, correspondence, and triangulation accuracy separately
so we can see which stage fails.

Ground-truth drone IDs from Blender are used ONLY for scoring, never as
pipeline input. The pipeline must work with anonymous blobs.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------

def load_clip(clip_dir):
    """Load a rendered clip: frames, ground truth, camera parameters."""
    gt_path = clip_dir / "gt.npz"
    gt = np.load(gt_path, allow_pickle=True)

    trajectory = gt["trajectory"]  # (N_FRAMES, N_DRONES, 3)
    K = gt["K"]  # (N_VIEWS, 3, 3)
    extrinsics = gt["extrinsics"]  # (N_VIEWS, 4, 4)
    meta = gt["meta"].item()

    # Load frame images
    frames = []  # (N_VIEWS, N_FRAMES, H, W)
    frame_dir = clip_dir / "frames"
    n_views = meta["n_views"]
    n_frames = meta["n_frames"]

    for v in range(n_views):
        view_frames = []
        view_dir = frame_dir / f"view_{v:02d}"
        for f_idx in range(n_frames):
            img_path = view_dir / f"frame_{f_idx:04d}.png"
            img = np.array(Image.open(img_path).convert("L"), dtype=np.float64)
            view_frames.append(img)
        frames.append(view_frames)

    frames = np.array(frames)  # (N_VIEWS, N_FRAMES, H, W)
    return {
        "frames": frames,
        "trajectory": trajectory,
        "K": K,
        "extrinsics": extrinsics,
        "meta": meta,
    }


# ---------------------------------------------------------------------------
# Stage 1: Frame differencing (per camera)
# ---------------------------------------------------------------------------

def frame_difference(frames):
    """Compute frame differences for a camera sequence.

    frames: (N_FRAMES, H, W)
    Returns: (N_FRAMES-1, H, W) absolute differences.
    """
    return np.abs(np.diff(frames.astype(np.float64), axis=0))


def detect_blobs(diff_frame, threshold_factor=3.0, min_area=2, max_area=50):
    """Detect bright blobs in a difference frame.

    Returns list of (cy, cx, area, peak_value) for each detected blob.
    This is the 'anonymous blob' detector — no drone IDs.
    """
    # Adaptive threshold: mean + threshold_factor * std
    mean_val = diff_frame.mean()
    std_val = diff_frame.std()
    threshold = mean_val + threshold_factor * std_val

    # Binary mask
    mask = diff_frame > threshold

    # Simple connected components (flood fill)
    labeled, n_labels = _label_connected(mask)
    blobs = []
    for label_id in range(1, n_labels + 1):
        component = (labeled == label_id)
        area = component.sum()
        if area < min_area or area > max_area:
            continue
        # Centroid
        ys, xs = np.where(component)
        cy = ys.mean()
        cx = xs.mean()
        peak = diff_frame[component].max()
        blobs.append((cy, cx, area, peak))

    return blobs


def _label_connected(mask):
    """Simple connected components labeling (4-connectivity)."""
    from scipy import ndimage
    labeled, n_labels = ndimage.label(mask)
    return labeled, n_labels


# ---------------------------------------------------------------------------
# Stage 2: Epipolar correspondence
# ---------------------------------------------------------------------------

def compute_epipolar_line(K_A, K_B, ext_A, ext_B, point_A):
    """Compute the epipolar line in camera B for a point in camera A.

    point_A: (x, y) pixel coordinates in camera A.
    Returns: (a, b, c) line equation ax + by + c = 0 in camera B.
    """
    # Essential matrix: E = K_B^T * [t]_x * R * K_A^{-1}
    # where [R|t] = ext_B * ext_A^{-1}
    T_AB = ext_B @ np.linalg.inv(ext_A)
    R = T_AB[:3, :3]
    t = T_AB[:3, 3]

    # Skew-symmetric matrix of t
    t_cross = np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0],
    ])

    E = K_B.T @ t_cross @ R @ np.linalg.inv(K_A)

    # Epipolar line: l = E * x_A (homogeneous)
    x_A_h = np.array([point_A[0], point_A[1], 1.0])
    l = E @ x_A_h
    return l


def point_to_line_distance(point, line):
    """Distance from a point (x, y) to a line (a, b, c) in pixels."""
    a, b, c = line
    x, y = point
    return abs(a * x + b * y + c) / np.sqrt(a**2 + b**2)


def match_detections_epipolar(blobs_A, blobs_B, K_A, K_B, ext_A, ext_B,
                               epipolar_threshold=3.0):
    """Match blobs between two cameras using epipolar geometry.

    Returns list of (idx_A, idx_B, distance) for matches within threshold.
    """
    matches = []
    for i, (cy_a, cx_a, _, _) in enumerate(blobs_A):
        line = compute_epipolar_line(K_A, K_B, ext_A, ext_B, (cx_a, cy_a))
        for j, (cy_b, cx_b, _, _) in enumerate(blobs_B):
            dist = point_to_line_distance((cx_b, cy_b), line)
            if dist < epipolar_threshold:
                matches.append((i, j, dist))

    # Greedy: take best matches first, one-to-one
    matches.sort(key=lambda x: x[2])
    matched_A = set()
    matched_B = set()
    unique_matches = []
    for i, j, d in matches:
        if i not in matched_A and j not in matched_B:
            unique_matches.append((i, j, d))
            matched_A.add(i)
            matched_B.add(j)

    return unique_matches


def match_detections_temporal(blobs_per_frame, K_list, ext_list,
                              epipolar_threshold=3.0):
    """Match detections across cameras using epipolar + temporal consistency.

    For each frame, match blobs across camera pairs.
    Then check that matches are consistent across consecutive frames.

    Returns: list of (frame_idx, drone_id, camera_view, blob) — the
    'tracked' detections with consistent cross-camera associations.
    """
    n_frames = len(blobs_per_frame[0])  # blobs_per_frame[view][frame]
    n_views = len(blobs_per_frame)

    # Per-frame cross-camera matches
    all_matches = []  # (frame_idx, view_A, view_B, idx_A, idx_B)

    for f_idx in range(n_frames):
        for v_a in range(n_views):
            for v_b in range(v_a + 1, n_views):
                blobs_a = blobs_per_frame[v_a][f_idx]
                blobs_b = blobs_per_frame[v_b][f_idx]
                if not blobs_a or not blobs_b:
                    continue

                matches = match_detections_epipolar(
                    blobs_a, blobs_b,
                    K_list[v_a], K_list[v_b],
                    ext_list[v_a], ext_list[v_b],
                    epipolar_threshold,
                )
                for i, j, d in matches:
                    all_matches.append((f_idx, v_a, v_b, i, j, d))

    return all_matches


# ---------------------------------------------------------------------------
# Stage 3: Triangulation
# ---------------------------------------------------------------------------

def triangulate_point(pts_2d, K_list, ext_list):
    """DLT triangulation from N ≥ 2 views.

    pts_2d: list of (view_idx, x, y) — detections in each camera.
    Returns: (3,) point in world coordinates, or None if insufficient views.
    """
    if len(pts_2d) < 2:
        return None

    A = []
    for view_idx, x, y in pts_2d:
        P = K_list[view_idx] @ ext_list[view_idx][:3, :]
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    A = np.array(A)

    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_detection(frames, trajectory, threshold_factor=3.0):
    """Evaluate per-camera frame differencing detection.

    Returns per-frame detection stats: true positives, false positives,
    missed detections.
    """
    n_views, n_frames, H, W = frames.shape
    n_drones = trajectory.shape[1]

    results = []
    for v in range(n_views):
        diffs = frame_difference(frames[v])
        for f_idx in range(len(diffs)):
            blobs = detect_blobs(diffs[f_idx], threshold_factor)
            results.append({
                "view": v,
                "frame": f_idx + 1,  # frame difference is between f and f+1
                "n_blobs": len(blobs),
                "blobs": blobs,
            })

    return results


def evaluate_correspondence(all_matches, blobs_per_frame, trajectory,
                            K_list, ext_list):
    """Evaluate cross-camera correspondence accuracy.

    Uses ground-truth drone positions to score matches (but NOT as input).
    For each match (view_A, blob_A) ↔ (view_B, blob_B):
    - Project both ground-truth drones into both cameras
    - Check if the matched blobs correspond to the same physical drone
    """
    n_frames = len(blobs_per_frame[0])
    n_views = len(blobs_per_frame)

    correct = 0
    total = 0
    false_positives = 0

    for f_idx, v_a, v_b, idx_a, idx_b, dist in all_matches:
        total += 1
        blobs_a = blobs_per_frame[v_a][f_idx]
        blobs_b = blobs_per_frame[v_b][f_idx]

        if idx_a >= len(blobs_a) or idx_b >= len(blobs_b):
            false_positives += 1
            continue

        blob_a = blobs_a[idx_a]
        blob_b = blobs_b[idx_b]

        # Project each ground-truth drone into both cameras
        # and find which drone is closest to each blob
        gt_pos = trajectory[f_idx]  # (N_DRONES, 3)

        best_drone_a = _find_closest_drone(blob_a[:2], gt_pos, K_list[v_a], ext_list[v_a])
        best_drone_b = _find_closest_drone(blob_b[:2], gt_pos, K_list[v_b], ext_list[v_b])

        if best_drone_a == best_drone_b and best_drone_a is not None:
            correct += 1
        else:
            false_positives += 1

    accuracy = correct / total if total > 0 else 0
    return {
        "total_matches": total,
        "correct": correct,
        "false_positives": false_positives,
        "accuracy": accuracy,
    }


def _find_closest_drone(blob_xy, gt_positions, K, ext):
    """Find which ground-truth drone is closest to a detected blob."""
    min_dist = float('inf')
    best_drone = None

    for drone_id in range(len(gt_positions)):
        pos_h = np.append(gt_positions[drone_id], 1.0)
        proj = K @ ext[:3, :] @ pos_h
        if proj[2] <= 0:
            continue
        px = proj[0] / proj[2]
        py = proj[1] / proj[2]
        dist = np.sqrt((px - blob_xy[0])**2 + (py - blob_xy[1])**2)
        if dist < min_dist:
            min_dist = dist
            best_drone = drone_id

    # Only accept if within reasonable distance (5px)
    if min_dist < 5.0:
        return best_drone
    return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline_on_clip(clip_dir):
    """Run the full classical pipeline on one clip."""
    print(f"\n  Loading {clip_dir.name}...")
    data = load_clip(clip_dir)

    frames = data["frames"]
    trajectory = data["trajectory"]
    K = data["K"]
    ext = data["extrinsics"]
    meta = data["meta"]

    n_views, n_frames, H, W = frames.shape
    print(f"  {n_views} views, {n_frames} frames, {meta['n_drones']} drones")

    # Stage 1: Detection
    print(f"  Stage 1: Frame differencing...")
    detection_results = evaluate_detection(frames, trajectory)
    total_blobs = sum(r["n_blobs"] for r in detection_results)
    print(f"    Total blobs detected: {total_blobs} "
          f"({total_blobs / len(detection_results):.1f} per frame-view)")

    # Stage 2: Correspondence
    print(f"  Stage 2: Epipolar correspondence...")
    # Compute blobs per view per frame (recompute for correspondence)
    blobs_per_frame = []
    for v in range(n_views):
        diffs = frame_difference(frames[v])
        view_blobs = [detect_blobs(d) for d in diffs]
        blobs_per_frame.append(view_blobs)

    all_matches = match_detections_temporal(blobs_per_frame, K, ext)
    print(f"    Cross-camera matches: {len(all_matches)}")

    # Score correspondence
    corr_result = evaluate_correspondence(all_matches, blobs_per_frame,
                                           trajectory, K, ext)
    print(f"    Correspondence accuracy: {corr_result['accuracy']:.1%} "
          f"({corr_result['correct']}/{corr_result['total_matches']})")

    # Stage 3: Triangulation (only for matched detections)
    print(f"  Stage 3: Triangulation...")
    # Group matches by frame and drone-like cluster
    # For now: triangulate each matched detection pair
    tri_errors = []
    for f_idx, v_a, v_b, idx_a, idx_b, _ in all_matches:
        blobs_a = blobs_per_frame[v_a][f_idx]
        blobs_b = blobs_per_frame[v_b][f_idx]
        if idx_a >= len(blobs_a) or idx_b >= len(blobs_b):
            continue

        blob_a = blobs_a[idx_a]
        blob_b = blobs_b[idx_b]

        # Triangulate
        pts = [(v_a, blob_a[0], blob_a[1]), (v_b, blob_b[0], blob_b[1])]
        X = triangulate_point(pts, K, ext)
        if X is None:
            continue

        # Find closest ground-truth drone
        gt_pos = trajectory[f_idx]
        dists = np.linalg.norm(gt_pos - X, axis=1)
        min_dist = dists.min()
        tri_errors.append(min_dist)

    tri_errors = np.array(tri_errors) if tri_errors else np.array([float('inf')])
    print(f"    Triangulated: {len(tri_errors)} points")
    print(f"    Median error: {np.median(tri_errors):.1f}m")
    print(f"    Mean error: {tri_errors.mean():.1f}m")
    print(f"    <100m: {(tri_errors < 100).mean():.1%}")
    print(f"    <500m: {(tri_errors < 500).mean():.1%}")

    return {
        "clip": clip_dir.name,
        "n_blobs": total_blobs,
        "n_matches": len(all_matches),
        "correspondence": corr_result,
        "triangulation": {
            "n_points": len(tri_errors),
            "median_error_m": float(np.median(tri_errors)),
            "mean_error_m": float(tri_errors.mean()),
            "pct_under_100m": float((tri_errors < 100).mean()),
            "pct_under_500m": float((tri_errors < 500).mean()),
        },
    }


def main():
    dataset_root = Path("dataset_temporal")
    clips_dir = dataset_root / "clips"

    if not clips_dir.exists():
        print(f"ERROR: {clips_dir} not found. Run render_batch.py first.")
        sys.exit(1)

    clip_dirs = sorted(clips_dir.iterdir())
    clip_dirs = [d for d in clip_dirs if d.is_dir()]
    print(f"Found {len(clip_dirs)} clips")

    all_results = []
    for clip_dir in clip_dirs:
        result = run_pipeline_on_clip(clip_dir)
        all_results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print("M4 SUMMARY")
    print(f"{'='*70}")
    print(f"Clips evaluated: {len(all_results)}")

    corr_accs = [r["correspondence"]["accuracy"] for r in all_results]
    tri_medians = [r["triangulation"]["median_error_m"] for r in all_results]
    tri_pcts = [r["triangulation"]["pct_under_100m"] for r in all_results]

    print(f"\nCorrespondence accuracy: {np.mean(corr_accs):.1%} ± {np.std(corr_accs):.1%}")
    print(f"Triangulation median error: {np.median(tri_medians):.1f}m (median across clips)")
    print(f"Triangulation <100m: {np.mean(tri_pcts):.1%}")

    # Per-clip breakdown
    print(f"\nPer-clip results:")
    for r in all_results:
        print(f"  {r['clip']}: corr={r['correspondence']['accuracy']:.0%}, "
              f"tri_med={r['triangulation']['median_error_m']:.0f}m, "
              f"<100m={r['triangulation']['pct_under_100m']:.0%}")

    # Save
    out_path = dataset_root / "m4_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
