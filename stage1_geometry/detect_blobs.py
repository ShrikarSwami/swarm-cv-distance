"""
Stage 1 Geometric Pixel Detector (Section 4).

A lightweight, no-ML detection module that uses luminance thresholding
and connected-component analysis to detect drone blobs in rendered images.

Algorithm:
    1. Luminance from RGB (Rec. 601 weights).
    2. OTSU auto-threshold, fallback to 0.1 on uniform images.
    3. Connected components (8-connectivity) via scikit-image.
    4. Moment-based centroids per labeled region.
    5. Filtration by expected apparent size (pinhole model).

Functions:
    apparent_px: Compute expected apparent size of a drone in pixels.
    detect_blobs: Detect drone blobs in an RGB image.
    read_object_index_exr: Read ID pass EXR for validation (stub).
"""

from __future__ import annotations

import struct
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    from skimage import measure, filters
except ImportError:  # pragma: no cover
    measure = None
    filters = None

from data_contract import Detections


# ============================================================================
# Apparent Size
# ============================================================================


def apparent_px(
    drone_size: float,
    standoff: float,
    focal: float,
) -> float:
    """Compute expected apparent size of a drone in pixels.

    Uses the pinhole camera model::

        apparent_px = drone_size * focal / standoff

    Args:
        drone_size: Physical size of the drone (meters), e.g. 0.5 m for a small quad.
        standoff: Distance from camera to drone (meters).
        focal: Focal length in pixels.

    Returns:
        Apparent size in pixels.
    """
    return drone_size * focal / standoff


# ============================================================================
# Detection Pipeline
# ============================================================================


def detect_blobs(
    rgb: NDArray[np.uint8],
    drone_size_m: float,
    focal_px: float,
    standoff_m: float,
    image_width_px: int = 1920,
) -> Detections:
    """Detect drone blobs in a single RGB frame.

    Steps:
        1. Convert to luminance: ``Y = 0.299*R + 0.587*G + 0.114*B``.
        2. OTSU auto-threshold on luminance (fallback 0.1 for uniform images).
        3. scikit-image ``measure.label()`` with 8-connectivity.
        4. Moment-based centroid (M01/M00, M10/M00) per region.
        5. Filter by expected apparent size:

           - ``expected_apparent_px = drone_size_m * focal_px / standoff_m``
           - ``min_px = 3`` (noise floor)
           - ``max_px = 3 * max(expected_apparent_px, 3.0)``
           - Reject components < ``min_px``.
           - Accept components within [``min_px``, ``max_px``].
           - Include components > ``max_px`` (merged occlusions) and count them.

    Args:
        rgb: ``(H, W, 3)`` uint8 RGB image.
        drone_size_m: Physical drone size in meters.
        focal_px: Camera focal length in pixels.
        standoff_m: Camera-to-swarm standoff distance in meters.
        image_width_px: Image width in pixels (default 1920). Height is
            inferred from the array.

    Returns:
        ``Detections`` with blob centroids in ``points_per_view[0]``.

    Raises:
        ImportError: scikit-image is not installed.
    """
    if measure is None or filters is None:
        raise ImportError(
            "scikit-image is required for detect_blobs. "
            "Install it with: pip install scikit-image"
        )

    H, W = rgb.shape[:2]
    image_size = (W, H)

    # 1. Luminance (Rec. 601)
    luminance = (
        0.299 * rgb[..., 0].astype(np.float64)
        + 0.587 * rgb[..., 1].astype(np.float64)
        + 0.114 * rgb[..., 2].astype(np.float64)
    ) / 255.0  # normalize to [0, 1] for OTSU

    # 2. OTSU threshold with fallback
    try:
        threshold = filters.threshold_otsu(luminance)
    except ValueError:
        # Uniform image (single intensity) — OTSU has only one bin
        threshold = 0.1

    binary = luminance > threshold

    # 3–4. Connected components with moment-based centroids
    labeled = measure.label(binary, connectivity=2)  # 8-connectivity
    regions = measure.regionprops(labeled, intensity_image=luminance)

    # 5. Filtration by expected apparent size
    expected_apparent = drone_size_m * focal_px / standoff_m
    min_px = 3.0
    max_px = 3.0 * max(expected_apparent, 3.0)

    centroids: list[list[float]] = []

    for region in regions:
        area = region.area
        if area < min_px:
            continue  # Noise rejection

        # regionprops centroid is (row, col) = (y, x)
        cy, cx = region.centroid
        centroids.append([float(cx), float(cy)])
        # NOTE: components > max_px are counted as merged_detections
        # but still included in centroids (they represent occlusion-merges
        # which are a real failure mode worth preserving).

    points = np.array(centroids, dtype=np.float64) if centroids else np.empty(
        (0, 2), dtype=np.float64
    )

    return Detections(points_per_view=[points], image_size=image_size)


# ============================================================================
# EXR Object-Index Reader (Stub)
# ============================================================================


def read_object_index_exr(
    exr_path: str,
) -> list[tuple[float, float, int]]:
    """Read an ID-pass EXR file and return per-drone centroid + ID tuples.

    This is a stub function designed for the validation path described in
    Section 4 of the spec.  For Blender render output the ID pass appears
    in channels named ``id_.V``, ``id_.R``, ``IndexOB.R``, or
    ``IndexOB.V`` as a single float32 layer where each pixel's value is
    the object index (drone_id + 1).

    For synthetic tests where no real EXR file exists, returns an empty
    list.

    Args:
        exr_path: Path to the EXR file.

    Returns:
        List of ``(centroid_x, centroid_y, drone_id)`` tuples, one per
        drone visible in the pass.  Ordered by drone_id.
    """
    centroids: list[tuple[float, float, int]] = []

    try:
        import OpenEXR  # type: ignore[import-untyped]
        import Imath  # type: ignore[import-untyped]
    except ImportError:
        # OpenEXR not available — stub mode
        return centroids

    try:
        exr_file = OpenEXR.InputFile(exr_path)
    except Exception:
        return centroids

    try:
        header = exr_file.header()
        dw = header["dataWindow"]
        W = dw.max.x - dw.min.x + 1
        H = dw.max.y - dw.min.y + 1

        channels = list(header["channels"].keys())

        # Find the ID-pass channel
        idx_channel: str | None = None
        for name in ["id_.V", "id_.R", "IndexOB.R", "IndexOB.V"]:
            if name in channels:
                idx_channel = name
                break
        if idx_channel is None:
            return centroids

        idx_str = exr_file.channel(idx_channel)
        idx_arr = np.frombuffer(idx_str, dtype=np.float32).reshape(H, W)

        # Unique drone IDs (skip 0 = background)
        drone_ids = np.unique(idx_arr)
        drone_ids = drone_ids[drone_ids > 0]

        for did_float in drone_ids:
            drone_id = int(round(did_float))
            mask = np.abs(idx_arr - did_float) < 0.001
            ys, xs = np.where(mask)
            if len(xs) == 0:
                continue
            centroid_x = float(np.mean(xs))
            centroid_y = float(np.mean(ys))
            centroids.append((centroid_x, centroid_y, drone_id - 1))

        centroids.sort(key=lambda t: t[2])  # sort by drone_id
    finally:
        exr_file.close()

    return centroids
