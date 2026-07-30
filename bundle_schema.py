"""
Bundle format schema for the Stage 1 geometry pipeline.

Defines Pydantic v2 models for manifest.json, poses.json, and ground_truth.json
files in the bundle directory structure specified in spec Section 2.

Usage:
    from bundle_schema import BundleManifest, BundlePoses, BundleGroundTruth

    # Validate from file
    manifest = BundleManifest.validate_file("path/to/manifest.json")

    # Create from dict
    manifest = BundleManifest(**manifest_dict)

    # Serialize to dict (matches original JSON structure)
    data = manifest.model_dump()

    # Get test fixtures
    m, p, gt = bundle_minimal()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator


# ============================================================================
# BundleManifest — schema for manifest.json
# ============================================================================

class BundleManifest(BaseModel):
    """Schema for the bundle manifest.json file.

    Describes the scene, camera configuration, capture metadata, and
    generation provenance for a rendered bundle.
    """

    bundle_version: Literal["1.0"]
    """Bundle format version (currently "1.0")."""

    scene_id: str
    """Unique identifier for this scene / capture session."""

    format: Literal["png", "mp4"]
    """Output format for view frames."""

    n_views: int = Field(gt=0)
    """Number of camera views (positive integer)."""

    n_frames: int = Field(gt=0)
    """Number of frames per view (positive integer)."""

    frame_indices: list[int]
    """Explicit list of frame indices rendered in this bundle."""

    image_size_px: list[int] = Field(min_length=2, max_length=2)
    """Image dimensions as [width, height] in pixels."""

    focal_px: float = Field(gt=0)
    """Focal length in pixels (positive)."""

    sensor_width_mm: float = Field(default=36.0, gt=0)
    """Camera sensor width in millimetres (positive)."""

    units: str = Field(default="meters")
    """Unit of measurement for all spatial quantities."""

    has_ground_truth: bool = Field(default=False)
    """Whether a ground_truth.json file is present in the bundle."""

    coverage_pct: float = Field(ge=0, le=100)
    """Percentage of drones framed by the camera rig (0-100)."""

    sync_convention: str = Field(default="all cameras render same frame indices")
    """Convention for frame synchronisation across views."""

    generated_by: Optional[dict] = None
    """Provenance metadata about the software that generated this bundle."""

    @classmethod
    def validate_file(cls, path: str | Path) -> "BundleManifest":
        """Load and validate a manifest.json file from disk.

        Args:
            path: Filesystem path to the manifest JSON file.

        Returns:
            A validated BundleManifest instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            pydantic.ValidationError: If the file content does not match
                the manifest schema.
        """
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


# ============================================================================
# CameraView — sub-model used by BundlePoses
# ============================================================================

class CameraView(BaseModel):
    """Schema for a single camera view entry in poses.json."""

    view_idx: int
    """Zero-based index of this camera view."""

    K: list[list[float]]
    """3x3 intrinsic calibration matrix.

    Example:
        [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]
    """

    c2w: list[list[float]]
    """4x4 camera-to-world transformation matrix (Blender convention)."""

    w2c_R: Optional[list[list[float]]] = None
    """3x3 world-to-camera rotation matrix (optional, derivable from c2w)."""

    w2c_t: Optional[list[float]] = None
    """3-element world-to-camera translation vector (optional)."""

    @field_validator("K")
    @classmethod
    def _check_K_shape(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) != 3 or any(len(row) != 3 for row in v):
            raise ValueError(
                f"K must be a 3x3 matrix, got shape ({len(v)}, "
                f"{len(v[0]) if v else 0})"
            )
        return v

    @field_validator("c2w")
    @classmethod
    def _check_c2w_shape(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) != 4 or any(len(row) != 4 for row in v):
            raise ValueError(
                f"c2w must be a 4x4 matrix, got shape ({len(v)}, "
                f"{len(v[0]) if v else 0})"
            )
        return v

    @field_validator("w2c_R")
    @classmethod
    def _check_w2c_R_shape(
        cls, v: Optional[list[list[float]]]
    ) -> Optional[list[list[float]]]:
        if v is not None and (len(v) != 3 or any(len(row) != 3 for row in v)):
            raise ValueError(
                f"w2c_R must be a 3x3 matrix or None, got shape ({len(v)}, "
                f"{len(v[0]) if v else 0})"
            )
        return v

    @field_validator("w2c_t")
    @classmethod
    def _check_w2c_t_shape(
        cls, v: Optional[list[float]]
    ) -> Optional[list[float]]:
        if v is not None and len(v) != 3:
            raise ValueError(
                f"w2c_t must be a 3-element vector or None, got {len(v)} elements"
            )
        return v


# ============================================================================
# BundlePoses — schema for poses.json
# ============================================================================

class BundlePoses(BaseModel):
    """Schema for the bundle poses.json file.

    Contains the intrinsic and extrinsic calibration for all camera views.
    """

    convention: str = Field(default="blender_c2w")
    """Convention tag for the pose representation (e.g. "blender_c2w")."""

    views: list[CameraView]
    """List of camera view entries, one per view."""

    @classmethod
    def validate_file(cls, path: str | Path) -> "BundlePoses":
        """Load and validate a poses.json file from disk.

        Args:
            path: Filesystem path to the poses JSON file.

        Returns:
            A validated BundlePoses instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            pydantic.ValidationError: If the file content does not match
                the poses schema.
        """
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


# ============================================================================
# BundleGroundTruth — schema for ground_truth.json
# ============================================================================

class BundleGroundTruth(BaseModel):
    """Schema for the bundle ground_truth.json file.

    Contains the ground-truth 3D positions of all drones for every frame.
    This file is optional in a bundle (flagged by manifest.has_ground_truth)
    and MUST NOT be passed to the solver.
    """

    drone_ids: list[int]
    """List of drone identifiers (parallels the first dimension of positions)."""

    positions: list[list[list[float]]]
    """Ground-truth 3D positions with shape (n_frames, n_drones, 3).

    Each inner list has shape [n_drones, 3] for one frame, containing
    (x, y, z) coordinates in the unit specified by manifest.units.
    """

    @field_validator("positions")
    @classmethod
    def _check_positions_shape(
        cls, v: list[list[list[float]]]
    ) -> list[list[list[float]]]:
        if not v:
            raise ValueError("positions must be a non-empty list of frames")
        for fi, frame in enumerate(v):
            for di, pt in enumerate(frame):
                if len(pt) != 3:
                    raise ValueError(
                        f"Frame {fi}, drone {di}: position must have 3 elements "
                        f"(x, y, z), got {len(pt)}"
                    )
        return v

    @classmethod
    def validate_file(cls, path: str | Path) -> "BundleGroundTruth":
        """Load and validate a ground_truth.json file from disk.

        Args:
            path: Filesystem path to the ground truth JSON file.

        Returns:
            A validated BundleGroundTruth instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            pydantic.ValidationError: If the file content does not match
                the ground truth schema.
        """
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


# ============================================================================
# Test fixtures
# ============================================================================

def bundle_minimal() -> tuple[dict, dict, dict]:
    """Return a minimal valid bundle fixture suitable for tests.

    All three dicts can be passed directly to their respective Pydantic model
    constructors:

        manifest, poses, gt = bundle_minimal()
        m = BundleManifest(**manifest)

    Returns:
        Tuple of (manifest_dict, poses_dict, ground_truth_dict) each
        representing a valid bundle with minimum configuration.
    """
    manifest = {
        "bundle_version": "1.0",
        "scene_id": "test-minimal",
        "format": "png",
        "n_views": 2,
        "n_frames": 1,
        "frame_indices": [0],
        "image_size_px": [1920, 1080],
        "focal_px": 2666.67,
        "sensor_width_mm": 36.0,
        "units": "meters",
        "has_ground_truth": False,
        "coverage_pct": 100.0,
        "sync_convention": "all cameras render same frame indices",
        "generated_by": None,
    }
    poses = {
        "convention": "blender_c2w",
        "views": [
            {
                "view_idx": 0,
                "K": [
                    [2666.67, 0.0, 960.0],
                    [0.0, 2666.67, 540.0],
                    [0.0, 0.0, 1.0],
                ],
                "c2w": [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
                "w2c_R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "w2c_t": [0.0, 0.0, 1000.0],
            }
        ],
    }
    ground_truth = {
        "drone_ids": [0, 1],
        "positions": [
            [
                [100.0, 200.0, 50.0],
                [300.0, 400.0, 100.0],
            ]
        ],
    }
    return manifest, poses, ground_truth
