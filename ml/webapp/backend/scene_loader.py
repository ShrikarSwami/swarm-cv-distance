"""Scene loading and management for the web frontend.

Loads scenes from ~/swarm_ml/manifest.jsonl and provides the data needed
for the frontend: scene metadata, camera poses, and renders.
"""

import json
import os
import sys
from typing import List, Dict, Optional
from pathlib import Path

# Add the repo root to sys.path to import stage1_geometry modules
_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_STAGE1 = _REPO_ROOT / "stage1_geometry"
for _p in (_STAGE1, _REPO_ROOT):
    _p_str = str(_p)
    if _p_str not in sys.path:
        sys.path.insert(0, _p_str)

from ml.recon_app import load_manifest, find_scene, _load_scene_data
from ml.recon_app import camera_positions, selected_view_angles

# Default data root - must be ~/swarm_ml
DEFAULT_ROOT = Path.home() / "swarm_ml"


def get_manifest_root() -> Path:
    """Get the manifest root directory, ensuring ~/swarm_ml exists."""
    if not DEFAULT_ROOT.exists() or not (DEFAULT_ROOT / "manifest.jsonl").exists():
        raise FileNotFoundError(
            f"manifest not found: {DEFAULT_ROOT / 'manifest.jsonl'}. "
            f"Please ensure ~/swarm_ml is present with manifest.jsonl and scenes/"
        )
    return DEFAULT_ROOT


def load_scenes_from_manifest(
    root: Optional[str] = None,
    split: Optional[str] = None,
    cell: Optional[str] = None,
    limit: int = 40
) -> List[Dict]:
    """Load scenes from manifest.jsonl with optional filtering.

    Args:
        root: Path to data root (default: ~/swarm_ml)
        split: Filter by split (test/train/val)
        cell: Filter by cell (primary/secondary)
        limit: Maximum number of scenes to return

    Returns:
        List of scene metadata dictionaries
    """
    root = Path(root) if root else get_manifest_root()
    scenes = load_manifest(str(root))

    # Apply filters
    if split:
        scenes = [s for s in scenes if s.get("split") == split]
    if cell:
        scenes = [s for s in scenes if s.get("cell") == cell]

    return scenes[:limit]


def get_scene_info(seed: int, root: Optional[str] = None) -> Dict:
    """Get complete scene information for a seed.

    Args:
        seed: Scene seed number
        root: Path to data root (default: ~/swarm_ml)

    Returns:
        Dictionary with scene metadata, cameras, and computed info
    """
    root = Path(root) if root else get_manifest_root()

    # Load manifest to find scene
    scenes = load_manifest(str(root))
    scene_meta = find_scene(scenes, seed)

    # Load scene data (ground truth and cameras)
    gt, cam = _load_scene_data(str(root), seed)

    # Get all 24 cameras
    all_view_idxs = list(range(len(cam["views"])))
    camera_pos = camera_positions(cam, all_view_idxs)
    cameras_meta = selected_view_angles(cam, all_view_idxs)

    # Compute tier composition (ground, level, aerial)
    tier_counts = {"ground": 0, "level": 0, "aerial": 0}
    for cam_meta in cameras_meta:
        tier = cam_meta["tier"]
        if tier in tier_counts:
            tier_counts[tier] += 1

    return {
        "seed": seed,
        "metadata": scene_meta,
        "cameras": cameras_meta,
        "camera_positions": camera_pos,
        "tier_composition": tier_counts,
        "n_drones": scene_meta.get("n_drones", 0),
        "a_max": scene_meta.get("a_max", 0.0),
    }


def get_thumbnail_path(seed: int) -> Path:
    """Get path to scene thumbnail render.

    For now, return a placeholder. In a full implementation, this would
    serve actual thumbnail renders from the scene renders directory.
    """
    root = get_manifest_root()
    # Look for renders in the scene directory
    scene_dir = root / "scenes" / f"{seed // 100:02d}" / f"{seed:05d}"

    # Check for common render extensions
    for ext in [".png", ".jpg", ".jpeg"]:
        for pattern in [f"render_*.{ext}", f"*render*.{ext}", f"*.{ext}"]:
            matches = list(scene_dir.glob(pattern))
            if matches:
                return matches[0]

    # Return a placeholder path
    return root / "renders" / f"scene_{seed}.png"


def validate_scene_exists(seed: int, root: Optional[str] = None) -> bool:
    """Check if a scene exists in the manifest.

    Args:
        seed: Scene seed number
        root: Path to data root (default: ~/swarm_ml)

    Returns:
        True if scene exists, False otherwise
    """
    try:
        root = Path(root) if root else get_manifest_root()
        scenes = load_manifest(str(root))
        find_scene(scenes, seed)
        return True
    except (FileNotFoundError, SystemExit):
        return False