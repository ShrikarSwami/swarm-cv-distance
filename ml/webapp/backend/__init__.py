"""FastAPI backend package for the web frontend.

This package provides the REST API endpoints that wrap the geometric
reconstruction pipeline (ml/recon_app.py) for browser-based visualization.
"""

from .main import app
from .models import (
    SceneMetadata, SceneInfo, ReconstructionRequest, ReconstructionResult,
    DronePrediction, Point3D, MetricDict, AngleSelection, Tier
)
from .scene_loader import (
    load_scenes_from_manifest, get_scene_info, get_thumbnail_path,
    validate_scene_exists
)

__all__ = [
    "app",
    "SceneMetadata",
    "SceneInfo",
    "ReconstructionRequest",
    "ReconstructionResult",
    "DronePrediction",
    "Point3D",
    "MetricDict",
    "AngleSelection",
    "Tier",
    "load_scenes_from_manifest",
    "get_scene_info",
    "get_thumbnail_path",
    "validate_scene_exists"
]