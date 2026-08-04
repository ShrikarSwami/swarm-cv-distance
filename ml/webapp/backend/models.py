"""Pydantic models for the web frontend API.

These define the request/response shapes for the FastAPI endpoints wrapping
ml/recon_app.py and the frozen pipeline.
"""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
from enum import Enum


class AngleMode(str, Enum):
    ALL = "all"
    EXACT = "exact"
    RANDOM_N = "random-n"
    RANDOM_RANDOM = "random-random"


class Tier(str, Enum):
    GROUND = "ground"
    LEVEL = "level"
    AERIAL = "aerial"


class CameraView(BaseModel):
    angle_idx: int
    tier: Tier
    elevation_deg: float
    azimuth_deg: float


class SceneMetadata(BaseModel):
    seed: int
    split: str
    cell: str
    n_drones: int
    n_views: int
    # Additional fields from manifest
    a_max: Optional[float] = None
    description: Optional[str] = None


class SceneInfo(BaseModel):
    seed: int
    metadata: SceneMetadata
    cameras: List[CameraView]


class AngleSelection(BaseModel):
    mode: AngleMode
    view_indices: Optional[List[int]] = None
    n: Optional[int] = None  # for random-n mode
    max_views: Optional[int] = None  # for random-random mode
    exact_indices: Optional[List[int]] = None  # for exact mode


class ReconstructionRequest(BaseModel):
    scene_seed: int
    angle_selection: AngleSelection
    recall_radius_px: float = 5.0


class Point3D(BaseModel):
    x: float
    y: float
    z: float

    @classmethod
    def from_array(cls, arr):
        return cls(x=float(arr[0]), y=float(arr[1]), z=float(arr[2]))


class DronePrediction(BaseModel):
    position: Point3D
    true_position: Optional[Point3D] = None
    matched: bool = False
    distance_m: Optional[float] = None
    tier: Optional[Tier] = None


class EdgeInfo(BaseModel):
    source: int
    target: int
    distance: float
    within_d_max: bool


class MetricDict(BaseModel):
    n_true: int
    n_pred: int
    per_tau: Dict[Union[str, float], Dict[str, Any]]  # frozen evaluate() keys per_tau by float tau
    mAP: float
    median_err_m: float
    chamfer_m: float
    count_err: int


class ReconstructionResult(BaseModel):
    scene_seed: int
    metrics: MetricDict
    true_positions: List[Point3D]
    pred_positions: List[Point3D]
    matched_pairs: List[tuple[int, int]]
    ghost_indices: List[int]
    missed_indices: List[int]
    camera_positions: List[Point3D]
    view_indices: List[int]
    wall_clock_s: float
    tier_composition: Dict[str, int]
    detector_recall: Optional[float] = None
    adjacency_f1: Optional[Dict[str, float]] = None