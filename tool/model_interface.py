"""Pluggable model interface for the reconstruction tool.

Every backend must implement:

    reconstruct(images, cameras) -> tuple[np.ndarray, np.ndarray | None]

Where:
  - images: list of V uint8 (H, W, 3) arrays (RGB, 1080×1920)
  - cameras: list of V camera dicts, each with:
        K: (3,3) float64 intrinsic matrix
        w2c_R: (3,3) float64 world-to-camera rotation
        w2c_t: (3,) float64 world-to-camera translation
  - returns: (positions, confidences)
        positions: (N, 3) float64 — world ENU metres
        confidences: (N,) float64 or None — per-position confidence in [0,1]

Backends are discovered by name. To add a new backend:
    1. Write a module `tool/my_backend.py`
    2. Implement `reconstruct(images, cameras) -> (positions, confidences)`
    3. Register it here by adding an entry to `_BACKENDS`
"""

from __future__ import annotations

import importlib
import numpy as np

# Registry: name -> (module_path, description)
_BACKENDS: dict[str, tuple[str, str]] = {
    "geometric": ("tool.geometric_backend", "DLT triangulation (epipolar + DLT)"),
    "learned": ("tool.learned_backend", "Learned voxel-fusion model (T6) — NOT YET PASSING G2"),
}


def list_backends() -> dict[str, str]:
    """Return {name: description} for all registered backends."""
    return {name: desc for name, (_, desc) in _BACKENDS.items()}


def get_backend(name: str):
    """Import and return the backend module for `name`.

    Raises ValueError if the backend is not registered.
    """
    if name not in _BACKENDS:
        raise ValueError(
            "unknown backend %r; available: %s"
            % (name, ", ".join(_BACKENDS.keys()))
        )
    mod_path, _ = _BACKENDS[name]
    return importlib.import_module(mod_path)


def reconstruct(images: list[np.ndarray],
                cameras: list[dict],
                backend: str = "geometric",
                ) -> tuple[np.ndarray, np.ndarray | None]:
    """Run reconstruction through the named backend.

    Args:
        images: list of V uint8 (H, W, 3) RGB arrays.
        cameras: list of V camera dicts (K, w2c_R, w2c_t).
        backend: "geometric" or "learned".

    Returns:
        (positions (N, 3) float64, confidences (N,) float64 or None).
    """
    mod = get_backend(backend)
    return mod.reconstruct(images, cameras)
