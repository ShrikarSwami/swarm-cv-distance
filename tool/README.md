# Swarm Reconstruction Tool

**Distributable multi-view drone-swarm reconstruction and adjacency-matrix export.**

Given rendered camera views of a drone swarm, this tool reconstructs 3D drone
positions, computes pairwise distances, thresholds at a comms range (`d_max`)
to produce an inferred adjacency matrix, and exports it in the exact format
the GA/PSO critical-node search expects.

---

## Quick start

```bash
cd swarm-tool
pip install -r tool/requirements.txt

# List the 20 bundled test scenes
python -m tool --list

# Reconstruct scene 0 with all 24 views, threshold at 15 m
python -m tool --scene 0 --mode all --d-max 15.0

# 8 random views, export adjacency matrix + positions + metrics
python -m tool --scene 5 --mode random-n --n 8 --d-max 15.0 --export results/

# Pick specific views
python -m tool --scene 10 --mode exact --angles 0,3,7,12,18

# Interactive mode (no flags)
python -m tool
```

This produces:
- **Terminal output:** full metric table (median error, count error, mAP,
  per-tau precision/recall) and adjacency-graph statistics.
- **3D overlay PNG:** ground truth (green), predictions (blue), error vectors
  (red), ghosts (orange ×), missed drones (hollow gray), camera positions
  (gray triangles).
- **Export directory** (with `--export`): `adjacency.json` (GA/PSO edge list),
  `positions.csv`, `ground_truth.csv`, `metrics.json`, `summary.txt`.

## Backends

| Backend     | Flag                    | Status    | Description |
|-------------|-------------------------|-----------|-------------|
| Geometric   | `--backend geometric`   | **Default** | Blob detection → epipolar correspondence → DLT triangulation. Pure geometry, no learned model. |
| Learned     | `--backend learned`     | Stub      | T6 voxel-fusion model. Currently returns empty results — the model has not yet passed the G2 acceptance gate. See `MODEL_DETAILS.md` and the main repo's `ml/FIX_QUEUE.md`. |

## Bundled scenes

20 test-split scenes (seeds 0–19, 12–57 drones each) with 24 rendered camera
views per scene (1080×1920 PNG). No Blender or external data dependency —
everything is in `tool/scenes/`.

## Output format (GA/PSO)

The adjacency matrix is exported as a JSON edge list:

```json
{
  "n_nodes": 45,
  "n_edges": 312,
  "d_max_m": 15.0,
  "adjacency": [[0, 3], [0, 7], [1, 2], ...]
}
```

This is a **drop-in replacement** for the simulation-derived adjacency matrix
in the `drone-swarm-splitting` GA/PSO pipeline.

## View-angle modes

| Mode           | Flag                             | Description |
|----------------|----------------------------------|-------------|
| `all`          | `--mode all`                     | All 24 angles (8 ground + 8 level + 8 aerial). |
| `exact`        | `--mode exact --angles 0,3,7`    | Specific comma-separated indices 0–23. |
| `random-n`     | `--mode random-n --n 8`          | N distinct random angles. |
| `random-random`| `--mode random-random --max-views 12` | k random angles, k ~ U(1, max). |

## Pluggable model interface

To add a new backend:

1. Write a module implementing `reconstruct(images, cameras) -> (positions, confidences)`.
2. Register it in `tool/model_interface.py` in the `_BACKENDS` dict.

See `tool/geometric_backend.py` for the reference implementation and
`MODEL_DETAILS.md` for architecture details.

## Requirements

- Python ≥ 3.10
- numpy, scipy, scikit-image, matplotlib, Pillow
- No torch, no Blender, no CUDA/MPS GPU required
- Tested on macOS (Apple Silicon) and Linux

## Architecture

```
tool/
  __main__.py          CLI entry point
  model_interface.py   Backend registry + reconstruct() dispatch
  geometric_backend.py Blob detect → epipolar → DLT (frozen pipeline)
  learned_backend.py   T6 voxel-fusion stub (not yet passing G2)
  scene_loader.py      Read bundled scene data
  adjacency.py         Pairwise distance → adjacency matrix → export
  visualizer.py        3D overlay plots
  export.py            GA/PSO-compatible export
  scenes/              20 bundled test scenes (PNG + JSON)
  MODEL_DETAILS.md     Backend architecture documentation
  README.md            This file
  requirements.txt     Python dependencies
```

## Related repositories

- **`drone-swarm-splitting`**: GA/PSO critical-node search. Consumes the
  adjacency matrix this tool produces.
- **`swarm-cv-distance`**: Main research repository. Contains the rendering
  pipeline, ML model training, and the frozen geometric baseline this tool
  imports.
