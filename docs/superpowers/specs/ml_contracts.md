# ML Contracts — Wave 0 Freeze

**Date:** 2026-07-31
**Status:** proposed for freeze — escalated for acceptance before Wave 1
**Source of truth for I/O:** `docs/superpowers/specs/2026-07-31-ml-swarm-reconstruction-design.md`
(extends `2026-07-30-end-to-end-demo-design.md`, which remains source of truth for the
geometric track).
**Operating rules:** `docs/superpowers/ORCHESTRATOR_ML.md`.
**Freeze rule:** this file defines exact signatures, shapes, dtypes, and dict/shard keys —
**no implementations**. Every Wave-1 agent codes against this contract. Once accepted,
any change to a signature, shape, dtype, or key requires human approval. A component is
done when its acceptance command exits 0, run by the orchestrator, never because a
subagent said so.

Frozen constants referenced throughout (source: `data_contract.py` (frozen), `calib.json`
(frozen), `ml/scene_gen.py` (T1)):

| Constant | Value | Source |
|---|---|---|
| `IMAGE_W`, `IMAGE_H` | `1920`, `1080` | `data_contract.IMAGE_SIZE` |
| `FOCAL_PX` | `2666.67` | `data_contract.DEFAULT_FOCAL_PX` |
| `N_VIEWS` | `24` (8 per tier) | spec §5 / `scene_gen.N_VIEWS` |
| `VOXEL_GRID_RES` | `64` | `tests/test_predictions_ml.py` |
| `DRONE_SIZE_M` | `0.5` | `calib.json` |
| `DEFAULT_TAUS` (m) | `[0.5, 1.0, 2.0, 5.0]` | spec §6 |
| Seed ranges (PATCH 7) | test `0–999`, val `1000–1999`, train `2000+` | spec §7 / `scene_gen.SEED_*` |

---

## 1. `ml/metrics.py` — shared metrics

**Freeze:** `ml/metrics.py` freezes on acceptance. One implementation, both tracks
consume it identically. There is no per-track code path; the two named entry points are
thin wrappers over one shared core so the comparison is never retrofitted.

### 1.1 Public functions and signatures

```python
DEFAULT_TAUS = (0.5, 1.0, 2.0, 5.0)          # metres; frozen (spec §6)

def evaluate(pred, true, taus=DEFAULT_TAUS, confidence=None) -> dict
    """Shared core — the single code path both tracks execute."""

def ml_evaluate(pred, true, taus=DEFAULT_TAUS, confidence=None) -> dict
    """ML-track entry point. MUST return a dict identical to `evaluate` on
    identical inputs (frozen test `test_metric_path_identity`)."""

def geometric_evaluate(pred, true, taus=DEFAULT_TAUS, confidence=None) -> dict
    """Geometric-track entry point. MUST return a dict identical to `evaluate`
    on identical inputs (frozen test `test_metric_path_identity`)."""

def ap_at_tau(pred, true, tau, confidence=None) -> float
    """Confidence-aware AP at one tau (spec §6, headline mAP uses this with
    real confidences). Called internally by `evaluate`."""
```

### 1.2 Argument contracts (both entry points and the core)

| Arg | Shape | Dtype | Meaning |
|---|---|---|---|
| `pred` | `(K, 3)` | `float64` (accepts `float32`) | predicted 3D positions, world metres. **Rows with any non-finite value are dropped before matching**; they do not count toward `n_pred` and are never matched (the geometric track emits NaN rows for unreconstructed drones). |
| `true` | `(N, 3)` | `float64` (accepts `float32`) | ground-truth 3D positions, world metres. No non-finite rows expected. |
| `taus` | sequence of float | — | matching thresholds, **metres**, ascending. |
| `confidence` | `(K,)` or `None` | `float64` | per-predicted-point confidence, aligned to `pred` **before** NaN-drop. ML track: heatmap peak value. Geometric track: views used in triangulation (or `None`). `None` → uniform confidence (single operating point). |

Return values are Python `float`/`int` (JSON-serializable) — both tracks write them to
tables without further conversion.

### 1.3 Return dict schema (exact keys)

```python
{
    "n_true": int,            # N
    "n_pred": int,            # K after NaN-drop
    "per_tau": {
        0.5: {"precision": float, "recall": float, "f1": float,
              "ap": float, "n_matched": int},
        1.0: { ... },
        2.0: { ... },
        5.0: { ... },
    },
    "mAP": float,             # mean of per_tau[t]["ap"] over the given taus
    "median_err_m": float,    # metres, over matched pairs; NaN if no pairs matched
    "chamfer_m": float,       # metres, symmetric two-sided chamfer (defn below)
    "count_err": int,         # n_pred - n_true
}
```

`per_tau` is keyed by the **float tau value** (they serialize to strings in JSON).

### 1.4 Definitions (pinned, so no two tracks can disagree)

- **Matching:** Hungarian min-cost assignment (`scipy.optimize.linear_sum_assignment`)
  on pairwise Euclidean distances (metres), restricted to `distance <= tau`. Unmatched
  `pred` rows are false positives at that tau; unmatched `true` rows are false negatives.
- **precision** = `n_matched / n_pred`, **recall** = `n_matched / n_true`,
  **f1** = `2·p·r / (p + r)` (0 when `p + r == 0`).
- **AP@tau:** confidence-sorted greedy matching (highest confidence first, nearest
  unmatched truth within tau), VOC-style PR integration over all predictions.
  With `confidence=None`, every point has equal confidence and the curve is a single
  operating point: `ap = precision@tau` after matching all predictions.
- **median_err_m:** median of the Euclidean matched-pair distances.
- **chamfer_m:** `(1/N)·Σ_true min_{pred} ||t−p|| + (1/K)·Σ_pred min_{true} ||p−t||`
  computed on non-dropped rows (threshold-free set-level check).
- **count_err:** `n_pred − n_true`.

### 1.5 Path-identity test (frozen)

`tests/test_predictions_ml.py::test_metric_path_identity` asserts
`ml_evaluate(pred, true, taus) == geometric_evaluate(pred, true, taus)` for identical
`(K,3)` arrays. Equality is by dict value. Any divergence in the shared core fails it.

### 1.6 Freeze

Tau set, match threshold, AP rule, and dict keys are frozen. Never loosened to make a
number pass — escalate instead (ORCHESTRATOR_ML escalation trigger 7).

---

## 2. `ml/pack_dataset.py` — shard format (T4)

**Freeze:** `ml/splits.json` freezes once written. The shard schema below is frozen on
acceptance of T4. Shards store **PNG BYTES, never decoded arrays** (spec PATCH 2 / §7
CRITICAL): raw 1080p uint8 is 6.2 MB/image and packs 5,000 scenes to ~560 GB; PNG bytes
pack to ~33 KB/image.

### 2.1 Input

A data root in the PATCH 2 layout written by `ml/render_harness.py`:

```
<root>/scenes/SS/NNNNN/
    ground_truth.json     # schema: ml/scene_gen.serialize_scene()
    cameras.json          # schema: ml/scene_gen.serialize_scene()
    angle_00.png ... angle_23.png
<root>/manifest.jsonl     # render-harness source of truth: one JSON record per done scene
```

The set of scenes to pack is taken from `manifest.jsonl` (seed, split, cell). The packer
never guesses or scans ad hoc.

### 2.2 CLI and output layout

```
python -m ml.pack_dataset --root <data-root> --out <packed-root> [--splits train val test]
                          [--shard-size N]
```

```
<packed-root>/
    splits.json          # copy of the canonical split manifest (below)
    shard_0000.npz
    shard_0001.npz
    ...
```

- Shards are ordered by ascending scene seed; each holds up to `--shard-size` scenes
  (default 32). A shard may mix splits — split is per scene, the loader filters.
- Container: `np.savez` (uncompressed). PNG bytes are already compressed; recompressing
  wastes CPU.

### 2.3 `splits.json` schema (canonical, frozen)

Canonical copy lives at **`ml/splits.json`** in the repo (frozen; consumed by the frozen
test `test_split_disjointness`). A byte-identical copy is written to `<packed-root>/splits.json`.

```json
{
  "schema_version": 1,
  "generated_at": "2026-07-31T00:00:00+00:00",
  "train": [2000, 2001, 2002, "..."],
  "val":   [1000, 1001, 1002, "..."],
  "test":  [0, 1, 2, "..."]
}
```

- `train`/`val`/`test`: ascending lists of scene **seeds** (ints) that were packed.
- Split is assigned by `scene_gen.split_for_seed(seed)` (PATCH 7): test `0–999`,
  val `1000–1999`, train `2000+`. A seed appears in exactly one split (G1).
- **G1 invariant:** zero seed overlap across the three lists — asserted by the frozen
  test `test_split_disjointness`, and the packer must verify before writing.

### 2.4 Shard file schema (`shard_XXXX.npz` exact keys/shapes/dtypes)

```python
{
    "scene_ids":    np.ndarray  int64     (S,)              # scene seeds in the shard
    "splits":       np.ndarray  object    (S,)              # "train" | "val" | "test"
    "cells":        np.ndarray  object    (S,)              # "primary" | "secondary"
    "n_drones":     np.ndarray  int64     (S,)
    "radius_m":     np.ndarray  float64   (S,)
    "swarm_center": np.ndarray  float64   (S, 3)            # world metres
    "positions":    np.ndarray  object    (S,)              # elt i: float64 (n_i, 3) ground truth
    "png_bytes":    np.ndarray  object    (S, N_VIEWS)      # elt (i,j): raw bytes of angle_%02d.png
    "cameras":      np.ndarray  object    (S,)              # elt i: list of N_VIEWS camera dicts
}
```

- `S` = scenes per shard (≤ `--shard-size`). `N_VIEWS = 24` for every scene.
- `png_bytes[i, j]` is the **file bytes of `angle_%02d.png` read from disk** — a `bytes`
  object, never a decoded array. The dataloader decodes on load.
- `cameras[i]` is the exact `cameras.json["views"]` list (per-view dicts: `angle_idx`,
  `tier`, `elevation_deg`, `azimuth_deg`, `K` 3×3, `c2w` 4×4, `w2c_R` 3×3, `w2c_t` (3,)).
- `positions[i]` is the exact `ground_truth.json["positions"]` list as float64 (n_i, 3).
- Object arrays load with `np.load(path, allow_pickle=True)`.
- The schema is extensible by **adding** keys only; the meaning of an existing key never
  changes.

### 2.5 PNG-bytes invariant (acceptance check)

After packing, on-disk shard size must be ≈ the sum of the raw PNG bytes it holds (plus
a few KB of JSON/npz metadata). A shard at ~6.2 MB/image means decoded arrays were
packed — that is a defect and the disk will fill. Verify, do not eyeball.

### 2.6 Dataloader read contract

```python
shard = np.load(path, allow_pickle=True)
# per scene i:
#   im_j = PIL.Image.open(io.BytesIO(shard["png_bytes"][i, j])).convert("RGB")
#   views[j] = np.asarray(im_j, dtype=np.float32) / 255.0      # (3, 1080, 1920)
#   positions = shard["positions"][i]                           # (n_i, 3) float64
#   cameras  = shard["cameras"][i]                              # 24 camera dicts
#   center   = shard["swarm_center"][i];  radius = shard["radius_m"][i]
```

---

## 3. `ml/model.py` — model I/O (architecture only, T6/T7 consumer)

**Freeze:** this I/O contract is frozen on acceptance. `ml/model.py` itself is not frozen
(it evolves through T6), but no public signature, shape, or dtype below changes without
re-escalation.

### 3.1 Public functions

```python
VOXEL_GRID_RES = 64

def forward(views, cameras=None, grid=None) -> np.ndarray
    """Fused 3D occupancy heatmap. Returns np.ndarray float32 (64, 64, 64), >= 0."""

def extract_positions(heatmap, grid) -> np.ndarray
    """Local maxima + per-peak soft-argmax -> predicted 3D positions.
    Returns np.ndarray float32 (K, 3) world metres."""
```

`forward` accepts either a list of numpy arrays or a stacked torch tensor; it returns a
**numpy** array (the frozen prediction test compares with `np.allclose`).

### 3.2 Inputs

| Arg | Shape | Dtype | Meaning |
|---|---|---|---|
| `views` | list of V arrays each `(3, 1080, 1920)`, or one tensor `(V, 3, 1080, 1920)` | `float32` | V views, V ∈ {1..24}. Variable view count is handled by the pooling step — no architecture change between 2 and 8 views. |
| `cameras` | `None` or list of V camera dicts | — | pose per view, `views[i]` ↔ `cameras[i]` by position. |
| `grid` | `None` or dict | — | the back-projection volume (see 3.3). |

Camera dict format — exactly the `cameras.json["views"]` entry (scene_gen), so the
dataloader passes it through unchanged:

```python
{
    "angle_idx": int,          # informational, unused by the model
    "tier": "ground",          # informational
    "elevation_deg": float,    # informational
    "azimuth_deg": float,      # informational
    "K":     [[float]*3]*3,    # 3x3 intrinsics, float64
    "c2w":   [[float]*4]*4,    # 4x4 camera-to-world, blender convention (scene_gen output)
    "w2c_R": [[float]*3]*3,    # 3x3 world-to-camera rotation, float64
    "w2c_t": [float, float, float],   # (3,) world-to-camera translation
}
```

### 3.3 Grid (back-projection volume)

- `grid = {"center": (3,) float32 world metres, "radius_m": float}` — a cube of side
  `2 * radius_m` centred on `center`. **Required when `cameras` is given** (the model
  back-projects into this volume). Supplied by the dataloader from `swarm_center` /
  `radius_m` of the scene.
- Voxel cell `i` on an axis has world coordinate
  `center[axis] + (i + 0.5) * cell - radius_m`, where `cell = 2 * radius_m / VOXEL_GRID_RES`
  (matches the frozen `_voxel_centers` convention in `tests/test_predictions_ml.py`).
- If `cameras is None` (the pose-blind ablation control), `grid` is ignored and a
  nominal default volume is used; the output shape is unchanged.

### 3.4 Permutation invariance (frozen test `test_permutation_invariance`)

**Subject of the test is the REAL path** — `forward(views, cameras, grid)` with pose
(Ruling 1, 2026-07-31). A pose-blind bare call would validate code that never runs in
training or evaluation.

- `forward(views, cameras, grid)` MUST be invariant to a **joint** permutation of
  (view, camera) pairs — voxel fusion with symmetric mean+max pooling across views gives
  this by construction. The frozen test shuffles the pairs JOINTLY and asserts
  `np.allclose(out, out_jointly_shuffled, atol=1e-6)` on the real path.
- The (view, camera) pairing must be **load-bearing**: the frozen test additionally
  asserts that permuting views ALONE (cameras fixed) CHANGES the output — a model whose
  output is unchanged under mispairing is ignoring camera geometry and fails the test.
- **Pose-blind control (labelled ablation, NOT the test's subject):** `forward(views)`
  (no cameras) is the "does the model use camera geometry or just count blobs?" control.
  With no cameras every view is encoded by the shared-weight encoder and pooled
  **without any pose warp** (each view treated identically). It must remain order-
  invariant (`forward(views) == forward(views[::-1])`), asserted as a control in the
  same test — but it is not what the permutation test is about.

### 3.5 Output heatmap and position extraction

- Heatmap: `(64, 64, 64)` float32, values ≥ 0, one fused occupancy volume regardless of V.
  Training target is the 3D Gaussian heatmap centred on each ground-truth drone
  (spec §4); loss is MSE on the heatmap plus a count-regularising term (defined in T6).
- `extract_positions(heatmap, grid)`: 3×3×3 local maxima above a small threshold,
  clustered, each peak refined by soft-argmax over its neighbourhood; world position via
  the affine map in 3.3. `K` = number of detected peaks = the model's predicted count
  (feeds `count_err`). Differentiable soft-argmax is used during training; extraction is
  the inference/evaluation path.

### 3.6 Encoder stride — HARD constraint (PATCH 5)

Encoder stride must not exceed **8**. At a_max ~9.6–12 px, stride-16 leaves <0.5 feature
pixels per drone and the drone vanishes before the 3D head. At most 3 downsampling
stages, or dilated convolutions at full stride. This is a hard architectural constraint
from T0, not a tuning choice.

### 3.7 MPS and timing

- The model must run on the MPS backend (Apple Silicon; torch `mps`). CPU fallback for
  tests is permitted, but the timing deliverable must be measured on MPS.
- **Required deliverable (Wave 1, Agent C):** measured MPS seconds per forward pass and
  per training step at batch size 1, from an actual timed run — a single batch
  forward + backward against a random target heatmap (no training loop, no training run;
  `ml/train.py` is T6). This is the largest unknown in the project and the 
  training wall-clock estimate for 3,000 scenes derives from it.
- **Timing is measured in a CLEAN window only (Ruling 2026-07-31).** The render campaign
  renders on the same Metal GPU throughout; any timing taken while it runs is
  contaminated and must not be reported even with a caveat. Wave 1 (Agent C) builds and
  validates `ml/model.py` normally — shape checks and permutation invariance against the
  real path — then STOPS before timing. The orchestrator escalates for a clean
  measurement window; the human pauses the render, the timing is measured, the human
  resumes. No estimate may substitute for the measurement.

---

## 4. Cross-section invariants (all three contracts)

1. **No decoded arrays in shards** — PNG bytes on disk, decode on load (contract §2.5).
2. **One metrics code path** — `ml_evaluate`/`geometric_evaluate` delegate to `evaluate`;
   identical inputs → identical dicts (contract §1.5).
3. **PATCH 7 ranges are structural** — enforced by `scene_gen.split_for_seed`, checked by
   the frozen `test_split_disjointness`, written by T4 (contract §2.3).
4. **Permutation invariance is non-negotiable** for a set predictor (contract §3.4).
5. **Freeze escalations** — any proposal to change a signature, key, tau set, threshold,
   or seed range goes to the human. Never loosen in place (ORCHESTRATOR_ML trigger 7).
