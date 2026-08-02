"""T6 — training loop for the VoxelFusionModel (owned file: `ml/train.py`).

Training loop for `ml.model.VoxelFusionModel` against the packed real dataset
(`~/swarm_ml_packed/shard_XXXX.npz`, contract §2.4). Frozen model I/O contract
is consumed unchanged: the differentiable path `forward_volume(views, cameras,
grid) -> (1,1,64,64,64)`. This file implements the *loop only*: dataloader,
target construction, loss, sampling policy, checkpointing, CLI.

====================================================================
LOSS (batch size 1; a "step" is one scene with a random view subset)
====================================================================
    pred      = model.forward_volume(views, cameras, grid)[0, 0]    # (64,64,64), >= 0
    target    = max over drones of the 3D Gaussian heatmap          # (64,64,64), >= 0
                g(p) = exp(-||C - p||^2 / (2 sigma^2)),  sigma = 1.5 * cell
                cell = 2 * radius_m / 64,  C = voxel-cell centres
                (contract §3.3 affine map; reference tests/test_predictions_ml.py)
    w         = 1 + pos_weight * target                 # target-proportional weight
    mse       = mean(w * (pred - target)^2)             # contract §3.5
                # sparse-target dilution fix: the 64^3 target's meaningful
                # support is well under 1%, so plain MSE makes a flat near-zero
                # output a strong local optimum (mean(target^2) ~= 0.002, below
                # the OLD 0.05 gate — the G2 false-pass). Weight is proportional
                # to the target value, NOT thresholded at target > 0: float64
                # Gaussians never underflow across a swarm-diameter box, so
                # target > 0 is ~63% of the grid, and upweighting that would put
                # a spurious DOWNWARD force on the tail voxels surrounding the
                # peaks (measured: in_mass stuck at ~25% of n_drones). The peak
                # core (target ~ 1) gets ~1+pos_weight, the tail gets ~1.
    peak_mask = (target > COUNT_MASK_T)              # 0/1 spatial support of the
                # drones' Gaussians (a constant per scene, no gradient through it)
    in_mass   = (pred * peak_mask).sum() / vpp       # peak-support mass ("drones")
    bg_excess = relu(pred - COUNT_FLOOR) * (1 - peak_mask)   # background above floor
    bg_drones = bg_excess.sum() / vpp                # background excess ("drones")
    count     = (in_mass - n_drones)^2 + COUNT_BG_WEIGHT * bg_drones^2
    loss      = mse + count_weight * count

`vpp = (target * peak_mask).sum() / n_drones` (peak-support mass per drone, per
scene) makes both count sub-terms exactly calibrated: at pred == target,
in_mass == n_drones (the count sub-term is zero) and bg_excess == 0 (the
background sub-term is zero), so the whole count term is zero.

Two sub-terms, two complementary jobs, neither with the failure modes of a
single naive count:
- `in_mass` (peak-support mass) has a strong *uniform* gradient on every voxel
  inside the drones' support at ANY intensity — no floor dead zone, so dim or
  narrow peaks are pushed toward the target Gaussian shape and count. It is a
  genuine count regulariser: a spurious ~1-drone blob inside the support adds
  ~1 to in_mass.
- `bg_drones` (background excess) crushes the uniform softplus background once
  it rises above COUNT_FLOOR and self-terminates when it drops below the floor.
  A raw all-grid mass count cannot reach zero because of that floor; splitting
  background out with a floor lets it, while still suppressing the diffuse
  phase-1 output that plain MSE (diluted over the 64^3 grid) removes only
  slowly.

`count_weight` defaults to 0.1 and `COUNT_BG_WEIGHT` defaults to 1.0. The count
term is ACTIVE in `--overfit` mode too: it is the peak-mass counter-pressure
that forces real mass on the drone support (zeroing it in overfit — the pre-fix
behaviour — removed the only force opposing the flat-output optimum and was half
of the G2 false-pass). Both components are logged every step.

====================================================================
G2 GATE (metric-based verdict; the old loss-threshold gate is FALSIFIED)
====================================================================
The old gate ("recent-mean loss < 0.05") passed trivially on a diffuse
near-flat volume: achieved recent-mean 0.002008 vs a zero-baseline mean(t^2)
~= 0.0020, 0 voxels > 0.1, peak-support MSE 0.128 vs target mean(t^2)=0.13.
The verdict is now computed from the FROZEN `ml.metrics.evaluate(pred, true)`
path on the FINAL checkpoint:
  - train the 8 overfit scenes (seeds 2000-2007) for a FIXED step budget
    (`OVERFIT_DEFAULT_STEPS`, default 600) — no loss-based early stop;
  - evaluate each scene at the fixed mixed 8-view subset `EVAL_VIEWS`
    (3 ground + 3 level + 2 aerial, spanning all three tiers);
  - decode positions with `ml.model.extract_positions`, score with
    `ml.metrics.evaluate`;
  - PASS iff aggregate median_err_m < 1.0 m AND every scene's count_err in
    [-1, +1]. A flat volume yields no peaks, so count_err = -N — the gate
    cannot be gamed.

====================================================================
DATALOADER (prefetching, real shards)
====================================================================
`torch.utils.data.DataLoader` over a `SceneDataset` that maps
`(scene_id, view_indices) -> (views, cameras, grid, target, ...)`.
PNG decode (`PIL -> float32/255 -> CHW`) runs in worker processes; shards are
loaded lazily per worker into a module-global cache (only paths/row indices are
pickled to workers, never open file handles or decoded frames). num_workers>=2,
prefetch_factor>1, persistent_workers=True, batch_size=1 (no batching anywhere:
batch growth is superlinear, measured 1.3/2.7/7.0/21.9/87 s/step for B=1/2/4/6/8).
Conservative for a 24 GB M4 Pro: default num_workers=2, prefetch_factor=2
(<= 4 decoded scenes in flight; one V=24 scene is ~600 MB of float32 frames).
The I/O-only cost per step is reported separately (see `--bench-io` and the
per-step `io` field, which is the time the loop waits on the dataloader).

====================================================================
RANDOM VIEW-SUBSET SAMPLING (one model, all compositions)
====================================================================
Each step the *main process* sampler draws:
  - scene_id  : uniform over the configured scene set (with replacement)
  - v         : uniform integer in [min_views, max_views] (default 2..8; >= 2
                enforced, a single view cannot localise depth). Cap of 8 matches
                the eval sweep's V in 2..8: training on V in [9,24] spends
                compute on configurations never evaluated (per-view
                backprojection is the dominant step cost; mean V ~= 17 -> ~5 is
                ~3x faster per step).
  - view mix  : each of the v slots picks a tier uniformly among the three
                tiers (ground/level/aerial, 8 cameras each) and a distinct
                camera within that tier. Pairing is load-bearing: the (view,
                camera) pair travels together through the dataset.
The sampling RNG is a `random.Random(seed)` owned by the sampler and saved in
every checkpoint, so resume continues the exact step sequence.

====================================================================
SCOPE CONDITIONS (non-negotiable)
====================================================================
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is REQUIRED: `aten::grid_sampler_2d_backward`
  has no MPS implementation in torch 2.12.1 and falls back to CPU with correct
  gradients. Set at import time (before torch).
- Device = MPS (Apple Silicon M4 Pro, 24 GB). Grid-sample forward runs natively
  on MPS; its backward falls back via the env var above.
- Batch size is ALWAYS 1.
- The full 3,000-scene training run is launched detached by the human; this
  module's run scope is `--smoke` + `--bench-io` + the G2 `--overfit` gate.

====================================================================
CLI
====================================================================
    python3 -m ml.train --root ~/swarm_ml_packed --split train --scenes N \
        [--epochs E | --max-steps S] [--checkpoint-dir DIR] [--resume] \
        [--seed N] [--device mps] [--overfit] [--smoke] [--bench-io N]

    --overfit   G2 gate: fixed 8-scene train subset (default seeds 2000-2007),
                fixed step budget (default OVERFIT_DEFAULT_STEPS), then the
                frozen-metric verdict on the final checkpoint:
                `G2 OVERFIT: PASS/FAIL (median_err_m < 1.0 m, per-scene
                count_err in [-1,+1])` plus a V=24 diagnostic.
                count_weight and pos_weight are BOTH active (see "G2 GATE").
    --smoke     cheap CI path: 2 steps on 2 scenes, exits 0.
    --bench-io N  dataloader-only benchmark: N timed fetches, reports the
                I/O-only cost per step (decode + IPC), no model involved.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import signal
import sys
import time
from collections import OrderedDict

# MUST precede the torch import: grid_sampler_2d_backward has no MPS kernel in
# torch 2.12.1; the fallback routes it to CPU with correct gradients.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ml import metrics  # noqa: E402
from ml.model import (  # noqa: E402
    VOXEL_GRID_RES,
    VoxelFusionModel,
    extract_positions,
)

N_VIEWS = 24                     # contract §2.4 (scene_gen.N_VIEWS, frozen)
TIERS = ("ground", "level", "aerial")
PER_TIER = 8
DEFAULT_SHARD_SIZE = 32          # pack_dataset.DEFAULT_SHARD_SIZE (frozen)
# G2 gate: fixed step budget (no loss-threshold early stop). The old 0.05
# recent-mean gate was FALSIFIED — a flat near-zero output sits at
# mean(target^2) ~= 0.002, ~25x below it, so it passed with ZERO peaks
# (0 voxels > 0.1, peak-support MSE 0.128 vs target mean(t^2)=0.13). The
# verdict now comes from the frozen metric path on the final checkpoint.
OVERFIT_DEFAULT_STEPS = 600
# Positive-voxel upweight for the sparse-target MSE (see "LOSS" docstring).
# Positive support is < 1% of the 64^3 grid, so plain MSE makes a near-zero
# output a strong local optimum; ~500x on target>0 voxels restores a real
# shape gradient there while background stays at weight 1.
POS_WEIGHT_DEFAULT = 500.0
# Fixed mixed 8-view evaluation subset for the G2 verdict — spans all three
# tiers, balanced 3 ground + 3 level + 2 aerial (camera indices 0-7 ground,
# 8-15 level, 16-23 aerial, verified on real shards):
EVAL_VIEWS = (0, 3, 6, 8, 11, 14, 17, 21)
EVAL_VIEWS_DOC = "ground[0,3,6] level[8,11,14] aerial[17,21]"
# Gaussian target sigma in voxel cells (see _build_target).
TARGET_SIGMA_CELLS = 1.5

# Count-term hyper-parameters (see module docstring "LOSS").
# COUNT_MASK_T is the target-heatmap threshold defining the per-scene peak
# support mask: voxels where the target exceeds it are inside the drones'
# Gaussian support and enter the in-support mass; everything else is treated as
# background. Target peaks are ~0.99; 0.1 captures the full Gaussian shoulder
# (radius ~3.2 voxels) while excluding the flat background.
COUNT_MASK_T = 0.1
# COUNT_FLOOR is the background-excess floor: background voxels below it are
# invisible to the background sub-term (observed converged background is
# ~0.04-0.08), and voxels above it are pushed down until they drop below.
COUNT_FLOOR = 0.1
# Relative weight of the background-excess sub-term inside `count`.
COUNT_BG_WEIGHT = 1.0

# ---------------------------------------------------------------------------
# Worker-side caches (module globals: created fresh in every spawn worker)
# ---------------------------------------------------------------------------

_SHARD_CACHE = {}
_TARGET_CACHE = OrderedDict()
_TARGET_CACHE_MAX = 128


def _load_shard(path):
    """Load (and cache) one packed shard in the calling process.

    Only called inside DataLoader workers (or the main process for --bench-io
    with workers=0); the cache is process-local so the pickle sent to workers
    never carries open file handles or decoded frames.
    """
    if path not in _SHARD_CACHE:
        _SHARD_CACHE[path] = np.load(path, allow_pickle=True)
    return _SHARD_CACHE[path]


def _decode_png_bytes(data):
    """RAW PNG bytes -> float32 (3, 1080, 1920) CHW in [0, 1] (contract §2.6)."""
    img = Image.open(io.BytesIO(data))
    a = np.asarray(img, np.float32) / 255.0
    return np.ascontiguousarray(a.transpose(2, 0, 1))


def _voxel_centers(center, radius):
    """(64^3, 3) float64 voxel-cell world centres (contract §3.3 affine map).

    Identical convention to the reference `_voxel_centers` in
    tests/test_predictions_ml.py (linspace midpoints == (i+0.5)*cell - radius).
    """
    res = VOXEL_GRID_RES
    cell = 2.0 * radius / res
    idx = np.arange(res, dtype=np.float64)
    xx, yy, zz = np.meshgrid(idx, idx, idx, indexing="ij")
    centers = (np.stack([xx, yy, zz], axis=-1) + 0.5) * cell - radius
    return (centers + np.asarray(center, dtype=np.float64)).reshape(-1, 3)


def _build_target(positions, center, radius):
    """3D Gaussian heatmap target (64,64,64) float32, values in [0, 1].

    max (not sum) over drones so overlapping blobs never exceed peak 1 and the
    count regulariser stays meaningful. sigma = TARGET_SIGMA_CELLS * voxel
    (1.5 == reference test_heatmap_target_fidelity).
    """
    res = VOXEL_GRID_RES
    cell = 2.0 * radius / res
    sigma = TARGET_SIGMA_CELLS * cell
    centers = _voxel_centers(center, radius)          # (64^3, 3) float64
    h = np.zeros(centers.shape[0], dtype=np.float64)
    for p in np.asarray(positions, dtype=np.float64):
        g = np.exp(-np.sum((centers - p) ** 2, axis=1) / (2.0 * sigma ** 2))
        np.maximum(h, g, out=h)
    return h.reshape(res, res, res).astype(np.float32)


def _target_for_scene(scene_id, positions, center, radius):
    """Cached per-scene target (deterministic; independent of the view subset)."""
    if scene_id not in _TARGET_CACHE:
        _TARGET_CACHE[scene_id] = _build_target(positions, center, radius)
        while len(_TARGET_CACHE) > _TARGET_CACHE_MAX:
            _TARGET_CACHE.popitem(last=False)
    return _TARGET_CACHE[scene_id]


def _peak_support_mask(target):
    """0/1 float32 spatial support of the drones' Gaussians (per scene constant).

    Voxels where the target exceeds COUNT_MASK_T. The mask is a constant in the
    loss (no gradient through it); it restricts the count term to the drones'
    peak support so the uniform softplus background never enters the count.
    """
    return (np.asarray(target, dtype=np.float32) > COUNT_MASK_T).astype(np.float32)


def _identity_collate(batch):
    """batch_size == 1: return the single item untouched (no field mangling)."""
    return batch[0]


def sample_view_indices(rng, v, per_tier=PER_TIER, n_tiers=3):
    """v distinct camera indices with a random tier mix.

    Each of the v slots picks a tier uniformly among the currently non-empty
    tiers and a distinct camera within it, so mixes range from balanced to
    pure-tier. Camera indices are grouped by tier in the frozen scene_gen
    layout (0..7 ground, 8..15 level, 16..23 aerial — verified on real shards).
    """
    available = [list(range(t * per_tier, (t + 1) * per_tier))
                 for t in range(n_tiers)]
    chosen = []
    for _ in range(v):
        nonempty = [t for t in range(n_tiers) if available[t]]
        tier = nonempty[rng.randrange(len(nonempty))]
        k = rng.randrange(len(available[tier]))
        chosen.append(available[tier].pop(k))
    return tuple(chosen)


def tier_mix_of(view_indices, cameras):
    """{tier: count} for the selected (view, camera) pairs."""
    mix = {t: 0 for t in TIERS}
    for j in view_indices:
        tier = cameras[j]["tier"]
        mix[tier] = mix.get(tier, 0) + 1
    return mix


# ---------------------------------------------------------------------------
# Dataset + sampler
# ---------------------------------------------------------------------------


class SceneDataset(torch.utils.data.Dataset):
    """Maps (scene_id, view_indices) -> one training step (a full scene batch).

    Decoding happens in DataLoader workers; the dataset object itself holds only
    picklable paths/indices. The returned dict:
      views     torch.Tensor (V, 3, 1080, 1920) float32
      cameras   list of V camera dicts (exact shard dicts, pairing load-bearing)
      grid      {"center": float32 (3,), "radius_m": float}
      target    torch.Tensor (64, 64, 64) float32
      peak_mask torch.Tensor (64, 64, 64) float32 0/1 (drones' Gaussian support)
      n_drones  int
      vpp       float  (peak-support mass per drone, from the target) — count ref
      scene_id / view_indices / tier_mix for logging
    """

    def __init__(self, scenes):
        # scenes: list of (scene_id, shard_path, row)
        self.scenes = scenes
        self.sids = [s[0] for s in scenes]
        self._row = {s[0]: (s[1], s[2]) for s in scenes}

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):
        scene_id, view_indices = index
        shard_path, row = self._row[scene_id]
        d = _load_shard(shard_path)
        png = d["png_bytes"][row]
        views = np.stack([_decode_png_bytes(png[j]) for j in view_indices])
        cameras = [d["cameras"][row][j] for j in view_indices]
        grid = {
            "center": np.asarray(d["swarm_center"][row], dtype=np.float32),
            "radius_m": float(d["radius_m"][row]),
        }
        positions = d["positions"][row]
        center = d["swarm_center"][row]
        radius = float(d["radius_m"][row])
        n_drones = int(d["n_drones"][row])
        target = _target_for_scene(int(scene_id), positions, center, radius)
        peak_mask = _peak_support_mask(target)
        vpp = float((target * peak_mask).sum() / n_drones) if n_drones > 0 \
            else 1.0
        return {
            "views": torch.from_numpy(views),
            "cameras": cameras,
            "grid": grid,
            "target": torch.from_numpy(target.copy()),
            "peak_mask": torch.from_numpy(peak_mask),
            "n_drones": n_drones,
            "vpp": vpp,
            "scene_id": int(scene_id),
            "view_indices": list(view_indices),
            "tier_mix": tier_mix_of(view_indices, d["cameras"][row]),
        }


class ViewSubsetSampler(torch.utils.data.Sampler):
    """Deterministic per-step (scene, view-subset) generator.

    Owns a `random.Random(seed)` whose state is checkpointed, so resume
    continues the exact step sequence. Yields (scene_id, view_indices) tuples
    which the DataLoader passes through to the dataset (workers decode only the
    sampled views; sampling itself is main-process and reproducible).
    """

    def __init__(self, scene_ids, min_views, max_views, seed, total_steps,
                 rng_state=None):
        if min_views < 2:
            raise ValueError("min_views must be >= 2 (a single view cannot "
                             "localise depth), got %d" % min_views)
        if max_views > N_VIEWS:
            raise ValueError("max_views must be <= %d, got %d"
                             % (N_VIEWS, max_views))
        if min_views > max_views:
            raise ValueError("min_views %d > max_views %d"
                             % (min_views, max_views))
        self.scene_ids = list(scene_ids)
        self.min_views = min_views
        self.max_views = max_views
        self.total_steps = int(total_steps)
        self.rng = random.Random(seed)
        if rng_state is not None:
            self.rng.setstate(rng_state)

    def __iter__(self):
        for _ in range(self.total_steps):
            sid = self.rng.choice(self.scene_ids)
            v = self.rng.randint(self.min_views, self.max_views)
            yield (sid, sample_view_indices(self.rng, v))

    def __len__(self):
        return self.total_steps

    def rng_state(self):
        return self.rng.getstate()


# ---------------------------------------------------------------------------
# Shard map + scene resolution
# ---------------------------------------------------------------------------


def build_shard_map(root):
    """scene_id -> (shard_path, row) by scanning shard headers.

    np.load keeps the npz mmapped; only the tiny scene_ids column is read per
    shard, so the scan is fast (~157 shards). Robust to any shard-size/layout
    change.
    """
    shard_map = {}
    for name in sorted(os.listdir(root)):
        if not name.startswith("shard_") or not name.endswith(".npz"):
            continue
        path = os.path.join(root, name)
        d = np.load(path, allow_pickle=True)
        ids = d["scene_ids"]
        for row, sid in enumerate(ids.tolist()):
            shard_map[int(sid)] = (path, row)
        d.close()
    return shard_map


def _parse_seed_spec(spec):
    """'2000-2007' / '2000,2002,2005' / '2000' -> sorted list of ints."""
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def load_splits(root):
    """Read the frozen split manifest (byte-identical copy at <root>/splits.json)."""
    path = os.path.join(root, "splits.json")
    if not os.path.isfile(path):
        # fall back to the canonical copy in the repo
        path = os.path.join(_REPO_ROOT, "ml", "splits.json")
    with open(path) as f:
        return json.load(f)


def resolve_scenes(args, splits):
    """Return (split_name, [scene_ids], [ (sid, shard_path, row) ])."""
    if args.overfit:
        split_name = "train"  # G2 runs only on train seeds (never test/val)
        ids = _parse_seed_spec(args.overfit_seeds)
        train_set = set(splits["train"])
        missing = [s for s in ids if s not in train_set]
        if missing:
            raise ValueError("--overfit seeds not in train split: %r"
                             % (missing,))
        scene_ids = ids
    else:
        split_name = args.split
        pool = splits[split_name]
        if args.scenes is None:
            scene_ids = list(pool)
        else:
            if args.scenes < 1:
                raise ValueError("--scenes must be >= 1, got %d" % args.scenes)
            scene_ids = list(pool[:args.scenes])
    if len(scene_ids) < 1:
        raise ValueError("no scenes selected for split %r" % split_name)

    shard_map = build_shard_map(args.root)
    missing = [s for s in scene_ids if s not in shard_map]
    if missing:
        raise ValueError("scenes missing from packed shards under %r: %r"
                         % (args.root, missing[:10]))
    scenes = [(sid, shard_map[sid][0], shard_map[sid][1]) for sid in scene_ids]
    return split_name, scene_ids, scenes


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def _save_checkpoint(path, model, optimizer, sampler, global_step, epoch, cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": int(global_step),
        "epoch": int(epoch),
        "sampler_rng_state": sampler.rng_state(),
        "torch_rng": torch.random.get_rng_state(),
        "np_rng": np.random.get_state(),
        "py_rng": random.getstate(),
        "cfg": cfg,
    }, tmp)
    os.replace(tmp, path)


def _load_checkpoint(path):
    if not os.path.isfile(path):
        raise FileNotFoundError("resume checkpoint not found: %s" % path)
    # weights_only=False: the checkpoint carries numpy/random RNG state tuples
    # (np_rng / sampler_rng_state) that weights_only=True rejects. The file is
    # self-produced by this module (trusted), not an untrusted artifact.
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# I/O benchmark (dataloader-only; no model)
# ---------------------------------------------------------------------------


def bench_io(args):
    """Measure the real I/O-only cost per step (decode + IPC), no model."""
    splits = load_splits(args.root)
    split_name, scene_ids, scenes = resolve_scenes(args, splits)
    epoch_len = len(scene_ids)
    total = max(int(args.bench_io), 1)
    dataset = SceneDataset(scenes)
    sampler = ViewSubsetSampler(scene_ids, args.min_views, args.max_views,
                                args.seed, total + 1)  # +1 warmup
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, sampler=sampler,
        num_workers=args.workers,
        prefetch_factor=args.prefetch if args.workers > 0 else None,
        persistent_workers=args.workers > 0,
        collate_fn=_identity_collate)

    it = iter(loader)
    _ = next(it)  # warmup: worker spawn + shard load + PIL import
    times = []
    views_n = []
    for _ in range(total):
        t0 = time.perf_counter()
        b = next(it)
        times.append(time.perf_counter() - t0)
        views_n.append(len(b["view_indices"]))
    it._reset(loader)

    times = np.asarray(times)
    print("IO BENCHMARK split=%s scenes=%d steps=%d workers=%d prefetch=%d"
          % (split_name, len(scene_ids), total, args.workers, args.prefetch))
    print("  per-step I/O (decode+IPC): mean %.3fs  median %.3fs  "
          "min %.3fs  max %.3fs  total %.3fs"
          % (times.mean(), np.median(times), times.min(), times.max(),
             times.sum()))
    print("  sampled views/step: mean %.1f  range [%d, %d]"
          % (np.mean(views_n), min(views_n), max(views_n)))
    print("  decode throughput ~ %.0f views/s"
          % (float(np.sum(views_n)) / max(times.sum(), 1e-9)))
    return 0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class _GracefulStop(Exception):
    pass


def _make_train_step(model, optimizer, batch, device, count_weight, pos_weight):
    views = batch["views"].to(device)
    target = batch["target"].to(device)
    peak_mask = batch["peak_mask"].to(device)
    optimizer.zero_grad(set_to_none=True)
    pred = model.forward_volume(views, batch["cameras"], batch["grid"])[0, 0]
    # Sparse-target dilution fix (see module "LOSS" docstring): upweight the
    # drone-support voxels so a flat near-zero output is no longer a local
    # optimum. The weight is PROPORTIONAL to the target value (w = 1 +
    # pos_weight * target): the peak core (target ~ 1) gets ~1+pos_weight, the
    # Gaussian shoulder a smaller upweight, and voxels where the float64
    # Gaussian tail has merely not underflowed to zero (target ~ 1e-300; ~63%
    # of the 64^3 grid — NOT the sparse support) get weight ~ 1 and no spurious
    # downward force on the region surrounding the peaks.
    w = 1.0 + pos_weight * target
    mse = ((pred - target).square() * w).mean()
    in_mass = (pred * peak_mask).sum() / batch["vpp"]
    bg_excess = torch.relu(pred - COUNT_FLOOR) * (1.0 - peak_mask)
    bg_drones = bg_excess.sum() / batch["vpp"]
    count = ((in_mass - batch["n_drones"]).square()
             + COUNT_BG_WEIGHT * bg_drones.square())
    loss = mse + count_weight * count
    loss.backward()
    optimizer.step()
    return loss, mse, count, in_mass, bg_drones


# ---------------------------------------------------------------------------
# G2 evaluation (frozen metric path on the final checkpoint)
# ---------------------------------------------------------------------------


def evaluate_overfit(model, scenes, device, view_idxs, tag):
    """Score one trained model on a fixed view subset via the frozen metric path.

    For each scene: decode the selected views (exact shard bytes, same decode
    as training), run `forward_volume` in eval mode, decode peak positions with
    `ml.model.extract_positions`, and score them with the frozen
    `ml.metrics.evaluate`. The metric dict is the single source of truth for
    the G2 verdict (median position error + count error).

    Returns a list of per-scene dicts:
        {"tag", "scene_id", "n_true", "n_pred", "count_err",
         "median_err_m", "pred_max"}
    """
    model.eval()
    rows = []
    for (sid, shard_path, row) in scenes:
        d = _load_shard(shard_path)
        png = d["png_bytes"][row]
        views = np.stack([_decode_png_bytes(png[j]) for j in view_idxs])
        cameras = [d["cameras"][row][j] for j in view_idxs]
        grid = {
            "center": np.asarray(d["swarm_center"][row], dtype=np.float32),
            "radius_m": float(d["radius_m"][row]),
        }
        positions = np.asarray(d["positions"][row], dtype=np.float64)
        with torch.no_grad():
            vol = model.forward_volume(
                torch.from_numpy(views), cameras, grid)[0, 0].cpu().numpy()
        pred = extract_positions(vol, grid)
        met = metrics.evaluate(pred, positions)
        rows.append({
            "tag": tag,
            "scene_id": int(sid),
            "n_true": int(met["n_true"]),
            "n_pred": int(met["n_pred"]),
            "count_err": int(met["count_err"]),
            "median_err_m": float(met["median_err_m"]),
            "pred_max": float(vol.max()),
        })
    return rows


def _print_eval_table(rows):
    """Print a per-scene table and return (median_err_m, count_err_all_in_range)."""
    print("  %-7s %-6s %-6s %-9s %-13s %-8s"
          % ("scene", "n_true", "n_pred", "count_err", "median_err_m",
             "pred_max"))
    for r in rows:
        print("  %-7d %-6d %-6d %-9d %-13.4f %-8.4f"
              % (r["scene_id"], r["n_true"], r["n_pred"], r["count_err"],
                 r["median_err_m"], r["pred_max"]))
    med = float(np.median([r["median_err_m"] for r in rows]))
    mean = float(np.mean([r["median_err_m"] for r in rows]))
    count_ok = all(-1 <= r["count_err"] <= 1 for r in rows)
    print("  aggregate: median_err_m=%.4f  mean=%.4f  "
          "count_err all in [-1,+1]: %s" % (med, mean, count_ok))
    return med, count_ok


def emit_g2_verdict(model, scenes, device):
    """Run the G2 gate: frozen-metric evaluation at the fixed mixed 8-view
    subset, plus the V=24 diagnostic. Returns exit code (0 = PASS, 1 = FAIL).

    PASS iff aggregate median_err_m < 1.0 m AND every scene's count_err in
    [-1, +1]. A flat volume yields no peaks, so count_err = -N per scene — the
    gate cannot be gamed by the diffuse output that passed the old loss gate.
    """
    print("G2 EVAL mixed8 (%s):" % EVAL_VIEWS_DOC)
    rows8 = evaluate_overfit(model, scenes, device, EVAL_VIEWS, "mixed8")
    agg_med8, count_ok8 = _print_eval_table(rows8)
    print("G2 EVAL V24 diagnostic (all 24 views):")
    rows24 = evaluate_overfit(model, scenes, device, tuple(range(N_VIEWS)),
                              "V24")
    agg_med24, _ = _print_eval_table(rows24)

    passed = agg_med8 < 1.0 and count_ok8
    if passed:
        print("G2 OVERFIT: PASS (aggregate median_err_m=%.4f m < 1.0 m; "
              "every scene count_err in [-1,+1]; V24 diag median_err_m=%.4f)"
              % (agg_med8, agg_med24), flush=True)
        return 0
    print("G2 OVERFIT: FAIL (aggregate median_err_m=%.4f m; "
          "count_err all in [-1,+1]: %s; V24 diag median_err_m=%.4f m)"
          % (agg_med8, count_ok8, agg_med24), flush=True)
    return 1


def run_training(args):
    splits = load_splits(args.root)
    split_name, scene_ids, scenes = resolve_scenes(args, splits)
    epoch_len = len(scene_ids)

    if args.max_steps is not None:
        total_steps = int(args.max_steps)
    elif args.epochs is not None:
        total_steps = int(args.epochs) * epoch_len
    elif args.overfit:
        total_steps = OVERFIT_DEFAULT_STEPS
    else:
        total_steps = epoch_len
    if total_steps < 1:
        raise ValueError("training budget is 0 steps")

    device = torch.device(args.device)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available")

    # --- deterministic init -------------------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    model = VoxelFusionModel(feat_channels=64).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_step = 0
    sampler_rng_state = None
    ckpt_path = os.path.join(args.checkpoint_dir, "latest.pt") \
        if args.checkpoint_dir else None

    if args.resume:
        if not ckpt_path:
            raise ValueError("--resume requires --checkpoint-dir")
        ckpt = _load_checkpoint(ckpt_path)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt["global_step"])
        sampler_rng_state = ckpt["sampler_rng_state"]
        torch.random.set_rng_state(ckpt["torch_rng"])
        np.random.set_state(ckpt["np_rng"])
        random.setstate(ckpt["py_rng"])
        print("resumed from %s at global_step %d (epoch %.2f)"
              % (ckpt_path, start_step, start_step / epoch_len))
    if start_step >= total_steps:
        print("already at global_step %d >= total_steps %d; nothing to do"
              % (start_step, total_steps))
        return 0

    dataset = SceneDataset(scenes)
    sampler = ViewSubsetSampler(scene_ids, args.min_views, args.max_views,
                                args.seed, total_steps,
                                rng_state=sampler_rng_state)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, sampler=sampler,
        num_workers=args.workers,
        prefetch_factor=args.prefetch if args.workers > 0 else None,
        persistent_workers=args.workers > 0,
        collate_fn=_identity_collate)

    cfg = {
        "root": os.path.abspath(args.root),
        "split": split_name,
        "scene_ids": scene_ids,
        "min_views": args.min_views,
        "max_views": args.max_views,
        "seed": args.seed,
        "lr": args.lr,
        "count_weight": args.count_weight,
        "pos_weight": args.pos_weight,
        "target_sigma_cells": TARGET_SIGMA_CELLS,
        "total_steps": total_steps,
        "epoch_len": epoch_len,
    }

    losses = []
    io_waits = []
    step_times = []
    model_times = []
    t_run_start = time.perf_counter()

    def _sigterm(signum, frame):
        raise _GracefulStop()

    old_sigterm = signal.signal(signal.SIGTERM, _sigterm)
    loader_iter = iter(loader)
    completed = False

    try:
        for global_step in range(start_step, total_steps):
            epoch = global_step / epoch_len
            t0 = time.perf_counter()
            batch = next(loader_iter)
            t1 = time.perf_counter()
            loss, mse, count, in_mass, bg_drones = _make_train_step(
                model, optimizer, batch, device, args.count_weight,
                args.pos_weight)
            if args.device == "mps":
                torch.mps.synchronize()
            t2 = time.perf_counter()

            io_wait = t1 - t0
            step_time = t2 - t0
            model_time = t2 - t1
            loss_v = float(loss.detach().cpu())
            losses.append(loss_v)
            io_waits.append(io_wait)
            step_times.append(step_time)
            model_times.append(model_time)

            v = len(batch["view_indices"])
            mix = batch["tier_mix"]
            if args.overfit or global_step % args.log_every == 0:
                print("step %d/%d ep %.2f loss %.6f mse %.6f count %.6f "
                      "inmass %.2f/%d bg %.2f V %d tiers %d/%d/%d step %.3fs "
                      "io %.3fs"
                      % (global_step + 1, total_steps, epoch, loss_v,
                         float(mse.detach().cpu()),
                         float(count.detach().cpu()),
                         float(in_mass.detach().cpu()),
                         batch["n_drones"],
                         float(bg_drones.detach().cpu()), v,
                         mix.get("ground", 0), mix.get("level", 0),
                         mix.get("aerial", 0), step_time, io_wait),
                     flush=True)

            if ckpt_path and (global_step + 1) % args.ckpt_every == 0:
                _save_checkpoint(ckpt_path, model, optimizer, sampler,
                                 global_step + 1, epoch, cfg)
                print("checkpoint saved at step %d -> %s"
                      % (global_step + 1, ckpt_path), flush=True)

        completed = True
    except _GracefulStop:
        print("received SIGTERM; checkpointing and exiting", flush=True)
    finally:
        if old_sigterm is not None:
            signal.signal(signal.SIGTERM, old_sigterm)
        if ckpt_path and losses:
            _save_checkpoint(ckpt_path, model, optimizer, sampler,
                             start_step + len(losses),
                             (start_step + len(losses)) / epoch_len, cfg)
            print("final checkpoint saved -> %s" % ckpt_path, flush=True)

    wall = time.perf_counter() - t_run_start
    recent = float(np.mean(losses[-args.overfit_window:])) if losses else float("nan")
    print("TRAINING DONE steps=%d wall=%.1fs step-mean=%.3fs model-mean=%.3fs "
          "io-mean=%.3fs io-total=%.1fs final-loss=%.6f"
          % (len(losses), wall,
             float(np.mean(step_times)) if step_times else 0.0,
             float(np.mean(model_times)) if model_times else 0.0,
             float(np.mean(io_waits)) if io_waits else 0.0,
             float(np.sum(io_waits)), losses[-1] if losses else float("nan")),
          flush=True)

    if args.overfit:
        if completed:
            # G2 verdict from the frozen metric path on the final checkpoint.
            return emit_g2_verdict(model, scenes, device)
        print("G2 OVERFIT: FAIL (training interrupted before the fixed "
              "step budget; no metric verdict emitted)", flush=True)
        return 1
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser():
    p = argparse.ArgumentParser(
        description="T6 training loop for the ML VoxelFusionModel "
                    "(smoke / G2 overfit / detached full training).")
    p.add_argument("--root", default=os.path.expanduser("~/swarm_ml_packed"),
                   help="packed shard root (shard_XXXX.npz + splits.json)")
    p.add_argument("--split", default="train", choices=("train", "val", "test"))
    p.add_argument("--scenes", type=int, default=None,
                   help="use the first N seeds of the split (default: all)")
    p.add_argument("--epochs", type=int, default=None,
                   help="training budget in epochs (one epoch == len(scenes) steps)")
    p.add_argument("--max-steps", type=int, default=None,
                   help="training budget in steps (takes precedence over --epochs)")
    p.add_argument("--checkpoint-dir", default=None,
                   help="dir for latest.pt checkpoints (default: none for "
                        "smoke/bench; checkpoints/ for training)")
    p.add_argument("--resume", action="store_true",
                   help="resume from --checkpoint-dir/latest.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="mps", choices=("mps", "cpu"))
    p.add_argument("--min-views", type=int, default=2)
    p.add_argument("--max-views", type=int, default=8,
                   help="training view-subset cap (default 8; eval sweep only "
                        "uses V in 2..8, so training above 8 wastes compute on "
                        "configurations never evaluated)")
    p.add_argument("--lr", type=float, default=None,
                   help="Adam learning rate (default: 1e-3; 1e-2 for --overfit)")
    p.add_argument("--pos-weight", type=float, default=POS_WEIGHT_DEFAULT,
                   help="positive-voxel upweight for the sparse-target MSE "
                        "(target > 0 voxels get this weight, background keeps "
                        "1.0; see 'LOSS' docstring)")
    p.add_argument("--count-weight", type=float, default=0.1,
                   help="weight of the count term (in-support mass + background "
                        "excess, each normalised to drones). ACTIVE in --overfit "
                        "mode too: it is the peak-mass counter-pressure that "
                        "forces real mass on the drone support")
    p.add_argument("--workers", type=int, default=2,
                   help="dataloader worker processes (>= 2 required)")
    p.add_argument("--prefetch", type=int, default=2,
                   help="dataloader prefetch_factor (> 1 required)")
    p.add_argument("--ckpt-every", type=int, default=250,
                   help="save a checkpoint every N steps")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--overfit", action="store_true",
                   help="G2 gate: fixed train-scene subset, fixed step budget, "
                        "then the frozen-metric verdict on the final "
                        "checkpoint (median_err_m < 1.0 m AND per-scene "
                        "count_err in [-1,+1])")
    p.add_argument("--overfit-seeds", default="2000-2007",
                   help="seed spec for the G2 subset (train split only)")
    p.add_argument("--overfit-threshold", type=float,
                   default=0.05,
                   help="INFORMATIONAL ONLY: the old loss-threshold gate was "
                        "falsified (a flat output passes mean(target^2)~0.002 "
                        "trivially); the G2 verdict is metric-based now")
    p.add_argument("--overfit-window", type=int, default=25,
                   help="recent-mean window for the loss log (not the verdict)")
    p.add_argument("--overfit-min-steps", type=int, default=50)
    p.add_argument("--smoke", action="store_true",
                   help="2 steps on 2 scenes, exit 0 (CI path)")
    p.add_argument("--bench-io", type=int, default=0, metavar="N",
                   help="dataloader-only I/O benchmark over N steps (no model)")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    if args.lr is None:
        # Overfit gates converge faster at a higher LR; full training stays
        # at the conservative 1e-3 default.
        args.lr = 1e-2 if args.overfit else 1e-3
    # The count term stays ACTIVE in --overfit mode: disabling it was half of
    # the G2 false-pass (a flat near-zero output then had no counter-pressure).
    if args.smoke:
        args.overfit = False
        args.scenes = 2
        args.max_steps = 2
        args.checkpoint_dir = None
        args.log_every = 1
        print("SMOKE: split=%s scenes=%d steps=%d workers=%d" %
              (args.split, args.scenes, args.max_steps, args.workers))
        return run_training(args)
    if args.bench_io:
        return bench_io(args)
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
    return run_training(args)


if __name__ == "__main__":
    sys.exit(main())
