"""T1 — ML scene + camera generator.

Owner: scene agent. Owned file: `ml/scene_gen.py`. Conventions imported from
the frozen geometric track (`data_contract`, `detect_blobs`) only; geometry is
self-contained.

Per the patched spec (docs/superpowers/specs/2026-07-31-ml-swarm-reconstruction-design.md,
Section 8 T1, PATCH 3/4/7):

- Random swarm within the chosen radius, N in [5, 60], plain background.
- 24 cameras, 8 per tier (ground / level / aerial), controlled azimuth spread.
- Both operating cells: primary R=50 m (a_max 9.6 px, standoff 139 m) and
  secondary R=100 m (a_max 4.8 px, standoff 278 m), W=1920, f=2666.67 px.
  Standoff = R * 2f/W = 2.778 * R (framing constraint). Scenes are tagged with
  their cell.
- Minimum inter-drone spacing enforced **at generation time** (rejection
  sampling that never crashes a run).
- Reserved seed ranges (PATCH 7): 0-999 TEST, 1000-1999 VAL, 2000+ TRAIN.
  Scenes are seed-indexed and deterministic: same seed -> same scene, forever.

Swarm geometry: drones uniformly in an oblate ellipsoid (horizontal radius R,
vertical half-axis 0.25 R, aspect matching the frozen geometric track's swarm),
centred at a per-scene altitude. At the framing standoff the swarm's 2R diameter
fills the horizontal FOV; the flattened vertical profile keeps most drones inside
the (narrower 16:9) vertical FOV for level/ground/aerial views. Coverage is
reported, not gated.

Usage (repo root):
    python -m ml.scene_gen --root DIR --seed 2000 --cell primary
    python -m ml.scene_gen --selfcheck   # 3-scene acceptance-style check
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "stage1_geometry"))

from data_contract import (  # noqa: E402  frozen conventions
    IMAGE_SIZE,
    make_K,
    opencv_w2c_to_blender_c2w,
)

# ---------------------------------------------------------------------------
# Frozen operating constants (calib.json, data_contract, PATCH 4)
# ---------------------------------------------------------------------------

DRONE_SIZE_M = 0.5
IMAGE_W, IMAGE_H = IMAGE_SIZE  # (1920, 1080)
FOCAL_PX = 50.0 * IMAGE_W / 36.0  # 2666.67 px (repo default, ~40 deg HFOV)
DEFAULT_MIN_SPACING_M = 3.0
VERTICAL_AXIS_FRACTION = 0.25  # oblate ellipsoid: vertical half-axis = 0.25 * R
CENTROID_ALTITUDE_RANGE_M = (300.0, 500.0)  # keep every camera above z=0

# Camera tiers (spec Section 5). Elevation of the camera POSITION relative to
# the swarm centroid: ground looks up (< -20), level (-20..+20), aerial looks
# down (> +20). Sampled ranges are interior to the tier bands so a recomputed
# elevation can never straddle a boundary.
TIERS = ("ground", "level", "aerial")
TIER_ELEV_RANGES_DEG = {
    "ground": (-35.0, -20.0),
    "level": (-15.0, 15.0),
    "aerial": (20.0, 35.0),
}
TIER_BOUNDS_DEG = {"ground": (-90.0, -20.0), "level": (-20.0, 20.0), "aerial": (20.0, 90.0)}

PER_TIER_VIEWS = 8  # PATCH 3
N_VIEWS = PER_TIER_VIEWS * len(TIERS)  # 24

# PATCH 7 reserved seed ranges.
SEED_TEST = (0, 999)
SEED_VAL = (1000, 1999)
SEED_TRAIN = (2000, None)  # extendable indefinitely


def _cell_standoff(radius_m: float) -> float:
    """Framing standoff = R * 2f/W = 2.778 * R (PATCH 4)."""
    return 2.0 * radius_m * FOCAL_PX / IMAGE_W


def _cell_a_max(radius_m: float) -> float:
    """a_max = d*W/(2R) — apparent size of the centroid drone at the framing standoff."""
    return DRONE_SIZE_M * IMAGE_W / (2.0 * radius_m)


OPERATING_CELLS = {
    "primary": {
        "radius_m": 50.0,
        "standoff_m": _cell_standoff(50.0),
        "a_max_px": _cell_a_max(50.0),
    },
    "secondary": {
        "radius_m": 100.0,
        "standoff_m": _cell_standoff(100.0),
        "a_max_px": _cell_a_max(100.0),
    },
}


def split_for_seed(seed: int) -> str:
    """PATCH 7: reserved seed ranges -> split."""
    if 0 <= seed <= SEED_TEST[1]:
        return "test"
    if SEED_VAL[0] <= seed <= SEED_VAL[1]:
        return "val"
    if SEED_TRAIN[0] <= seed:
        return "train"
    raise ValueError("seed must be non-negative, got %d" % seed)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _look_at_w2c(eye, target):
    """OpenCV-convention world-to-camera (R, t) looking from eye to target."""
    forward = np.asarray(target, dtype=float) - np.asarray(eye, dtype=float)
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    R = np.vstack([right, -up, forward]).astype(np.float64)
    t = (-R @ np.asarray(eye, dtype=float)).astype(np.float64)
    return R, t


def _sample_positions(rng, center, radius_m, vert_axis_m, n, min_spacing_m):
    """Uniform-in-ellipsoid positions with generation-time min-spacing rejection.

    Rejection never crashes a run: if spacing cannot be satisfied within a large
    attempt budget (impossible at this density), the candidate is accepted anyway
    and the shortfall is recorded in the returned achieved spacing. This is the
    "enforce at generation time, never assert afterwards" contract — an unlucky
    seed slows the sampler, it does not kill a multi-day run.
    """
    positions = np.empty((0, 3), dtype=np.float64)

    def draw():
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        r = rng.random() ** (1.0 / 3.0)  # uniform volume in the unit ball
        return center + np.array([
            radius_m * r * v[0], radius_m * r * v[1], vert_axis_m * r * v[2],
        ])

    budget = max(2000 * n + 100, 20000)
    for _ in range(n):
        candidate = None
        for _attempt in range(budget):
            c = draw()
            if positions.shape[0] == 0 or min_spacing_m <= 0:
                candidate = c
                break
            if np.all(np.linalg.norm(positions - c, axis=1) >= min_spacing_m):
                candidate = c
                break
        if candidate is None:  # budget exhausted — accept rather than crash
            candidate = draw()
        positions = np.vstack([positions, candidate])

    achieved = np.inf
    if positions.shape[0] > 1:
        d = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        achieved = float(d.min())
    return positions, achieved


def _build_cameras(rng, center, standoff_m):
    """24 cameras: 8 per tier, controlled azimuth spread (even + jitter)."""
    views = []
    idx = 0
    for tier in TIERS:
        lo, hi = TIER_ELEV_RANGES_DEG[tier]
        base_az = float(rng.uniform(0.0, 2.0 * np.pi))
        for k in range(PER_TIER_VIEWS):
            az = base_az + 2.0 * np.pi * k / PER_TIER_VIEWS + float(rng.uniform(-0.12, 0.12))
            elev_rad = math.radians(float(rng.uniform(lo, hi)))
            dirv = np.array([
                math.cos(elev_rad) * math.cos(az),
                math.cos(elev_rad) * math.sin(az),
                math.sin(elev_rad),
            ])
            eye = center + standoff_m * dirv
            R, t = _look_at_w2c(eye, center)
            c2w = opencv_w2c_to_blender_c2w(R, t)
            K = make_K(FOCAL_PX, (IMAGE_W, IMAGE_H))
            views.append({
                "angle_idx": idx,
                "tier": tier,
                "elevation_deg": round(math.degrees(elev_rad), 4),
                "azimuth_deg": round(math.degrees(az), 4),
                "K": np.round(K, 6).tolist(),
                "c2w": np.round(c2w, 9).tolist(),
                "w2c_R": np.round(R, 9).tolist(),
                "w2c_t": np.round(t, 6).tolist(),
            })
            idx += 1
    return views


# ---------------------------------------------------------------------------
# Scene generation
# ---------------------------------------------------------------------------


def generate_scene(seed: int, cell="primary", n_drones=None, min_spacing_m=DEFAULT_MIN_SPACING_M,
                   centroid_altitude_m=None):
    """Deterministically generate one scene for the given seed and operating cell.

    Returns the scene dict (never writes to disk). Same seed + cell -> identical
    scene, forever. `n_drones` overrides the random N in [5, 60] (used for the
    pilot's fixed-density measurement; the campaign uses random N).
    """
    if seed < 0:
        raise ValueError("seed must be non-negative, got %d" % seed)
    if isinstance(cell, str):
        if cell not in OPERATING_CELLS:
            raise ValueError("unknown cell %r (choose %s)"
                             % (cell, sorted(OPERATING_CELLS)))
        cell_name = cell
        cell_cfg = OPERATING_CELLS[cell]
    else:  # dict
        cell_name = cell.get("name", "custom")
        cell_cfg = cell

    radius_m = float(cell_cfg["radius_m"])
    standoff_m = float(cell_cfg["standoff_m"])

    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 61)) if n_drones is None else int(n_drones)
    if not (5 <= n <= 60):
        raise ValueError("n_drones must be in [5, 60], got %d" % n)
    if centroid_altitude_m is None:
        centroid_altitude_m = float(rng.uniform(*CENTROID_ALTITUDE_RANGE_M))
    center = np.array([0.0, 0.0, float(centroid_altitude_m)])

    positions, spacing_achieved = _sample_positions(
        rng, center, radius_m, VERTICAL_AXIS_FRACTION * radius_m, n, min_spacing_m)

    cameras = _build_cameras(rng, center, standoff_m)

    scene = {
        "schema_version": 1,
        "scene_id": int(seed),
        "seed": int(seed),
        "split": split_for_seed(int(seed)),
        "cell": cell_name,
        "radius_m": radius_m,
        "standoff_m": round(standoff_m, 4),
        "a_max_px": round(_cell_a_max(radius_m), 4),
        "focal_px": FOCAL_PX,
        "image_size_px": [IMAGE_W, IMAGE_H],
        "n_views": N_VIEWS,
        "swarm_center": [0.0, 0.0, float(centroid_altitude_m)],
        "n_drones": n,
        "min_inter_drone_spacing_m": min_spacing_m,
        "min_inter_drone_spacing_achieved_m": round(spacing_achieved, 4),
        "positions": [list(map(float, p)) for p in positions],
        "cameras": cameras,
        "generated_by": "ml/scene_gen.py",
    }
    errors = scene_schema_validate(scene)
    if errors:
        raise RuntimeError("generated scene failed schema validation:\n- "
                           + "\n- ".join(errors))
    return scene


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def recompute_elevation_deg(c2w, centroid):
    """Elevation of the camera position relative to the centroid plane."""
    eye = np.asarray(c2w)[:3, 3]
    rel = eye - np.asarray(centroid, dtype=float)
    return float(np.degrees(np.arcsin(rel[2] / np.linalg.norm(rel))))


def tier_for_elevation(elev_deg):
    """Spec Section 5 tier bands, from a recomputed elevation (not a stored label)."""
    for tier, (lo, hi) in TIER_BOUNDS_DEG.items():
        if lo <= elev_deg < hi:
            return tier
    return None


def scene_schema_validate(scene):
    """Return a list of schema-error strings (empty == valid). Raises nothing."""
    errors = []
    required = ("schema_version", "scene_id", "seed", "split", "cell", "radius_m",
                "standoff_m", "a_max_px", "focal_px", "image_size_px",
                "swarm_center", "n_drones", "min_inter_drone_spacing_m",
                "min_inter_drone_spacing_achieved_m", "positions", "cameras",
                "generated_by")
    for key in required:
        if key not in scene:
            errors.append("missing key %r" % key)
            continue
    if errors:
        return errors

    if scene["scene_id"] != scene["seed"]:
        errors.append("scene_id != seed")
    if split_for_seed(scene["seed"]) != scene["split"]:
        errors.append("split %r inconsistent with PATCH 7 seed ranges" % scene["split"])
    n = scene["n_drones"]
    if not (5 <= n <= 60):
        errors.append("n_drones out of [5,60]: %d" % n)
    center = np.asarray(scene["swarm_center"], dtype=float)
    R = scene["radius_m"]
    positions = np.asarray(scene["positions"], dtype=float)
    if positions.shape != (n, 3):
        errors.append("positions shape %s != (%d, 3)" % (positions.shape, n))
    elif not np.all(np.isfinite(positions)):
        errors.append("non-finite position")
    else:
        horiz = np.linalg.norm(positions[:, :2] - center[:2], axis=1)
        vert = np.abs(positions[:, 2] - center[2])
        if horiz.max() > R * (1 + 1e-9):
            errors.append("max horizontal offset %.3f m exceeds radius %.3f m"
                          % (horiz.max(), R))
        if vert.max() > VERTICAL_AXIS_FRACTION * R * (1 + 1e-9):
            errors.append("max vertical offset %.3f m exceeds vertical half-axis"
                          % vert.max())
        achieved = scene["min_inter_drone_spacing_achieved_m"]
        if achieved < scene["min_inter_drone_spacing_m"] - 1e-9:
            errors.append("achieved min spacing %.3f < requested %.3f"
                          % (achieved, scene["min_inter_drone_spacing_m"]))

    cams = scene["cameras"]
    if len(cams) != N_VIEWS:
        errors.append("expected %d cameras, got %d" % (N_VIEWS, len(cams)))
    else:
        from collections import Counter
        tier_counts = Counter(c["tier"] for c in cams)
        for tier in TIERS:
            if tier_counts[tier] != PER_TIER_VIEWS:
                errors.append("tier %r has %d cameras, expected %d"
                              % (tier, tier_counts[tier], PER_TIER_VIEWS))
        idxs = [c["angle_idx"] for c in cams]
        if sorted(idxs) != list(range(N_VIEWS)):
            errors.append("angle_idx not a permutation of 0..%d" % (N_VIEWS - 1))
        for c in cams:
            K = np.asarray(c["K"], dtype=float)
            c2w = np.asarray(c["c2w"], dtype=float)
            Rw = np.asarray(c["w2c_R"], dtype=float)
            tw = np.asarray(c["w2c_t"], dtype=float)
            if K.shape != (3, 3) or c2w.shape != (4, 4) or Rw.shape != (3, 3) or tw.shape != (3,):
                errors.append("view %d: bad matrix shapes" % c["angle_idx"])
                continue
            if not np.allclose(Rw @ Rw.T, np.eye(3), atol=1e-6):
                errors.append("view %d: w2c_R not orthonormal" % c["angle_idx"])
            eye = -Rw.T @ tw
            if not np.allclose(c2w[:3, 3], eye, atol=1e-6):
                errors.append("view %d: c2w translation != camera position" % c["angle_idx"])
            elev = recompute_elevation_deg(c2w, center)
            if tier_for_elevation(elev) != c["tier"]:
                errors.append("view %d: declared tier %r != recomputed tier %r (elev %.2f)"
                              % (c["angle_idx"], c["tier"], tier_for_elevation(elev), elev))
    return errors


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def scene_dir(root, seed):
    """Nested scene directory (PATCH 2 layout): scenes/00/00001/."""
    seed = int(seed)
    return os.path.join(root, "scenes", "%02d" % (seed // 100), "%05d" % seed)


def _write_json_atomic(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_scene(root, scene):
    """Write ground_truth.json + cameras.json for a scene dict. Returns the dir."""
    d = scene_dir(root, scene["scene_id"])
    os.makedirs(d, exist_ok=True)

    gt = {k: scene[k] for k in (
        "schema_version", "scene_id", "seed", "split", "cell", "radius_m",
        "swarm_center", "n_drones", "min_inter_drone_spacing_m",
        "min_inter_drone_spacing_achieved_m", "generated_by")}
    gt["drone_ids"] = list(range(scene["n_drones"]))
    gt["positions"] = scene["positions"]

    cam = {k: scene[k] for k in (
        "schema_version", "scene_id", "seed", "split", "cell", "radius_m",
        "standoff_m", "a_max_px", "focal_px", "image_size_px", "n_views",
        "swarm_center", "generated_by")}
    cam["views"] = scene["cameras"]

    _write_json_atomic(os.path.join(d, "ground_truth.json"), gt)
    _write_json_atomic(os.path.join(d, "cameras.json"), cam)
    return d


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _selfcheck(args):
    """Acceptance-style check: 3 scenes, both cells, write, reload, re-verify."""
    print("=== T1 scene_gen self-check ===")
    root = args.root or os.path.join(os.path.expanduser("~"), "swarm_ml_selfcheck")
    from collections import Counter
    ok = True
    for cell, seeds in (("primary", [2000, 2001]), ("secondary", [2002])):
        for seed in seeds:
            scene = generate_scene(seed=seed, cell=cell)
            d = write_scene(root, scene)
            gt = json.load(open(os.path.join(d, "ground_truth.json")))
            cam = json.load(open(os.path.join(d, "cameras.json")))
            # reload-from-disk validation (what the render harness consumes)
            reloaded = dict(scene)
            reloaded["positions"] = gt["positions"]
            reloaded["cameras"] = cam["views"]
            errs = scene_schema_validate(reloaded)
            n_views = len(cam["views"])
            tiers = Counter(v["tier"] for v in cam["views"])
            center = np.asarray(scene["swarm_center"], dtype=float)
            tier_ok = all(
                tier_for_elevation(recompute_elevation_deg(v["c2w"], center)) == v["tier"]
                for v in cam["views"]
            )
            print("seed %d cell %-9s N=%2d split=%-5s views=%d tiers=%s "
                  "standoff=%.1fm a_max=%.1fpx spacing=%.2fm tier_recompute=%s"
                  % (seed, cell, scene["n_drones"], scene["split"], n_views,
                     dict(tiers), scene["standoff_m"], scene["a_max_px"],
                     scene["min_inter_drone_spacing_achieved_m"],
                     "OK" if tier_ok else "MISMATCH"))
            if errs:
                ok = False
                print("  SCHEMA ERRORS:")
                for e in errs:
                    print("   -", e)
            if not tier_ok:
                ok = False
    print("SELF-CHECK: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="T1 ML scene + camera generator")
    parser.add_argument("--root", default=os.path.join(os.path.expanduser("~"), "swarm_ml"),
                        help="data root (PATCH 2 layout: root/scenes/..)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cell", default="primary", choices=sorted(OPERATING_CELLS))
    parser.add_argument("--n-drones", type=int, default=None, help="override N (pilot only)")
    parser.add_argument("--selfcheck", action="store_true", help="3-scene acceptance check")
    parser.add_argument("--json", action="store_true", help="print scene JSON to stdout")
    args = parser.parse_args(argv)

    if args.selfcheck:
        return _selfcheck(args)
    if args.seed is None:
        parser.error("--seed required unless --selfcheck")
    scene = generate_scene(seed=args.seed, cell=args.cell, n_drones=args.n_drones)
    if args.json:
        print(json.dumps(scene, indent=2))
        return 0
    d = write_scene(args.root, scene)
    print("wrote %s" % d)
    print("scene_id=%d cell=%s split=%s n_drones=%d standoff=%.2fm a_max=%.2fpx"
          % (scene["scene_id"], scene["cell"], scene["split"], scene["n_drones"],
             scene["standoff_m"], scene["a_max_px"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
