#!/usr/bin/env python3
"""T4 — dataset packing + splits (contract §2).

Packs the scenes referenced by a render-harness manifest (`manifest.jsonl`)
into npz shards, split by scene seed.

CRITICAL (spec §7 / contract §2.5): shards store PNG BYTES, never decoded
arrays. Raw uint8 at 1080p is ~6.2 MB/image and packs 5,000 scenes to
~560 GB; PNG bytes pack to ~33 KB/image. The dataloader decodes on load
(contract §2.6).

CLI (contract §2.2):
    python -m ml.pack_dataset --root <data-root> --out <packed-root>
                              [--splits train val test] [--shard-size N]

The set of scenes to pack is taken from `manifest.jsonl` (seed, split, cell).
The packer never guesses or scans ad hoc. Shards are ordered by ascending
scene seed; each holds up to `--shard-size` scenes; a shard may mix splits.

Writes:
    ml/splits.json                    canonical split manifest (frozen, §2.3)
    <packed-root>/splits.json         byte-identical copy
    <packed-root>/shard_XXXX.npz      uncompressed np.savez, schema §2.4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

from ml import scene_gen

N_VIEWS = scene_gen.N_VIEWS                # 24 (frozen, contract §2)
DEFAULT_SHARD_SIZE = 32
VALID_SPLITS = ("train", "val", "test")
SPLITS_SCHEMA_VERSION = 1
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# paths / manifest
# ---------------------------------------------------------------------------


def canonical_splits_path():
    """Canonical ml/splits.json — frozen, consumed by test_split_disjointness."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits.json")


def read_manifest(root):
    """manifest.jsonl -> list of records (render_harness _manifest_record schema).

    The manifest is the source of truth for what was rendered: one JSON record
    per done scene, with (seed, split, cell). The packer never scans or guesses.
    """
    path = os.path.join(root, "manifest.jsonl")
    if not os.path.isfile(path):
        raise FileNotFoundError("manifest not found: %s" % path)
    records = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "manifest line %d is not valid JSON (%s)" % (lineno, exc))
    return records


def _require_fields(rec, fields, where):
    missing = [f for f in fields if f not in rec]
    if missing:
        raise ValueError("%s record missing required field(s): %s"
                         % (where, ", ".join(missing)))


def select_scenes(root, manifest, want_splits):
    """Filter manifest records to the requested splits, ordered by ascending seed.

    Each returned dict: {"seed", "split", "cell", "scene_dir"}. Locates the
    scene dir by seed via the PATCH 2 layout (scenes/SS/NNNNN, SS = seed // 100).
    Verifies the manifest split agrees with scene_gen.split_for_seed(seed)
    (PATCH 7 ranges are structural) before anything is packed.
    """
    scenes = []
    for rec in manifest:
        _require_fields(rec, ("seed", "split", "cell"), "manifest")
        seed = int(rec["seed"])
        split = str(rec["split"])
        if split not in want_splits:
            continue
        expected = scene_gen.split_for_seed(seed)
        if split != expected:
            raise ValueError(
                "seed %d manifest split %r != scene_gen.split_for_seed() %r "
                "(PATCH 7 violated)" % (seed, split, expected))
        scenes.append({
            "seed": seed,
            "split": split,
            "cell": str(rec["cell"]),
            "scene_dir": scene_gen.scene_dir(root, seed),
        })
    scenes.sort(key=lambda s: s["seed"])
    return scenes


# ---------------------------------------------------------------------------
# per-scene shard rows
# ---------------------------------------------------------------------------


def load_scene(scene):
    """Read one scene's shard row from disk (contract §2.4).

    - positions[i]  = exact ground_truth.json["positions"] as float64 (n_i, 3)
    - cameras[i]    = exact cameras.json["views"] list (N_VIEWS dicts)
    - png_bytes[i,j]= raw file bytes of angle_%02d.png (bytes, never decoded)
    """
    d = scene["scene_dir"]
    if not os.path.isdir(d):
        raise FileNotFoundError("scene dir missing for seed %d: %s"
                                % (scene["seed"], d))
    gt_path = os.path.join(d, "ground_truth.json")
    cam_path = os.path.join(d, "cameras.json")
    if not os.path.isfile(gt_path):
        raise FileNotFoundError("ground_truth.json missing for seed %d: %s"
                                % (scene["seed"], gt_path))
    if not os.path.isfile(cam_path):
        raise FileNotFoundError("cameras.json missing for seed %d: %s"
                                % (scene["seed"], cam_path))

    with open(gt_path) as f:
        gt = json.load(f)
    with open(cam_path) as f:
        cam = json.load(f)

    positions = np.asarray(gt["positions"], dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("seed %d: positions must be (n_i, 3), got %r"
                         % (scene["seed"], positions.shape))
    n_drones = int(gt["n_drones"])
    if len(positions) != n_drones:
        raise ValueError("seed %d: n_drones=%d != len(positions)=%d"
                         % (scene["seed"], n_drones, len(positions)))

    swarm_center = np.asarray(gt["swarm_center"], dtype=np.float64)
    if swarm_center.shape != (3,):
        raise ValueError("seed %d: swarm_center must be (3,), got %r"
                         % (scene["seed"], swarm_center.shape))

    views = cam["views"]
    if len(views) != N_VIEWS:
        raise ValueError("seed %d: cameras has %d views, expected %d"
                         % (scene["seed"], len(views), N_VIEWS))

    png = [None] * N_VIEWS
    for j in range(N_VIEWS):
        png_path = os.path.join(d, "angle_%02d.png" % j)
        with open(png_path, "rb") as f:
            data = f.read()
        if not data.startswith(PNG_MAGIC):
            raise ValueError("seed %d angle_%02d: file is not a PNG"
                             % (scene["seed"], j))
        png[j] = data

    return {
        "scene_ids": np.int64(scene["seed"]),
        "splits": str(scene["split"]),
        "cells": str(scene["cell"]),
        "n_drones": np.int64(n_drones),
        "radius_m": np.float64(gt["radius_m"]),
        "swarm_center": swarm_center,
        "positions": positions,
        "png_bytes": png,
        "cameras": views,
    }


# ---------------------------------------------------------------------------
# shard writing
# ---------------------------------------------------------------------------


def _clear_stale_shards(out):
    """Remove prior shard_*.npz in the out dir so no stale shard lingers."""
    for name in os.listdir(out):
        if name.startswith("shard_") and name.endswith(".npz"):
            os.remove(os.path.join(out, name))


def _write_shard(chunk, out_path):
    """Write one npz shard (contract §2.4: exact keys/shapes/dtypes).

    Container is uncompressed np.savez (PNG bytes are already compressed;
    recompressing wastes CPU). Object arrays (splits/cells/positions/
    png_bytes/cameras) pickle inside the .npy entries and load with
    np.load(path, allow_pickle=True).
    """
    s = len(chunk)
    scene_ids = np.empty(s, dtype=np.int64)
    splits = np.empty(s, dtype=object)
    cells = np.empty(s, dtype=object)
    n_drones = np.empty(s, dtype=np.int64)
    radius_m = np.empty(s, dtype=np.float64)
    swarm_center = np.empty((s, 3), dtype=np.float64)
    positions = np.empty(s, dtype=object)
    png_bytes = np.empty((s, N_VIEWS), dtype=object)
    cameras = np.empty(s, dtype=object)

    for i, scene in enumerate(chunk):
        row = load_scene(scene)
        scene_ids[i] = row["scene_ids"]
        splits[i] = row["splits"]
        cells[i] = row["cells"]
        n_drones[i] = row["n_drones"]
        radius_m[i] = row["radius_m"]
        swarm_center[i] = row["swarm_center"]
        positions[i] = row["positions"]
        cameras[i] = row["cameras"]
        for j in range(N_VIEWS):
            png_bytes[i, j] = row["png_bytes"][j]

    np.savez(out_path,
             scene_ids=scene_ids,
             splits=splits,
             cells=cells,
             n_drones=n_drones,
             radius_m=radius_m,
             swarm_center=swarm_center,
             positions=positions,
             png_bytes=png_bytes,
             cameras=cameras)


# ---------------------------------------------------------------------------
# splits.json (canonical, frozen)
# ---------------------------------------------------------------------------


def build_splits(scenes):
    """Group packed seeds by split; verify G1 BEFORE anything is written."""
    splits = {"train": [], "val": [], "test": []}
    for sc in scenes:
        splits[sc["split"]].append(int(sc["seed"]))
    for key in splits:
        splits[key].sort()
    verify_splits(splits)
    return splits


def verify_splits(splits):
    """G1 + PATCH 7 range membership — raises ValueError on violation.

    Mirrors the frozen test_split_disjointness: zero seed overlap across
    train/val/test, test 0-999, val 1000-1999, train 2000+.
    """
    train = set(splits["train"])
    val = set(splits["val"])
    test = set(splits["test"])
    overlap = (train & val) | (train & test) | (val & test)
    if overlap:
        raise ValueError("G1 violated: seed(s) in more than one split: %r"
                         % sorted(overlap))
    bad_test = [s for s in test if not (0 <= s <= 999)]
    bad_val = [s for s in val if not (1000 <= s <= 1999)]
    bad_train = [s for s in train if s < 2000]
    if bad_test or bad_val or bad_train:
        raise ValueError(
            "PATCH 7 range violation — test=%r val=%r train=%r "
            "(test 0-999, val 1000-1999, train 2000+)"
            % (sorted(bad_test), sorted(bad_val), sorted(bad_train)))


def render_splits_bytes(splits):
    """Serialise the canonical splits.json payload (schema §2.3), once."""
    payload = {
        "schema_version": SPLITS_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
    }
    return json.dumps(payload, indent=2).encode("utf-8") + b"\n"


def _write_bytes_atomic(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# top-level pack + CLI
# ---------------------------------------------------------------------------


def pack(root, out, want_splits, shard_size):
    """Pack manifest scenes under root into shards under out.

    Returns (splits, n_scenes, n_shards). Writes the canonical ml/splits.json
    and a byte-identical copy at <out>/splits.json. G1 is verified before any
    splits.json is written.
    """
    manifest = read_manifest(root)
    scenes = select_scenes(root, manifest, want_splits)
    if not scenes:
        raise ValueError(
            "no manifest scenes match requested splits %r under %r "
            "(manifest has %d record(s))" % (sorted(want_splits), root, len(manifest)))

    os.makedirs(out, exist_ok=True)
    _clear_stale_shards(out)

    n_shards = (len(scenes) + shard_size - 1) // shard_size
    for si in range(n_shards):
        chunk = scenes[si * shard_size:(si + 1) * shard_size]
        out_path = os.path.join(out, "shard_%04d.npz" % si)
        _write_shard(chunk, out_path)
        print("shard_%04d.npz: %d scene(s), %d bytes on disk"
              % (si, len(chunk), os.path.getsize(out_path)))

    splits = build_splits(scenes)  # verifies G1 before writing
    data = render_splits_bytes(splits)
    canonical = canonical_splits_path()
    _write_bytes_atomic(canonical, data)
    _write_bytes_atomic(os.path.join(out, "splits.json"), data)
    return splits, len(scenes), n_shards


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="T4 dataset packing + splits (contract §2). "
                    "Packs manifest scenes into npz shards storing PNG BYTES.")
    parser.add_argument("--root", required=True,
                        help="data root (PATCH 2 layout: scenes/SS/NNNNN/"
                             "ground_truth.json + cameras.json + angle_*.png, "
                             "manifest.jsonl)")
    parser.add_argument("--out", required=True,
                        help="packed root (shard_XXXX.npz + splits.json copy)")
    parser.add_argument("--splits", nargs="*", choices=VALID_SPLITS,
                        default=list(VALID_SPLITS),
                        help="splits to pack (default: train val test)")
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE,
                        help="max scenes per shard (default %d)"
                             % DEFAULT_SHARD_SIZE)
    args = parser.parse_args(argv)
    if args.shard_size < 1:
        parser.error("--shard-size must be >= 1, got %d" % args.shard_size)

    want_splits = set(args.splits) if args.splits else set(VALID_SPLITS)
    splits, n_scenes, n_shards = pack(args.root, args.out, want_splits,
                                      args.shard_size)
    print("packed %d scene(s) into %d shard(s) at %s"
          % (n_scenes, n_shards, args.out))
    print("splits: train=%d val=%d test=%d (G1 verified)"
          % (len(splits["train"]), len(splits["val"]), len(splits["test"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
