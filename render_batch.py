"""
Batch render 30 temporal clips for M3/M4 validation.

Renders 30 clips × 20 frames × 12 views at 24mm/2km true scale.
Each clip: different seed, same camera config.

Usage:
    python render_batch.py [--n_clips 30] [--start 0]
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

BLENDER = str(Path.home() / "Library/Application Support/Steam/steamapps/common/Blender/Blender.app/Contents/MacOS/Blender")
RENDER_SCRIPT = str(Path(__file__).parent / "render_sequence.py")
DATASET_ROOT = "dataset_temporal"

CLIP_CONFIG = {
    "dataset_root": DATASET_ROOT,
    "n_frames": 20,
    "fps": 10,
    "n_views": 12,
    "focal_mm": 24,
    "sensor_width_mm": 36.0,
    "resolution": [1920, 1080],
    "standoff_m": 2000,
    "n_drones": 20,
    "samples": 32,
}


def main():
    n_clips = 30
    start = 0
    if "--n_clips" in sys.argv:
        n_clips = int(sys.argv[sys.argv.index("--n_clips") + 1])
    if "--start" in sys.argv:
        start = int(sys.argv[sys.argv.index("--start") + 1])

    print(f"Rendering {n_clips} clips (start={start})")
    print(f"Config: {CLIP_CONFIG['n_frames']} frames, {CLIP_CONFIG['n_views']} views, "
          f"{CLIP_CONFIG['n_drones']} drones, {CLIP_CONFIG['samples']} samples")
    print(f"Blender: {BLENDER}")
    print()

    total_renders = n_clips * CLIP_CONFIG["n_frames"] * CLIP_CONFIG["n_views"]
    print(f"Total renders: {total_renders}")
    print()

    t_start = time.time()
    results = []

    for i in range(start, start + n_clips):
        clip_name = f"seq_{i:03d}"
        seed = 1000 + i * 37  # deterministic, spread seeds

        config = {**CLIP_CONFIG, "clip_name": clip_name, "seed": seed}
        config_path = Path(DATASET_ROOT) / f"config_{clip_name}.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(config, f)

        print(f"[{clip_name}] Rendering... ", end="", flush=True)
        t0 = time.time()

        proc = subprocess.run(
            [BLENDER, "--background", "--python", RENDER_SCRIPT, "--", str(config_path)],
            capture_output=True, text=True, timeout=600,
        )

        t_elapsed = time.time() - t0
        clip_idx = i - start + 1

        if proc.returncode == 0:
            # Extract timing from output
            timing = "?"
            for line in proc.stdout.split("\n"):
                if line.startswith("TIMING:"):
                    timing = line.split(":")[1].strip()
            print(f"done ({t_elapsed:.0f}s, timing={timing}s)")
            results.append({"clip": clip_name, "status": "ok", "time": t_elapsed})
        else:
            print(f"FAILED ({t_elapsed:.0f}s)")
            # Print last 10 lines of stderr for debugging
            stderr_lines = proc.stderr.strip().split("\n")
            for line in stderr_lines[-5:]:
                print(f"  {line}")
            results.append({"clip": clip_name, "status": "failed", "time": t_elapsed})

        # Progress
        done = clip_idx
        elapsed = time.time() - t_start
        rate = done / elapsed if elapsed > 0 else 0
        remaining = (n_clips - done) / rate if rate > 0 else 0
        print(f"  Progress: {done}/{n_clips} ({rate:.2f} clips/s, "
              f"~{remaining/60:.0f}min remaining)")

    t_total = time.time() - t_start
    ok = sum(1 for r in results if r["status"] == "ok")
    failed = sum(1 for r in results if r["status"] != "ok")

    print(f"\n{'='*60}")
    print(f"Batch complete: {ok}/{n_clips} clips rendered, {failed} failed")
    print(f"Total time: {t_total:.0f}s ({t_total/60:.1f}min)")
    print(f"Dataset: {DATASET_ROOT}/clips/")

    # Save manifest
    manifest = {
        "config": CLIP_CONFIG,
        "n_clips": n_clips,
        "results": results,
        "total_time_s": t_total,
    }
    manifest_path = Path(DATASET_ROOT) / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
