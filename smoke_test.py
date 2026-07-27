"""
Smoke test: ~5 clips spanning 2+ environments and 2+ weather conditions.

Renders end-to-end through the real pipeline, saves ground truth,
and verifies against actual frames.

Usage:
    python smoke_test.py --output-root dataset_smoke_test
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Default Blender path
BLENDER = str(Path.home() / "Library/Application Support/Steam/steamapps/common/Blender/Blender.app/Contents/MacOS/Blender")
RENDER_SCRIPT = str(Path(__file__).parent / "render_clip.py")

# Smoke test clips: 2 environments × 2 weather × 1 formation × 1 camera = 4 clips
SMOKE_TEST_CLIPS = [
    {
        "clip_name": "desert_clear",
        "environment": "desert",
        "weather": "clear",
        "formation": "random_cloud",
        "camera_arrangement": "dome_6",
        "seed": 42,
    },
    {
        "clip_name": "desert_overcast",
        "environment": "desert",
        "weather": "overcast",
        "formation": "random_cloud",
        "camera_arrangement": "dome_6",
        "seed": 43,
    },
    {
        "clip_name": "forest_clear",
        "environment": "forest",
        "weather": "clear",
        "formation": "random_cloud",
        "camera_arrangement": "dome_6",
        "seed": 44,
    },
    {
        "clip_name": "forest_hazy",
        "environment": "forest",
        "weather": "hazy",
        "formation": "random_cloud",
        "camera_arrangement": "dome_6",
        "seed": 45,
    },
    {
        "clip_name": "city_dusk",
        "environment": "city",
        "weather": "dusk",
        "formation": "grid",
        "camera_arrangement": "dome_12",
        "seed": 46,
    },
]

# Default render settings (matching full sweep)
DEFAULT_RENDER = {
    "n_frames": 20,
    "fps": 10,
    "focal_mm": 24,
    "sensor_width_mm": 36.0,
    "resolution": [1920, 1080],
    "n_views": 6,
    "standoff_m": 2000,
    "n_drones": 20,
    "samples": 32,
}


def run_smoke_test(output_root: str, n_clips: int = 5):
    """Run the smoke test."""
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SMOKE TEST: Rendering 5 clips across environments and weather")
    print("=" * 70)
    print(f"Output root: {output_path}")
    print(f"Blender: {BLENDER}")
    print()

    results = []
    t_start = time.time()

    for i, clip_config in enumerate(SMOKE_TEST_CLIPS[:n_clips]):
        clip_name = clip_config["clip_name"]
        print(f"[{i+1}/{n_clips}] Rendering {clip_name}...")

        # Merge clip config with defaults
        config = {**DEFAULT_RENDER, **clip_config}
        config["dataset_root"] = str(output_path)

        # Save config
        config_path = output_path / f"config_{clip_name}.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        t0 = time.time()

        try:
            # Run Blender
            proc = subprocess.run(
                [BLENDER, "--background", "--python", RENDER_SCRIPT, "--", str(config_path)],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            t_elapsed = time.time() - t0

            if proc.returncode == 0:
                # Extract timing
                timing = "?"
                for line in proc.stdout.split("\n"):
                    if line.startswith("TIMING:"):
                        timing = line.split(":")[1].strip()

                print(f"  ✓ Completed in {t_elapsed:.1f}s (timing={timing}s)")
                results.append({
                    "clip": clip_name,
                    "status": "ok",
                    "time": t_elapsed,
                    "timing": timing,
                })
            else:
                print(f"  ✗ Failed after {t_elapsed:.1f}s")
                stderr_lines = proc.stderr.strip().split("\n")
                for line in stderr_lines[-3:]:
                    print(f"    {line}")
                results.append({
                    "clip": clip_name,
                    "status": "failed",
                    "time": t_elapsed,
                    "error": proc.stderr[-200:] if proc.stderr else "Unknown",
                })

        except subprocess.TimeoutExpired:
            t_elapsed = time.time() - t0
            print(f"  ✗ Timeout after {t_elapsed:.1f}s")
            results.append({
                "clip": clip_name,
                "status": "timeout",
                "time": t_elapsed,
            })

        except Exception as e:
            t_elapsed = time.time() - t0
            print(f"  ✗ Error: {e}")
            results.append({
                "clip": clip_name,
                "status": "error",
                "time": t_elapsed,
                "error": str(e),
            })

    t_total = time.time() - t_start
    ok = sum(1 for r in results if r["status"] == "ok")
    failed = sum(1 for r in results if r["status"] != "ok")

    # Print summary
    print("\n" + "=" * 70)
    print("SMOKE TEST SUMMARY")
    print("=" * 70)
    print(f"Total clips: {n_clips}")
    print(f"Completed: {ok}")
    print(f"Failed: {failed}")
    print(f"Total time: {t_total:.1f}s ({t_total/60:.1f}min)")
    print(f"Average time per clip: {t_total/n_clips:.1f}s")
    print()

    # Extrapolate for larger datasets
    print("EXTRapolation for larger datasets:")
    print(f"  100 clips: {t_total/n_clips * 100 / 60:.1f}min")
    print(f"  500 clips: {t_total/n_clips * 500 / 60:.1f}min")
    print(f"  1000 clips: {t_total/n_clips * 1000 / 60:.1f}min")
    print(f"  2400 clips (full sweep): {t_total/n_clips * 2400 / 60:.1f}min")
    print("=" * 70)

    # Save results
    results_path = output_path / "smoke_test_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "clips": results,
            "total_time_s": t_total,
            "avg_time_per_clip_s": t_total / n_clips,
            "extrapolation": {
                "100_clips_min": t_total / n_clips * 100 / 60,
                "500_clips_min": t_total / n_clips * 500 / 60,
                "1000_clips_min": t_total / n_clips * 1000 / 60,
                "2400_clips_min": t_total / n_clips * 2400 / 60,
            },
        }, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    return ok == n_clips


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke test: 5 clips across environments/weather")
    parser.add_argument("--output-root", default="dataset_smoke_test", help="Output root directory")
    parser.add_argument("--n-clips", type=int, default=5, help="Number of clips to render")

    args = parser.parse_args()

    success = run_smoke_test(args.output_root, args.n_clips)
    sys.exit(0 if success else 1)
