"""
Config generator for the full sweep: environment × weather × formation × camera arrangement × seed.

Produces job specs the batch queue can consume.

Usage:
    python config_generator.py --output sweep_config.json --n-seeds 5
"""

import json
import itertools
from pathlib import Path
from typing import List, Dict, Optional

# Environment presets
ENVIRONMENTS = ["desert", "grassland", "forest", "city"]

# Weather presets with sky luminance
WEATHER = {
    "clear": {"luminance_cd_m2": 8000, "energy_mult": 1.0},
    "overcast": {"luminance_cd_m2": 2000, "energy_mult": 0.3},
    "hazy": {"luminance_cd_m2": 4000, "energy_mult": 0.7},
    "dusk": {"luminance_cd_m2": 500, "energy_mult": 0.4},
    "night": {"luminance_cd_m2": 0.001, "energy_mult": 0.1},
}

# Formation presets
FORMATIONS = ["random_cloud", "grid", "sphere", "herringbone",
              "lightshow_circle", "lightshow_star"]

# Camera arrangements
CAMERA_ARRANGEMENTS = [
    {"name": "dome_6", "n_views": 6, "focal_mm": 24, "standoff_m": 2000},
    {"name": "dome_12", "n_views": 12, "focal_mm": 24, "standoff_m": 2000},
    {"name": "wide_6", "n_views": 6, "focal_mm": 16, "standoff_m": 1500},
    {"name": "narrow_12", "n_views": 12, "focal_mm": 50, "standoff_m": 3000},
]

# Default render settings
DEFAULT_RENDER = {
    "n_frames": 20,
    "fps": 10,
    "sensor_width_mm": 36.0,
    "resolution": [1920, 1080],
    "n_drones": 20,
    "samples": 32,
}


def generate_clip_name(env: str, weather: str, formation: str,
                       camera: str, seed: int) -> str:
    """Generate a unique clip name from parameters."""
    return f"{env}_{weather}_{formation}_{camera}_s{seed:04d}"


def generate_clip_config(
    env: str,
    weather: str,
    formation: str,
    camera: Dict,
    seed: int,
    output_root: str,
) -> Dict:
    """Generate a single clip configuration."""
    clip_name = generate_clip_name(env, weather, formation, camera["name"], seed)

    config = {
        "clip_name": clip_name,
        "dataset_root": output_root,
        "environment": env,
        "weather": weather,
        "formation": formation,
        "seed": seed,
        **camera,
        **DEFAULT_RENDER,
    }

    return config


def generate_sweep(
    output_root: str,
    n_seeds: int = 5,
    environments: Optional[List[str]] = None,
    weather: Optional[List[str]] = None,
    formations: Optional[List[str]] = None,
    camera_arrangements: Optional[List[Dict]] = None,
) -> Dict:
    """Generate the full sweep configuration."""
    envs = environments or ENVIRONMENTS
    weathers = weather or list(WEATHER.keys())
    forms = formations or FORMATIONS
    cams = camera_arrangements or CAMERA_ARRANGEMENTS

    clips = []
    for env, weath, form, cam, seed in itertools.product(
        envs, weathers, forms, cams, range(n_seeds)
    ):
        config = generate_clip_config(env, weath, form, cam, seed, output_root)
        clips.append(config)

    return {
        "description": "Full sweep: environment × weather × formation × camera × seed",
        "n_clips": len(clips),
        "sweep": {
            "environments": envs,
            "weather": weathers,
            "formations": forms,
            "cameras": [c["name"] for c in cams],
            "n_seeds": n_seeds,
        },
        "clips": clips,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate sweep configuration")
    parser.add_argument("--output", required=True, help="Output config JSON path")
    parser.add_argument("--output-root", default="dataset_sweep", help="Output root for renders")
    parser.add_argument("--n-seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument("--environments", nargs="+", help="Environments to include")
    parser.add_argument("--weather", nargs="+", help="Weather conditions to include")
    parser.add_argument("--formations", nargs="+", help="Formations to include")
    parser.add_argument("--cameras", nargs="+", help="Camera arrangements to include")
    parser.add_argument("--summary", action="store_true", help="Print summary only")

    args = parser.parse_args()

    # Filter camera arrangements if specified
    cams = CAMERA_ARRANGEMENTS
    if args.cameras:
        cams = [c for c in CAMERA_ARRANGEMENTS if c["name"] in args.cameras]

    config = generate_sweep(
        output_root=args.output_root,
        n_seeds=args.n_seeds,
        environments=args.environments,
        weather=args.weather,
        formations=args.formations,
        camera_arrangements=cams,
    )

    if args.summary:
        print(f"Total clips: {config['n_clips']}")
        print(f"Environments: {config['sweep']['environments']}")
        print(f"Weather: {config['sweep']['weather']}")
        print(f"Formations: {config['sweep']['formations']}")
        print(f"Cameras: {config['sweep']['cameras']}")
        print(f"Seeds: {config['sweep']['n_seeds']}")
        return

    # Save config
    with open(args.output, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Generated {config['n_clips']} clip configs")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
