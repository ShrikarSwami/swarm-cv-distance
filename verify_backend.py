#!/usr/bin/env python3
"""Verify the web frontend backend produces identical results to ml/recon_app.py.

This is the acceptance-critical step (Step 2 from the requirements). It compares
the FastAPI backend results with the existing CLI implementation on the same
scene and view set inputs.
"""

import json
import os
import sys
import numpy as np
import subprocess
import tempfile
from pathlib import Path

# Add the repo root to sys.path to import stage1_geometry modules
REPO_ROOT = Path(__file__).parent
STAGE1 = REPO_ROOT / "stage1_geometry"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(STAGE1))

# Import from ml.recon_app
from ml.recon_app import (
    load_manifest, find_scene, choose_angles, camera_positions,
    selected_view_angles, _load_scene_data, _overlay_pred_positions,
    process_scene, scene_metadata_line, N_VIEWS_TOTAL, DEFAULT_ROOT,
    DEFAULT_RECALL_RADIUS_PX, parse_args, run_recon
)

# Import the backend models to verify structure
from ml.webapp.backend.models import ReconstructionResult


def run_backend_reconstruction(scene_seed, view_idxs, recall_radius_px=5.0):
    """Run reconstruction using the FastAPI backend.

    This imports the backend code and calls the reconstruction function
    directly (bypassing the API server for testing).
    """
    # Import the backend reconstruction function
    sys.path.insert(0, str(REPO_ROOT / "ml" / "webapp" / "backend"))
    from main import reconstruct_scene

    # Create a mock request object
    from ml.webapp.backend.models import ReconstructionRequest, AngleSelection, AngleMode

    request = ReconstructionRequest(
        scene_seed=scene_seed,
        angle_selection=AngleSelection(
            mode=AngleMode.ALL if len(view_idxs) == N_VIEWS_TOTAL else AngleMode.EXACT,
            view_indices=view_idxs if len(view_idxs) != N_VIEWS_TOTAL else None,
            exact_indices=view_idxs if len(view_idxs) == N_VIEWS_TOTAL else None
        ),
        recall_radius_px=recall_radius_px
    )

    # Run the reconstruction
    return reconstruct_scene(request)


def run_cli_reconstruction(scene_seed, view_idxs, mode="exact", recall_radius_px=5.0):
    """Run reconstruction using the original ml/recon_app.py CLI.

    This runs the CLI as a subprocess to ensure we're comparing apples to apples.
    """
    # Create a temporary script that calls ml.recon_app directly
    script_content = f'''import sys
import os
sys.path.insert(0, "{REPO_ROOT}")

from ml.recon_app import load_manifest, find_scene, _load_scene_data, process_scene
from ml.recon_app import _overlay_pred_positions, camera_positions
from ml.recon_app import DEFAULT_ROOT, DEFAULT_RECALL_RADIUS_PX
import numpy as np

# Load scene
scenes = load_manifest(DEFAULT_ROOT)
scene = find_scene(scenes, {scene_seed})

# Load scene data
gt, cam = _load_scene_data(DEFAULT_ROOT, {scene_seed})
true = np.asarray(gt["positions"], dtype=np.float64)

# Run reconstruction
result = process_scene(DEFAULT_ROOT, {scene_seed}, {view_idxs}, {recall_radius_px})
result["recall_radius_px"] = {recall_radius_px}

# Get predicted positions for overlay
pred_positions = _overlay_pred_positions(DEFAULT_ROOT, {scene_seed}, {view_idxs}, cam=cam)

# Get camera positions
cam_pos = camera_positions(cam, {view_idxs})

# Print metrics in a parseable format
import json
output = {{
    "metrics": result["metrics"],
    "true_positions": true.tolist(),
    "pred_positions": pred_positions.tolist(),
    "camera_positions": cam_pos.tolist(),
    "view_indices": {view_idxs},
    "wall_clock_s": result.get("wall_clock_s", 0.0),
    "detector_recall": result.get("detector_recall"),
    "n_true": len(true),
    "n_pred": len(pred_positions)
}}
print(json.dumps(output, indent=2))
'''

    # Write script to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        script_path = f.name

    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT)
        )

        if result.returncode != 0:
            print(f"CLI reconstruction failed: {result.stderr}")
            return None

        # Parse output
        output_lines = result.stdout.strip().split('\n')
        json_start = output_lines[0].find('{')
        if json_start == -1:
            print("Unexpected output format from CLI:")
            print(result.stdout)
            return None

        json_str = result.stdout[json_start:]
        return json.loads(json_str)

    finally:
        # Clean up temp file
        os.unlink(script_path)


def compare_results(backend_result, cli_result, tolerance=1e-6):
    """Compare backend and CLI results.

    Returns True if results are equivalent within tolerance, False otherwise.
    """
    print("Comparing backend and CLI results...")

    # Compare metrics
    backend_metrics = backend_result.get("metrics", {})
    cli_metrics = cli_result.get("metrics", {})

    metric_keys = ["n_true", "n_pred", "mAP", "median_err_m", "chamfer_m", "count_err"]

    for key in metric_keys:
        if key not in backend_metrics or key not in cli_metrics:
            print(f"  Missing metric key: {key}")
            continue

        backend_val = backend_metrics[key]
        cli_val = cli_metrics[key]

        if isinstance(backend_val, (int, float)) and isinstance(cli_val, (int, float)):
            if abs(backend_val - cli_val) > tolerance:
                print(f"  Metric {key} differs: backend={backend_val}, CLI={cli_val}")
                return False
        elif backend_val != cli_val:
            print(f"  Metric {key} differs: backend={backend_val}, CLI={cli_val}")
            return False

    # Compare per-tau metrics
    backend_per_tau = backend_metrics.get("per_tau", {})
    cli_per_tau = cli_metrics.get("per_tau", {})

    backend_tau_keys = set(backend_per_tau.keys())
    cli_tau_keys = set(cli_per_tau.keys())

    if backend_tau_keys != cli_tau_keys:
        print(f"  Per-tau keys differ: backend={backend_tau_keys}, CLI={cli_tau_keys}")
        return False

    for tau in backend_tau_keys:
        backend_tau = backend_per_tau[tau]
        cli_tau = cli_per_tau[tau]

        for metric in ["precision", "recall", "f1", "ap", "n_matched"]:
            if metric not in backend_tau or metric not in cli_tau:
                print(f"  Missing tau metric {metric} for tau {tau}")
                continue

            backend_val = backend_tau[metric]
            cli_val = cli_tau[metric]

            if isinstance(backend_val, (int, float)) and isinstance(cli_val, (int, float)):
                if abs(backend_val - cli_val) > tolerance:
                    print(f"  Tau {tau} metric {metric} differs: backend={backend_val}, CLI={cli_val}")
                    return False
            elif backend_val != cli_val:
                print(f"  Tau {tau} metric {metric} differs: backend={backend_val}, CLI={cli_val}")
                return False

    # Compare positions (if present)
    for pos_key in ["true_positions", "pred_positions", "camera_positions"]:
        if pos_key in backend_result and pos_key in cli_result:
            backend_pos = backend_result[pos_key]
            cli_pos = cli_result[pos_key]

            if len(backend_pos) != len(cli_pos):
                print(f"  Position array {pos_key} length differs: backend={len(backend_pos)}, CLI={len(cli_pos)}")
                return False

            for i, (backend_pt, cli_pt) in enumerate(zip(backend_pos, cli_pos)):
                if isinstance(backend_pt, dict) and isinstance(cli_pt, dict):
                    for coord in ["x", "y", "z"]:
                        if abs(backend_pt.get(coord, 0) - cli_pt.get(coord, 0)) > tolerance:
                            print(f"  Position {pos_key}[{i}].{coord} differs: backend={backend_pt.get(coord)}, CLI={cli_pt.get(coord)}")
                            return False

    print("  ✓ Results match within tolerance")
    return True


def main():
    """Main verification function."""
    print("Starting backend verification...")

    # Check if ~/swarm_ml exists
    if not os.path.exists(DEFAULT_ROOT):
        print(f"Error: Data root not found: {DEFAULT_ROOT}")
        print("Please ensure ~/swarm_ml is present with manifest.jsonl and scenes/")
        return 1

    # Load scenes
    scenes = load_manifest(DEFAULT_ROOT)
    print(f"Loaded {len(scenes)} scenes from manifest")

    # Select a test scene (use scene 0 if available)
    test_scene = scenes[0]
    scene_seed = int(test_scene["seed"])
    print(f"Testing with scene seed: {scene_seed}")

    # Test Case 1: All 24 views
    print("\n--- Test Case 1: All 24 views ---")
    view_idxs = list(range(N_VIEWS_TOTAL))

    print("Running CLI reconstruction...")
    cli_result = run_cli_reconstruction(scene_seed, view_idxs)
    if cli_result is None:
        print("CLI reconstruction failed")
        return 1

    print("Running backend reconstruction...")
    backend_result = run_backend_reconstruction(scene_seed, view_idxs)
    if backend_result is None:
        print("Backend reconstruction failed")
        return 1

    if compare_results(backend_result, cli_result):
        print("✓ Test Case 1 PASSED")
    else:
        print("✗ Test Case 1 FAILED")
        return 1

    # Test Case 2: Random subset (6 views)
    print("\n--- Test Case 2: Random subset (6 views) ---")
    import random
    rng = random.Random(42)  # Deterministic
    view_idxs = sorted(rng.sample(range(N_VIEWS_TOTAL), 6))

    print("Running CLI reconstruction...")
    cli_result = run_cli_reconstruction(scene_seed, view_idxs)
    if cli_result is None:
        print("CLI reconstruction failed")
        return 1

    print("Running backend reconstruction...")
    backend_result = run_backend_reconstruction(scene_seed, view_idxs)
    if backend_result is None:
        print("Backend reconstruction failed")
        return 1

    if compare_results(backend_result, cli_result):
        print("✓ Test Case 2 PASSED")
    else:
        print("✗ Test Case 2 FAILED")
        return 1

    print("\n✓ All verification tests PASSED!")
    print("The backend produces identical results to ml/recon_app.py")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)