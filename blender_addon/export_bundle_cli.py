"""
CLI variant of SWARM_OT_export_bundle for headless rendering in CI or automation.

Renders synchronized frames across all cameras, checks framing coverage, and
writes the bundle directory -- all without the Blender UI.

Usage:
    blender -b scene.blend -P export_bundle_cli.py -- \\
        --output-dir /path/to/bundle

    blender -b scene.blend -P export_bundle_cli.py -- \\
        --output-dir /path/to/bundle \\
        --scene-id my_scene_001 \\
        --samples 256

    blender -b scene.blend -P export_bundle_cli.py --help
"""

import argparse
import os
import sys

# Add the repo root so we can import the addon and stage1 modules.
# This mirrors blender_addon/swarm_scanner/__init__.py's own sys.path setup.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "stage1_geometry"))
sys.path.insert(0, _REPO_ROOT)

import bpy


def main():
    parser = argparse.ArgumentParser(
        description="Export a capture bundle from a Blender scene."
    )
    parser.add_argument(
        "--blend-file",
        default="",
        help="Path to .blend file (optional; uses already-open file if omitted)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write the bundle to",
    )
    parser.add_argument(
        "--scene-id",
        default="export_cli_001",
        help="Unique identifier for this capture session",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=128,
        help="Cycles render samples per view (default: 128)",
    )
    args, unknown = parser.parse_known_args()

    # Open blend file if specified
    if args.blend_file:
        if not os.path.exists(args.blend_file):
            print(f"ERROR: Blend file not found: {args.blend_file}", file=sys.stderr)
            sys.exit(1)
        bpy.ops.wm.open_mainfile(filepath=args.blend_file)

    # Register the addon so the operator and its settings properties exist.
    from blender_addon.swarm_scanner import register as swarm_register
    swarm_register()

    # Set export settings on the scene's settings group.
    scene = bpy.context.scene
    scene.swarm_scan.export_output_dir = args.output_dir
    scene.swarm_scan.export_scene_id = args.scene_id
    scene.swarm_scan.export_render_samples = args.samples

    # Run the export operator by bl_idname.
    result = bpy.ops.swarm_scan.export_bundle()

    if result == {"FINISHED"}:
        print(f"OK: Bundle exported to {args.output_dir}")
    else:
        print(f"ERROR: Export returned {result}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
