#!/usr/bin/env bash
# Build the self-contained setup zip for Linux machines.
# Includes ONLY the interactive addon + the two numpy-only stage1 files it
# imports + the checker/launcher/README. Excludes all research-pipeline,
# dataset-generation, batch-queue, CV/EXR-detection, and Stage 2 code.
#
# Run from the repo root:  ./linux_quickstart/make_package.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$REPO/linux_quickstart/dist"
STAGE="$OUT/swarm-scan-linux"

rm -rf "$STAGE"
mkdir -p "$STAGE/stage1_geometry"

# 1. the addon (whole dir)
cp -r "$REPO/blender_addon" "$STAGE/blender_addon"

# 2. ONLY the two stage1 files the addon imports at load (both numpy-only)
cp "$REPO/stage1_geometry/scene_config.py" "$STAGE/stage1_geometry/"
cp "$REPO/stage1_geometry/multiview_triangulation_test.py" "$STAGE/stage1_geometry/"

# 3. user-facing files
cp "$REPO/linux_quickstart/check_env.py"    "$STAGE/"
cp "$REPO/linux_quickstart/launch.sh"       "$STAGE/"     # manual-launch fallback
cp "$REPO/linux_quickstart/setup_linux.sh"  "$STAGE/"     # one-command installer
cp "$REPO/linux_quickstart/README_LINUX.md" "$STAGE/README.md"
chmod +x "$STAGE/launch.sh" "$STAGE/setup_linux.sh"

# strip caches
find "$STAGE" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$STAGE" -name '*.pyc' -delete 2>/dev/null || true

( cd "$OUT" && rm -f swarm-scan-linux.zip && zip -rq swarm-scan-linux.zip swarm-scan-linux )
echo "Built: $OUT/swarm-scan-linux.zip"
echo "Contents:"
( cd "$OUT" && find swarm-scan-linux -type f | sort | sed 's/^/  /' )
