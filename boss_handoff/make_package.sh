#!/usr/bin/env bash
# Build the self-contained handoff zip for the boss.
# Includes ONLY the interactive addon + the two numpy-only stage1 files it
# imports + the checker/launcher/README. Excludes all research-pipeline,
# dataset-generation, batch-queue, CV/EXR-detection, and Stage 2 code.
#
# Run from the repo root:  ./boss_handoff/make_package.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$REPO/boss_handoff/dist"
STAGE="$OUT/swarm-scan-boss"

rm -rf "$STAGE"
mkdir -p "$STAGE/stage1_geometry"

# 1. the addon (whole dir)
cp -r "$REPO/blender_addon" "$STAGE/blender_addon"

# 2. ONLY the two stage1 files the addon imports at load (both numpy-only)
cp "$REPO/stage1_geometry/scene_config.py" "$STAGE/stage1_geometry/"
cp "$REPO/stage1_geometry/multiview_triangulation_test.py" "$STAGE/stage1_geometry/"

# 3. boss-facing files
cp "$REPO/boss_handoff/check_env.py"    "$STAGE/"
cp "$REPO/boss_handoff/launch.sh"       "$STAGE/"     # manual-launch fallback
cp "$REPO/boss_handoff/setup_linux.sh"  "$STAGE/"     # one-command installer
cp "$REPO/boss_handoff/README_Linux.md" "$STAGE/README.md"
chmod +x "$STAGE/launch.sh" "$STAGE/setup_linux.sh"

# strip caches
find "$STAGE" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$STAGE" -name '*.pyc' -delete 2>/dev/null || true

( cd "$OUT" && rm -f swarm-scan-boss.zip && zip -rq swarm-scan-boss.zip swarm-scan-boss )
echo "Built: $OUT/swarm-scan-boss.zip"
echo "Contents:"
( cd "$OUT" && find swarm-scan-boss -type f | sort | sed 's/^/  /' )
