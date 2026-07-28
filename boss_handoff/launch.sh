#!/usr/bin/env bash
# Launch Blender with the Swarm Scan addon loaded.
# Works on Linux and macOS. Finds Blender automatically; override with:
#   SWARM_BLENDER=/path/to/blender ./launch.sh
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOADER="$DIR/blender_addon/dev_load.py"

if [[ ! -f "$LOADER" ]]; then
  echo "ERROR: cannot find blender_addon/dev_load.py next to this script."
  echo "Make sure you unzipped the whole package and are running it from inside that folder."
  exit 1
fi

find_blender() {
  if [[ -n "${SWARM_BLENDER:-}" && -x "$SWARM_BLENDER" ]]; then echo "$SWARM_BLENDER"; return; fi
  if command -v blender >/dev/null 2>&1; then command -v blender; return; fi
  for c in \
      /opt/blender/blender \
      /usr/local/bin/blender \
      /snap/bin/blender \
      "$HOME/blender/blender" \
      "$HOME/Downloads/blender/blender" \
      /Applications/Blender.app/Contents/MacOS/Blender \
      "$HOME/Applications/Blender.app/Contents/MacOS/Blender"; do
    [[ -x "$c" ]] && { echo "$c"; return; }
  done
  return 1
}

BLENDER="$(find_blender || true)"
if [[ -z "$BLENDER" ]]; then
  echo "ERROR: could not find Blender."
  echo "Install it (see README_BOSS.md step 1), or point this script at it:"
  echo "    SWARM_BLENDER=/full/path/to/blender ./launch.sh"
  exit 1
fi

echo "Using Blender: $BLENDER"
exec "$BLENDER" --python "$LOADER"
