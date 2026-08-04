#!/usr/bin/env bash
#
# Overnight render campaign, both operating cells.
#
# Scene directories are keyed by SEED ALONE, not seed+cell (verified by scratch
# test 2026-07-31). Rendering the same seed under a second --cell is a silent
# no-op. Therefore each cell gets a disjoint seed range.
#
# Seed plan, aligned to the PATCH 7 split boundaries:
#   test   0-999      primary 0-499       secondary 500-999
#   val    1000-1999  primary 1000-1499   secondary 1500-1999
#   train  2000-4999  primary 2000-3499   secondary 3500-4999
#
# Total 5000 scenes at ~3.6s = ~5 hours.
#
# Launch:
#   nohup caffeinate -i bash start_render.sh > "$HOME/swarm_ml/render.log" 2>&1 &
#
# Resume after a pause or interruption: just run it again. Completed stages are
# skipped, partial stages continue.
#
# Pause:  echo PAUSED  > "$HOME/swarm_ml/control.state"
# Resume: echo RUNNING > "$HOME/swarm_ml/control.state"

set -uo pipefail

ROOT="$HOME/swarm_ml"
HARNESS="ml/render_harness.py"

mkdir -p "$ROOT"

if [ ! -f "$HARNESS" ]; then
  echo "FATAL: $HARNESS not found. Run from the repo root." >&2
  exit 1
fi

# Count manifest entries whose seed falls in [start, end).
count_range() {
  local start=$1 end=$2
  python3 - "$ROOT/manifest.jsonl" "$start" "$end" <<'PY'
import json, sys, os
path, start, end = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
if not os.path.exists(path):
    print(0); raise SystemExit
n = 0
with open(path) as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            seed = json.loads(line)["seed"]
        except Exception:
            continue
        if start <= seed < end:
            n += 1
print(n)
PY
}

run_stage() {
  local label=$1 cell=$2 start=$3 count=$4
  local end=$(( start + count ))
  local have

  have=$(count_range "$start" "$end")

  if [ "$have" -ge "$count" ]; then
    echo "=== SKIP  $label  (already $have/$count in seeds $start-$((end-1))) ==="
    return 0
  fi

  echo "=== START $label  cell=$cell  seeds $start-$((end-1))  have=$have/$count ==="
  python "$HARNESS" --root "$ROOT" --cell "$cell" --start-seed "$start" --target "$count"
  local rc=$?

  local now
  now=$(count_range "$start" "$end")
  echo "=== END   $label  $have -> $now / $count  (exit $rc) ==="

  # Control-file pause exits cleanly without finishing. Distinguish that from a
  # genuine no-op by checking whether anything was added at all.
  if [ "$now" -le "$have" ]; then
    echo "FATAL: '$label' added zero scenes." >&2
    echo "  Either the run is PAUSED, or --target counts globally rather than" >&2
    echo "  from --start-seed. Check $ROOT/control.state and the harness log." >&2
    exit 1
  fi

  if [ "$now" -lt "$count" ]; then
    echo "PAUSED or interrupted during '$label' ($now/$count). Re-run to continue." >&2
    exit 0
  fi
}

echo "########## render campaign starting $(date) ##########"
df -h /System/Volumes/Data | tail -1

# Test and val first, so training can start on partial data.
run_stage "test-primary"    primary      0    500
run_stage "test-secondary"  secondary  500    500
run_stage "val-primary"     primary   1000    500
run_stage "val-secondary"   secondary 1500    500
run_stage "train-primary"   primary   2000   1500
run_stage "train-secondary" secondary 3500   1500

echo "########## render campaign complete $(date) ##########"
count_range 0 5000 | xargs -I{} echo "total scenes in manifest: {}"
du -sh "$ROOT"
