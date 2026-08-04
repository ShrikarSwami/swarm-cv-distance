#!/usr/bin/env bash
#
# ml_overnight.sh — unattended Claude Code loop for the ML model fix (G2 gate)
#
# Each iteration is a fresh `claude -p` process doing ONE queued fix attempt.
# A stall kills that process only; the loop continues.
#
# Usage:  caffeinate -dims ./scripts/ml_overnight.sh
#
# Pause/resume (control file, same pattern as ml/render_harness.py):
#   echo PAUSED  > .overnight/control.state
#   echo RUNNING > .overnight/control.state
#   echo STOP    > .overnight/control.state
# The control file is polled at the top of each iteration, never mid-unit.
# PAUSED waits (paused time does not count against WALL_CLOCK_MAX); STOP ends
# the run cleanly so the morning summary still prints.
#
set -uo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
CLAUDE_BIN=/Users/shrishrimacbook/.local/bin/claude
PROMPT_FILE="prompts/ml-unit-prompt.md"
QUEUE_FILE="ml/FIX_QUEUE.md"
GATES_FILE="ml/GATES.md"
PROGRESS_FILE="docs/PROGRESS.md"

# ── Safety parameters ────────────────────────────────────────────────────────
BRANCH="overnight/ml-g2-fix"
LOG_DIR=".overnight"
MAX_ITER=12
UNIT_TIMEOUT=3600        # 60 min — a G2 attempt at ~1.3 s/step needs the room
COOLDOWN=10
SENTINEL="OVERNIGHT_COMPLETE"
MAX_CONSECUTIVE_FAILS=3
WALL_CLOCK_MAX=28800     # 8 hours
MIN_FREE_GB=10
# ─────────────────────────────────────────────────────────────────────────────

export PYTORCH_ENABLE_MPS_FALLBACK=1

# ── Frozen set: any change aborts the entire run ─────────────────────────────
#
# This is the mechanism that replaces the human reviewer. Unattended, facing a
# gate it cannot pass, an agent's locally rational move is to weaken the gate.
# Checksums make that mechanically impossible rather than merely forbidden.
FROZEN_FILES=(
  "ml/GATES.md"
  "ml/metrics.py"
  "ml/splits.json"
  "calib.json"
  "tests/test_predictions_ml.py"
  "ml/baseline_adapter.py"
  "ml/eval_sweep.py"
  "ml/adjacency_eval.py"
  "ml/recon_app.py"
)

freeze_manifest() {
  {
    for f in "${FROZEN_FILES[@]}"; do
      [ -f "$f" ] && shasum -a 256 "$f"
    done
    if [ -d stage1_geometry ]; then
      find stage1_geometry -type f -name '*.py' -exec shasum -a 256 {} \; 2>/dev/null
    fi
  } | sort
}

# ── Tracking / exit summary ───────────────────────────────────────────────────
# log_summary, on_exit and the trap live ABOVE preflight: any early exit —
# including a preflight failure — must still write the summary without a
# "log_summary: command not found" error.
ITERATIONS_DONE=0
COMMITS_MADE=0
EXIT_REASON=""
START_EPOCH=$(date +%s)
SUMMARY_LOG="logs/ml-overnight-summary.log"
TOTAL_PAUSED_SECONDS=0
PAUSE_START=""

log_summary() {
  mkdir -p "$(dirname "$SUMMARY_LOG")" 2>/dev/null || true
  local elapsed=$(( $(date +%s) - START_EPOCH - TOTAL_PAUSED_SECONDS ))
  {
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="
    echo "  iterations:   $ITERATIONS_DONE"
    echo "  commits:      $COMMITS_MADE"
    echo "  exit reason:  ${EXIT_REASON:-unknown}"
    echo "  wall clock:   $((elapsed/60))m$((elapsed%60))s"
    echo ""
  } >> "$SUMMARY_LOG" 2>/dev/null || true
}
on_exit() { log_summary; }
trap on_exit EXIT

# ── Preflight ────────────────────────────────────────────────────────────────
preflight_ok=1

if [ ! -x "$CLAUDE_BIN" ]; then
  echo "PREFLIGHT FAIL: $CLAUDE_BIN missing or not executable"; preflight_ok=0
fi

for f in "$PROMPT_FILE" "$QUEUE_FILE" "$GATES_FILE"; do
  if [ ! -f "$f" ]; then
    echo "PREFLIGHT FAIL: $f not found"; preflight_ok=0
  fi
done

if [ ! -d "$HOME/swarm_ml_packed" ]; then
  echo "PREFLIGHT FAIL: ~/swarm_ml_packed not found — nothing to train on"
  preflight_ok=0
fi

free_gb=$(df -g /System/Volumes/Data | awk 'NR==2 {print $4}')
if [ "${free_gb:-0}" -lt "$MIN_FREE_GB" ]; then
  echo "PREFLIGHT FAIL: only ${free_gb}GB free, need ${MIN_FREE_GB}GB for checkpoints"
  preflight_ok=0
fi

if [ -n "$(git status --porcelain)" ]; then
  echo "PREFLIGHT FAIL: working tree is dirty"
  git status --short | head -20
  echo ""
  echo "Commit or stash first. This repo has untracked debris; consider:"
  echo "  git add -A && git commit -m 'wip: handoff to overnight run'"
  preflight_ok=0
fi

current_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)
if [ "$current_branch" = "main" ]; then
  echo "PREFLIGHT FAIL: on main — overnight runs must not target main"
  echo "  git checkout -b $BRANCH"
  preflight_ok=0
fi

if ! grep -q "PENDING" "$QUEUE_FILE" 2>/dev/null; then
  echo "PREFLIGHT FAIL: no PENDING tasks in $QUEUE_FILE"; preflight_ok=0
fi

if grep -q "$SENTINEL" "$PROGRESS_FILE" 2>/dev/null; then
  echo "PREFLIGHT FAIL: sentinel already present in $PROGRESS_FILE"
  echo "  Remove it before starting a new run."
  preflight_ok=0
fi

if [ "$preflight_ok" -ne 1 ]; then
  echo ""; echo "PREFLIGHT FAIL — fix the above and retry"; exit 1
fi

# macOS ships BSD userland; `brew install coreutils` provides gtimeout.
if command -v timeout >/dev/null 2>&1; then TIMEOUT_CMD="timeout"
elif command -v gtimeout >/dev/null 2>&1; then TIMEOUT_CMD="gtimeout"
else echo "ERROR: no timeout command. brew install coreutils"; exit 1; fi

mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/ml-run-$(date +%Y%m%d-%H%M%S).log"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$RUN_LOG"; }

# ── Pause/resume control file ────────────────────────────────────────────────
# Same pattern as ml/render_harness.py. Polled at the top of each iteration.
CONTROL_FILE="$LOG_DIR/control.state"
if [ ! -f "$CONTROL_FILE" ]; then
  echo "RUNNING" > "$CONTROL_FILE"
fi

log "on branch $current_branch"
START_COMMIT=$(git rev-parse --short HEAD)
log "starting at $START_COMMIT"

FROZEN_BASELINE=$(freeze_manifest)
log "frozen baseline: $(echo "$FROZEN_BASELINE" | wc -l | tr -d ' ') files checksummed"

fails=0
LAST_HEAD=$(git rev-parse HEAD)

# ── Pause/resume: polled at the TOP of each iteration, never mid-unit ────────
# PAUSED sleeps 30s and re-polls (resuming needs no re-preflight); STOP breaks
# the loop so the morning summary still prints. Paused seconds accumulate in
# TOTAL_PAUSED_SECONDS and are subtracted from the wall-clock budget.
poll_control() {
  while true; do
    state=$(cat "$CONTROL_FILE" 2>/dev/null | tr -d '[:space:]' || echo RUNNING)
    case "$state" in
      PAUSED)
        if [ -z "$PAUSE_START" ]; then
          PAUSE_START=$(date +%s)
          log "control file PAUSED — pausing (polling every 30s)"
        fi
        sleep 30
        ;;
      STOP)
        EXIT_REASON="stopped via control file"
        log "control file STOP — ending run"
        return 1
        ;;
      *)
        if [ -n "$PAUSE_START" ]; then
          local paused=$(( $(date +%s) - PAUSE_START ))
          TOTAL_PAUSED_SECONDS=$(( TOTAL_PAUSED_SECONDS + paused ))
          log "control file RUNNING — resumed after ${paused}s paused (${TOTAL_PAUSED_SECONDS}s total)"
          PAUSE_START=""
        fi
        return 0
        ;;
    esac
  done
}

for i in $(seq 1 "$MAX_ITER"); do

  # ── Pause/resume control: polled before any stop-check or launch ──────────
  if ! poll_control; then
    break
  fi

  # ── Stop conditions ───────────────────────────────────────────────────────
  if grep -q "$SENTINEL" "$PROGRESS_FILE" 2>/dev/null; then
    log "sentinel found — queue complete"
    EXIT_REASON="sentinel found"; break
  fi

  if ! grep -q "PENDING" "$QUEUE_FILE" 2>/dev/null; then
    log "no PENDING tasks remain"
    EXIT_REASON="queue exhausted"; break
  fi

  if [ "$fails" -ge "$MAX_CONSECUTIVE_FAILS" ]; then
    log "ABORT: $fails consecutive process failures"
    EXIT_REASON="$fails consecutive failures"; break
  fi

  elapsed=$(( $(date +%s) - START_EPOCH - TOTAL_PAUSED_SECONDS ))
  if [ "$elapsed" -ge "$WALL_CLOCK_MAX" ]; then
    log "WALL CLOCK LIMIT reached (${elapsed}s)"
    EXIT_REASON="wall clock limit"; break
  fi

  # ── Frozen-path check: the human-substitute ───────────────────────────────
  CURRENT_FROZEN=$(freeze_manifest)
  if [ "$CURRENT_FROZEN" != "$FROZEN_BASELINE" ]; then
    log "!!! ABORT: a frozen file changed !!!"
    diff <(echo "$FROZEN_BASELINE") <(echo "$CURRENT_FROZEN") | tee -a "$RUN_LOG"
    EXIT_REASON="FROZEN PATH MODIFIED — manual review required"
    break
  fi

  # ── Guard against a sneaked-in gate change with matching checksum ────────
  if ! grep -q "median position error  <  1.0 m" "$GATES_FILE" 2>/dev/null; then
    log "!!! ABORT: G2 definition not found verbatim in $GATES_FILE !!!"
    EXIT_REASON="GATE DEFINITION ALTERED"
    break
  fi

  ITER_LOG="$LOG_DIR/ml-iter-$(printf '%03d' "$i").log"
  next_task=$(grep -B4 "PENDING" "$QUEUE_FILE" | grep -m1 "^## FIX-" || echo "unknown")
  log "── iteration $i ── next: $next_task"

  "$TIMEOUT_CMD" "$UNIT_TIMEOUT" "$CLAUDE_BIN" -p "$(cat "$PROMPT_FILE")" \
      --dangerously-skip-permissions \
      > "$ITER_LOG" 2>&1
  code=$?

  case $code in
    0)   log "iteration $i ok"; fails=0 ;;
    124) log "iteration $i TIMED OUT after ${UNIT_TIMEOUT}s"
         echo "TIMEOUT" >> "$ITER_LOG"; fails=$((fails+1)) ;;
    *)   log "iteration $i exited $code — see $ITER_LOG"; fails=$((fails+1)) ;;
  esac

  # A FAILED gate is a normal outcome and must not count as a process failure.
  if grep -qE "^\s*verdict:\s*(FAIL|FAILED)" "$QUEUE_FILE" 2>/dev/null; then
    : # expected; nothing to do
  fi

  # ── Stash anything uncommitted, push what landed ──────────────────────────
  if [ -n "$(git status --porcelain)" ]; then
    log "WARNING: uncommitted changes after iteration $i"
    git stash push -m "ml-overnight-iter-$i" >/dev/null 2>&1 \
      && log "stashed uncommitted work"
  fi

  NEW_HEAD=$(git rev-parse HEAD)
  if [ "$NEW_HEAD" != "$LAST_HEAD" ]; then
    COMMITS_MADE=$((COMMITS_MADE + 1))
    if ! git push origin "$current_branch" 2>>"$RUN_LOG"; then
      log "WARNING: push failed — commits remain local"
    else
      log "pushed after iteration $i"
    fi
  else
    log "NOTE: iteration $i produced no commit"
  fi
  LAST_HEAD="$NEW_HEAD"

  ITERATIONS_DONE=$i
  sleep "$COOLDOWN"
done

[ -z "$EXIT_REASON" ] && EXIT_REASON="completed all $MAX_ITER iterations"

# ── Morning summary ──────────────────────────────────────────────────────────
END_COMMIT=$(git rev-parse --short HEAD)
{
  echo ""
  echo "════════════════════════════════════════"
  echo "  ML OVERNIGHT RUN COMPLETE"
  echo "════════════════════════════════════════"
  echo "  branch:  $current_branch"
  echo "  commits: $START_COMMIT..$END_COMMIT"
  echo "  reason:  $EXIT_REASON"
  echo ""
  echo "  Commits this run:"
  git log --oneline "$START_COMMIT..$END_COMMIT" 2>/dev/null | sed 's/^/    /'
  echo ""
  echo "  Queue status:"
  grep -E "^(## FIX-|\*\*Status:\*\*)" "$QUEUE_FILE" 2>/dev/null | sed 's/^/    /'
  echo ""
  echo "  Timeouts:"
  grep -l "TIMEOUT" "$LOG_DIR"/ml-iter-*.log 2>/dev/null | sed 's/^/    /' || echo "    none"
  echo ""
  echo "  Review in this order:"
  echo "    1. $QUEUE_FILE       (predicted vs observed per fix)"
  echo "    2. $PROGRESS_FILE    (session log)"
  echo "    3. git log --oneline $START_COMMIT..$END_COMMIT"
  echo "    4. any timed-out iteration log"
  echo "    5. git stash list    (recovered partial work)"
  echo ""
  echo "  If EXIT_REASON mentions FROZEN PATH or GATE: read that first."
  echo "════════════════════════════════════════"
} | tee -a "$RUN_LOG"
