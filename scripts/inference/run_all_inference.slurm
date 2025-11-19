#!/usr/bin/env bash
# run_inference_per_script_looper.sh
# For each script below, submit/run it; when it finishes, immediately submit/run it again,
# until it has run N times. Each script is managed independently & concurrently.
#
# Default scripts (edit as needed):
#   - carpark-inference.sh        (unified_carpark_runner)
#   - crossword-inference.sh      (unified_crossword_runner)
#   - math7b-inference.sh         (unified_math_runner)
#   - math8b-inference.sh         (unified_math_runner)
#   - math-inference.sh           (unified_math_runner)
#
# Usage:
#   bash run_inference_per_script_looper.sh
#   bash run_inference_per_script_looper.sh --loops 5 --sleep 15
#   bash run_inference_per_script_looper.sh --force-sbatch         # submit all via sbatch
#   bash run_inference_per_script_looper.sh --force-bash           # run all via bash
#   bash run_inference_per_script_looper.sh --dry-run
#   bash run_inference_per_script_looper.sh --fail-fast            # stop a script's loop on failure
#   bash run_inference_per_script_looper.sh --scripts "a.sh b.sh"  # override list

set -euo pipefail

# -------------------- Configurable defaults --------------------
LOOPS=5
SLEEP_SECS=15
force_mode="auto"   # auto | bash | sbatch
dry_run=false
fail_fast=false
SCRIPTS=("math8b-inference.sh")

# -------------------- Args --------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --loops)        LOOPS="${2:-5}"; shift 2;;
    --sleep)        SLEEP_SECS="${2:-15}"; shift 2;;
    --force-bash)   force_mode="bash"; shift;;
    --force-sbatch) force_mode="sbatch"; shift;;
    --dry-run)      dry_run=true; shift;;
    --fail-fast)    fail_fast=true; shift;;
    --scripts)      IFS=' ' read -r -a SCRIPTS <<< "${2:-}"; shift 2;;
    -h|--help)
      sed -n '1,120p' "$0"; exit 0;;
    *)
      echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

# -------------------- Helpers --------------------
ts() { date +'%F %T'; }
log() { echo "[$(ts)] $*"; }
err() { echo "[$(ts)] ERROR: $*" >&2; }

detect_mode() {
  local f="$1"
  if [[ "$force_mode" == "bash" ]];   then echo "bash";   return; fi
  if [[ "$force_mode" == "sbatch" ]]; then echo "sbatch"; return; fi
  # Auto: consider it a Slurm script if it declares #SBATCH near the top
  if head -n 50 "$f" 2>/dev/null | grep -qE '^\s*#SBATCH'; then
    echo "sbatch"
  else
    echo "bash"
  fi
}

submit_or_run() {
  # Prints a job id (for sbatch) or empty string (for bash).
  local f="$1" mode="$2"
  if [[ "$mode" == "sbatch" ]]; then
    sbatch --parsable "$f"
  else
    bash "$f"
    echo ""
  fi
}

job_is_active() {
  local jid="$1"
  [[ -n "$(squeue -h -j "$jid" 2>/dev/null || true)" ]]
}

wait_for_job_finish() {
  local jid="$1" script_name="$2"
  log "[$script_name] Waiting for Slurm job $jid to finish..."
  while job_is_active "$jid"; do
    sleep "$SLEEP_SECS"
  done
  log "[$script_name] Slurm job $jid finished."
}

check_job_state() {
  # Returns 0 if COMPLETED or sacct unavailable; 1 otherwise.
  local jid="$1" script_name="$2"
  if ! command -v sacct >/dev/null 2>&1; then
    log "[$script_name] sacct not available; skipping job state check."
    return 0
  fi
  # Get the top entry (batch), consider COMPLETED* as success
  local line state
  line=$(sacct -j "$jid" --format=JobID,State,ExitCode -n -P 2>/dev/null | awk -F'|' 'NR==1{print}')
  [[ -z "$line" ]] && { log "[$script_name] sacct returned no data for $jid"; return 0; }
  state=$(echo "$line" | awk -F'|' '{print $2}')
  log "[$script_name] sacct: $line"
  [[ "$state" == COMPLETED* ]]
}

# -------------------- Per-script manager --------------------
manage_script() {
  local f="$1"
  local loops="$2"

  if [[ ! -f "$f" ]]; then
    err "Missing script: $f"
    return 1
  fi

  local mode
  mode=$(detect_mode "$f")
  log "[$f] Managing in '$mode' mode for $loops loop(s)."

  local i=1
  while (( i <= loops )); do
    if $dry_run; then
      log "[DRY RUN][$f] Would launch iteration $i/$loops via $mode."
      ((i++))
      continue
    fi

    if [[ "$mode" == "sbatch" ]]; then
      # Submit and wait for that specific job to finish, then loop.
      local jid
      log "[$f] Submitting iteration $i/$loops..."
      if ! jid=$(submit_or_run "$f" "sbatch"); then
        err "[$f] sbatch submit failed (iteration $i)."
        $fail_fast && return 1
        ((i++))
        continue
      fi
      log "[$f] Submitted job: $jid"
      wait_for_job_finish "$jid" "$f"
      if ! check_job_state "$jid" "$f"; then
        err "[$f] Job $jid did not complete successfully."
        $fail_fast && return 1
      fi
    else
      # Run locally and check exit code.
      log "[$f] Running locally (iteration $i/$loops)..."
      if ! bash "$f"; then
        rc=$?
        err "[$f] Local run failed (rc=$rc) on iteration $i."
        $fail_fast && return $rc
      fi
    fi

    log "[$f] Iteration $i/$loops done. Restarting..."
    ((i++))
  done

  log "[$f] Completed $loops iteration(s)."
}

# -------------------- Launch managers concurrently --------------------
pids=()
for f in "${SCRIPTS[@]}"; do
  manage_script "$f" "$LOOPS" &
  pids+=("$!")
done

# Clean up: on SIGINT/SIGTERM, forward to children
trap 'echo "Signal received, terminating child processes..."; kill "${pids[@]}" 2>/dev/null || true' INT TERM

# Wait for all managers to finish
rc_overall=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    rc=$?
    rc_overall=$(( rc_overall || rc ))
  fi
done

log "All script managers finished ($LOOPS loop(s) each)."
exit $rc_overall
