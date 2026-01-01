#!/usr/bin/env bash
# Backfill missing `"pass2": null` / missing pass2 rows for many math result JSONLs,
# using a single multi-GPU Slurm job (no arrays required).
#
# Typical usage (Llama8B only):
#   python scripts/inference/write_math_backfill_manifest.py \
#     --only_family Llama8B \
#     --out tmp/llama8b_pass2_manifest.tsv
#   sbatch --account mltheory --partition gpu --gres gpu:a100:8 --ntasks 8 \
#     scripts/inference/math-backfill-pass2-manifest-parallel.slurm \
#     MANIFEST=tmp/llama8b_pass2_manifest.tsv BATCH_SIZE=1
#
# Env overrides / args:
#   MANIFEST=<path>            (required)
#   CONDA_ENV=<env name/path>  (default: $PROJECT_ROOT/openr1)
#   BATCH_SIZE=<int>           (default: 1)
#   MAX_PROBLEMS=<int>         (optional; passed to backfill CLI)
#   MAX_ROWS=<int>             (optional; passed to backfill CLI)
#   FLUSH_EVERY=<int>          (optional; rewrite the JSONL every N backfilled rows, e.g. 8)
#   DRY_RUN=1                  (print commands only)
#
# Notes:
# - Shards by manifest line index: line_idx % world_size == rank
# - Each task runs sequentially through its assigned targets.

#SBATCH --job-name=math_backfill_p2_8gpu
#SBATCH --output=logs/math_backfill_p2_8gpu_%j.out
#SBATCH --error=logs/math_backfill_p2_8gpu_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00

set -euo pipefail
ulimit -n 4096

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
fi
cd "$PROJECT_ROOT"

source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
CONDA_ENV="${CONDA_ENV:-$PROJECT_ROOT/openr1}"
conda activate "$CONDA_ENV"
module load cudatoolkit/12.6

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$PROJECT_ROOT/logs"

MANIFEST="${MANIFEST:-}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_PROBLEMS="${MAX_PROBLEMS:-}"
MAX_ROWS="${MAX_ROWS:-}"
FLUSH_EVERY="${FLUSH_EVERY:-}"
DRY_RUN="${DRY_RUN:-0}"

for arg in "$@"; do
  case "$arg" in
    MANIFEST=*) MANIFEST="${arg#MANIFEST=}" ;;
    BATCH_SIZE=*) BATCH_SIZE="${arg#BATCH_SIZE=}" ;;
    MAX_PROBLEMS=*) MAX_PROBLEMS="${arg#MAX_PROBLEMS=}" ;;
    MAX_ROWS=*) MAX_ROWS="${arg#MAX_ROWS=}" ;;
    FLUSH_EVERY=*) FLUSH_EVERY="${arg#FLUSH_EVERY=}" ;;
    DRY_RUN=*) DRY_RUN="${arg#DRY_RUN=}" ;;
  esac
done

if [[ -z "$MANIFEST" ]]; then
  echo "Missing required: MANIFEST=<path>" >&2
  echo "Example:" >&2
  echo "  python scripts/inference/write_math_backfill_manifest.py --only_family Llama8B --out tmp/llama8b.tsv" >&2
  echo "  sbatch --gres gpu:a100:8 --ntasks 8 scripts/inference/math-backfill-pass2-manifest-parallel.slurm MANIFEST=tmp/llama8b.tsv BATCH_SIZE=1" >&2
  exit 2
fi

if [[ "$MANIFEST" != /* ]]; then
  MANIFEST="$PROJECT_ROOT/$MANIFEST"
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST" >&2
  exit 1
fi

export MANIFEST BATCH_SIZE MAX_PROBLEMS MAX_ROWS DRY_RUN
export FLUSH_EVERY

echo "→ Backfill pass2 (manifest-parallel)"
echo "   project_root: $PROJECT_ROOT"
echo "   manifest    : $MANIFEST"
echo "   ntasks      : ${SLURM_NTASKS:-unset}"
echo "   batch_size  : $BATCH_SIZE"
if [[ -n "$MAX_PROBLEMS" ]]; then echo "   max_problems: $MAX_PROBLEMS"; fi
if [[ -n "$MAX_ROWS" ]]; then echo "   max_rows    : $MAX_ROWS"; fi
if [[ -n "$FLUSH_EVERY" ]]; then echo "   flush_every : $FLUSH_EVERY"; fi
if [[ "$DRY_RUN" == "1" ]]; then echo "   DRY_RUN=1"; fi

# Some clusters (or user shells) end up with multiple SLURM_MEM_PER_* variables
# set simultaneously; `srun` treats those as mutually exclusive and aborts.
# If we detect multiple are set, unset them all for the job step.
mem_vars_set=0
for v in SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE; do
  if [[ -n "${!v:-}" ]]; then
    mem_vars_set=$((mem_vars_set + 1))
  fi
done
if (( mem_vars_set > 1 )); then
  unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE
fi

srun --label --kill-on-bad-exit=1 --gpus-per-task=1 --gpu-bind=single:1 \
  bash -lc '
    set -euo pipefail
    extra=()
    if [[ -n "${MAX_PROBLEMS:-}" ]]; then extra+=(--max_problems "${MAX_PROBLEMS}"); fi
    if [[ -n "${MAX_ROWS:-}" ]]; then extra+=(--max_rows "${MAX_ROWS}"); fi
    if [[ -n "${FLUSH_EVERY:-}" ]]; then extra+=(--flush_every "${FLUSH_EVERY}"); fi
    dry=()
    if [[ "${DRY_RUN:-0}" == "1" ]]; then dry+=(--dry_run); fi
    python scripts/inference/run_math_backfill_manifest_shard.py \
      --manifest "${MANIFEST}" \
      --rank "${SLURM_PROCID}" \
      --world_size "${SLURM_NTASKS}" \
      --project_root "." \
      --batch_size "${BATCH_SIZE}" \
      "${extra[@]}" \
      "${dry[@]}"
  '
