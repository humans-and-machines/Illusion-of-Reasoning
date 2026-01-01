#!/usr/bin/env bash
# Fill missing MATH-500 questions/samples for many result JSONLs using one multi-GPU job.
#
# This runs `src/inference/cli/unified_math.py --two_pass` on dataset chunks listed in a manifest.
# It is the non-array equivalent of `scripts/inference/math-fill-missing-questions-array.slurm`.
#
# Typical usage (focus one temp/step set):
#   python scripts/inference/plan_math_missing_questions.py \
#     --only_family Llama8B \
#     --only_temp 0.7 \
#     --only_steps 100,150,250,300,350,400,450 \
#     --chunk_size 25 \
#     --out_manifest tmp/llama8b_t07_fill_missing.tsv
#   sbatch --account mltheory --partition gpu --gres gpu:a100:8 --ntasks 8 --time 24:00:00 \
#     scripts/inference/math-fill-missing-questions-manifest-parallel.slurm \
#     MANIFEST=tmp/llama8b_t07_fill_missing.tsv
#
# Env overrides / args:
#   MANIFEST=<path>            (required)
#   CONDA_ENV=<env name/path>  (default: $PROJECT_ROOT/openr1)
#   DRY_RUN=1                  (print commands only)
#   NUM_SAMPLES=8              (passed to unified_math)
#   TOP_P=0.95                 (passed to unified_math)
#   DTYPE=float16              (passed to unified_math)
#   ENTROPY_MODE=full          (passed to unified_math)
#   ATTN_IMPLEMENTATION=sdpa   (passed to unified_math)
#   THINK_CAP=750              (passed to unified_math)
#   ANSWER_CAP=50              (passed to unified_math)
#   SECOND_PASS_PHRASE="..."   (passed to unified_math)

#SBATCH --job-name=math_fill_missing_8gpu
#SBATCH --output=logs/math_fill_missing_8gpu_%j.out
#SBATCH --error=logs/math_fill_missing_8gpu_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00

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
export TRANSFORMERS_NO_TORCHVISION=1
export TRANSFORMERS_NO_PYTORCH_IMAGE_TRANSFORMS=1

mkdir -p "$PROJECT_ROOT/logs"

MANIFEST="${MANIFEST:-}"
DRY_RUN="${DRY_RUN:-0}"

for arg in "$@"; do
  case "$arg" in
    MANIFEST=*) MANIFEST="${arg#MANIFEST=}" ;;
    DRY_RUN=*) DRY_RUN="${arg#DRY_RUN=}" ;;
  esac
done

if [[ -z "$MANIFEST" ]]; then
  echo "Missing required: MANIFEST=<path>" >&2
  exit 2
fi
if [[ "$MANIFEST" != /* ]]; then
  MANIFEST="$PROJECT_ROOT/$MANIFEST"
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST" >&2
  exit 1
fi

NUM_SAMPLES="${NUM_SAMPLES:-8}"
TOP_P="${TOP_P:-0.95}"
DTYPE="${DTYPE:-float16}"
ENTROPY_MODE="${ENTROPY_MODE:-full}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
THINK_CAP="${THINK_CAP:-750}"
ANSWER_CAP="${ANSWER_CAP:-50}"
SECOND_PASS_PHRASE="${SECOND_PASS_PHRASE:-Hold on, this reasoning might be wrong. Let's go back and check each step carefully. Actually, this approach doesn't look correct. Let's restart and work through the solution more systematically. Wait, we need to reconsider. Let's think this through step by step.}"

export MANIFEST NUM_SAMPLES TOP_P DTYPE ENTROPY_MODE ATTN_IMPLEMENTATION THINK_CAP ANSWER_CAP SECOND_PASS_PHRASE DRY_RUN

echo "→ Fill missing questions (manifest-parallel)"
echo "   project_root: $PROJECT_ROOT"
echo "   manifest    : $MANIFEST"
echo "   ntasks      : ${SLURM_NTASKS:-unset}"
echo "   num_samples : $NUM_SAMPLES"
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
    dry=()
    if [[ "${DRY_RUN:-0}" == "1" ]]; then dry+=(--dry_run); fi
    python scripts/inference/run_math_fill_missing_manifest_shard.py \
      --manifest "${MANIFEST}" \
      --rank "${SLURM_PROCID}" \
      --world_size "${SLURM_NTASKS}" \
      --project_root "." \
      --num_samples "${NUM_SAMPLES}" \
      --top_p "${TOP_P}" \
      --dtype "${DTYPE}" \
      --entropy_mode "${ENTROPY_MODE}" \
      --attn_implementation "${ATTN_IMPLEMENTATION}" \
      --think_cap "${THINK_CAP}" \
      --answer_cap "${ANSWER_CAP}" \
      --second_pass_phrase "${SECOND_PASS_PHRASE}" \
      "${dry[@]}"
  '
