#!/usr/bin/env bash
# Run unified_math on missing-question chunks (one chunk per Slurm array task).
#
# Required env:
#   MANIFEST=<tsv>  lines: family<TAB>temp<TAB>step<TAB>output_dir<TAB>dataset_start<TAB>num_examples<TAB>dataset_path
#
# Example:
#   python scripts/inference/plan_math_missing_questions.py --out_manifest tmp/m.tsv
#   N=$(wc -l < tmp/m.tsv)
#   sbatch --array=0-$((N-1))%2000 scripts/inference/math-fill-missing-questions-array.slurm MANIFEST=tmp/m.tsv
#   TWO_PASS=0 sbatch ...   # run pass1 only

#SBATCH --job-name=math_fill_missing_qs
#SBATCH --output=logs/math_fill_missing_qs_%A_%a.out
#SBATCH --error=logs/math_fill_missing_qs_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00

set -euo pipefail
ulimit -n 4096

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
fi
cd "$PROJECT_ROOT"

MANIFEST="${MANIFEST:-}"
# Allow passing MANIFEST as a positional arg for convenience, e.g.:
#   sbatch ... scripts/inference/math-fill-missing-questions-array.slurm MANIFEST=tmp/m.tsv
# or:
#   sbatch ... scripts/inference/math-fill-missing-questions-array.slurm tmp/m.tsv
if [[ -z "$MANIFEST" && $# -ge 1 ]]; then
  case "$1" in
    MANIFEST=*)
      MANIFEST="${1#MANIFEST=}"
      ;;
    *)
      MANIFEST="$1"
      ;;
  esac
fi
if [[ -z "$MANIFEST" || ! -f "$MANIFEST" ]]; then
  echo "Missing MANIFEST=<path> (file not found: $MANIFEST)"
  exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
LINE="$(sed -n "$((TASK_ID + 1))p" "$MANIFEST" || true)"
if [[ -z "$LINE" ]]; then
  echo "No manifest line for task $TASK_ID"
  exit 0
fi

IFS=$'\t' read -r FAMILY TEMP STEP OUTDIR DATASET_START NUM_EXAMPLES DATASET_PATH <<< "$LINE"
if [[ -z "${FAMILY:-}" || -z "${TEMP:-}" || -z "${STEP:-}" || -z "${OUTDIR:-}" || -z "${DATASET_START:-}" || -z "${NUM_EXAMPLES:-}" || -z "${DATASET_PATH:-}" ]]; then
  echo "Malformed manifest line: $LINE"
  exit 1
fi
if [[ ! -f "$DATASET_PATH" ]]; then
  echo "Dataset path not found: $DATASET_PATH"
  exit 1
fi

if [[ "${DEBUG_MANIFEST:-0}" == "1" ]]; then
  echo "manifest=$MANIFEST"
  echo "manifest_line=$((TASK_ID + 1))"
  echo "manifest_raw=$LINE"
fi

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

SCRIPT_PATH="$PROJECT_ROOT/src/inference/cli/unified_math.py"

NUM_SAMPLES="${NUM_SAMPLES:-8}"
TOP_P="${TOP_P:-0.95}"
DTYPE="${DTYPE:-float16}"
BATCH_SIZE="${BATCH_SIZE:-$NUM_EXAMPLES}"

THINK_CAP="${THINK_CAP:-750}"
ANSWER_CAP="${ANSWER_CAP:-50}"
ENTROPY_MODE="${ENTROPY_MODE:-full}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

RECONSIDER_CUE="${SECOND_PASS_PHRASE:-Hold on, this reasoning might be wrong. Let's go back and check each step carefully. Actually, this approach doesn't look correct. Let's restart and work through the solution more systematically. Wait, we need to reconsider. Let's think this through step by step.}"
TWO_PASS="${TWO_PASS:-1}"

SECOND_PASS_ARGS=()
if [[ "$TWO_PASS" == "1" ]]; then
  SECOND_PASS_ARGS=(--two_pass --second_pass_phrase "$RECONSIDER_CUE")
fi

BASE_MODEL=""
TOKENIZER_PATH=""
CKPT_ROOT=""

case "$FAMILY" in
  1.5B)
    BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
    TOKENIZER_PATH="Qwen/Qwen2.5-1.5B-Instruct"
    CKPT_ROOT="$PROJECT_ROOT/artifacts/models/Qwen2.5-1.5B/Qwen2.5-1.5B-Open-R1-GRPO-math-v1"
    ;;
  7B)
    BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
    TOKENIZER_PATH="Qwen/Qwen2.5-7B-Instruct"
    CKPT_ROOT="$PROJECT_ROOT/artifacts/models/Qwen2.5-7B-Open-R1-GRPO-math-7b"
    ;;
  Llama8B)
    BASE_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
    TOKENIZER_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct"
    CKPT_ROOT="$PROJECT_ROOT/artifacts/models/open-r1/Llama-8B-Open-R1-GRPO-math-v2"
    ;;
  *)
    echo "Unknown FAMILY=$FAMILY"
    exit 1
    ;;
esac

MODEL_NAME_OR_PATH="$BASE_MODEL"
if [[ "$STEP" != "0" ]]; then
  CKPT_DIR="$CKPT_ROOT/checkpoint-$STEP"
  if [[ ! -d "$CKPT_DIR" ]]; then
    echo "Missing checkpoint dir: $CKPT_DIR"
    exit 1
  fi
  MODEL_NAME_OR_PATH="$CKPT_DIR"
fi

mkdir -p "$OUTDIR"

CACHEROOT="$OUTDIR/hf_cache"
mkdir -p "$CACHEROOT" "$OUTDIR/.triton" "$OUTDIR/.torchinductor" "$OUTDIR/.tmp"
export HF_HOME="$CACHEROOT"
export TRANSFORMERS_CACHE="$CACHEROOT/transformers"
export HF_HUB_CACHE="$CACHEROOT/hub"
export TRITON_CACHE_DIR="$OUTDIR/.triton"
export TORCHINDUCTOR_CACHE_DIR="$OUTDIR/.torchinductor"
export TMPDIR="$OUTDIR/.tmp"

echo "→ fill missing questions | family=$FAMILY temp=$TEMP step=$STEP"
echo "   outdir=$OUTDIR start=$DATASET_START num_examples=$NUM_EXAMPLES num_samples=$NUM_SAMPLES"
echo "   model=$MODEL_NAME_OR_PATH"

python -u "$SCRIPT_PATH" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --output_dir "$OUTDIR" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --attn_implementation "$ATTN_IMPLEMENTATION" \
  --batch_size "$BATCH_SIZE" \
  --entropy_mode "$ENTROPY_MODE" \
  --num_examples "$NUM_EXAMPLES" \
  --dataset_start "$DATASET_START" \
  --num_samples "$NUM_SAMPLES" \
  --temperature "$TEMP" \
  --top_p "$TOP_P" \
  --seed 42 \
  --dtype "$DTYPE" \
  --dataset_id MATH-500 \
  --dataset_path "$DATASET_PATH" \
  --split test \
  "${SECOND_PASS_ARGS[@]}" \
  --think_cap "$THINK_CAP" \
  --answer_cap "$ANSWER_CAP" \
  --step "$STEP"

echo "✓ Done task=$TASK_ID"
