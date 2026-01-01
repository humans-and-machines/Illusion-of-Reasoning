#!/usr/bin/env bash
# Resume/fill the temp=0.7 Llama-8B math sweep at steps 250 and 300 in
# single-problem chunks. Existing progress:
#   step-250: problems 0-93 done → run array 94-499
#   step-300: problems 0-88 done → run array 89-499
# This script hardcodes CHUNK_SIZE=1 and the corresponding array ranges:
#   SLURM_ARRAY_TASK_ID mapping: step_idx = idx / NUM_CHUNKS, chunk_idx = idx % NUM_CHUNKS
#   Array indices: 94-499 (step 250) and 589-999 (step 300)
#
# Usage (from repo root):
#   sbatch scripts/inference/math8b-temp07-step250-300-single.slurm

#SBATCH --job-name=math8b_fill_t07_s250_300
#SBATCH --output=logs/math8b_fill_t07_s250_300_%A_%a.out
#SBATCH --error=logs/math8b_fill_t07_s250_300_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --account=mltheory
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:59:00
# 500 problems * 2 steps = 1000 tasks; we only run the missing ranges below.
#SBATCH --array=94-499,589-999

set -euo pipefail
ulimit -n 4096

# ── Project root ─────────────────────────────────────────────────────────────
# Prefer the Slurm submission directory so the script can be staged in /var/spool.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
fi

export LOGLEVEL=DEBUG
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Conda env ───────────────────────────────────────────────────────────────
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
CONDA_ENV="${CONDA_ENV:-$PROJECT_ROOT/openr1}"
conda activate "$CONDA_ENV"
module load cudatoolkit/12.6
echo "✅ Conda env: $(which python)"; python --version

# Avoid user site pkgs
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT"
export PIP_DISABLE_PIP_VERSION_CHECK=1

# HF online (base model pull + datasets)
export TRANSFORMERS_OFFLINE=0
export HF_HUB_OFFLINE=0
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_REQUEST_TIMEOUT=60
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-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}"

# W&B (optional)
export WANDB_MODE=online
export WANDB_DIR=/n/fs/similarity/wandb-offload/tmp
export WANDB_ARTIFACT_DIR=/n/fs/similarity/wandb-offload/artifacts
export WANDB_CACHE_DIR=/n/fs/similarity/wandb-offload/cache
mkdir -p "$WANDB_DIR" "$WANDB_ARTIFACT_DIR" "$WANDB_CACHE_DIR" logs

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_PATH="${SCRIPT_PATH:-$PROJECT_ROOT/src/inference/cli/unified_math.py}"
MODEL_ROOT="${MODEL_ROOT:-$PROJECT_ROOT/artifacts/models/open-r1/Llama-8B-Open-R1-GRPO-math-v2}"
TOKENIZER_PATH="${TOKENIZER_PATH:-meta-llama/Meta-Llama-3.1-8B-Instruct}"

# Target steps
STEPS=(${STEPS:-250 300})
N_STEPS=${#STEPS[@]}

# Single-problem tasks; keep aligned with the array ranges above.
CHUNK_SIZE="${CHUNK_SIZE:-1}"
if [[ "$CHUNK_SIZE" -ne 1 ]]; then
  echo "This script assumes CHUNK_SIZE=1; adjust --array if you override it."
  exit 1
fi
TOTAL_EXAMPLES="${TOTAL_EXAMPLES:-500}"
NUM_CHUNKS="${NUM_CHUNKS:-$(( (TOTAL_EXAMPLES + CHUNK_SIZE - 1) / CHUNK_SIZE ))}"
TOTAL_TASKS=$(( N_STEPS * NUM_CHUNKS ))

TASK="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK < 0 || TASK >= TOTAL_TASKS )); then
  echo "TASK ${TASK} outside 0..$((TOTAL_TASKS - 1)); exiting."
  exit 0
fi

STEP_IDX=$(( TASK / NUM_CHUNKS ))
CHUNK_IDX=$(( TASK % NUM_CHUNKS ))
STEP="${STEPS[$STEP_IDX]}"

if [[ "$STEP" -eq 0 ]]; then
  MODEL_NAME_OR_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct"
else
  CKPT_DIR="$MODEL_ROOT/checkpoint-$STEP"
  if [[ ! -d "$CKPT_DIR" ]]; then
    echo "Missing tuned checkpoint: $CKPT_DIR"
    exit 1
  fi
  MODEL_NAME_OR_PATH="$CKPT_DIR"
fi

TEMP="${TEMP:-0.7}"
OUTPUT_ROOT="${INFER_OUTPUT_ROOT:-$PROJECT_ROOT/artifacts/results/GRPO-Llama8B-math-temp-0.7}"
OUTDIR="$OUTPUT_ROOT/step-${STEP}"

# Per-task caches (under outdir to localize HF/Trtion/TI)
CACHEROOT="$OUTDIR/hf_cache"
mkdir -p "$OUTDIR" "$CACHEROOT" "$OUTDIR/.triton" "$OUTDIR/.torchinductor" "$OUTDIR/.tmp"
export HF_HOME="$CACHEROOT"
export TRANSFORMERS_CACHE="$CACHEROOT/transformers"
export HF_HUB_CACHE="$CACHEROOT/hub"
export TRITON_CACHE_DIR="$OUTDIR/.triton"
export TORCHINDUCTOR_CACHE_DIR="$OUTDIR/.torchinductor"
export TMPDIR="$OUTDIR/.tmp"

DATASET_START=$(( CHUNK_IDX * CHUNK_SIZE ))
BATCH_SIZE="${BATCH_SIZE:-$CHUNK_SIZE}"

RECONSIDER_CUE="Hold on, this reasoning might be wrong. Let's go back and check each step carefully. Actually, this approach doesn't look correct. Let's restart and work through the solution more systematically. Wait, we need to reconsider. Let's think this through step by step."

echo "→ step=${STEP} temp=${TEMP} chunk=${CHUNK_IDX}/${NUM_CHUNKS} (start=${DATASET_START} size=${CHUNK_SIZE})"
echo "   model : $MODEL_NAME_OR_PATH"
echo "   outdir: $OUTDIR"

python -u "$SCRIPT_PATH" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --output_dir "$OUTDIR" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --batch_size "$BATCH_SIZE" \
  --entropy_mode full \
  --num_examples "$CHUNK_SIZE" \
  --dataset_start "$DATASET_START" \
  --num_samples "${NUM_SAMPLES:-8}" \
  --temperature "$TEMP" \
  --top_p 0.95 \
  --seed 42 \
  --dtype "${DTYPE:-float16}" \
  --dataset_id MATH-500 \
  --split test \
  --two_pass \
  --second_pass_phrase "$RECONSIDER_CUE" \
  --think_cap 750 \
  --answer_cap 50 \
  --step "$STEP"

RESULTS_BASENAME=$(printf "step%04d_test.jsonl" "$STEP")
echo "✓ Done: step=${STEP} temp=${TEMP} chunk=${CHUNK_IDX} → $OUTDIR/${RESULTS_BASENAME}"
