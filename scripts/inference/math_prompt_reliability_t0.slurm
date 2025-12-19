#!/usr/bin/env bash
# Qwen-1.5B MATH-500 prompt-reliability sweep at T=0.
# Runs 4 epochs × 5 prompt variants in chunks of questions and writes:
#   artifacts/results/prompt_reliability/epoch{0..3}/variant{1..5}/step{STEP}/stepXXXX_test.jsonl
# Each Slurm array task processes CHUNK_SIZE questions (default: 5).

# Usage (from repo root, with defaults: 500 test questions, chunks of 5):
#   sbatch --array=0-1999%20 scripts/inference/math_prompt_reliability_t0.slurm

#SBATCH --job-name=math_prompt_reliab_t0
#SBATCH --output=logs/math_prompt_reliab_t0_%A_%a.out
#SBATCH --error=logs/math_prompt_reliab_t0_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --array=0-1999

set -euo pipefail
ulimit -n 4096

# ── Project root ─────────────────────────────────────────────────────────────
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

# HF online
export TRANSFORMERS_OFFLINE=0
export HF_HUB_OFFLINE=0
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_REQUEST_TIMEOUT=60

# W&B (optional)
export WANDB_MODE=online
export WANDB_DIR=/n/fs/similarity/wandb-offload/tmp
export WANDB_ARTIFACT_DIR=/n/fs/similarity/wandb-offload/artifacts
export WANDB_CACHE_DIR=/n/fs/similarity/wandb-offload/cache
mkdir -p "$WANDB_DIR" "$WANDB_ARTIFACT_DIR" "$WANDB_CACHE_DIR" logs

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_PATH="${SCRIPT_PATH:-$PROJECT_ROOT/src/inference/cli/unified_math.py}"
MODEL_ROOT="${MODEL_ROOT:-$PROJECT_ROOT/artifacts/models/open-r1/Qwen2.5-1.5B-Open-R1-GRPO-math-v1}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

# Epochs and variants
# Epoch 0 = pre (base model), 1 = ckpt 500, 2 = ckpt 1000, 3 = final (1500 by default).
EPOCH_STEPS=(0 500 1000 1500)
NUM_EPOCHS=${#EPOCH_STEPS[@]}
VARIANTS=(1 2 3 4 5)
NUM_VARIANTS=${#VARIANTS[@]}

# Chunking over the 500-example test split
CHUNK_SIZE="${CHUNK_SIZE:-5}"
TOTAL_EXAMPLES="${TOTAL_EXAMPLES:-500}"
NUM_CHUNKS="${NUM_CHUNKS:-$(( (TOTAL_EXAMPLES + CHUNK_SIZE - 1) / CHUNK_SIZE ))}"

TASK="${SLURM_ARRAY_TASK_ID:-0}"
TOTAL_TASKS=$(( NUM_EPOCHS * NUM_VARIANTS * NUM_CHUNKS ))
if (( TASK < 0 || TASK >= TOTAL_TASKS )); then
  echo "TASK ${TASK} outside 0..$((TOTAL_TASKS - 1)); exiting."
  exit 0
fi

EPOCH_BLOCK=$(( NUM_VARIANTS * NUM_CHUNKS ))
EPOCH_IDX=$(( TASK / EPOCH_BLOCK ))
REMAINDER=$(( TASK % EPOCH_BLOCK ))
VARIANT_IDX=$(( REMAINDER / NUM_CHUNKS ))
CHUNK_IDX=$(( REMAINDER % NUM_CHUNKS ))

STEP="${EPOCH_STEPS[$EPOCH_IDX]}"
VARIANT="${VARIANTS[$VARIANT_IDX]}"

# Select model: base for epoch 0, checkpoint for others
if [[ "$STEP" -eq 0 ]]; then
  MODEL_NAME_OR_PATH="$BASE_MODEL"
else
  CKPT_DIR="$MODEL_ROOT/checkpoint-$STEP"
  if [[ ! -d "$CKPT_DIR" ]]; then
    echo "Missing tuned checkpoint: $CKPT_DIR"
    exit 1
  fi
  MODEL_NAME_OR_PATH="$CKPT_DIR"
fi

TEMP=0.0

# Dataset slice for this chunk
DATASET_START=$(( CHUNK_IDX * CHUNK_SIZE ))
NUM_EXAMPLES="$CHUNK_SIZE"

# Structured output root:
#   artifacts/results/prompt_reliability/epoch{E}/variant{V}/step{STEP}/stepXXXX_test.jsonl
OUTPUT_ROOT="$PROJECT_ROOT/artifacts/results/prompt_reliability/epoch${EPOCH_IDX}/variant${VARIANT}"
OUTDIR="$OUTPUT_ROOT/step${STEP}"

# Per-task caches
CACHEROOT="$OUTDIR/hf_cache"
mkdir -p "$OUTDIR" "$CACHEROOT" "$OUTDIR/.triton" "$OUTDIR/.torchinductor" "$OUTDIR/.tmp"
export HF_HOME="$CACHEROOT"
export TRANSFORMERS_CACHE="$CACHEROOT/transformers"
export HF_HUB_CACHE="$CACHEROOT/hub"
export TRITON_CACHE_DIR="$OUTDIR/.triton"
export TORCHINDUCTOR_CACHE_DIR="$OUTDIR/.torchinductor"
export TMPDIR="$OUTDIR/.tmp"

# Second-pass cue (same as main math runs)
RECONSIDER_CUE="Hold on, this reasoning might be wrong. Let's go back and check each step carefully. ||| Actually, this approach doesn't look correct. Let's restart and work through the solution more systematically. ||| Wait, something is not right, we need to reconsider. Let's think this through step by step."

# Expose variant ID so downstream code can select the appropriate TaskSpec / system prompt.
export PROMPT_VARIANT_ID="$VARIANT"

echo "→ epoch=${EPOCH_IDX} step=${STEP} variant=${VARIANT} temp=${TEMP} chunk=${CHUNK_IDX}/${NUM_CHUNKS} start=${DATASET_START} size=${NUM_EXAMPLES}"
echo "   model : $MODEL_NAME_OR_PATH"
echo "   outdir: $OUTDIR"

python -u "$SCRIPT_PATH" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --output_dir "$OUTDIR" \
  --batch_size 1 \
  --entropy_mode full \
  --num_examples "$NUM_EXAMPLES" \
  --dataset_start "$DATASET_START" \
  --num_samples 8 \
  --temperature "$TEMP" \
  --top_p 0.95 \
  --seed 42 \
  --dtype bfloat16 \
  --dataset_id MATH-500 \
  --split test \
  --two_pass \
  --second_pass_phrase "$RECONSIDER_CUE" \
  --think_cap 750 \
  --answer_cap 50 \
  --step "$STEP"

RESULTS_BASENAME=$(printf "step%04d_test.jsonl" "$STEP")
echo "✓ Done: epoch=${EPOCH_IDX} step=${STEP} variant=${VARIANT} → $OUTDIR/${RESULTS_BASENAME}"
