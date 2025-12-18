#!/usr/bin/env bash
# Step-0 Rush Hour (car-park) baseline in 100-problem chunks across 4 temperatures.
# Uses the unified_carpark runner, matching env setup from carpark-inference.slurm.
#
# Usage (from repo root):
#   CONDA_ENV="$OPENR1_ENV" sbatch scripts/inference/carpark-step0-25.slurm

#SBATCH --job-name=carpark_step0_25
#SBATCH --output=logs/carpark_step0_25_%A_%a.out
#SBATCH --error=logs/carpark_step0_25_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:59:00
# 4 temps × 5 chunks (500 / 100) = 20 tasks → indices 0..19
# You can cap concurrency with e.g. 0-19%4 if needed.
#SBATCH --array=0-19

set -euo pipefail
ulimit -n 4096

# ── Project root ─────────────────────────────────────────────────────────────
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
fi

# ── Conda env ───────────────────────────────────────────────
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
CONDA_ENV="${CONDA_ENV:-$PROJECT_ROOT/openr1}"
conda activate "$CONDA_ENV"
module load cudatoolkit/12.6

# ── Env hygiene ──────────────────────────────────────────────
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export TRANSFORMERS_NO_TORCHVISION=1
export TRANSFORMERS_NO_PYTORCH_IMAGE_TRANSFORMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MALLOC_ARENA_MAX=2

# ── Paths & dataset ─────────────────────────────────────────
SCRIPT_PATH="${SCRIPT_PATH:-$PROJECT_ROOT/src/inference/cli/unified_carpark.py}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"

DATASET_ID="${DATASET_ID:-od2961/rush4-5-6-balanced}"
DATASET_PROMPT_COL="${DATASET_PROMPT_COL:-messages}"
DATASET_SOLUTION_COL="${DATASET_SOLUTION_COL:-solution}"

TEMPS=("0" "0.05" "0.3" "0.7")
# Allow CHUNK_SIZE / NUM_CHUNKS to be overridden via env, defaulting to 5×100=500.
CHUNK_SIZE="${CHUNK_SIZE:-100}"
NUM_CHUNKS="${NUM_CHUNKS:-5}"  # 500 / 100 by default

TASK=${SLURM_ARRAY_TASK_ID}
TEMP_IDX=$(( TASK / NUM_CHUNKS ))
CHUNK_IDX=$(( TASK % NUM_CHUNKS ))

if (( TEMP_IDX < 0 || TEMP_IDX >= ${#TEMPS[@]} )); then
  echo "TASK $TASK beyond temp range; exiting."
  exit 0
fi

TEMP="${TEMPS[$TEMP_IDX]}"
DATASET_START=$(( CHUNK_IDX * CHUNK_SIZE ))

OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/artifacts/results/GRPO-1.5B-carpark-temp-${TEMP}}"
OUTDIR="$OUTPUT_ROOT/base-step0"
mkdir -p "$OUTDIR"

RECONSIDER_CUE="Hold on, this reasoning might be wrong. Let's go back and check each step carefully. Actually, this approach doesn't look correct. Let's restart and work through the solution more systematically. Wait, we need to reconsider. Let's think this through step by step."

echo "→ carpark step=0 temp=$TEMP chunk=$CHUNK_IDX (start=$DATASET_START)"
echo "   model: $MODEL_NAME_OR_PATH"
echo "   data : $DATASET_ID"
echo "   out  : $OUTDIR"

python -u "$SCRIPT_PATH" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --output_dir "$OUTDIR" \
  --batch_size 1 \
  --entropy_mode full \
  --num_examples "$CHUNK_SIZE" \
  --dataset_start "$DATASET_START" \
  --num_samples 8 \
  --temperature "$TEMP" \
  --top_p 0.95 \
  --seed 42 \
  --dtype bfloat16 \
  --dataset_id "$DATASET_ID" \
  --dataset_prompt_column "$DATASET_PROMPT_COL" \
  --dataset_solution_column "$DATASET_SOLUTION_COL" \
  --split test \
  --two_pass \
  --second_pass_phrase "$RECONSIDER_CUE" \
  --think_cap 750 \
  --answer_cap 50 \
  --step 0

echo "✓ Done: carpark step=0 temp=$TEMP chunk=$CHUNK_IDX → $OUTDIR"
