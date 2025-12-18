#!/usr/bin/env bash
# Step-0 crossword baseline on local Guardian split, all temps, 8 samples × 130.
# Uses the same unified_crossword runner as the main sweep.
# Requires: DATASET_PATH pointing to the 130-problem JSONL.
#
# Example submission (from repo root):
#   CONDA_ENV="$OPENR1_ENV" \
#   DATASET_PATH="$PWD/data/crossword/guardian_cryptonite_test.jsonl" \
#   sbatch scripts/inference/crossword-step0-local.slurm

#SBATCH --job-name=xword_step0_local
#SBATCH --output=logs/xword_step0_local_%A_%a.out
#SBATCH --error=logs/xword_step0_local_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:59:00
#SBATCH --array=0-3

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
echo "✅ Conda: $(which python)"; python --version

# Optional CUDA module if your cluster needs it
module load cudatoolkit/12.6

# ── Env hygiene ─────────────────────────────────────────────
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export TRANSFORMERS_NO_TORCHVISION=1
export TRANSFORMERS_NO_PYTORCH_IMAGE_TRANSFORMS=1
export TOKENIZERS_PARALLELISM=false
export LOGLEVEL=DEBUG
export PYTHONUNBUFFERED=1

# ── Project-local caches (per-temp, under artifacts/results) ────────────────
SCRIPT_PATH="${SCRIPT_PATH:-$PROJECT_ROOT/src/inference/cli/unified_crossword.py}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"

TEMPS=("0" "0.05" "0.3" "0.7")
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#TEMPS[@]} )); then
  echo "TASK ${SLURM_ARRAY_TASK_ID} beyond temps; exiting."
  exit 0
fi
TEMP="${TEMPS[$SLURM_ARRAY_TASK_ID]}"

DATASET_ID="CROSSWORD-LOCAL"
DATASET_PATH="${DATASET_PATH:-}"
if [[ -z "$DATASET_PATH" ]]; then
  echo "ERROR: DATASET_PATH is required for CROSSWORD-LOCAL." >&2
  exit 1
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/artifacts/results/GRPO-1.5B-xword-temp-${TEMP}}"
OUTDIR="$OUTPUT_ROOT/base-step0"
mkdir -p "$OUTDIR"

CACHEROOT="$OUTDIR/hf_cache"
mkdir -p "$CACHEROOT" "$OUTDIR/.triton" "$OUTDIR/.torchinductor" "$OUTDIR/.tmp"
export HF_HOME="$CACHEROOT"
export TRANSFORMERS_CACHE="$CACHEROOT/transformers"
export HF_HUB_CACHE="$CACHEROOT/hub"
export TRITON_CACHE_DIR="$OUTDIR/.triton"
export TORCHINDUCTOR_CACHE_DIR="$OUTDIR/.torchinductor"
export TMPDIR="$OUTDIR/.tmp"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MALLOC_ARENA_MAX=2

RECONSIDER_CUE="Hold on, this reasoning might be wrong. Let's go back and check each step carefully. Actually, this approach doesn't look correct. Let's restart and work through the solution more systematically. Wait, we need to reconsider. Let's think this through step by step."

echo "→ step=0 temp=$TEMP"
echo "   model : $MODEL_NAME_OR_PATH"
echo "   data  : $DATASET_ID @ $DATASET_PATH"
echo "   out   : $OUTDIR"

python -u "$SCRIPT_PATH" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --output_dir "$OUTDIR" \
  --batch_size 1 \
  --entropy_mode full \
  --num_examples 1000000 \
  --num_samples 8 \
  --temperature "$TEMP" \
  --top_p 0.95 \
  --seed 42 \
  --dtype bfloat16 \
  --dataset_id "$DATASET_ID" \
  --dataset_path "$DATASET_PATH" \
  --split test \
  --two_pass \
  --second_pass_phrase "$RECONSIDER_CUE" \
  --think_cap 750 \
  --answer_cap 50 \
  --step 0

echo "✓ Done: step=0 temp=$TEMP → $OUTDIR"

