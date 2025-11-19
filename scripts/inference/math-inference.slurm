#!/usr/bin/env bash
#SBATCH --job-name=infer_math_ckpts_local_4temps
#SBATCH --output=logs/infer_ckpts_local_%A_%a.out
#SBATCH --error=logs/infer_ckpts_local_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:59:00
#SBATCH --array=0-123   # 31 steps * 4 temps = 124 tasks (0..123). Add %N to cap concurrency if needed.

set -euo pipefail
ulimit -n 4096

# ── Repo root ────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT}"

export LOGLEVEL=DEBUG
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Conda env ───────────────────────────────────────────────
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-openr1}"
echo "✅ Conda env: $(which python)"; python --version

# Avoid user site pkgs
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PIP_DISABLE_PIP_VERSION_CHECK=1

# HF online (datasets ok)
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

# ── Paths ───────────────────────────────────────────────────
MODEL_ROOT="${MODEL_ROOT:-$PROJECT_ROOT/models/open-r1/Qwen2.5-1.5B-Open-R1-GRPO-math-v1}"
SCRIPT_PATH="${SCRIPT_PATH:-$PROJECT_ROOT/src/inference/unified_math_runner.py}"

# Steps and temps
CHECKPOINT_STEPS=(50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000 1050 1100 1150 1200 1250 1300 1350 1400 1450 1500 1550)
TEMPS=("0" "0.05" "0.3" "0.7")

NUM_STEPS=${#CHECKPOINT_STEPS[@]}
NUM_TEMPS=${#TEMPS[@]}

# Map array index → (step_idx, temp_idx)
TASK_ID=${SLURM_ARRAY_TASK_ID}
STEP_IDX=$(( TASK_ID / NUM_TEMPS ))
TEMP_IDX=$(( TASK_ID % NUM_TEMPS ))

# Safety: ignore overflow tasks if you tweak --array size
if (( STEP_IDX >= NUM_STEPS )); then
  echo "TASK ${TASK_ID}: step index ${STEP_IDX} out of range (NUM_STEPS=${NUM_STEPS}); exiting."
  exit 0
fi

STEP=${CHECKPOINT_STEPS[$STEP_IDX]}
TEMP=${TEMPS[$TEMP_IDX]}
CKPT_DIR="$MODEL_ROOT/checkpoint-$STEP"
test -d "$CKPT_DIR" || { echo "Missing $CKPT_DIR"; exit 1; }

# Map temperature → output root
temp_to_root () {
  case "$1" in
    0.7)  echo "$PROJECT_ROOT/results/GRPO-1.5B-math-temp-0.7" ;;
    0.3)  echo "$PROJECT_ROOT/results/GRPO-1.5B-low-temp" ;;
    0.05) echo "$PROJECT_ROOT/results/GRPO-1.5B-math-temp-0.05" ;;
    0|0.0) echo "$PROJECT_ROOT/results/GRPO-1.5B-math-temp-0.0" ;;
    *)    echo "$PROJECT_ROOT/results/GRPO-1.5B-math-temp-${1}" ;;
  esac
}
OUTPUT_ROOT="$(temp_to_root "$TEMP")"
OUTDIR="$OUTPUT_ROOT/step${STEP}"

# Per-task caches (unique per temp+step)
CACHEROOT="$OUTDIR/hf_cache"
mkdir -p "$OUTDIR" "$CACHEROOT" "$OUTDIR/.triton" "$OUTDIR/.torchinductor" "$OUTDIR/.tmp"
export HF_HOME="$CACHEROOT"
export TRANSFORMERS_CACHE="$CACHEROOT/transformers"
export HF_HUB_CACHE="$CACHEROOT/hub"
export TRITON_CACHE_DIR="$OUTDIR/.triton"
export TORCHINDUCTOR_CACHE_DIR="$OUTDIR/.torchinductor"
export TMPDIR="$OUTDIR/.tmp"

echo "→ step=${STEP} temp=${TEMP} → $OUTDIR"

python -u "$SCRIPT_PATH" \
  --model_name_or_path "$CKPT_DIR" \
  --output_dir "$OUTDIR" \
  --batch_size 1 \
  --entropy_mode full \
  --num_examples 1000 \
  --num_samples 8 \
  --temperature "$TEMP" \
  --top_p 0.95 \
  --seed 42 \
  --dtype bfloat16 \
  --dataset_id MATH-500 \
  --split test \
  --two_pass \
  --second_pass_phrase "Wait, we need to reconsider. Let's think this through step by step." \
  --think_cap 750 \
  --answer_cap 50 \
  --step "$STEP"

echo "✓ Done: step=${STEP} temp=${TEMP} → $OUTDIR"
