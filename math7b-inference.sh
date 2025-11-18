#!/usr/bin/env bash
#SBATCH --job-name=math_compare_qwen7b_4tempsXckpts
#SBATCH --output=logs/math_compare_qwen7b_%A_%a.out
#SBATCH --error=logs/math_compare_qwen7b_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:59:00
#SBATCH --array=0 # 4 temps * 15 ckpts

# Array mapping:
#   temps=(0 0.05 0.3 0.7)
#   ckpts=(0 50 100 150 200 250 300 350 400 450 500)
#   temp_idx = idx % 4, ckpt_idx = idx / 4

set -euo pipefail
ulimit -n 4096

export LOGLEVEL=DEBUG
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Conda env ───────────────────────────────────────────────────────────────
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
export ROOT_DIR="$PWD"
export ENV_NAME="openr1"
export ENV_DIR="$ROOT_DIR/$ENV_NAME"
conda activate "$ENV_DIR"
echo "✅ Conda env: $(which python)"
python --version

module load cudatoolkit/12.6

# Avoid user site pkgs
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PIP_DISABLE_PIP_VERSION_CHECK=1

# HF online (for base model)
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
SCRIPT_PATH="/n/fs/similarity/open-r1/math-inference.py"
MODEL_ROOT="/n/fs/similarity/open-r1/data/Qwen2.5-7B-Open-R1-GRPO-math-7b"
BASE_RESULTS_ROOT="/n/fs/similarity/open-r1/results2"

# Tags for directory naming (override if you want e.g. MODEL_TAG=1.5B DOMAIN_TAG=xword)
MODEL_TAG="${MODEL_TAG:-7B}"
DOMAIN_TAG="${DOMAIN_TAG:-math}"

# ── Grid ────────────────────────────────────────────────────────────────────
temps=(0.05)
ckpts=(500)

n_temps=${#temps[@]}
n_ckpts=${#ckpts[@]}
n_total=$(( n_temps * n_ckpts ))

# Safety: ensure array index in range
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= n_total )); then
  echo "Array index ${SLURM_ARRAY_TASK_ID} out of range (0..$((n_total-1)))"
  exit 1
fi

# Map array index -> (temp, ckpt)
idx="${SLURM_ARRAY_TASK_ID}"
temp_idx=$(( idx % n_temps ))
ckpt_idx=$(( idx / n_temps ))

TEMP="${temps[$temp_idx]}"
STEP="${ckpts[$ckpt_idx]}"

# Model selection
if [[ "$STEP" -eq 0 ]]; then
  MODEL_NAME_OR_PATH="Qwen/Qwen2.5-7B-Instruct"   # base
  TAG="base-step0"
  STEP_FLAG="--step 0"
else
  CKPT_DIR="$MODEL_ROOT/checkpoint-$STEP"
  test -d "$CKPT_DIR" || { echo "Missing tuned checkpoint: $CKPT_DIR"; exit 1; }
  MODEL_NAME_OR_PATH="$CKPT_DIR"
  TAG="tuned-step${STEP}"
  STEP_FLAG="--step ${STEP}"
fi

# Desired output style:
#   /.../results/GRPO-<MODEL_TAG>-<DOMAIN_TAG>-temp-<TEMP>/step-<STEP>
OUT_PREFIX="${BASE_RESULTS_ROOT}/GRPO-${MODEL_TAG}-${DOMAIN_TAG}-temp-${TEMP}"
OUTDIR="${OUT_PREFIX}/step-${STEP}"
mkdir -p "$OUTDIR"

# Per-run caches (avoid contention)
CACHEROOT="$OUTDIR/hf_cache"
mkdir -p "$CACHEROOT" "$OUTDIR/.triton" "$OUTDIR/.torchinductor" "$OUTDIR/.tmp"
export HF_HOME="$CACHEROOT"
export TRANSFORMERS_CACHE="$CACHEROOT/transformers"
export HF_HUB_CACHE="$CACHEROOT/hub"
export TRITON_CACHE_DIR="$OUTDIR/.triton"
export TORCHINDUCTOR_CACHE_DIR="$OUTDIR/.torchinductor"
export TMPDIR="$OUTDIR/.tmp"

echo "→ Task ${SLURM_ARRAY_TASK_ID}/${n_total}: TEMP=${TEMP}, STEP=${STEP}"
echo "→ Model:  $MODEL_NAME_OR_PATH"
echo "→ Output: $OUTDIR"

# ── Inference (1 sample, no second pass) ────────────────────────────────────
DTYPE="${DTYPE:-float16}"

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
  --dtype "$DTYPE" \
  --dataset_id MATH-500 \
  --split test \
  --two_pass \
  --second_pass_phrase "Wait, we need to reconsider. Let's think this through step by step." \
  --think_cap 750 \
  --answer_cap 50 \
  $STEP_FLAG

echo "✓ Done: ${TAG} → $OUTDIR"
