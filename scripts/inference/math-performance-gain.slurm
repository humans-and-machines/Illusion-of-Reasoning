#!/usr/bin/env bash
# Experimental comparison script: 1-shot math baseline vs tuned checkpoint.
# Canonical full-sweep math inference lives in math-inference.slurm.
#SBATCH --job-name=math_compare_qwen
#SBATCH --output=logs/math_compare_qwen_%A_%a.out
#SBATCH --error=logs/math_compare_qwen_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:59:00
#SBATCH --array=0-1
set -euo pipefail
ulimit -n 4096

# ── Repo root ────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT}"

export LOGLEVEL=DEBUG
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# ── Conda env ───────────────────────────────────────────────
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
CONDA_ENV="${CONDA_ENV:-$PROJECT_ROOT/openr1}"
conda activate "$CONDA_ENV"
echo "✅ Conda env: $(which python)"
python --version

# Avoid user site pkgs
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PIP_DISABLE_PIP_VERSION_CHECK=1

# HF online (for base model)
export TRANSFORMERS_OFFLINE=0
export HF_HUB_OFFLINE=0
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_REQUEST_TIMEOUT=60

# Caches
export WANDB_MODE=online
export WANDB_DIR=/n/fs/similarity/wandb-offload/tmp
export WANDB_ARTIFACT_DIR=/n/fs/similarity/wandb-offload/artifacts
export WANDB_CACHE_DIR=/n/fs/similarity/wandb-offload/cache
export TMPDIR="${TMPDIR:-$PROJECT_ROOT/.tmp}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$PROJECT_ROOT/.torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PROJECT_ROOT/.triton}"
mkdir -p "$WANDB_DIR" "$WANDB_ARTIFACT_DIR" "$WANDB_CACHE_DIR" "$TMPDIR" logs "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" .hf_cache

# ── Paths ───────────────────────────────────────────────────
SCRIPT_PATH="${SCRIPT_PATH:-$PROJECT_ROOT/src/inference/unified_math_runner.py}"

# Tuned (local) root and step to compare
MODEL_ROOT="${MODEL_ROOT:-$PROJECT_ROOT/artifacts/models/open-r1/Qwen2.5-1.5B-Open-R1-GRPO-math-v1}"
CKPT_STEP=500
CKPT_DIR="$MODEL_ROOT/checkpoint-$CKPT_STEP"

# Output & HF caches (localized under artifacts/results)
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/artifacts/results/GRPO-1.5B-math-compare-1shot}"
CACHEROOT="$OUTPUT_ROOT/hf_cache"
mkdir -p "$OUTPUT_ROOT" "$CACHEROOT"
export HF_HOME="$CACHEROOT"
export TRANSFORMERS_CACHE="$CACHEROOT/transformers"
export HF_HUB_CACHE="$CACHEROOT/hub"

# ── Select model (array: 0=base, 1=tuned@1000) ─────────────
if [[ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]]; then
  MODEL_NAME_OR_PATH="Qwen/Qwen2.5-1.5B-Instruct"   # base, no finetuning
  TAG="base-step0"
  STEP_FLAG="--step 0"
else
  test -d "$CKPT_DIR" || { echo "Missing tuned checkpoint: $CKPT_DIR"; exit 1; }
  MODEL_NAME_OR_PATH="$CKPT_DIR"
  TAG="tuned-step${CKPT_STEP}"
  STEP_FLAG="--step ${CKPT_STEP}"
fi

OUTDIR="$OUTPUT_ROOT/$TAG"
mkdir -p "$OUTDIR"

echo "→ Running ${TAG}"
echo "→ Model: $MODEL_NAME_OR_PATH"
echo "→ Output: $OUTDIR"

# ── Inference (1 sample, no second pass) ────────────────────
python -u "$SCRIPT_PATH" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --output_dir "$OUTDIR" \
  --batch_size 1 \
  --entropy_mode full \
  --num_examples 500 \
  --num_samples 1 \
  --temperature 0.05 \
  --top_p 0.95 \
  --seed 42 \
  --dtype bfloat16 \
  --dataset_id MATH-500 \
  --split test \
  --think_cap 750 \
  --answer_cap 50 \
  $STEP_FLAG

echo "✓ Done: ${TAG} → $OUTDIR"
