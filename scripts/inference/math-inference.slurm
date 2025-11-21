#!/usr/bin/env bash
#SBATCH --job-name=infer_math_step950_t0p05
#SBATCH --output=logs/infer_math_step950_t0p05_%j.out
#SBATCH --error=logs/infer_math_step950_t0p05_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --account=mltheory

set -euo pipefail
ulimit -n 4096

# ── Project root ─────────────────────────────────────────────────────────────
# Prefer the original submission directory under Slurm; fall back to $PWD.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
fi

export LOGLEVEL=DEBUG
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Conda env ───────────────────────────────────────────────
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
# Default to the project-local env created by `make install` unless CONDA_ENV is set explicitly.
CONDA_ENV="${CONDA_ENV:-$PROJECT_ROOT/openr1}"
conda activate "$CONDA_ENV"
echo "✅ Conda env: $(which python)"; python --version

# Avoid user site pkgs
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONPATH="$PROJECT_ROOT"

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
# Default to the explicit Qwen2.5-1.5B math checkpoint root; override via MODEL_ROOT if needed.
MODEL_ROOT="${MODEL_ROOT:-/n/fs/similarity/Illusion-of-Reasoning/artifacts/models/Qwen2.5-1.5B/Qwen2.5-1.5B-Open-R1-GRPO-math-v1}"
SCRIPT_PATH="${SCRIPT_PATH:-$PROJECT_ROOT/src/inference/unified_math_runner.py}"

# Single step and temperature
CHECKPOINT_STEPS=(950)
TEMPS=("0.05")

STEP=${CHECKPOINT_STEPS[0]}
TEMP=${TEMPS[0]}
CKPT_DIR="$MODEL_ROOT/checkpoint-$STEP"
test -d "$CKPT_DIR" || { echo "Missing $CKPT_DIR"; exit 1; }

# Map temperature → output root (under artifacts/results)
temp_to_root () {
  case "$1" in
    0.7)  echo "$PROJECT_ROOT/artifacts/results/GRPO-1.5B-math-temp-0.7-3" ;;
    0.3)  echo "$PROJECT_ROOT/artifacts/results/GRPO-1.5B-low-temp-3" ;;
    0.05) echo "$PROJECT_ROOT/artifacts/results/GRPO-1.5B-math-temp-0.05-3" ;;
    0|0.0) echo "$PROJECT_ROOT/artifacts/results/GRPO-1.5B-math-temp-0.0-3" ;;
    *)    echo "$PROJECT_ROOT/artifacts/results/GRPO-1.5B-math-temp-${1}-3" ;;
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

RECONSIDER_CUE="Hold on, this reasoning might be wrong. Let's go back and check each step carefully. ||| Actually, this approach doesn't look correct. Let's restart and work through the solution more systematically. ||| Wait, we need to reconsider. Let's think this through step by step."

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
  --second_pass_phrase "$RECONSIDER_CUE" \
  --think_cap 750 \
  --answer_cap 50 \
  --step "$STEP"

echo "✓ Done: step=${STEP} temp=${TEMP} → $OUTDIR"
