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

# ── Project root ─────────────────────────────────────────────────────────────
# Use Slurm submission directory when available; otherwise fall back to $PWD.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
fi

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
export PYTHONPATH="$PROJECT_ROOT"
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

# Localized compute caches (force inside repo; ignore external CACHE_ROOT)
CACHE_ROOT="$PROJECT_ROOT/.caches"
mkdir -p "$CACHE_ROOT"/{torchinductor,triton} "$PROJECT_ROOT/.tmp" logs
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TMPDIR="$PROJECT_ROOT/.tmp"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Ensure W&B dirs exist on /n/fs
mkdir -p "$WANDB_DIR" "$WANDB_ARTIFACT_DIR" "$WANDB_CACHE_DIR"

# ── Reconsider cue (match main sweep) ──────────────────────
RECONSIDER_CUE="Hold on, this reasoning might be wrong. Let's go back and check each step carefully. ||| Actually, this approach doesn't look correct. Let's restart and work through the solution more systematically. ||| Wait, something is not right, we need to reconsider. Let's think this through step by step."

# ── Paths ───────────────────────────────────────────────────
SCRIPT_PATH="${SCRIPT_PATH:-$PROJECT_ROOT/src/inference/cli/unified_math.py}"

# Tuned (local) root and step to compare
# Use the same default checkpoint root as math-inference.slurm
MODEL_ROOT="${MODEL_ROOT:-$PROJECT_ROOT/artifacts/models/Qwen2.5-1.5B/Qwen2.5-1.5B-Open-R1-GRPO-math-v1}"
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

# Allow overriding temperature + output root per run so we can match the
# main sweep's temp grid and folder layout.
BASE_TEMP="${BASE_TEMP:-0.05}"

# If OUTPUT_ROOT is unset, default to the temp-specific GRPO layout so
# base-step0 lands alongside other temps (e.g., GRPO-1.5B-math-temp-0.0).
if [[ -z "${OUTPUT_ROOT:-}" ]]; then
  case "$BASE_TEMP" in
    0|0.0)  OUTPUT_ROOT="$PROJECT_ROOT/artifacts/results/GRPO-1.5B-math-temp-0.0" ;;
    0.05)   OUTPUT_ROOT="$PROJECT_ROOT/artifacts/results/GRPO-1.5B-math-temp-0.05" ;;
    0.3)    OUTPUT_ROOT="$PROJECT_ROOT/artifacts/results/GRPO-1.5B-math-temp-0.3" ;;
    0.7)    OUTPUT_ROOT="$PROJECT_ROOT/artifacts/results/GRPO-1.5B-math-temp-0.7" ;;
    *)      OUTPUT_ROOT="$PROJECT_ROOT/artifacts/results/GRPO-1.5B-math-temp-${BASE_TEMP}" ;;
  esac
fi

OUTDIR="$OUTPUT_ROOT/$TAG"
mkdir -p "$OUTDIR"

echo "→ Running ${TAG}"
echo "→ Model: $MODEL_NAME_OR_PATH"
echo "→ Output: $OUTDIR"

# Per-run dataset slice (match main sweep semantics)
NUM_EXAMPLES="${NUM_EXAMPLES:-500}"
DATASET_START="${DATASET_START:-0}"

# ── Inference (8 samples, two-pass) ─────────────────────────
python -u "$SCRIPT_PATH" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --output_dir "$OUTDIR" \
  --batch_size 1 \
  --entropy_mode full \
  --num_examples "$NUM_EXAMPLES" \
  --dataset_start "$DATASET_START" \
  --num_samples 8 \
  --temperature "$BASE_TEMP" \
  --top_p 0.95 \
  --seed 42 \
  --dtype bfloat16 \
  --dataset_id MATH-500 \
  --split test \
  --two_pass \
  --second_pass_phrase "$RECONSIDER_CUE" \
  --think_cap 750 \
  --answer_cap 50 \
  $STEP_FLAG

echo "✓ Done: ${TAG} → $OUTDIR"
