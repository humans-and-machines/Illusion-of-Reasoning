#!/usr/bin/env bash
# Experimental comparison script: 1-shot crossword baseline vs tuned checkpoint.
# Canonical full-sweep crossword inference lives in crossword-inference.slurm.
#SBATCH --job-name=xword_compare_qwen
#SBATCH --output=logs/xword_compare_qwen_%A_%a.out
#SBATCH --error=logs/xword_compare_qwen_%A_%a.err
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

# ── Conda env ───────────────────────────────────────────────
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-openr1}"
echo "✅ Conda: $(which python)"; python --version

# Optional CUDA module (use if your cluster needs it)
# module load cudatoolkit/12.6

# ── Env hygiene ─────────────────────────────────────────────
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PIP_DISABLE_PIP_VERSION_CHECK=1
export TRANSFORMERS_NO_TORCHVISION=1
export TRANSFORMERS_NO_PYTORCH_IMAGE_TRANSFORMS=1
export TOKENIZERS_PARALLELISM=false
export LOGLEVEL=DEBUG
export PYTHONUNBUFFERED=1

# ── Project-local caches ────────────────────────────────────
CACHE_ROOT="${CACHE_ROOT:-$PROJECT_ROOT/.caches}"
mkdir -p "$CACHE_ROOT"/{hf,xdg,torchinductor,triton} "$PROJECT_ROOT/.tmp" logs
export HF_HOME="$CACHE_ROOT/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MALLOC_ARENA_MAX=2

# WANDB
export WANDB_MODE=online
export WANDB_DIR=/n/fs/similarity/wandb-offload/tmp
export WANDB_ARTIFACT_DIR=/n/fs/similarity/wandb-offload/artifacts
export WANDB_CACHE_DIR=/n/fs/similarity/wandb-offload/cache
mkdir -p "$WANDB_DIR" "$WANDB_ARTIFACT_DIR" "$WANDB_CACHE_DIR"

# ── Paths ───────────────────────────────────────────────────
SCRIPT_PATH="${SCRIPT_PATH:-$PROJECT_ROOT/src/inference/unified_crossword_runner.py}"
DATA_JSONL="${DATA_JSONL:-$PROJECT_ROOT/data/data.jsonl}"  # your local eval set

# Tuned checkpoint root + step (override with env if needed)
MODEL_ROOT="${MODEL_ROOT:-$PROJECT_ROOT/artifacts/models/open-r1/Qwen2.5-1.5B-Open-R1-GRPO-Crosswords-v07}"
CKPT_STEP="${CKPT_STEP:-1000}"
CKPT_DIR="$MODEL_ROOT/checkpoint-$CKPT_STEP"

# Output (under artifacts/results)
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/artifacts/results/GRPO-1.5B-xword-compare-1shot}"
mkdir -p "$OUTPUT_ROOT"

# HF online so we can pull the base model
export TRANSFORMERS_OFFLINE=0
export HF_HUB_OFFLINE=0
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_REQUEST_TIMEOUT=60

# ── Select model (array: 0=base, 1=tuned@CKPT_STEP) ─────────
if [[ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]]; then
  MODEL_NAME_OR_PATH="Qwen/Qwen2.5-1.5B-Instruct"   # base model (no FT)
  TAG="base-step0"
  STEP_FLAG="--step 0"
else
  if [[ ! -d "$CKPT_DIR" ]]; then
    echo "Missing tuned checkpoint: $CKPT_DIR" >&2
    exit 1
  fi
  MODEL_NAME_OR_PATH="$CKPT_DIR"
  TAG="tuned-step${CKPT_STEP}"
  STEP_FLAG="--step ${CKPT_STEP}"
fi

OUTDIR="$OUTPUT_ROOT/$TAG"
mkdir -p "$OUTDIR"

echo "→ Running crossword compare: $TAG"
echo "   Model: $MODEL_NAME_OR_PATH"
echo "   Data : $DATA_JSONL"
echo "   Out  : $OUTDIR"

# ── Inference: 1 sample, NO second pass ─────────────────────
python -u "$SCRIPT_PATH" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --output_dir "$OUTDIR" \
  --batch_size 1 \
  --entropy_mode full \
  --num_examples 1000000 \
  --num_samples 1 \
  --temperature 0.0 \
  --top_p 0.95 \
  --seed 42 \
  --dtype bfloat16 \
  --dataset_id CROSSWORD-LOCAL \
  --dataset_path "$DATA_JSONL" \
  --split test \
  --think_cap 750 \
  --answer_cap 50 \
  $STEP_FLAG

echo "✓ Done: ${TAG} → $OUTDIR"
