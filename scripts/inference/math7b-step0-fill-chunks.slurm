#!/usr/bin/env bash
# Resume/fill the temp=0.7 step=0 math sweep (7B) in 5-problem chunks.
# Existing results file:
#   artifacts/results/GRPO-7B-math-temp-0.7/step-0/step0000_test.jsonl
# That file currently has 3896/4000 rows (MATH-500 × 8 samples), so the
# unified runner's resume logic will only generate the missing samples.
#
# Usage (from repo root):
#   sbatch scripts/inference/math7b-step0-fill-chunks.slurm
#   CHUNK_SIZE=5 NUM_SAMPLES=8 sbatch ...   # optional overrides

#SBATCH --job-name=math7b_step0_fill_t07
#SBATCH --output=logs/math7b_step0_fill_t07_%A_%a.out
#SBATCH --error=logs/math7b_step0_fill_t07_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:59:00
# Default: 500 problems / CHUNK_SIZE=5 → 100 array tasks.
# Adjust CHUNK_SIZE/array bounds if desired.
#SBATCH --array=0-99

set -euo pipefail
ulimit -n 4096

# ── Project root ─────────────────────────────────────────────────────────────
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
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
# Allow override; fall back to same default token used elsewhere.
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-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}"

# W&B (optional)
export WANDB_MODE=online
export WANDB_DIR=/n/fs/similarity/wandb-offload/tmp
export WANDB_ARTIFACT_DIR=/n/fs/similarity/wandb-offload/artifacts
export WANDB_CACHE_DIR=/n/fs/similarity/wandb-offload/cache
mkdir -p "$WANDB_DIR" "$WANDB_ARTIFACT_DIR" "$WANDB_CACHE_DIR" logs

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_PATH="${SCRIPT_PATH:-$PROJECT_ROOT/src/inference/cli/unified_math.py}"
MODEL_ROOT="${MODEL_ROOT:-$PROJECT_ROOT/artifacts/models/Qwen2.5-7B-Open-R1-GRPO-math-7b}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-7B-Instruct}"  # step 0 = base model
STEP="${STEP:-0}"
TEMP="${TEMP:-0.7}"

OUTPUT_ROOT="${INFER_OUTPUT_ROOT:-$PROJECT_ROOT/artifacts/results/GRPO-7B-math-temp-0.7}"
OUTDIR="$OUTPUT_ROOT/step-${STEP}"

# Per-task caches (isolated under the results dir)
CACHEROOT="$OUTDIR/hf_cache"
mkdir -p "$OUTDIR" "$CACHEROOT" "$OUTDIR/.triton" "$OUTDIR/.torchinductor" "$OUTDIR/.tmp"
export HF_HOME="$CACHEROOT"
export TRANSFORMERS_CACHE="$CACHEROOT/transformers"
export HF_HUB_CACHE="$CACHEROOT/hub"
export TRITON_CACHE_DIR="$OUTDIR/.triton"
export TORCHINDUCTOR_CACHE_DIR="$OUTDIR/.torchinductor"
export TMPDIR="$OUTDIR/.tmp"

# ── Chunking (default 5 problems per task) ──────────────────────────────────
CHUNK_SIZE="${CHUNK_SIZE:-5}"
TOTAL_EXAMPLES="${TOTAL_EXAMPLES:-500}"  # MATH-500 test split
NUM_CHUNKS="${NUM_CHUNKS:-$(( (TOTAL_EXAMPLES + CHUNK_SIZE - 1) / CHUNK_SIZE ))}"
TASK="${SLURM_ARRAY_TASK_ID}"
if (( TASK < 0 || TASK >= NUM_CHUNKS )); then
  echo "TASK ${TASK} outside 0..$((NUM_CHUNKS - 1)); exiting."
  exit 0
fi
DATASET_START=$(( TASK * CHUNK_SIZE ))
BATCH_SIZE="${BATCH_SIZE:-$CHUNK_SIZE}"

echo "→ step=${STEP} temp=${TEMP} chunk=${TASK}/${NUM_CHUNKS} (start=${DATASET_START} size=${CHUNK_SIZE})"
echo "   model : $MODEL_NAME_OR_PATH"
echo "   outdir: $OUTDIR"

python -u "$SCRIPT_PATH" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --output_dir "$OUTDIR" \
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
  --think_cap 750 \
  --answer_cap 50 \
  --step "$STEP"

RESULTS_BASENAME=$(printf "step%04d_test.jsonl" "$STEP")
echo "✓ Done: step=${STEP} temp=${TEMP} chunk=${TASK} → $OUTDIR/${RESULTS_BASENAME}"
