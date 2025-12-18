#!/usr/bin/env bash
# Resume/fill the temp=0.0 Llama-8B math sweep in small chunks.
# Existing result files (resume-aware):
#   artifacts/results/GRPO-Llama8B-math-temp-0/step-400/step0400_test.jsonl (3808/4000 rows)
#   artifacts/results/GRPO-Llama8B-math-temp-0/step-450/step0450_test.jsonl (3832/4000 rows)
#   artifacts/results/GRPO-Llama8B-math-temp-0/step-500/step0500_test.jsonl (3648/4000 rows)
# The unified runner only generates missing sample indices, so it is safe to
# re-run across all chunks.
#
# Usage (from repo root):
#   sbatch scripts/inference/math8b-step-fill-chunks.slurm
#   CHUNK_SIZE=5 STEPS="400 450 500" sbatch ...     # optional overrides
#   SLURM_ARRAY_TASK_ID mapping: step_idx = idx / NUM_CHUNKS, chunk_idx = idx % NUM_CHUNKS

#SBATCH --job-name=math8b_fill_t00
#SBATCH --output=logs/math8b_fill_t00_%A_%a.out
#SBATCH --error=logs/math8b_fill_t00_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:59:00

set -euo pipefail
ulimit -n 4096

# ── Project root ─────────────────────────────────────────────────────────────
# Prefer the Slurm submission directory so the script can be staged in /var/spool.
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

# HF online (base model pull + datasets)
export TRANSFORMERS_OFFLINE=0
export HF_HUB_OFFLINE=0
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_REQUEST_TIMEOUT=60
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-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}"

# W&B (optional)
export WANDB_MODE=online
export WANDB_DIR=/n/fs/similarity/wandb-offload/tmp
export WANDB_ARTIFACT_DIR=/n/fs/similarity/wandb-offload/artifacts
export WANDB_CACHE_DIR=/n/fs/similarity/wandb-offload/cache
mkdir -p "$WANDB_DIR" "$WANDB_ARTIFACT_DIR" "$WANDB_CACHE_DIR" logs

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_PATH="${SCRIPT_PATH:-$PROJECT_ROOT/src/inference/cli/unified_math.py}"
MODEL_ROOT="${MODEL_ROOT:-$PROJECT_ROOT/artifacts/models/open-r1/Llama-8B-Open-R1-GRPO-math-v2}"
TOKENIZER_PATH="${TOKENIZER_PATH:-meta-llama/Meta-Llama-3.1-8B-Instruct}"

# Target steps (override with STEPS env as a space-separated list)
STEPS=(${STEPS:-400 450 500})
N_STEPS=${#STEPS[@]}

# Chunking defaults: 5 problems → 100 chunks for the 500-example test split
CHUNK_SIZE="${CHUNK_SIZE:-5}"
TOTAL_EXAMPLES="${TOTAL_EXAMPLES:-500}"
NUM_CHUNKS="${NUM_CHUNKS:-$(( (TOTAL_EXAMPLES + CHUNK_SIZE - 1) / CHUNK_SIZE ))}"

# Array mapping: idx -> (step_idx, chunk_idx)
TASK="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK < 0 || TASK >= N_STEPS * NUM_CHUNKS )); then
  echo "TASK ${TASK} outside 0..$((N_STEPS * NUM_CHUNKS - 1)); exiting."
  exit 0
fi
STEP_IDX=$(( TASK / NUM_CHUNKS ))
CHUNK_IDX=$(( TASK % NUM_CHUNKS ))
STEP="${STEPS[$STEP_IDX]}"

if [[ "$STEP" -eq 0 ]]; then
  MODEL_NAME_OR_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct"
else
  CKPT_DIR="$MODEL_ROOT/checkpoint-$STEP"
  if [[ ! -d "$CKPT_DIR" ]]; then
    echo "Missing tuned checkpoint: $CKPT_DIR"
    exit 1
  fi
  MODEL_NAME_OR_PATH="$CKPT_DIR"
fi

TEMP="${TEMP:-0.0}"

DATASET_START=$(( CHUNK_IDX * CHUNK_SIZE ))
BATCH_SIZE="${BATCH_SIZE:-$CHUNK_SIZE}"

OUTPUT_ROOT="${INFER_OUTPUT_ROOT:-$PROJECT_ROOT/artifacts/results/GRPO-Llama8B-math-temp-0}"
PER_PROBLEM_SUBDIR="${PER_PROBLEM_SUBDIR:-0}"
if [[ "$PER_PROBLEM_SUBDIR" == "1" ]]; then
  SUBDIR_SUFFIX=$(printf "/problem-%04d" "$DATASET_START")
else
  SUBDIR_SUFFIX=""
fi
OUTDIR="$OUTPUT_ROOT/step-${STEP}${SUBDIR_SUFFIX}"

# Per-task caches (under outdir to localize HF/Trtion/TI). Set PER_PROBLEM_SUBDIR=1
# alongside CHUNK_SIZE=1 if you want every question to land in its own directory.
CACHEROOT="$OUTDIR/hf_cache"
mkdir -p "$OUTDIR" "$CACHEROOT" "$OUTDIR/.triton" "$OUTDIR/.torchinductor" "$OUTDIR/.tmp"
export HF_HOME="$CACHEROOT"
export TRANSFORMERS_CACHE="$CACHEROOT/transformers"
export HF_HUB_CACHE="$CACHEROOT/hub"
export TRITON_CACHE_DIR="$OUTDIR/.triton"
export TORCHINDUCTOR_CACHE_DIR="$OUTDIR/.torchinductor"
export TMPDIR="$OUTDIR/.tmp"

RECONSIDER_CUE="Hold on, this reasoning might be wrong. Let's go back and check each step carefully. Actually, this approach doesn't look correct. Let's restart and work through the solution more systematically. Wait, we need to reconsider. Let's think this through step by step."

echo "→ step=${STEP} temp=${TEMP} chunk=${CHUNK_IDX}/${NUM_CHUNKS} (start=${DATASET_START} size=${CHUNK_SIZE})"
echo "   model : $MODEL_NAME_OR_PATH"
echo "   outdir: $OUTDIR"

python -u "$SCRIPT_PATH" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --output_dir "$OUTDIR" \
  --tokenizer_path "$TOKENIZER_PATH" \
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
  --two_pass \
  --second_pass_phrase "$RECONSIDER_CUE" \
  --think_cap 750 \
  --answer_cap 50 \
  --step "$STEP"

RESULTS_BASENAME=$(printf "step%04d_test.jsonl" "$STEP")
echo "✓ Done: step=${STEP} temp=${TEMP} chunk=${CHUNK_IDX} → $OUTDIR/${RESULTS_BASENAME}"
