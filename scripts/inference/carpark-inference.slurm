#!/usr/bin/env bash
#SBATCH --job-name=infer_carpark_ckpts_local_4temps
#SBATCH --output=logs/infer_carpark_%A_%a.out
#SBATCH --error=logs/infer_carpark_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:59:00
# 20 steps × 4 temps = 80 tasks → indices 0..79
#SBATCH --array=0-79

set -euo pipefail
ulimit -n 4096

# ── Project root ─────────────────────────────────────────────────────────────
# Use Slurm submission directory when available; otherwise fall back to $PWD.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
fi

source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
CONDA_ENV="${CONDA_ENV:-$PROJECT_ROOT/openr1}"
conda activate "$CONDA_ENV"
module load cudatoolkit/12.6

# ── Env hygiene ──────────────────────────────────────────────────────────────
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export TRANSFORMERS_NO_TORCHVISION=1
export TRANSFORMERS_NO_PYTORCH_IMAGE_TRANSFORMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MALLOC_ARENA_MAX=2

# ── Paths & defaults ─────────────────────────────────────────────────────────
SCRIPT_PATH="${SCRIPT_PATH:-$PROJECT_ROOT/src/inference/cli/unified_carpark.py}"
MODEL_ROOT="${MODEL_ROOT:-$PROJECT_ROOT/artifacts/models/open-r1/Qwen2.5-1.5B-Open-R1-GRPO-carpark-v1}"

# Carpark (Rush Hour) dataset config
DATASET_ID="${DATASET_ID:-od2961/rush4-5-6-balanced}"
DATASET_PROMPT_COL="${DATASET_PROMPT_COL:-messages}"
DATASET_SOLUTION_COL="${DATASET_SOLUTION_COL:-solution}"

# Steps (20 total by default)
STEPS_DEFAULT="50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000"
read -r -a STEP_ARR <<< "${CHECKPOINT_STEPS:-$STEPS_DEFAULT}"
TEMPS=("0" "0.05" "0.3" "0.7")

NUM_STEPS=${#STEP_ARR[@]}
NUM_TEMPS=${#TEMPS[@]}

# ── Map array index → (step, temp) ───────────────────────────────────────────
TASK=${SLURM_ARRAY_TASK_ID}
STEP_IDX=$(( TASK / NUM_TEMPS ))
TEMP_IDX=$(( TASK % NUM_TEMPS ))

# Safety for overrides where array range > steps provided
if (( STEP_IDX >= NUM_STEPS )); then
  echo "TASK $TASK beyond steps (NUM_STEPS=$NUM_STEPS); exiting."
  exit 0
fi

STEP=${STEP_ARR[$STEP_IDX]}
TEMP=${TEMPS[$TEMP_IDX]}

CKPT="$MODEL_ROOT/checkpoint-$STEP"
if [[ ! -d "$CKPT" ]]; then
  echo "Missing checkpoint: $CKPT"
  exit 0
fi

# ── Output & per-task caches ─────────────────────────────────────────────────
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/artifacts/results/GRPO-1.5B-carpark-temp-${TEMP}}"
OUTDIR="$OUTPUT_ROOT/step${STEP}"
mkdir -p "$OUTDIR"

CACHEROOT="$OUTDIR/hf_cache"
mkdir -p "$CACHEROOT" "$OUTDIR/.triton" "$OUTDIR/.torchinductor" "$OUTDIR/.tmp"
export HF_HOME="$CACHEROOT"
export TRANSFORMERS_CACHE="$CACHEROOT/transformers"
export HF_HUB_CACHE="$CACHEROOT/hub"
export TRITON_CACHE_DIR="$OUTDIR/.triton"
export TORCHINDUCTOR_CACHE_DIR="$OUTDIR/.torchinductor"
export TMPDIR="$OUTDIR/.tmp"

RECONSIDER_CUE="Hold on, this reasoning might be wrong. Let's go back and check each step carefully. Actually, this approach doesn't look correct. Let's restart and work through the solution more systematically. Wait, we need to reconsider. Let's think this through step by step."

echo "→ carpark | step=$STEP temp=$TEMP"
echo "   ckpt: $CKPT"
echo "   out : $OUTDIR"
echo "   data: $DATASET_ID  ($DATASET_PROMPT_COL → $DATASET_SOLUTION_COL)"

# NOTE: If your carpark-inference.py isn't yet patched for temperature==0 → greedy,
# either add the small Python fix (preferred) or temporarily map 0→1e-6.
# TEMP_ARG="$TEMP"; if [[ "$TEMP" == "0" || "$TEMP" == "0.0" ]]; then TEMP_ARG="1e-6"; fi

python -u "$SCRIPT_PATH" \
  --model_name_or_path "$CKPT" \
  --output_dir "$OUTDIR" \
  --batch_size 1 \
  --entropy_mode full \
  --num_examples 500 \
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
  --step "$STEP"

echo "✓ Done: step=$STEP temp=$TEMP → $OUTDIR"
