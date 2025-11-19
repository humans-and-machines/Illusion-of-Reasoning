#!/usr/bin/env bash
#SBATCH --job-name=infer_xword_ckpts_local_4temps
#SBATCH --output=logs/infer_xword_%A_%a.out
#SBATCH --error=logs/infer_xword_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:59:00
#SBATCH --array=0-123

set -euo pipefail
ulimit -n 4096

# ── Repo root ────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT}"

source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-openr1}"
module load cudatoolkit/12.6

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PIP_DISABLE_PIP_VERSION_CHECK=1
export TRANSFORMERS_NO_TORCHVISION=1
export TRANSFORMERS_NO_PYTORCH_IMAGE_TRANSFORMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MALLOC_ARENA_MAX=2

SCRIPT_PATH="${SCRIPT_PATH:-$PROJECT_ROOT/src/inference/crossword-inference.py}"
MODEL_ROOT="${MODEL_ROOT:-$PROJECT_ROOT/models/open-r1/Qwen2.5-1.5B-Open-R1-GRPO-Crosswords-v07}"
DATA_JSONL="${DATA_JSONL:-$PROJECT_ROOT/data/data.jsonl}"

STEPS_DEFAULT="50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000 1050 1100 1150 1200 1250 1300 1350 1400 1450 1500 1550"
read -r -a STEP_ARR <<< "${CHECKPOINT_STEPS:-$STEPS_DEFAULT}"
TEMPS=("0" "0.05" "0.3" "0.7")

NUM_STEPS=${#STEP_ARR[@]}
NUM_TEMPS=${#TEMPS[@]}

TASK=${SLURM_ARRAY_TASK_ID}
STEP_IDX=$(( TASK / NUM_TEMPS ))
TEMP_IDX=$(( TASK % NUM_TEMPS ))

# Safety for odd --array sizes
if (( STEP_IDX >= NUM_STEPS )); then
  echo "TASK $TASK beyond steps (NUM_STEPS=$NUM_STEPS); exiting."
  exit 0
fi

STEP=${STEP_ARR[$STEP_IDX]}
TEMP=${TEMPS[$TEMP_IDX]}

CKPT="$MODEL_ROOT/checkpoint-$STEP"
[[ -d "$CKPT" ]] || { echo "Missing checkpoint: $CKPT"; exit 0; }

# Output & per-task caches
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/results/GRPO-1.5B-xword-temp-${TEMP}}"
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

echo "→ step=$STEP temp=$TEMP"
echo "   ckpt: $CKPT"
echo "   out : $OUTDIR"
echo "   data: $DATA_JSONL"

python -u "$SCRIPT_PATH" \
  --model_name_or_path "$CKPT" \
  --output_dir "$OUTDIR" \
  --batch_size 1 \
  --entropy_mode full \
  --num_examples 1000000 \
  --num_samples 8 \
  --temperature "$TEMP" \
  --top_p 0.95 \
  --seed 42 \
  --dtype bfloat16 \
  --dataset_id CROSSWORD-LOCAL \
  --dataset_path "$DATA_JSONL" \
  --split test \
  --two_pass \
  --second_pass_phrase "Wait, we need to reconsider. Let's think this through step by step." \
  --think_cap 750 \
  --answer_cap 50 \
  --step "$STEP"

echo "✓ Done: step=$STEP temp=$TEMP → $OUTDIR"
