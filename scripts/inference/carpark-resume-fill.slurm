#!/usr/bin/env bash
# Resume/fill outstanding Rush Hour (carpark) generations for specific (step, temp) pairs.
# This script runs the unified_carpark CLI once per spec listed in RUN_SPECS, relying on
# the built-in resume behaviour that only re-generates missing sample indices.
#
# Usage (from repo root):
#   CONDA_ENV="$OPENR1_ENV" sbatch scripts/inference/carpark-resume-fill.slurm
#
# Optional env knobs:
#   BASE_MODEL=<hf model name>                (default: Qwen/Qwen2.5-1.5B-Instruct)
#   CKPT_ROOT=<path to GRPO checkpoints>      (default: artifacts/models/.../carpark-v1)
#   RESULTS_PREFIX=<results root override>    (default: $PROJECT_ROOT/artifacts/results)
#   DATASET_ID=<HF dataset id>                (default: od2961/rush4-5-6-balanced)
#   DATASET_PROMPT_COL / DATASET_SOLUTION_COL (default: messages / solution)

#SBATCH --job-name=carpark_resume_fill
#SBATCH --output=logs/carpark_resume_fill_%A_%a.out
#SBATCH --error=logs/carpark_resume_fill_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
# Three targets (step0 @ temps 0.05/0.3 and step950 @ temp0.7) → indices 0..2
#SBATCH --array=0-2

set -euo pipefail
ulimit -n 4096

# ── Project root ─────────────────────────────────────────────────────────────
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
fi

# ── Conda env ────────────────────────────────────────────────────────────────
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

# ── Paths & dataset ─────────────────────────────────────────────────────────
SCRIPT_PATH="${SCRIPT_PATH:-$PROJECT_ROOT/src/inference/cli/unified_carpark.py}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
# Default points to the actual local checkpoint drop (moved out of open-r1/).
CKPT_ROOT="${CKPT_ROOT:-$PROJECT_ROOT/artifacts/models/Qwen2.5-1.5B/Qwen2.5-1.5B-Open-R1-GRPO-carpark-v1}"
RESULTS_PREFIX="${RESULTS_PREFIX:-$PROJECT_ROOT/artifacts/results}"

DATASET_ID="${DATASET_ID:-od2961/rush4-5-6-balanced}"
DATASET_PROMPT_COL="${DATASET_PROMPT_COL:-messages}"
DATASET_SOLUTION_COL="${DATASET_SOLUTION_COL:-solution}"

RECONSIDER_CUE="Hold on, this reasoning might be wrong. Let's go back and check each step carefully. Actually, this approach doesn't look correct. Let's restart and work through the solution more systematically. Wait, we need to reconsider. Let's think this through step by step."

# ── Targeted resume specs ───────────────────────────────────────────────────
# Format: "temp=<value> step=<value> kind=<base|ckpt> out_suffix=<subdir>"
RUN_SPECS=(
  "temp=0.05 step=0 kind=base out_suffix=base-step0"
  "temp=0.3 step=0 kind=base out_suffix=base-step0"
  "temp=0.7 step=950 kind=ckpt out_suffix=step950"
)

TASK=${SLURM_ARRAY_TASK_ID}
if (( TASK < 0 || TASK >= ${#RUN_SPECS[@]} )); then
  echo "TASK $TASK out of range (0..$(( ${#RUN_SPECS[@]} - 1 ))); exiting."
  exit 1
fi

spec="${RUN_SPECS[$TASK]}"
TEMP=""
STEP=""
KIND=""
OUT_SUFFIX=""
for kv in $spec; do
  key=${kv%%=*}
  val=${kv#*=}
  case "$key" in
    temp) TEMP="$val" ;;
    step) STEP="$val" ;;
    kind) KIND="$val" ;;
    out_suffix) OUT_SUFFIX="$val" ;;
  esac
done

if [[ -z "$TEMP" || -z "$STEP" || -z "$KIND" || -z "$OUT_SUFFIX" ]]; then
  echo "Spec '$spec' missing fields (temp=$TEMP step=$STEP kind=$KIND out_suffix=$OUT_SUFFIX)"
  exit 1
fi

TEMP_ROOT="$RESULTS_PREFIX/GRPO-1.5B-carpark-temp-${TEMP}"
mkdir -p "$TEMP_ROOT"

case "$KIND" in
  base)
    MODEL_PATH="$BASE_MODEL"
    OUTDIR="$TEMP_ROOT/$OUT_SUFFIX"
    ;;
  ckpt)
    CKPT_DIR="$CKPT_ROOT/checkpoint-$STEP"
    if [[ ! -d "$CKPT_DIR" ]]; then
      echo "Missing checkpoint directory: $CKPT_DIR"
      exit 1
    fi
    MODEL_PATH="$CKPT_DIR"
    OUTDIR="$TEMP_ROOT/$OUT_SUFFIX"
    ;;
  *)
    echo "Unknown run kind '$KIND' (spec: $spec)"
    exit 1
    ;;
esac

mkdir -p "$OUTDIR"

CACHEROOT="$OUTDIR/hf_cache"
mkdir -p "$CACHEROOT" "$OUTDIR/.triton" "$OUTDIR/.torchinductor" "$OUTDIR/.tmp"
export HF_HOME="$CACHEROOT"
export TRANSFORMERS_CACHE="$CACHEROOT/transformers"
export HF_HUB_CACHE="$CACHEROOT/hub"
export TRITON_CACHE_DIR="$OUTDIR/.triton"
export TORCHINDUCTOR_CACHE_DIR="$OUTDIR/.torchinductor"
export TMPDIR="$OUTDIR/.tmp"

echo "→ carpark resume | kind=$KIND step=$STEP temp=$TEMP"
echo "   model: $MODEL_PATH"
echo "   out  : $OUTDIR (root=$TEMP_ROOT)"
echo "   data : $DATASET_ID"

python -u "$SCRIPT_PATH" \
  --model_name_or_path "$MODEL_PATH" \
  --output_dir "$OUTDIR" \
  --batch_size 1 \
  --entropy_mode full \
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

echo "✓ Resume complete: step=$STEP temp=$TEMP → $OUTDIR"
