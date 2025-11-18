#!/bin/bash
#SBATCH --job-name=infer_math_ckpts
#SBATCH --output=logs/infer_ckpts_%A_%a.out
#SBATCH --error=logs/infer_ckpts_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1             # Request 1 GPU per job
#SBATCH --cpus-per-task=16       # CPU cores per task
#SBATCH --mem=128G               # Total memory
#SBATCH --time=12:00:00          # Max runtime hh:mm:ss
#SBATCH --array=0-21             # 20 revisions, max 4 running at once

# Enable unbuffered Python output and debug logging
export LOGLEVEL=DEBUG
export PYTHONUNBUFFERED=1

# ----------------------------
# Load CUDA toolkit and activate environment
# ----------------------------
module load cudatoolkit/12.4
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
# Activate the existing environment by path
export ROOT_DIR="$PWD"
export ENV_NAME="openr1"
export ENV_DIR="$ROOT_DIR/$ENV_NAME"

export CONDA_PKGS_DIRS="$ROOT_DIR/.conda_pkgs"
export CONDA_ENVS_DIRS="$ROOT_DIR/.conda_envs"
export CONDA_ENVS_DIRS="$ROOT_DIR/.conda_envs"
export CONDA_CACHEDIR="$ROOT_DIR/.conda_cache"
export PYTHONUSERBASE="$ROOT_DIR/.local"
export CONDARC="$ROOT_DIR/.condarc"
export PIP_CACHE_DIR="$ROOT_DIR/.pip_cache"

mkdir -p "$CONDA_PKGS_DIRS" "$CONDA_ENVS_DIRS" "$CONDA_CACHEDIR" "$PIP_CACHE_DIR"

# ─── Activate Environment ───────────────────────────────────────────────
conda activate "$ENV_DIR"
echo "✅ Conda env active at: $(which python)"
python --version


# Provide the HuggingFace token for authentication
export HF_TOKEN="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"
export TRANSFORMERS_OFFLINE=0
export HF_HUB_OFFLINE=0

# ----------------------------
# WandB cache and artifact dirs on /n/fs
# ----------------------------
export WANDB_DIR=/n/fs/similarity/wandb-offload/tmp
export WANDB_ARTIFACT_DIR=/n/fs/similarity/wandb-offload/artifacts
export WANDB_CACHE_DIR=/n/fs/similarity/wandb-offload/cache
export TMPDIR=/n/fs/similarity/wandb-offload/tmp
export HF_HUB_REQUEST_TIMEOUT=60

mkdir -p /n/fs/similarity/vllm
mkdir -p "$WANDB_DIR" "$WANDB_ARTIFACT_DIR" "$WANDB_CACHE_DIR" "$TMPDIR"
mkdir -p logs .cache .hf_cache .tmp .torchinductor .triton

# ----------------------------
# HF + Cache (local workspace)
# ----------------------------
export TRANSFORMERS_CACHE="$(pwd)/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="$(pwd)/.cache/huggingface/datasets"
export TMPDIR="$(pwd)/.tmp"
export TORCHINDUCTOR_CACHE_DIR="$(pwd)/.torchinductor"
export TRITON_CACHE_DIR="$(pwd)/.triton"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# W&B Online Mode
export WANDB_MODE=online

# ----------------------------
# Model ID and revision SHAs
# ----------------------------
MODEL_ID="od2961/Qwen2.5-1.5B-Open-R1-GRPO"
REVISIONS=(
  aef4ec7  # step 50
  9ac7e5e  # step 150
  ba99bc4  # step 250
  7cbc93a  # step 350
  4803ae9  # step 450
  5d0b169  # step 550
  334f623  # step 650
  f6b58de  # step 750
  14436ba  # step 850
  d5bb572  # step 950
  1e7973f  # step 1050
  4771a8f  # step 1150
  a472241  # step 1250
  76bee3d  # step 1350
  57fa3c6  # step 1450
  70ddc2c  # step 1550
  29605a8  # step 1650
  26044d8  # step 1750
  e8a648f  # step 1850
  df6030f  # step 1950
  3075e71  # step 2050
  c148b71  # step 2150
  bd98859  # step 2250
  08b3d89  # step 2350
  d1d3bf7  # step 2450
)

# Determine this job's revision
REV=${REVISIONS[$SLURM_ARRAY_TASK_ID]}
echo "→ Running revision: $REV"

# Paths and output roots
OUTPUT_ROOT="/n/fs/similarity/open-r1/results/od2961/Math220k/GRPO/1.5B"
CACHEROOT="$OUTPUT_ROOT/hf_cache"

# Ensure output and cache directories exist
mkdir -p "$OUTPUT_ROOT" "$CACHEROOT"

# Script location and per-revision output directory
SCRIPT_PATH="/n/fs/similarity/open-r1/inference.py"
OUTDIR="$OUTPUT_ROOT/$REV"
mkdir -p "$OUTDIR"

# HF cache envs
export HF_HOME="$CACHEROOT"
export TRANSFORMERS_CACHE="$CACHEROOT/transformers"
export HF_HUB_CACHE="$CACHEROOT/hub"

# Run inference at DEBUG logging
python -u "$SCRIPT_PATH" \
  --model_name_or_path "$MODEL_ID" \
  --revision "$REV" \
  --output_dir "$OUTDIR" \
  --batch_size 5 \
  --num_examples 500 \
  --num_samples 1 \
  --temperature 0

# Clean up snapshot to free space
echo "→ Cleaning cache for revision: $REV"
rm -rf "$CACHEROOT/hub/models--${MODEL_ID//\//--}--*/snapshots/$REV"

echo "✓ Revision $REV completed and cleaned up."
