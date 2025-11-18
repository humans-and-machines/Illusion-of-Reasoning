#!/bin/bash
#SBATCH --job-name=OpenR1_SFT
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=training_%j.out

set -euo pipefail

# -------------------------------
# MODULES & PYTHON ENVIRONMENT
# -------------------------------
module load cudatoolkit/12.4
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh

export ROOT_DIR="$PWD"
export ENV_NAME="openr1"
export ENV_DIR="$ROOT_DIR/$ENV_NAME"

# Set local pip/conda caches for reproducibility & speed
export CONDA_PKGS_DIRS="$ROOT_DIR/.conda_pkgs"
export CONDA_ENVS_DIRS="$ROOT_DIR/.conda_envs"
export CONDA_CACHEDIR="$ROOT_DIR/.conda_cache"
export PYTHONUSERBASE="$ROOT_DIR/.local"
export PIP_CACHE_DIR="$ROOT_DIR/.pip_cache"
export CONDARC="$ROOT_DIR/.condarc"

mkdir -p "$CONDA_PKGS_DIRS" "$CONDA_ENVS_DIRS" "$CONDA_CACHEDIR" "$PIP_CACHE_DIR"

# Activate environment
conda activate "$ENV_DIR"
echo "✅ Conda env active at: $(which python)"
python --version

# -------------------------------
# ENVIRONMENT & CACHE PATHS
# -------------------------------
export RUN_NAME="Qwen1.5B-SFT-Finetune-v2"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export CONFIG="recipes/Qwen2.5-1.5B-Instruct/sft/config_demo_liv.yaml"
export CONFIG_FILE="recipes/accelerate_configs/zero3.yaml"
export TRAINING_LOG="logs/liv_train_${RUN_NAME}_${TIMESTAMP}.log"

# Hugging Face/W&B cache & config dirs (use cluster persistent)
export HF_HOME="$ROOT_DIR/.hf_cache"
export HF_DATASETS_CACHE="$ROOT_DIR/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="$ROOT_DIR/.cache/huggingface/transformers"
export XDG_CACHE_HOME="$ROOT_DIR/.cache"
export TMPDIR="$ROOT_DIR/.tmp"
export TORCHINDUCTOR_CACHE_DIR="$ROOT_DIR/.torchinductor"
export TRITON_CACHE_DIR="$ROOT_DIR/.triton"
export WANDB_DIR="$ROOT_DIR/.wandb"
export WANDB_CACHE_DIR="$ROOT_DIR/.wandb_cache"
export WANDB_MODE="online"
export TORCH_LOAD_WEIGHTS_ONLY=0

mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME"
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" logs

# Project-specific persistent cache (shared on cluster, optional)
export WANDB_ARTIFACT_DIR="/n/fs/similarity/wandb-offload/artifacts"
export WANDB_CONFIG_DIR="/n/fs/similarity/wandb-offload/config"
export VLLM_USAGE_STATS_PATH="/n/fs/similarity/vllm/usage_stats.json"
mkdir -p "$WANDB_ARTIFACT_DIR" "$WANDB_CONFIG_DIR" "/n/fs/similarity/vllm"

# vLLM stats cache
export VLLM_API_KEY="dummy"
export VLLM_ATTENTION_BACKEND="xformers"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Hugging Face Authentication
export HUGGING_FACE_HUB_TOKEN="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"
huggingface-cli login --token "$HUGGING_FACE_HUB_TOKEN"
echo "✅ Logged into Hugging Face"

# -------------------------------
# PIP UPGRADE (if needed)
# -------------------------------
pip install --upgrade yq huggingface_hub

# -------------------------------
# PRINT ENV SUMMARY
# -------------------------------
echo "🟢 Setup complete. Ready to run SFT."
echo "Env:        $ENV_DIR"
echo "Config:     $CONFIG"
echo "Log Files:  $TRAINING_LOG"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# -------------------------------
# GPU/ACCELERATE CONFIG
# -------------------------------
ALL_GPUS=$CUDA_VISIBLE_DEVICES
TRAINING_GPUS=$ALL_GPUS
NUM_TRAINING=$(echo $TRAINING_GPUS | tr ',' '\n' | wc -l)
cp "${CONFIG_FILE}" "${CONFIG_FILE}.bak"
yq -y --in-place ".num_processes = $NUM_TRAINING" "${CONFIG_FILE}"

# -------------------------------
# LAUNCH TRAINING
# -------------------------------
export MASTER_PORT=29503
CUDA_VISIBLE_DEVICES=$ALL_GPUS \
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --main_process_port $MASTER_PORT \
  --config_file "${CONFIG_FILE}" \
  src/open_r1/sft.py \
    --output_dir /n/fs/similarity/open-r1/data/Qwen2.5-1.5B-Instruct-SFT-v4 \
    --config "$CONFIG" \
    --run_name "${RUN_NAME}-${TIMESTAMP}" \
  2>&1 | tee "${TRAINING_LOG}"
