#!/bin/bash
#SBATCH --job-name=OpenR1_GRPO
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=00:59:00
#SBATCH --output=logs/slurm_%j.out

set -euo pipefail

# ----------------------------
# MODULES & PYTHON ENVIRONMENT
# --------------------------
#module load cudatoolkit/12.4
export PATH="$HOME/.local/bin:$PATH"	
pip install --upgrade huggingface_hub
# ----------------------------
# Setup
# ----------------------------
export RUN_NAME="Qwen1.5B-GRPO-Finetune"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export CONFIG="recipes/Qwen2.5-1.5B-Instruct/grpo/config_math.yaml"
export CONFIG_FILE="recipes/accelerate_configs/zero3.yaml"
export SERVER_LOG="logs/liv_vllm_${RUN_NAME}_${TIMESTAMP}.log"
export TRAINING_LOG="logs/liv_train_${RUN_NAME}_${TIMESTAMP}.log"

# ensure old HF_TOKEN does not take precedence
unset HF_TOKEN

# configure HF cache locations before login
export HF_HOME="$(pwd)/.hf_cache"
export XDG_CACHE_HOME="$(pwd)/.cache"
#mkdir -p "$HF_HOME" "$XDG_CACHE_HOME"
export NLTK_DATA="$(pwd)/.cache/nltk_data"


# provide the new token
export HUGGING_FACE_HUB_TOKEN="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"
export TORCH_LOAD_WEIGHTS_ONLY=0


# ----------------------------
# Determine number of training GPUs (total GPUs – 1 for vLLM)
# ----------------------------
# Slurm will set CUDA_VISIBLE_DEVICES to something like "0,1,2,3,4,5,6,7"
ALL_GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_TOTAL=$(echo "$ALL_GPUS" | tr ',' '\n' | wc -l)
NUM_TRAINING=$(( NUM_TOTAL - 1 ))

# ----------------------------
# WandB cache and artifact dirs on /n/fs
# ----------------------------
export WANDB_DIR=/n/fs/similarity/wandb-offload/tmp
export WANDB_ARTIFACT_DIR=/n/fs/similarity/wandb-offload/artifacts
export WANDB_CACHE_DIR=/n/fs/similarity/wandb-offload/cache
export VLLM_USAGE_STATS_PATH=/n/fs/similarity/vllm/usage_stats.json
export TMPDIR=/n/fs/similarity/wandb-offload/tmp
export HF_HUB_REQUEST_TIMEOUT=60

#mkdir -p /n/fs/similarity/vllm
#mkdir -p "$WANDB_DIR" "$WANDB_ARTIFACT_DIR" "$WANDB_CACHE_DIR" "$TMPDIR"

# Optional: Set WANDB_CONFIG_DIR if needed (e.g. wandb/settings)
export WANDB_CONFIG_DIR=/n/fs/similarity/wandb-offload/config
#mkdir -p /n/fs/similarity/wandb-offload/{tmp,artifacts,cache,config}
#mkdir -p logs .cache .hf_cache .tmp .torchinductor .triton

# ----------------------------
# HF + Cache (local workspace)
# ----------------------------
export TRANSFORMERS_CACHE="$(pwd)/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="$(pwd)/.cache/huggingface/datasets"
export TMPDIR="$(pwd)/.tmp"
export VLLM_API_KEY="dummy"
export TORCHINDUCTOR_CACHE_DIR="$(pwd)/.torchinductor"
export TRITON_CACHE_DIR="$(pwd)/.triton"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# ✅ Force full state loading in PyTorch (not just weights)
export TORCH_LOAD_WEIGHTS_ONLY=0

# (Optional) prevent Triton cache slowdown warnings
#export TRITON_CACHE_DIR="/tmp/$USER/triton"
#mkdir -p "$TRITON_CACHE_DIR"

# W&B Online Mode
export WANDB_MODE=online
#export WANDB_PROJECT=your_project_name
#export WANDB_ENTITY=your_entity
#export WANDB_API_KEY=your_token_here  # or ensure ~/.netrc has token

# ─── Load modules and conda ─────────────────────────────────────────────
#module load cudatoolkit/12.4
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh

# ─── Localized Conda + Pip Setup ───────────────────────────────────────
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

#mkdir -p "$CONDA_PKGS_DIRS" "$CONDA_ENVS_DIRS" "$CONDA_CACHEDIR" "$PIP_CACHE_DIR"

# ─── Activate Environment ───────────────────────────────────────────────
conda activate "$ENV_DIR"
echo "✅ Conda env active at: $(which python)"
python --version

pip uninstall huggingface-hub -y
pip install huggingface-hub
python -m pip install yq

# Update accelerate config so that num_processes = num_training_gpus
cp "${CONFIG_FILE}" "${CONFIG_FILE}.bak"
python -m yq -y --in-place ".num_processes = $NUM_TRAINING" "$CONFIG_FILE"
echo "→ Set accelerate num_processes to $NUM_TRAINING (total GPUs: $NUM_TOTAL)"
# ----------------------------
# Log in to Hugging Face (first time)
# ----------------------------
#python -m huggingface-cli login --token "$HUGGING_FACE_HUB_TOKEN" --add-to-git-credential
#echo "✅ Logged into Hugging Face (step 1)"

# ─── (Optional) Second Hugging Face Authentication ─────────────────────
# If you need to switch to a different HF token later, unset HF_TOKEN again:
# unset HF_TOKEN
# export HUGGING_FACE_HUB_TOKEN="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"
# huggingface-cli login --token "$HUGGING_FACE_HUB_TOKEN" --add-to-git-credential
# echo "✅ Logged into Hugging Face (step 2)"

# ─── Environment Identifiers ────────────────────────────────────────────
export RUN_NAME="Qwen1.5B-GRPO-Finetune"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export CONFIG="recipes/Qwen2.5-1.5B-Instruct/grpo/config_math.yaml"
export CONFIG_FILE="recipes/accelerate_configs/zero3.yaml"
export SERVER_LOG="logs/liv_vllm_${RUN_NAME}_${TIMESTAMP}.log"
export TRAINING_LOG="logs/liv_train_${RUN_NAME}_${TIMESTAMP}.log"

# ─── Local cache + logging paths ────────────────────────────────────────
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

#mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME"
#mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" logs

# ─── Optional: disable vLLM usage stats ────────────────────────────────
export VLLM_API_KEY="dummy"
export VLLM_ATTENTION_BACKEND="xformers"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ─── Log summary ────────────────────────────────────────────────────────
echo "🟢 Setup complete. Ready to run GRPO."
echo "Env:        $ENV_DIR"
echo "Config:     $CONFIG"
echo "Log Files:  $SERVER_LOG, $TRAINING_LOG"
echo "CUDA_VISIBLE_DEVICES: $ALL_GPUS (using $NUM_TRAINING for training)"

# ----------------------------
# WandB cache and artifact dirs on /n/fs (again, if needed)
# ----------------------------
export WANDB_DIR=/n/fs/similarity/wandb-offload/tmp
export WANDB_ARTIFACT_DIR=/n/fs/similarity/wandb-offload/artifacts
export WANDB_CACHE_DIR=/n/fs/similarity/wandb-offload/cache
export VLLM_USAGE_STATS_PATH=/n/fs/similarity/vllm/usage_stats.json
export TMPDIR=/n/fs/similarity/wandb-offload/tmp

#mkdir -p /n/fs/similarity/vllm
#mkdir -p "$WANDB_DIR" "$WANDB_ARTIFACT_DIR" "$WANDB_CACHE_DIR" "$TMPDIR"

# Optional: Set WANDB_CONFIG_DIR if needed (e.g. wandb/settings)
export WANDB_CONFIG_DIR=/n/fs/similarity/wandb-offload/config
#mkdir -p /n/fs/similarity/wandb-offload/{tmp,artifacts,cache,config}
#mkdir -p logs .cache .hf_cache .tmp .torchinductor .triton

# W&B Online Mode
export WANDB_MODE=online

# -----------------------------------
# 1) Launch vLLM server on GPU 0
# -----------------------------------
export VLLM_ATTENTION_BACKEND=xformers
#export VLLM_ENGINE=v0
#unset VLLM_ATTENTION_BACKEND
export TORCH_FORCE_FULL_STATE_DICT=1
export FLASH_ATTENTION_FORCE_DISABLED=1
export TRANSFORMERS_NO_FLASH_ATTN=1
export WANDB_DATA_DIR=/n/fs/similarity/open-r1/wandb

# -----------------------------------
# Launch vLLM + trainer in one srun
# -----------------------------------
srun --gres=gpu:8 --cpus-per-task=64 bash -c '
set -euo pipefail

############################
# 1) vLLM on GPU-0
############################
export CUDA_VISIBLE_DEVICES=0
echo "Launching vLLM on GPU 0…"

trl vllm-serve \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dtype float16 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.90 \
  > "'"$SERVER_LOG"'" 2>&1 &

VLLM_PID=$!

# Health-check loop
until curl -sf http://localhost:8000/health > /dev/null; do
  echo "Waiting for vLLM…"
  sleep 2
done
echo "✅ vLLM is healthy"

############################
# 2) Training on GPU 1-7
############################
export CUDA_VISIBLE_DEVICES=$(echo "'"$ALL_GPUS"'" | cut -d"," -f2-)
echo "🚀 Launching training on GPUs $CUDA_VISIBLE_DEVICES"

accelerate launch \
  --main_process_port 29504 \
  --config_file "'"$CONFIG_FILE"'" \
  src/open_r1/grpo.py \
  --config "'"$CONFIG"'" \
  --use_vllm \
  --run_name "'"${RUN_NAME}-${TIMESTAMP}"'" \
  --ignore_data_skip \
  --overwrite_output_dir false \
  --resume_from_checkpoint /n/fs/similarity/open-r1/data/Qwen2.5-1.5B-Open-R1-GRPO-math-v1/checkpoint-850 \
  --seed 42 \
  > "$TRAINING_LOG" 2>&1

# Wait for vLLM to exit after training finishes
wait $VLLM_PID
'
