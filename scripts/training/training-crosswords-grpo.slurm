#!/bin/bash
#SBATCH --job-name=OpenR1_GRPO
#SBATCH --partition=mltheory        # ← use the mltheory partition
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm_%j.out
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

module use /usr/local/share/Modules/modsulefiles
# then:
module avail cudatoolkit
module purge && module load cudatoolkit/12.4
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-openr1}"
export PYTHONNOUSERSITE=1

module list
which nvcc && nvcc --version
echo "$CUDA_HOME"

# ----------------------------
# MODULES & PYTHON ENVIRONMENT
# --------------------------
pip install --upgrade huggingface_hub
# ----------------------------
# Setup
# ----------------------------
export RUN_NAME="Qwen1.5B-GRPO-Finetune"
export TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export CONFIG="recipes/Qwen2.5-1.5B-Instruct/grpo/config_crosswords.yaml"
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
ALL_GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
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

# make absolutely sure user site-packages are ignored
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PIP_USER=false
#pip install --upgrade pip

#pip uninstall -y torch torchvision torchaudio || true
#pip uninstall -y flash-attn || true
#pip uninstall vllm -y
#pip uninstall -y trl transformers || true
#pip uninstall -y flash-attn

#pip install vllm==0.8.4
#pip install "transformers==4.52.3" "trl[vllm]@git+https://github.com/huggingface/trl.git@9ac614fb081e17805f7f62ab3f5f7036bdefe7b0"
#pip install "protobuf==3.20.3"

#pip install --index-url https://download.pytorch.org/whl/cu124 \
#  torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124
#python - <<'PY'
#import torch; print("torch:", torch.__version__, "CUDA build:", torch.version.cuda)
#PY

#pip install setuptools
# sanity check: torch should come from the conda env
python - <<'PY'
import torch, site, sys
print("[torch file]", torch.__file__)
print("[user site]", site.getusersitepackages())
print("[sys.path head]", sys.path[:5])
PY
python - <<'PY'
import os, torch, vllm
print("torch:", torch.__version__, "cuda build:", torch.version.cuda)
print("CUDA_HOME:", os.environ.get("CUDA_HOME"))
print("vllm:", vllm.__version__)
PY

export VLLM_WORKER_MULTIPROC_METHOD=spawn
# (optional) don’t pick the older engine via xformers if you don’t need it
unset VLLM_ATTENTION_BACKEND


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
export CONFIG="recipes/Qwen2.5-1.5B-Instruct/grpo/config_crosswords.yaml"
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
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$PROJECT_ROOT/wandb}"

# -----------------------------------
# Ports + GPU selection
# -----------------------------------
export VLLM_PORT=8000
export MASTER_PORT=29510

ALL_GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
TRAIN_GPUS="$(echo "$ALL_GPUS" | cut -d',' -f2-)"            # everything except the first GPU
NUM_TRAINING=$(echo "$TRAIN_GPUS" | awk -F',' '{print NF}')  # number of training GPUs

# Make TRITON cache node-local to avoid NFS stalls
export TMPDIR="${TMPDIR:-$PWD/.tmp}"
mkdir -p "$TMPDIR"
export TRITON_CACHE_DIR="$TMPDIR/triton-cache"

# Make sure accelerate config has the right world size
ACC_CFG="recipes/accelerate_configs/zero3.yaml"
cp "$ACC_CFG" "${ACC_CFG}.bak"
python - <<PY
import yaml
p="$ACC_CFG"
y=yaml.safe_load(open(p))
y["num_processes"] = $NUM_TRAINING
open(p, "w").write(yaml.safe_dump(y, sort_keys=False))
print("accelerate num_processes =", $NUM_TRAINING)
PY

# Logs
RUN_NAME="Qwen1.5B-GRPO-Finetune"
TS=$(date +%Y%m%d_%H%M%S)
SERVER_LOG="logs/vllm_${RUN_NAME}_${TS}.log"
TRAINING_LOG="logs/train_${RUN_NAME}_${TS}.log"

# Clean up vLLM if something fails/exits
cleanup() { [[ -n "${VLLM_PID:-}" ]] && kill -9 "$VLLM_PID" 2>/dev/null || true; }
trap cleanup EXIT

# -----------------------------------
# Launch vLLM + trainer in one srun
# -----------------------------------
srun --gres="$SLURM_GPUS" --cpus-per-task="$SLURM_CPUS_PER_TASK" bash -lc "
set -euo pipefail

############################
# 1) vLLM on GPU-0
############################
export CUDA_VISIBLE_DEVICES=\$(echo '$ALL_GPUS' | cut -d',' -f1)
echo 'Launching vLLM on GPU(s):' \$CUDA_VISIBLE_DEVICES

trl vllm-serve \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dtype bf16 \
  --port $VLLM_PORT \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  > '$SERVER_LOG' 2>&1 &

VLLM_PID=\$!

# Health-check loop (up to ~2 minutes)
for i in \$(seq 1 120); do
  if curl -fsS http://127.0.0.1:$VLLM_PORT/health >/dev/null; then
    echo '✅ vLLM is healthy'
    break
  fi
  echo 'Waiting for vLLM…'
  sleep 1
done

############################
# 2) Training on GPUs 1..N
############################
export CUDA_VISIBLE_DEVICES=$TRAIN_GPUS
echo '🚀 Launching training on GPU(s):' \$CUDA_VISIBLE_DEVICES

accelerate launch \
  --main_process_port $MASTER_PORT \
  --config_file $ACC_CFG \
  src/training/grpo.py \
  --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_crosswords.yaml \
  --run_name ${RUN_NAME}-${TS} \
  --ignore_data_skip \
  --seed 42 \
  --use_vllm \
  --vllm_mode server \
  --vllm_server_base_url http://127.0.0.1:$VLLM_PORT \
  > '$TRAINING_LOG' 2>&1

# keep server tidy on normal exit
wait \$VLLM_PID
"
