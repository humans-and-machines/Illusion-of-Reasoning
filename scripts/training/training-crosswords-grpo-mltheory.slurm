#!/bin/bash
#SBATCH --job-name=OpenR1_GRPO
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=00:59:00
#SBATCH --output=logs/slurm_%j.out
set -euo pipefail

# ── Repo + Conda env ───────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT}"

source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-openr1}"
export PYTHONNOUSERSITE=1

# Optional sanity
python - <<'PY'
import torch; print("torch:", torch.__version__, "CUDA build:", torch.version.cuda)
PY

# ── IDs & logs ──────────────────────────────────────────────────────────
export RUN_NAME="Qwen1.5B-GRPO-Finetune"
export TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export CONFIG="recipes/Qwen2.5-1.5B-Instruct/grpo/config_crosswords.yaml"
export SERVER_LOG="logs/liv_vllm_${RUN_NAME}_${TIMESTAMP}.log"
export TRAINING_LOG="logs/liv_train_${RUN_NAME}_${TIMESTAMP}.log"

# ── Caches / W&B ────────────────────────────────────────────────────────
unset HF_TOKEN
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PROJECT_ROOT/.cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PROJECT_ROOT/.cache/huggingface/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$PROJECT_ROOT/.cache/huggingface/transformers}"
export TMPDIR="${TMPDIR:-$PROJECT_ROOT/.tmp}"
export HF_HUB_REQUEST_TIMEOUT=60

export WANDB_MODE=online
export WANDB_DIR=/n/fs/similarity/wandb-offload/tmp
export WANDB_ARTIFACT_DIR=/n/fs/similarity/wandb-offload/artifacts
export WANDB_CACHE_DIR=/n/fs/similarity/wandb-offload/cache
export WANDB_CONFIG_DIR=/n/fs/similarity/wandb-offload/config

export HUGGING_FACE_HUB_TOKEN="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"

# ── vLLM / Transformer safety ───────────────────────────────────────────
export VLLM_NO_USAGE_STATS=1
export VLLM_DO_NOT_TRACK=1
export DO_NOT_TRACK=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn     # required for CUDA+mp :contentReference[oaicite:2]{index=2}
unset VLLM_ATTENTION_BACKEND                  # let vLLM pick safe default
export TRANSFORMERS_NO_FLASH_ATTN=1           # avoid FA .so import on this stack

export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf_cache}"   # replaces TRANSFORMERS_CACHE


echo "🟢 Setup done — cfg=$CONFIG  logs: $SERVER_LOG | $TRAINING_LOG"

# ── Single srun step: vLLM (GPU#1) + training (rest) — DDP, NO DeepSpeed ──
srun --cpus-per-task=64 bash -c '
set -euo pipefail
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-openr1}"
export PYTHONNOUSERSITE=1

# Split exactly the GPUs Slurm assigned to this step (e.g., "0,5,7,8,3,2,6,1")
IFS=, read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:?}"
VLLM_GPU="${GPUS[0]}"
if (( ${#GPUS[@]} > 1 )); then
  TRAIN_GPUS="$(IFS=,; echo "${GPUS[*]:1}")"
  NUM_TRAINING=$(( ${#GPUS[@]} - 1 ))
else
  TRAIN_GPUS=""
  NUM_TRAINING=0
fi
echo "SLURM GPUs: ${CUDA_VISIBLE_DEVICES} | vLLM=${VLLM_GPU} | train=${TRAIN_GPUS:-<none>} (${NUM_TRAINING})"

# vLLM (must use spawn with CUDA + mp)
export VLLM_WORKER_MULTIPROC_METHOD=spawn
unset VLLM_ATTENTION_BACKEND
CUDA_VISIBLE_DEVICES="${VLLM_GPU}" trl vllm-serve \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dtype float16 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.90 \
  > "'"$SERVER_LOG"'" 2>&1 &

VLLM_PID=$!
until curl -sf http://localhost:8000/health >/dev/null; do
  echo "Waiting for vLLM…"; sleep 2
done
echo "✅ vLLM healthy"

# after "✅ vLLM healthy", inside the srun block
set +e
# after "✅ vLLM healthy"
CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" accelerate launch \
  --num_processes "${NUM_TRAINING}" \
  --num_machines 1 \
  --mixed_precision bf16 \
  --main_process_port 29510 \
  src/training/grpo.py \
    --config "$CONFIG" \
    --use_vllm \
    --run_name "${RUN_NAME}-${TIMESTAMP}" \
    --ignore_data_skip \
    --seed 42 \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
  > "$TRAINING_LOG" 2>&1
rc=$?
set -e
if (( rc != 0 )); then
  echo "❌ Training exited with code $rc — last 200 lines:"
  tail -n 200 "$TRAINING_LOG"
  kill $VLLM_PID || true
  exit $rc
fi


wait $VLLM_PID
'
