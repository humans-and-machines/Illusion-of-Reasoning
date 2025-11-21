#!/bin/bash
#SBATCH --job-name=OpenR1_GRPO
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%j.out

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT}"

# ── Conda env ───────────────────────────────────────────────────────────
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-openr1}"
export PYTHONNOUSERSITE=1

# ── IDs & logs ──────────────────────────────────────────────────────────
export RUN_NAME="Qwen1.5B-GRPO-Finetune"
export TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export CONFIG="recipes/Qwen2.5-1.5B-Instruct/grpo/config_carpark.yaml"
export SERVER_LOG="logs/liv_vllm_${RUN_NAME}_${TIMESTAMP}.log"
export TRAINING_LOG="logs/liv_train_${RUN_NAME}_${TIMESTAMP}.log"
mkdir -p logs

# ── HF / caches (non-NFS compile caches) ───────────────────────────────
unset HF_TOKEN
unset TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PROJECT_ROOT/.cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PROJECT_ROOT/.triton/${SLURM_JOB_ID}}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$PROJECT_ROOT/.torchinductor/${SLURM_JOB_ID}}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# Token (hardcoded, per your note)
export HUGGING_FACE_HUB_TOKEN="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"

# W&B
export WANDB_MODE=online
export WANDB_DIR=/n/fs/similarity/wandb-offload/tmp
export WANDB_ARTIFACT_DIR=/n/fs/similarity/wandb-offload/artifacts
export WANDB_CACHE_DIR=/n/fs/similarity/wandb-offload/cache
export WANDB_CONFIG_DIR=/n/fs/similarity/wandb-offload/config
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$PROJECT_ROOT/wandb}"
export HF_HUB_REQUEST_TIMEOUT=60

# General tmp
export TMPDIR="${TMPDIR:-$PROJECT_ROOT/.tmp}"
mkdir -p "$TMPDIR"

# ── vLLM / transformers safety ─────────────────────────────────────────
export VLLM_NO_USAGE_STATS=1
export VLLM_DO_NOT_TRACK=1
export DO_NOT_TRACK=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
unset VLLM_ATTENTION_BACKEND
export FLASH_ATTENTION_FORCE_DISABLED=1
export TRANSFORMERS_NO_FLASH_ATTN=1
export TORCH_FORCE_FULL_STATE_DICT=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── DeepSpeed: hard-disable ────────────────────────────────────────────
unset DEEPSPEED_CONFIG_FILE
export ACCELERATE_USE_DEEPSPEED=0
export TRANSFORMERS_USE_DEEPSPEED=0
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export DS_BUILD_CPU_ADAM=0
export DS_BUILD_FUSED_ADAM=0
export DS_BUILD_FUSED_LAMB=0
export DS_BUILD_TRANSFORMER_INFERENCE=0

# Strip any lingering `deepspeed:` keys in your YAML (silent)
python - <<'PY'
import os, yaml
p=os.environ["CONFIG"]
with open(p) as f: cfg=yaml.safe_load(f)
def strip_ds(x):
    if isinstance(x, dict):
        x.pop("deepspeed", None)
        for v in x.values(): strip_ds(v)
    elif isinstance(x, list):
        for v in x: strip_ds(v)
strip_ds(cfg)
with open(p,"w") as f: yaml.safe_dump(cfg, f, sort_keys=False)
PY


export RESUME_CKPT="${RESUME_CKPT:-$PROJECT_ROOT/artifacts/models/open-r1/Qwen2.5-1.5B-Open-R1-GRPO-carpark-v1/checkpoint-950}"


# ── Launch: vLLM (1 GPU) + Trainer (remaining GPUs) ────────────────────
srun --gres=gpu:8 --cpus-per-task=64 bash -c '
set -euo pipefail
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
CONDA_ENV="${CONDA_ENV:-'"$PROJECT_ROOT"'/openr1}"
conda activate "$CONDA_ENV"
export PYTHONNOUSERSITE=1

export VLLM_USE_V1=0
export FLASH_ATTENTION_FORCE_DISABLED=1
export TRANSFORMERS_NO_FLASH_ATTN=1
unset VLLM_ATTENTION_BACKEND

IFS=, read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:?}"
VLLM_GPU="${GPUS[0]}"
if (( ${#GPUS[@]} > 1 )); then
  TRAIN_GPUS="$(IFS=,; echo "${GPUS[*]:1}")"
  NUM_TRAINING=$(( ${#GPUS[@]} - 1 ))
else
  TRAIN_GPUS=""
  NUM_TRAINING=0
fi

CUDA_VISIBLE_DEVICES="${VLLM_GPU}" trl vllm-serve \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dtype float16 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.90 \
  > "'"$SERVER_LOG"'" 2>&1 &

VLLM_PID=$!
ok=0
for i in {1..120}; do
  curl -sf http://localhost:8000/health >/dev/null && ok=1 && break
  kill -0 "$VLLM_PID" 2>/dev/null || { wait $VLLM_PID; exit 1; }
  sleep 2
done
[[ ${ok} -eq 1 ]] || { kill $VLLM_PID 2>/dev/null || true; exit 1; }

if (( NUM_TRAINING > 0 )); then
  set +e
  CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" accelerate launch \
    --num_processes "${NUM_TRAINING}" \
    --num_machines 1 \
    --mixed_precision bf16 \
    --main_process_port 29518 \
    src/training/grpo.py \
      --config "'"$CONFIG"'" \
      --use_vllm \
      --run_name "'"${RUN_NAME}-${TIMESTAMP}"'" \
      --ignore_data_skip \
      --seed 42 \
      --vllm_server_host 127.0.0.1 \
      --vllm_server_port 8000 \
      --resume_from_checkpoint "'"$RESUME_CKPT"'" \
    > "'"$TRAINING_LOG"'" 2>&1
  rc=$?
  set -e
  (( rc == 0 )) || { kill $VLLM_PID 2>/dev/null || true; exit $rc; }
fi

wait $VLLM_PID
'
