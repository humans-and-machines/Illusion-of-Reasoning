#!/usr/bin/env bash
#SBATCH --job-name=math_compare_llama8b_4tempsXckpts
#SBATCH --output=logs/math_compare_llama8b_%A_%a.out
#SBATCH --error=logs/math_compare_llama8b_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:59:00
# NOTE: 4 temps (0,0.05,0.3,0.7) × 11 ckpts (0,50,...,500) = 44 tasks
#SBATCH --array=0-43

set -euo pipefail
ulimit -n 4096

# ── Repo root ────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT}"

# ── Logging/runtime ──────────────────────────────────────────────────────────
export LOGLEVEL=DEBUG
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Conda env ───────────────────────────────────────────────────────────────
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
CONDA_ENV="${CONDA_ENV:-$PROJECT_ROOT/openr1}"
conda activate "$CONDA_ENV"
echo "✅ Conda env: $(which python)"
python --version

module load cudatoolkit/12.6

# ── HF auth / behavior ───────────────────────────────────────────────────────
# Allow overriding the Hub token via env; fall back to this default only
# if nothing is set in the submission environment.
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-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}"

# HF online (for base model pull)
export TRANSFORMERS_OFFLINE=0
export HF_HUB_OFFLINE=0
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_REQUEST_TIMEOUT=60
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT"

# ── W&B (optional) ──────────────────────────────────────────────────────────
export WANDB_MODE=online
export WANDB_DIR=/n/fs/similarity/wandb-offload/tmp
export WANDB_ARTIFACT_DIR=/n/fs/similarity/wandb-offload/artifacts
export WANDB_CACHE_DIR=/n/fs/similarity/wandb-offload/cache
mkdir -p "$WANDB_DIR" "$WANDB_ARTIFACT_DIR" "$WANDB_CACHE_DIR" logs

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_PATH="${SCRIPT_PATH:-$PROJECT_ROOT/src/inference/cli/unified_math.py}"
MODEL_ROOT="${MODEL_ROOT:-$PROJECT_ROOT/artifacts/models/open-r1/Llama-8B-Open-R1-GRPO-math-v2}"
BASE_RESULTS_ROOT="${BASE_RESULTS_ROOT:-$PROJECT_ROOT/artifacts/results}"
TOKENIZER_PATH="${TOKENIZER_PATH:-meta-llama/Meta-Llama-3.1-8B-Instruct}"

# Tags for directory naming
MODEL_TAG="${MODEL_TAG:-Llama8B}"
DOMAIN_TAG="${DOMAIN_TAG:-math}"

# ── Grid ────────────────────────────────────────────────────────────────────
# Four temperatures across checkpoints 0..500 in steps of 50.
temps=(0 0.05 0.3 0.7)
ckpts=(0 50 100 150 200 250 300 350 400 450 500)

n_temps=${#temps[@]}
n_ckpts=${#ckpts[@]}
n_total=$(( n_temps * n_ckpts ))

# Safety: ensure array index in range
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= n_total )); then
  echo "Array index ${SLURM_ARRAY_TASK_ID} out of range (0..$((n_total-1)))"
  exit 1
fi

# Map array index -> (temp, ckpt)
idx="${SLURM_ARRAY_TASK_ID}"
temp_idx=$(( idx % n_temps ))
ckpt_idx=$(( idx / n_temps ))

TEMP="${temps[$temp_idx]}"
STEP="${ckpts[$ckpt_idx]}"

# ── Model selection ─────────────────────────────────────────────────────────
# Step 0 = base Llama-3.1 8B Instruct from HF; tuned steps load from $MODEL_ROOT/checkpoint-<STEP>
if [[ "$STEP" -eq 0 ]]; then
  MODEL_NAME_OR_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct"
  TAG="base-step0"
  STEP_FLAG="--step 0"
else
  CKPT_DIR="$MODEL_ROOT/checkpoint-$STEP"
  if [[ ! -d "$CKPT_DIR" ]]; then
    echo "Missing tuned checkpoint: $CKPT_DIR – downloading from Hugging Face..."
    STEP="$STEP" MODEL_ROOT="$MODEL_ROOT" python - << 'PY'
import os

from huggingface_hub import snapshot_download

repo_id = "od2961/Llama-8B-Open-R1-GRPO-math-v2"

# Map training step -> HF commit hash (from model card)
STEP_TO_REVISION = {
    50: "1d1ffd0",
    100: "fc9526c",
    150: "159ec0c",
    200: "6534664",
    250: "c1fcfaf",
    300: "dd22a84",
    350: "1c57740",
    400: "a86a5d7",
    450: "3c2b06c",
    500: "6e8fe74",
}

step = int(os.environ["STEP"])
if step == 0:
    raise SystemExit("STEP=0 should not trigger checkpoint download")

rev = STEP_TO_REVISION.get(step)
if rev is None:
    raise SystemExit(f"No revision mapping for step {step}")

model_root = os.environ["MODEL_ROOT"]
local_dir = os.path.join(model_root, f"checkpoint-{step}")
os.makedirs(local_dir, exist_ok=True)

print(f"→ snapshot_download(repo_id={repo_id!r}, revision={rev!r}) → {local_dir}")
snapshot_download(
    repo_id=repo_id,
    revision=rev,
    local_dir=local_dir,
    resume_download=True,
    max_workers=4,
)
PY
  fi
  MODEL_NAME_OR_PATH="$CKPT_DIR"
  TAG="tuned-step${STEP}"
  STEP_FLAG="--step ${STEP}"
fi

# ── Output layout ───────────────────────────────────────────────────────────
# /.../artifacts/results/GRPO-<MODEL_TAG>-<DOMAIN_TAG>-temp-<TEMP>/step-<STEP>
OUT_PREFIX="${BASE_RESULTS_ROOT}/GRPO-${MODEL_TAG}-${DOMAIN_TAG}-temp-${TEMP}"
OUTDIR="${OUT_PREFIX}/step-${STEP}"
mkdir -p "$OUTDIR"

# Per-run caches (avoid contention / NFS slowness)
CACHEROOT="$OUTDIR/hf_cache"
mkdir -p "$CACHEROOT" "$OUTDIR/.triton" "$OUTDIR/.torchinductor" "$OUTDIR/.tmp"
export HF_HOME="$CACHEROOT"
export TRANSFORMERS_CACHE="$CACHEROOT/transformers"
export HF_HUB_CACHE="$CACHEROOT/hub"
export TRITON_CACHE_DIR="$OUTDIR/.triton"
export TORCHINDUCTOR_CACHE_DIR="$OUTDIR/.torchinductor"
export TMPDIR="$OUTDIR/.tmp"

echo "→ Task ${SLURM_ARRAY_TASK_ID}/${n_total}: TEMP=${TEMP}, STEP=${STEP}"
echo "→ Model:  $MODEL_NAME_OR_PATH"
echo "→ Output: $OUTDIR"

# ── Inference (mirror your Qwen run) ────────────────────────────────────────
DTYPE="${DTYPE:-float16}"
RECONSIDER_CUE="Hold on, this reasoning might be wrong. Let's go back and check each step carefully. Actually, this approach doesn't look correct. Let's restart and work through the solution more systematically. Wait, we need to reconsider. Let's think this through step by step."

# Allow small-chunk sweeps via env overrides, mirroring math-inference.slurm.
NUM_EXAMPLES="${NUM_EXAMPLES:-1000000}"
DATASET_START="${DATASET_START:-0}"

python -u "$SCRIPT_PATH" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --output_dir "$OUTDIR" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --batch_size 1 \
  --entropy_mode full \
  --num_examples "$NUM_EXAMPLES" \
  --dataset_start "$DATASET_START" \
  --num_samples 8 \
  --temperature "$TEMP" \
  --top_p 0.95 \
  --seed 42 \
  --dtype "$DTYPE" \
  --dataset_id MATH-500 \
  --split test \
  --two_pass \
  --second_pass_phrase "$RECONSIDER_CUE" \
  --think_cap 750 \
  --answer_cap 50 \
  $STEP_FLAG

echo "✓ Done: ${TAG} → $OUTDIR"
