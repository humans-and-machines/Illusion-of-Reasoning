#!/usr/bin/env bash
#SBATCH --job-name=deepseek_math_azure
#SBATCH --output=logs/deepseek_math_azure_%j.out
#SBATCH --error=logs/deepseek_math_azure_%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
ulimit -n 4096

# ── Repo root ────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=0
export HF_HUB_OFFLINE=0
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_REQUEST_TIMEOUT=60

# ── Conda env ───────────────────────────────────────────────────────────────
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
CONDA_ENV="${CONDA_ENV:-$PROJECT_ROOT/openr1}"
conda activate "$CONDA_ENV"
echo "✅ Conda env: $(which python)"
python --version

# ── Azure auth check ────────────────────────────────────────────────────────
if [[ -z "${AZURE_OPENAI_API_KEY:-}" ]]; then
  echo "Missing AZURE_OPENAI_API_KEY env var." >&2
  exit 1
fi

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_PATH="${SCRIPT_PATH:-$PROJECT_ROOT/src/inference/math_deepseek_azure.py}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/artifacts/results/deepseek-r1-math-azure}"
mkdir -p "$OUTPUT_ROOT" logs

ENDPOINT="${AZURE_OPENAI_ENDPOINT:-https://example.openai.azure.com/}"
DEPLOYMENT="${AZURE_OPENAI_DEPLOYMENT:-deepseek-r1}"
API_VERSION="${AZURE_OPENAI_API_VERSION:-2024-12-01-preview}"

echo "→ Endpoint:   $ENDPOINT"
echo "→ Deployment: $DEPLOYMENT"
echo "→ Output:     $OUTPUT_ROOT"

python -u "$SCRIPT_PATH" \
  --output_dir "$OUTPUT_ROOT" \
  --dataset_id MATH-500 \
  --split test \
  --num_samples 1 \
  --temperature 0.6 \
  --top_p 0.95 \
  --max_output_tokens 900 \
  --step 0 \
  --endpoint "$ENDPOINT" \
  --deployment "$DEPLOYMENT" \
  --api_version "$API_VERSION" \
  --use_v1 1

echo "✓ Done → $OUTPUT_ROOT"
