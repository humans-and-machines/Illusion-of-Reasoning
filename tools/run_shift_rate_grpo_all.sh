#!/usr/bin/env bash
set -euo pipefail

# Run the default reasoning-shift aggregation over:
# - Qwen2.5-1.5B (Math, Crossword, Rush Hour) across all temps
# - Qwen2.5-7B (Math, all temps)
# - Llama-3.1-8B (Math, all temps)
#
# Usage (from repo root):
#   bash tools/run_shift_rate_grpo_all.sh

SCRIPT_DIR="$(
  cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd
)"

python "$SCRIPT_DIR/shift_rate_calc.py" \
  --root-group qwen15b_math Math \
    artifacts/results/GRPO-1.5B-math-temp-0.0 \
    artifacts/results/GRPO-1.5B-math-temp-0.05 \
    artifacts/results/GRPO-1.5B-math-temp-0.3 \
    artifacts/results/GRPO-1.5B-math-temp-0.7 \
  --root-group qwen15b_xword Crossword \
    artifacts/results/GRPO-1.5B-xword-temp-0 \
    artifacts/results/GRPO-1.5B-xword-temp-0.05 \
    artifacts/results/GRPO-1.5B-xword-temp-0.3 \
    artifacts/results/GRPO-1.5B-xword-temp-0.7 \
  --root-group qwen15b_rush "Rush Hour" \
    artifacts/results/GRPO-1.5B-carpark-temp-0 \
    artifacts/results/GRPO-1.5B-carpark-temp-0.05 \
    artifacts/results/GRPO-1.5B-carpark-temp-0.3 \
    artifacts/results/GRPO-1.5B-carpark-temp-0.7 \
  --root-group qwen7b_math Math \
    artifacts/results/GRPO-7B-math-temp-0 \
    artifacts/results/GRPO-7B-math-temp-0.05 \
    artifacts/results/GRPO-7B-math-temp-0.3 \
    artifacts/results/GRPO-7B-math-temp-0.7 \
  --root-group llama8b_math Math \
    artifacts/results/GRPO-Llama8B-math-temp-0 \
    artifacts/results/GRPO-Llama8B-math-temp-0.05 \
    artifacts/results/GRPO-Llama8B-math-temp-0.3 \
    artifacts/results/GRPO-Llama8B-math-temp-0.7

