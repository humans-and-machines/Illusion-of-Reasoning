#!/usr/bin/env bash

# Run Formal Aha threshold search across all Qwen2.5-1.5B roots and temperatures.
#
# Usage:
#   tools/run_threshold_search_q15b_all.sh [bootstrap_draws]
# If bootstrap_draws is omitted, defaults to 2000. Results are written to
# artifacts/threshold_search_q15b/, one CSV per root, and stdout per run is
# tee'd into matching .log files.

set -euo pipefail

BOOT=${1:-2000}
OUT_DIR="artifacts/threshold_search_q15b"
mkdir -p "${OUT_DIR}"

# Flat list so each token is a separate CLI arg
GRIDS=(
  --delta1_grid 0,0.125,0.25
  --delta2_grid 0,0.125,0.25
  --delta3_grid none,0.0,0.05,0.125
)

declare -a MATH_ROOTS=(
  artifacts/results/GRPO-1.5B-math-temp-0.0
  artifacts/results/GRPO-1.5B-math-temp-0.05
  artifacts/results/GRPO-1.5B-math-temp-0.3
  artifacts/results/GRPO-1.5B-math-temp-0.7
)

declare -a XWORD_ROOTS=(
  artifacts/results/GRPO-1.5B-xword-temp-0
  artifacts/results/GRPO-1.5B-xword-temp-0.05
  artifacts/results/GRPO-1.5B-xword-temp-0.3
  artifacts/results/GRPO-1.5B-xword-temp-0.7
)

declare -a RUSH_ROOTS=(
  artifacts/results/GRPO-1.5B-carpark-temp-0
  artifacts/results/GRPO-1.5B-carpark-temp-0.05
  artifacts/results/GRPO-1.5B-carpark-temp-0.3
  artifacts/results/GRPO-1.5B-carpark-temp-0.7
)

run_one() {
  local root="$1"
  local tag
  tag=$(basename "${root}")
  local csv="${OUT_DIR}/${tag}.csv"
  local log="${OUT_DIR}/${tag}.log"
  echo "[info] Running threshold search for ${root} -> ${csv}"
  python -m src.analysis.threshold_search \
    --results_root "${root}" \
    "${GRIDS[@]}" \
    --min_prior_steps 2 \
    --bootstrap_draws "${BOOT}" \
    --require_positive_ci \
    --min_events 5 \
    --top_k 10 \
    --out_csv "${csv}" | tee "${log}"
}

for r in "${MATH_ROOTS[@]}"; do run_one "${r}"; done
for r in "${XWORD_ROOTS[@]}"; do run_one "${r}"; done
for r in "${RUSH_ROOTS[@]}"; do run_one "${r}"; done

echo "[done] Wrote CSV/logs to ${OUT_DIR}"
