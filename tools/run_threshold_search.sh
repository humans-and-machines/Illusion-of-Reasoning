#!/usr/bin/env bash

# Run Formal Aha threshold grid search with bootstrap CIs.
#
# Usage:
#   tools/run_threshold_search.sh /path/to/pass1_jsonl_root /tmp/threshold_grid.csv
# Both arguments are optional; if OUT_CSV is omitted the grid is not saved.
#
# Notes:
# - Point RESULTS_ROOT at the directory containing PASS-1 JSONL logs
#   (typically a per-run "pass1/" folder, not the top-level "logs/").
# - Adjust the grid/flags below as needed for your dev slab.

set -euo pipefail

RESULTS_ROOT=${1:-}
OUT_CSV=${2:-}

if [[ -z "${RESULTS_ROOT}" ]]; then
  echo "Usage: $0 RESULTS_ROOT [OUT_CSV]" >&2
  exit 1
fi

CMD=(python -m src.analysis.threshold_search
  --results_root "${RESULTS_ROOT}"
  --delta1_grid "0,0.125,0.25"
  --delta2_grid "0,0.125,0.25"
  --delta3_grid "none,0.0,0.05,0.125"
  --min_prior_steps 2
  --bootstrap_draws 2000
  --require_positive_ci
  --min_events 5
  --top_k 10
)

# Optional outputs
if [[ -n "${OUT_CSV}" ]]; then
  CMD+=(--out_csv "${OUT_CSV}")
fi

echo "[info] Running threshold search over dev slab:"
printf '  %q ' "${CMD[@]}"
echo

"${CMD[@]}"
