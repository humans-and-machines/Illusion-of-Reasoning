#!/usr/bin/env bash
# Generate the balanced Rush Hour dataset (4x4/5x5/6x6) and optionally push to HF.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/openr1/bin/python}"

# Core knobs (override via env)
SIZES="${SIZES:-4 5 6}"
OUT_DIR="${OUT_DIR:-${ROOT}/data/car_park/rush4-5-6-balanced}"
LIMIT_PER_SIZE="${LIMIT_PER_SIZE:-200000}"
SAMPLE_SIZES="${SAMPLE_SIZES:-6}"   # space-separated for multiple sizes, e.g., "6" or "5 6"
TARGET_PER_SIZE="${TARGET_PER_SIZE:-6:200000}"
MAX_PIECES_PER_SIZE="${MAX_PIECES_PER_SIZE:-4:8,5:10,6:12}"
MIN_EMPTIES_PER_SIZE="${MIN_EMPTIES_PER_SIZE:-4:1,5:2,6:3}"
MAX_NODES="${MAX_NODES:-500000}"
NUM_WORKERS="${NUM_WORKERS:-60}"
DIFFICULTY_SCHEME="${DIFFICULTY_SCHEME:-fixed}"
FIXED_THRESHOLDS="${FIXED_THRESHOLDS:-3,5}"
BALANCE_PER_SIZE="${BALANCE_PER_SIZE:-1}"
SPLIT="${SPLIT:-0.8,0.1,0.1}"
SEED="${SEED:-42}"

# Hub settings
DATASET_ID="${DATASET_ID:-od2961/rush4-5-balanced}"
PUSH_TO_HUB="${PUSH_TO_HUB:-0}"  # set to 1 to push

mkdir -p "${OUT_DIR}"

read -r -a size_arr <<<"${SIZES}"
read -r -a sample_arr <<<"${SAMPLE_SIZES}"

cmd=("${PYTHON_BIN}" "${ROOT}/data/car_park/build_rush_small_balanced.py"
  --out_dir "${OUT_DIR}"
  --limit_per_size "${LIMIT_PER_SIZE}"
  --sample_sizes "${sample_arr[@]}"
  --target_per_size "${TARGET_PER_SIZE}"
  --max_pieces_per_size "${MAX_PIECES_PER_SIZE}"
  --min_empties_per_size "${MIN_EMPTIES_PER_SIZE}"
  --max_nodes "${MAX_NODES}"
  --num_workers "${NUM_WORKERS}"
  --difficulty_scheme "${DIFFICULTY_SCHEME}"
  --fixed_thresholds "${FIXED_THRESHOLDS}"
  --split "${SPLIT}"
  --seed "${SEED}"
)

for s in "${size_arr[@]}"; do
  cmd+=(--sizes "${s}")
done

if [[ "${BALANCE_PER_SIZE}" == "1" ]]; then
  cmd+=(--balance_per_size)
fi

if [[ "${PUSH_TO_HUB}" == "1" ]]; then
  [[ -n "${DATASET_ID}" ]] || { echo "PUSH_TO_HUB=1 requires DATASET_ID"; exit 1; }
  cmd+=(--push_to_hub --dataset_id "${DATASET_ID}")
fi

echo "Running: ${cmd[*]}"
PYTHONNOUSERSITE=1 "${cmd[@]}"

echo "Done. Outputs in ${OUT_DIR}"
