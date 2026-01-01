#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Submit a single pass2 backfill Slurm job for one results JSONL.

Example:
  scripts/inference/sbatch_math_backfill_pass2_single.sh \
    --jsonl artifacts/results/GRPO-Llama8B-math-temp-0.7/step-400/step0400_test.jsonl \
    --account mltheory \
    --gres gpu:a100:1 \
    --time 02:00:00

Notes:
  - This uses scripts/inference/math-backfill-pass2-all.slurm under the hood.
  - Leaves MAX_PROBLEMS/MAX_ROWS unset, so it backfills everything missing in the file.
EOF
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

jsonl=""
family=""
temp=""
step=""

account="mltheory"
partition="gpu"
gres="gpu:a100:1"
time_limit="02:00:00"
batch_size="1"
dry_run="0"

while (( $# > 0 )); do
  case "$1" in
    --jsonl) jsonl="${2:-}"; shift 2 ;;
    --family) family="${2:-}"; shift 2 ;;
    --temp) temp="${2:-}"; shift 2 ;;
    --step) step="${2:-}"; shift 2 ;;
    --account) account="${2:-}"; shift 2 ;;
    --partition) partition="${2:-}"; shift 2 ;;
    --gres) gres="${2:-}"; shift 2 ;;
    --time) time_limit="${2:-}"; shift 2 ;;
    --batch-size) batch_size="${2:-}"; shift 2 ;;
    --dry-run) dry_run="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$jsonl" ]]; then
  echo "Missing required: --jsonl <path>" >&2
  usage
  exit 2
fi

if [[ "$jsonl" != /* ]]; then
  jsonl_abs="$PROJECT_ROOT/$jsonl"
else
  jsonl_abs="$jsonl"
fi
if [[ ! -f "$jsonl_abs" ]]; then
  echo "JSONL not found: $jsonl_abs" >&2
  exit 1
fi

# Prefer writing a manifest line with a path relative to the repo root when possible.
jsonl_for_manifest="$jsonl_abs"
if [[ "$jsonl_abs" == "$PROJECT_ROOT/"* ]]; then
  jsonl_for_manifest="${jsonl_abs#"$PROJECT_ROOT/"}"
fi

if [[ -z "$family" || -z "$temp" ]]; then
  if [[ "$jsonl_for_manifest" =~ /GRPO-(1\.5B|7B|Llama8B)-math-temp-([^/]+)/ ]]; then
    family="${family:-${BASH_REMATCH[1]}}"
    temp="${temp:-${BASH_REMATCH[2]}}"
  fi
fi
if [[ -z "$step" ]]; then
  base="$(basename "$jsonl_for_manifest")"
  if [[ "$base" =~ ^step([0-9]{4})_ ]]; then
    step="${BASH_REMATCH[1]}"
    step=$((10#$step))
  fi
fi

if [[ -z "$family" || -z "$temp" || -z "$step" ]]; then
  echo "Could not infer --family/--temp/--step from: $jsonl_for_manifest" >&2
  echo "Provide them explicitly, e.g. --family Llama8B --temp 0.7 --step 400" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
mkdir -p tmp

manifest="tmp/math_backfill_pass2_single_${family}_temp-${temp}_step-${step}_$(date +%s).tsv"
printf "%s\t%s\t%s\t%s\n" "$family" "$temp" "$step" "$jsonl_for_manifest" > "$manifest"

cmd=(
  sbatch
  --account "$account"
  --partition "$partition"
  --gres "$gres"
  --time "$time_limit"
  "scripts/inference/math-backfill-pass2-all.slurm"
  "MANIFEST=$manifest"
  "BATCH_SIZE=$batch_size"
)

echo "Manifest: $manifest"
printf "Submitting:"
printf " %q" "${cmd[@]}"
echo
if [[ "$dry_run" == "1" ]]; then
  exit 0
fi

"${cmd[@]}"
