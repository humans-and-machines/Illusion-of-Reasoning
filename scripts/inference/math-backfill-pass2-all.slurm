#!/usr/bin/env bash
# Backfill missing `"pass2": null` rows for all existing math result JSONLs
# across temperatures and steps, using the default (standard) second_pass_phrase.
#
# Usage (from repo root):
#   sbatch scripts/inference/math-backfill-pass2-all.slurm
#
# One-file helper:
#   scripts/inference/sbatch_math_backfill_pass2_single.sh --jsonl <path>
#
# Array mode (one JSONL per task):
#   python scripts/inference/write_math_backfill_manifest.py --out tmp/math_backfill_manifest.tsv
#   N=$(wc -l < tmp/math_backfill_manifest.tsv)
#   sbatch --array=0-$((N-1))%2000 scripts/inference/math-backfill-pass2-all.slurm MANIFEST=tmp/math_backfill_manifest.tsv
#
# Single multi-GPU job (no arrays; shards manifest across ranks):
#   sbatch --gres gpu:a100:8 --ntasks 8 \
#     scripts/inference/math-backfill-pass2-manifest-parallel.slurm \
#     MANIFEST=tmp/math_backfill_manifest.tsv BATCH_SIZE=1
#
# Optional env overrides:
#   RESULTS_ROOT=<path>        (default: $PROJECT_ROOT/artifacts/results)
#   ONLY_FAMILY=<1.5B|7B|Llama8B>
#   ONLY_TEMP=<e.g. 0.3>
#   MANIFEST=<path>            (optional; if set, uses that manifest instead of scanning)
#   DRY_RUN=1                 (print commands, don't run)
#   CONDA_ENV=<env name/path> (default: $PROJECT_ROOT/openr1)
#   BATCH_SIZE=<int>          (default: 1; use 1 to reduce CUDA memory)
#   FLUSH_EVERY=<int>         (optional; rewrite the JSONL every N backfilled rows, e.g. 8)
#   MIN_STEP=<int>            (optional)
#   MAX_STEP=<int>            (optional)
#   MAX_PROBLEMS=<int>        (optional; backfill at most N distinct problems per run)
#   MAX_ROWS=<int>            (optional; backfill at most N rows per run)
#
# Notes:
# - Runs in-place and creates a one-time backup per file: *.bak_pass2
# - Skips files that do not contain `"pass2": null`

#SBATCH --job-name=math_backfill_pass2
#SBATCH --output=logs/math_backfill_pass2_%j.out
#SBATCH --error=logs/math_backfill_pass2_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:59:00

set -euo pipefail
ulimit -n 4096

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
fi

cd "$PROJECT_ROOT"

source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
CONDA_ENV="${CONDA_ENV:-$PROJECT_ROOT/openr1}"
conda activate "$CONDA_ENV"
module load cudatoolkit/12.6

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$PROJECT_ROOT/logs"

RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_ROOT/artifacts/results}"
ONLY_FAMILY="${ONLY_FAMILY:-}"
ONLY_TEMP="${ONLY_TEMP:-}"
MANIFEST="${MANIFEST:-}"
DRY_RUN="${DRY_RUN:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
FLUSH_EVERY="${FLUSH_EVERY:-}"
MIN_STEP="${MIN_STEP:-}"
MAX_STEP="${MAX_STEP:-}"
MAX_PROBLEMS="${MAX_PROBLEMS:-}"
MAX_ROWS="${MAX_ROWS:-}"

SCRIPT_MOD="src.inference.cli.backfill_math_pass2"

# Allow passing common knobs as positional args, e.g.:
#   sbatch ... math-backfill-pass2-all.slurm MANIFEST=tmp/m.tsv ONLY_FAMILY=7B ONLY_TEMP=0.3 DRY_RUN=1
for arg in "$@"; do
  case "$arg" in
    RESULTS_ROOT=*) RESULTS_ROOT="${arg#RESULTS_ROOT=}" ;;
    ONLY_FAMILY=*) ONLY_FAMILY="${arg#ONLY_FAMILY=}" ;;
    ONLY_TEMP=*) ONLY_TEMP="${arg#ONLY_TEMP=}" ;;
    MANIFEST=*) MANIFEST="${arg#MANIFEST=}" ;;
    DRY_RUN=*) DRY_RUN="${arg#DRY_RUN=}" ;;
    BATCH_SIZE=*) BATCH_SIZE="${arg#BATCH_SIZE=}" ;;
    FLUSH_EVERY=*) FLUSH_EVERY="${arg#FLUSH_EVERY=}" ;;
    MIN_STEP=*) MIN_STEP="${arg#MIN_STEP=}" ;;
    MAX_STEP=*) MAX_STEP="${arg#MAX_STEP=}" ;;
    MAX_PROBLEMS=*) MAX_PROBLEMS="${arg#MAX_PROBLEMS=}" ;;
    MAX_ROWS=*) MAX_ROWS="${arg#MAX_ROWS=}" ;;
  esac
done

if command -v rg >/dev/null 2>&1; then
  HAS_RG=1
else
  HAS_RG=0
fi

echo "→ Backfill pass2 (standard phrase)"
echo "   root: $RESULTS_ROOT"
if [[ -n "$ONLY_FAMILY" ]]; then echo "   only family: $ONLY_FAMILY"; fi
if [[ -n "$ONLY_TEMP" ]]; then echo "   only temp  : $ONLY_TEMP"; fi
if [[ -n "$MANIFEST" ]]; then echo "   manifest   : $MANIFEST"; fi
if [[ "$DRY_RUN" == "1" ]]; then echo "   DRY_RUN=1"; fi
echo "   batch_size : $BATCH_SIZE"
if [[ -n "$FLUSH_EVERY" ]]; then echo "   flush_every: $FLUSH_EVERY"; fi
if [[ -n "$MIN_STEP" ]]; then echo "   min step   : $MIN_STEP"; fi
if [[ -n "$MAX_STEP" ]]; then echo "   max step   : $MAX_STEP"; fi
if [[ -n "$MAX_PROBLEMS" ]]; then echo "   max_problems: $MAX_PROBLEMS"; fi
if [[ -n "$MAX_ROWS" ]]; then echo "   max_rows    : $MAX_ROWS"; fi

manifest_lines=()
if [[ -n "$MANIFEST" ]]; then
  if [[ ! -f "$MANIFEST" ]]; then
    echo "Manifest not found: $MANIFEST"
    echo "  pwd: $(pwd)"
    echo "  PROJECT_ROOT: $PROJECT_ROOT"
    echo "  Tip: pass an absolute path, e.g. MANIFEST=$PROJECT_ROOT/tmp/your.tsv"
    echo "  Listing manifest dir:"
    ls -lah "$(dirname "$MANIFEST")" 2>/dev/null || true
    exit 1
  fi
  while IFS=$'\t' read -r fam temp step jsonl_path; do
    [[ -n "${fam:-}" && -n "${temp:-}" && -n "${step:-}" && -n "${jsonl_path:-}" ]] || continue
    # Preserve real tab separators for later parsing.
    manifest_lines+=("${fam}"$'\t'"${temp}"$'\t'"${step}"$'\t'"${jsonl_path}")
  done < "$MANIFEST"
  if (( ${#manifest_lines[@]} == 0 )); then
    echo "Manifest is empty: $MANIFEST"
    exit 0
  fi
fi

temp_roots=()
if [[ -z "$MANIFEST" ]]; then
  for pat in \
    "$RESULTS_ROOT/GRPO-1.5B-math-temp-"* \
    "$RESULTS_ROOT/GRPO-7B-math-temp-"* \
    "$RESULTS_ROOT/GRPO-Llama8B-math-temp-"*; do
    [[ -d "$pat" ]] || continue
    temp_roots+=("$pat")
  done

  if (( ${#temp_roots[@]} == 0 )); then
    echo "No math temp roots found under: $RESULTS_ROOT"
    exit 0
  fi
fi

total_files=0
total_backfilled=0
total_skipped=0

run_one_target() {
  local family="$1"
  local temp="$2"
  local step="$3"
  local jsonl="$4"

  local base_model=""
  local tokenizer_path=""
  local ckpt_root=""

  case "$family" in
    1.5B)
      base_model="Qwen/Qwen2.5-1.5B-Instruct"
      tokenizer_path="Qwen/Qwen2.5-1.5B-Instruct"
      ckpt_root="$PROJECT_ROOT/artifacts/models/Qwen2.5-1.5B/Qwen2.5-1.5B-Open-R1-GRPO-math-v1"
      ;;
    7B)
      base_model="Qwen/Qwen2.5-7B-Instruct"
      tokenizer_path="Qwen/Qwen2.5-7B-Instruct"
      ckpt_root="$PROJECT_ROOT/artifacts/models/Qwen2.5-7B-Open-R1-GRPO-math-7b"
      ;;
    Llama8B)
      base_model="meta-llama/Meta-Llama-3.1-8B-Instruct"
      tokenizer_path="meta-llama/Meta-Llama-3.1-8B-Instruct"
      ckpt_root="$PROJECT_ROOT/artifacts/models/open-r1/Llama-8B-Open-R1-GRPO-math-v2"
      ;;
    *)
      echo "  [skip] unknown family: $family"
      total_skipped=$((total_skipped + 1))
      return 0
      ;;
  esac

  if [[ -n "$ONLY_FAMILY" && "$family" != "$ONLY_FAMILY" ]]; then
    total_skipped=$((total_skipped + 1))
    return 0
  fi
  if [[ -n "$ONLY_TEMP" && "$temp" != "$ONLY_TEMP" ]]; then
    total_skipped=$((total_skipped + 1))
    return 0
  fi
  if [[ -n "$MIN_STEP" && "$step" -lt "$MIN_STEP" ]]; then
    total_skipped=$((total_skipped + 1))
    return 0
  fi
  if [[ -n "$MAX_STEP" && "$step" -gt "$MAX_STEP" ]]; then
    total_skipped=$((total_skipped + 1))
    return 0
  fi

  if [[ ! -f "$jsonl" ]]; then
    echo "  [skip] missing file: $jsonl"
    total_skipped=$((total_skipped + 1))
    return 0
  fi

  # Fast skip if no missing pass2 in the file.
  local has_missing=0
  if (( HAS_RG == 1 )); then
    rg -q -m 1 "\"pass2\":\\s*null" "$jsonl" || has_missing=1
  else
    grep -q "\"pass2\":[[:space:]]*null" "$jsonl" || has_missing=1
  fi
  if (( has_missing != 0 )); then
    total_skipped=$((total_skipped + 1))
    return 0
  fi

  local model_name_or_path="$base_model"
  if (( step != 0 )); then
    local ckpt_dir="$ckpt_root/checkpoint-$step"
    if [[ ! -d "$ckpt_dir" ]]; then
      echo "  [skip] missing checkpoint dir: $ckpt_dir (file: $jsonl)"
      total_skipped=$((total_skipped + 1))
      return 0
    fi
    model_name_or_path="$ckpt_dir"
  fi

  local -a cmd
  cmd=(
    python -m "$SCRIPT_MOD"
    --model_name_or_path "$model_name_or_path"
    --tokenizer_path "$tokenizer_path"
    --input_jsonl "$jsonl"
    --inplace
    --batch_size "$BATCH_SIZE"
    --temperature "$temp"
    --top_p 0.95
    --entropy_mode full
    --think_cap 750
    --answer_cap 50
  )
  if [[ -n "$MAX_PROBLEMS" ]]; then
    cmd+=(--max_problems "$MAX_PROBLEMS")
  fi
  if [[ -n "$MAX_ROWS" ]]; then
    cmd+=(--max_rows "$MAX_ROWS")
  fi
  if [[ -n "$FLUSH_EVERY" ]]; then
    cmd+=(--flush_every "$FLUSH_EVERY")
  fi

  echo "  → backfill family=$family temp=$temp step=$step file=$jsonl"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf "    %q " "${cmd[@]}"; echo
  else
    "${cmd[@]}"
  fi
  total_backfilled=$((total_backfilled + 1))
}

if [[ -n "$MANIFEST" ]]; then
  total_files=${#manifest_lines[@]}
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    idx="${SLURM_ARRAY_TASK_ID}"
    if (( idx < 0 || idx >= ${#manifest_lines[@]} )); then
      echo "Array index $idx out of range (0..$(( ${#manifest_lines[@]} - 1 ))); exiting."
      echo "Manifest lines: ${#manifest_lines[@]}"
      echo "Tip: resubmit with:"
      echo "  N=$(wc -l < \"$MANIFEST\")"
      echo "  sbatch --array=0-$((N-1))%2000 scripts/inference/math-backfill-pass2-all.slurm MANIFEST=\"$MANIFEST\""
      exit 0
    fi
    IFS=$'\t' read -r family temp step jsonl <<< "${manifest_lines[$idx]}"
    run_one_target "$family" "$temp" "$step" "$jsonl"
    echo "✓ Done (array task $idx)"
    exit 0
  fi

  for line in "${manifest_lines[@]}"; do
    IFS=$'\t' read -r family temp step jsonl <<< "$line"
    run_one_target "$family" "$temp" "$step" "$jsonl"
  done

  echo "✓ Done"
  echo "  total jsonls listed   : $total_files"
  echo "  backfill commands run : $total_backfilled"
  echo "  skipped              : $total_skipped"
  exit 0
fi

for temp_root in "${temp_roots[@]}"; do
  root_name="$(basename "$temp_root")"
  temp="${root_name##*temp-}"

  case "$root_name" in
    GRPO-1.5B-math-temp-*) family="1.5B" ;;
    GRPO-7B-math-temp-*) family="7B" ;;
    GRPO-Llama8B-math-temp-*) family="Llama8B" ;;
    *) continue ;;
  esac

  if [[ -n "$ONLY_FAMILY" && "$family" != "$ONLY_FAMILY" ]]; then
    continue
  fi
  if [[ -n "$ONLY_TEMP" && "$temp" != "$ONLY_TEMP" ]]; then
    continue
  fi

  echo "— family=$family temp=$temp root=$temp_root"

  shopt -s nullglob
  step_dirs=("$temp_root"/base-step0 "$temp_root"/step-* "$temp_root"/step*)
  shopt -u nullglob

  for step_dir in "${step_dirs[@]}"; do
    [[ -d "$step_dir" ]] || continue

    shopt -s nullglob
    jsonls=("$step_dir"/step????_*.jsonl)
    shopt -u nullglob

    for jsonl in "${jsonls[@]}"; do
      [[ -f "$jsonl" ]] || continue
      total_files=$((total_files + 1))

      base="$(basename "$jsonl")"
      step_str="${base#step}"
      step_str="${step_str%%_*}"
      step=$((10#$step_str))
      run_one_target "$family" "$temp" "$step" "$jsonl"
    done
  done
done

echo "✓ Done"
echo "  total jsonls seen     : $total_files"
echo "  backfill commands run : $total_backfilled"
echo "  skipped              : $total_skipped"
