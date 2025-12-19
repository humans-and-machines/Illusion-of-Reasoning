#!/usr/bin/env bash
set -euo pipefail

cd /n/fs/similarity/Illusion-of-Reasoning

temps=(0 0.05 0.3 0.7)

for T in "${temps[@]}"; do
  python -m src.analysis.heatmap_1 \
    --root_crossword "artifacts/results/GRPO-1.5B-xword-temp-$T" \
    --root_math      "artifacts/results/GRPO-1.5B-math-temp-$T" \
    --root_math2     "artifacts/results/GRPO-7B-math-temp-$T" \
    --root_math3     "artifacts/results/GRPO-Llama8B-math-temp-$T" \
    --root_carpark   "artifacts/results/GRPO-1.5B-carpark-temp-$T" \
    --split test \
    --dataset_name "Crossword+Math+RushHour" \
    --model_name "Qwen2.5-1.5B_T$T" \
    --no_titles \
    --out_dir "paper_figs/aha_heatmaps/Q1p5B_T$T"
done

for T in "${temps[@]}"; do
  python -m src.analysis.heatmap_1 \
    --root_math2 "artifacts/results/GRPO-7B-math-temp-$T" \
    --root_math3 "artifacts/results/GRPO-Llama8B-math-temp-$T" \
    --split test \
    --dataset_name "MATH-500" \
    --model_name "Qwen2.5-7B+Llama8B_T$T" \
    --no_titles \
    --out_dir "paper_figs/aha_heatmaps/Q7B_L8B_Math_T$T"
done

