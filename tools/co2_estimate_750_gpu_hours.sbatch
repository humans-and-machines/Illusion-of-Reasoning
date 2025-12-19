#!/bin/bash
#SBATCH --job-name=co2_110kg
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/co2_estimate_750.out
#SBATCH --error=logs/co2_estimate_750.err

# Run from the submission directory (repo root).
cd "${SLURM_SUBMIT_DIR:-.}"

python src/utils/co2_estimator.py \
  --gpu-hours 750 \
  --per-gpu-kw 0.35 \
  --pue 1.2 \
  --kg-per-kwh 0.35 \
  --round-to 5

