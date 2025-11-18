#!/bin/bash
#SBATCH --job-name=RushBalanced           # <-- change as you like
#SBATCH --nodes=1                         # <-- you can change
#SBATCH --cpus-per-task=64                # <-- you can change
#SBATCH --mem=256G                        # <-- you can change
#SBATCH --time=128:00:00                   # <-- you can change
#SBATCH --output=logs/slurm_%j.out        # <-- you can change

set -euo pipefail

# (optional) load modules / activate env
# module load python/3.10  # or your site’s module
# source /path/to/conda.sh && conda activate openr1

# Hugging Face auth (required for --push_to_hub)
# export HF_TOKEN="paste_your_token_here"

# Keep Triton cache off NFS if possible
source activate ./openr1
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/${USER}/triton"
mkdir -p "$TRITON_CACHE_DIR"

# Make multiprocessing predictable on clusters
export OMP_NUM_THREADS=1

# You asked for exactly this command line:
#   python car_park_data/build_rush_small_balanced.py \
#     --sizes 4 5 6 \
#     --out_dir ./rush4-5-6-balanced \
#     --limit_per_size 150000 \
#     --max_pieces_per_size "4:8,5:10,6:12" \
#     --min_empties_per_size "4:1,5:2,6:3" \
#     --max_nodes 500000 \
#     --num_workers 32 \
#     --split 0.8,0.1,0.1 \
#     --dataset_id od2961/rush4-5-balanced \
#     --push_to_hub

# If you set --cpus-per-task differently, update NUM_WORKERS below if desired.
NUM_WORKERS=32

mkdir -p logs
srun --cpu-bind=cores \
  python car_park_data/build_rush_small_balanced.py \
    --sizes 4 5 6 \
    --out_dir ./rush4-5-6-balanced \
    --limit_per_size 150000 \
    --max_pieces_per_size "4:8,5:10,6:12" \
    --min_empties_per_size "4:1,5:2,6:3" \
    --max_nodes 500000 \
    --num_workers "${NUM_WORKERS}" \
    --split 0.8,0.1,0.1 \
    --dataset_id od2961/rush4-5-balanced \
    --push_to_hub
