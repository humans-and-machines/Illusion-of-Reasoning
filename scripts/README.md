# Scripts layout (launchers only)
- `inference/`: Slurm launchers for eval/entropy experiments (call Python in `src/inference/`).
- `training/`: Slurm launchers for GRPO/SFT runs.

Notes
- Python entrypoints for inference live under `src/inference/`; the `.slurm` files here just wrap them.
- Other helper trees (`analysis/`, `annotate/`, `utils/`, `viz/`) now live under `src/scripts/` if needed.
