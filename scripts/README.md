# Scripts layout (launchers only)
- `inference/`: Slurm launchers for eval/entropy experiments (call Python in `src/inference/`).
- `training/`: Slurm launchers for GRPO training runs (call Python in `src/training/`).

Notes
- Python entrypoints for inference live under `src/inference/`; the `.slurm` files here just wrap them.
- Training entrypoints live under `src/training/`; the `.slurm` files in `training/` are thin wrappers around those CLIs.
- Analysis, annotation, and plotting code lives in dedicated packages (for example, `src/analysis/` and `src/annotate/`); there is no separate `src/scripts/` package in this repo.
