# Scripts layout
- `annotate/`: LLM-as-judge + annotation entrypoints (`gpt-eval.py`, `evaluation.py`, `recheck.py`, `gpt-cohens-kappa.py`).
- `analysis/`: plotting/stats entrypoints; `analysis/legacy/` holds older one-offs kept for reference.
- `inference/`: generation/GRPO/inference runners.
- `viz/`: figure/table renderers.
- `utils/`: helpers (download/prune/summarize, list-results, etc.).
- `slurm/` (symlinked as `cluster/`): job specs for running on the cluster.
