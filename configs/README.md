# `configs/` directory

This directory centralizes configuration **templates** and shared tooling for the project. It is meant to be safe to track in git while keeping any machine‑specific or secret configuration files local and uncommitted.

The main tracked files are:

- `environment.yml` – Conda environment template used in the main README (create/update your local env from this file).
- `azure.example.yml` – Example Azure/OpenAI client config. If you need a real config, copy this to `configs/azure.yml` and fill in your details; `configs/azure.yml` is git‑ignored and must not be committed.
- `ds_zero3_safe.json` – A DeepSpeed ZeRO‑3 configuration tuned for low VRAM; you can copy/modify this for your own runs if needed.
- `.env.example` – Example environment variables (Azure/OpenAI, sandbox, etc.). Copy this to `.env` at the repo root (or to a local `.env` in your environment) and fill in real API keys; `.env` files are always ignored by git.
- `.condarc` – A sample `conda` configuration that pins caches and envs inside the repo. You can reuse these settings in your local `~/.condarc` or adapt them here.
- `.pylintrc` – Shared linting configuration for `pylint`.
- `.readthedocs.yaml` – Build configuration for Read the Docs.
- `pytest.ini` – Pytest configuration (test paths, markers, and warning filters).

## What should live here (but not be pushed)

It is appropriate to keep **local-only** configuration files in `configs/`, but they should stay out of version control. Examples:

- Real Azure/OpenAI client configs (e.g., `configs/azure.yml` with your true endpoint and API key).
- Any `.env` file containing secrets (API keys, tokens, private endpoints).
- Auto-generated Conda artifacts (e.g., `configs/condaenv.*.requirements.txt`) or other machine-specific configuration dumps.

These files are already covered by `.gitignore` (or should be added there if new). As a rule of thumb:

- If a file contains **secrets**, **tokens**, or **machine-specific paths**, keep it local and untracked.
- Only commit **generic templates** and **shared tooling configs** that are safe for others to reuse.

