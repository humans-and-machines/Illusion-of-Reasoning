# Source package overview

The `src` tree contains the main Python packages for this project. At a high
level, the workflow is:

1. **Training** – GRPO / SFT fine-tuning of base models.  
2. **Inference** – running trained (or baseline) models on math / carpark /
   crossword tasks to produce JSONL traces.  
3. **Annotation** – optional human or model-based labeling / inspection of
   traces.  
4. **Analysis** – aggregating step-level metrics, plotting, and building
   higher-level summaries.

This document orients you to the flow across `src/training`, `src/inference`,
`src/annotate`, and `src/analysis`.

---

## 1. Training (`src/training`)

The training stack implements GRPO-style reinforcement learning over
chain-of-thought traces, plus supporting generation utilities:

- Core scripts:
  - `grpo.py` – main GRPO training loop.
  - `generate.py` – batched generation for evaluation or dataset creation.
  - `rlif.py` – RLHF / RLIF-style helpers (if used).
  - `sft.py` – supervised fine-tuning entrypoint (present but not used for the
    main paper results).
- Reward and helper modules:
  - `rewards_*.py` – task-specific reward functions for math / carpark /
    crossword.
  - Shared utilities for loading configs, wiring vLLM / DeepSpeed, and logging.

Training jobs are typically launched via the Slurm or shell scripts under
`scripts/training/` and `scripts/inference/`, which pass a YAML recipe from
`recipes/` into the appropriate training entrypoint.

**Typical flow:**

1. Choose or author a recipe under `recipes/` (model, dataset, optimization
   settings).
2. Launch `grpo.py` (or `sft.py`) via the corresponding Slurm script.
3. Checkpoints and logs are written under `artifacts/models/` (or a configured
   output root).

Those checkpoints can then be plugged directly into the inference runners.

---

## 2. Inference (`src/inference`)

The inference package runs trained models (or baselines) on evaluation
benchmarks and logs detailed JSONL traces:

- **Cores:** `math_core.py`, `carpark_core.py`, `crossword_core.py` implement
  two-pass (or single-pass) loops with resume/fill behavior, entropy tracking,
  and reconsideration markers.
- **Unified CLIs:** `cli/unified_math.py`, `cli/unified_carpark.py`,
  `cli/unified_crossword.py` expose a consistent command-line interface,
  with shared wiring in `unified_runner_base.py`.
- **Backends and utilities:** `backends.py`, `common.py`, `gateway_utils.py`,
  `math_pass_utils.py`, and `text_utils.py` centralize model plumbing, sampling
  params, canonicalization, and gateway retry logic.
- **Gateway clients:** `gateways/providers/{azure,openrouter,portkey}.py`
  call remote APIs (Azure, OpenRouter, Portkey) instead
  of local HF checkpoints.
- **Summaries:** `src/inference/domains/summarize/summarize_inference_core.py`
  and `src/inference/runners/summarize_inference_runner.py`
  aggregate pass-1/pass-2 JSONL outputs into CSV tables and human-readable
  text summaries.

All public classes and functions in `src/inference` are documented with
Sphinx-style docstrings (`:param:`, `:returns:`), so you can plug them into a
Sphinx `autodoc` configuration if desired.

**Typical flow:**

1. Select a model:
   - HF checkpoint (trained or base) for math / carpark / crossword.
   - A gateway deployment for remote inference.
2. Run the appropriate unified runner or gateway script from `src/inference`:
   - `python -m src.inference.cli.unified_math ...`
   - `python -m src.inference.cli.unified_carpark ...`
   - `python -m src.inference.cli.unified_crossword ...`
   - or a gateway client such as `gateways/providers/azure.py`.
3. Inspect JSONL outputs under your chosen `--output_dir`. These files are the
   input to annotation and analysis.

For a more detailed walkthrough of the inference modules and CLIs, see
`src/inference/README.md`.

---

## 3. Annotation (`src/annotate`)

The annotation package supports human and model-based labeling / triage of
inference traces. While details may evolve, typical responsibilities include:

- Loading JSONL outputs from `src/inference` runs.
- Rendering prompts and model chains-of-thought for human raters.
- Applying task-specific quality labels or rationales.
- Optionally calling external models (e.g., via Azure or OpenAI) to score or
  tag reasoning behaviors.

The Azure helper functions used by `backends.py` and gateway runners are
implemented here (for example, loading Azure config and constructing preferred
clients).

**Typical flow:**

1. Point annotator scripts at the JSONL files emitted by inference.
2. Collect labels into new JSONL/CSV artifacts (e.g., per-sample quality
   scores, error categories, or “Aha!” markers).
3. Feed these labels into downstream analysis, or back into training recipes
   if you are iterating on reward functions.

---

## 4. Analysis (`src/analysis`)

The analysis package turns raw JSONL traces (and optional annotation labels)
into plots and aggregate tables used in the paper:

- Step-level metrics:
  - Accuracy by pass and by problem.
  - Entropy statistics for think / answer segments.
  - Reconsideration marker prevalence and placement.
- Prompt-level and dataset-level summaries:
  - Grouping by prompt family or task.
  - Filtering / capping by prompt variants.
  - Correlating annotation labels with model behavior.

Scripts in `src/analysis` typically use:

- The summarization helpers from `src/inference/domains/summarize/` and the
  CLI wiring in `src/inference/runners/summarize_inference_runner.py`.
- The same JSONL iteration and canonicalization utilities as the inference
  stack (via `src.inference.utils.common` / `src.inference.utils.gateway_utils`
  / `src.inference.utils.math_pass_utils`).

**Typical flow:**

1. Run `python -m src.inference.runners.summarize_inference_runner` over an
   inference results root to produce per-step CSV summaries and a
   human-readable table.
2. Use `src/analysis` notebooks or scripts to:
   - Slice results by configuration (step, prompt family, model, dataset).
   - Overlay annotation-based labels (e.g., whether a sample contains an
     “Aha!” moment) on top of entropy / correctness curves.
   - Save plots and tables used in the paper or internal reports.

---

## Putting it all together

In practice, a typical end-to-end run looks like:

1. **Train** a model with `src/training/grpo.py` using a recipe from `recipes/`.
2. **Run inference** on math / carpark / crossword with the unified runners in
   `src/inference`, producing JSONL traces under `artifacts/` or another
   output directory.
3. **Annotate** selected traces with `src/annotate` (or external tooling),
   writing additional labels back into JSONL/CSV.
4. **Analyze** results with `python -m src.inference.runners.summarize_inference_runner` and the
   scripts in `src/analysis` to compute metrics, visualize trends, and support
   conclusions about reasoning behavior.

This structure keeps the core experimental loop (training → inference →
annotation → analysis) explicit while allowing each layer to evolve
independently.
