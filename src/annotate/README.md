# Annotation utilities (`src/annotate`)

This package contains lightweight utilities for **post‑hoc annotation** of
JSONL inference results (e.g., detecting “shifts in reasoning” in `<think>`).
It is separate from the core training/inference code so that annotation runs
can evolve independently of model code.

Typical workflow:
- Run inference (e.g., `scripts/inference/math-inference.slurm`) to produce
  `stepXXXX_<split>.jsonl` files under `artifacts/results/…`.
- Use the shift annotator (CLI: `python -m src.annotate.cli.shift_cli`
  (backcompat: `python -m src.annotate.shift_cli`), or via the modules
  `src.annotate.core.shift_core` /
  `src.annotate.backcompat.tasks.shifts`) to add
  strict `shift_in_reasoning_v1` labels
  (and supporting metadata) to selected passes (`pass1`, `pass2`, `pass2a`, …).
- Optionally use `src/annotate/cli/clean_cli.py` (CLI: `clean-shift-fallbacks`)
  to strip
  conservative fallback labels created when the LLM judge fails.

---

## Public API

New code should generally import from the top-level package instead of
reaching into submodules directly, for example:

```python
from src.annotate import (
    AnnotateOpts,
    annotate_file,
    scan_jsonl,
    llm_judge_shift,
    load_azure_config,
    load_sandbox_config,
    build_preferred_client,
    SHIFT_JUDGE_SYSTEM_PROMPT,
    SHIFT_JUDGE_USER_TEMPLATE,
    clean_file,
    clean_root,
)
```

Direct imports from submodules (e.g., `src.annotate.shift_core`) are treated as
internal and may change more frequently.

---

## CLIs

When the package is installed, the main annotation utilities are exposed as
console commands:

- `annotate-shifts` – run the shift-in-reasoning annotator over JSONL results
  (wrapper around `src.annotate.cli.shift_cli:main`).
- `clean-shift-fallbacks` – strip fallback FALSE shift labels from JSONL
  results (wrapper around `src.annotate.cli.clean_cli:main`).

You can still invoke the modules directly with `python -m` if you prefer
not to install console scripts.

---

## Components

The modules in this package are grouped (and laid out on disk) into four
subpackages: **core**, **infra**, **cli**, and **backcompat**.

Core and CLI modules come in pairs, with a pure library implementation under
``core/`` and a thin CLI wrapper under ``cli/`` (for example,
``shift_core`` / ``shift_cli`` and ``clean_core`` / ``clean_cli``).

### Core modules

- `core/shift_core.py` – core **LLM‑based annotator** for shift detection:
  - Scans a `results_root` recursively for `*.jsonl` files (optionally filtered
    by `--split`), sorts them by `step`, and annotates selected passes.
  - For each record and pass (`pass1`, `pass2`, `pass2a`, …):
    1. Applies the regex prefilter (`prefilter.py`) to find candidate cues.
    2. When cues are present (or when you force relabel), calls the LLM judge
       (`llm_judge_shift`) with the strict prompts.
    3. Writes fields such as:
       - `shift_in_reasoning_v1` (bool)
       - `shift_markers_v1` (list of cue names/phrases)
       - `shift_first_marker_char`
       - `shift_before_excerpt`, `shift_after_excerpt`
       - `shift_rationale_gpt`, `shift_rationale_gpt_model`, `shift_rationale_gpt_time`
    4. If the model call fails or returns unparseable JSON, it defaults to a
       conservative FALSE label and logs the offending prompt to a local file.
  - Idempotent: rows already containing `shift_in_reasoning_v1` are skipped
    unless you pass `--force_relabel`.

- `core/prefilter.py` – regex‑based cue detector on `<think>` text:
  - `extract_think(txt)` pulls the `<think>…</think>` contents (if any).
  - `SHIFT_CAND_PATTERNS` enumerates conservative lexical cues (e.g. “wait”,
    “hold on”, “scratch that”, “I misread”, “re‑check”, “contradiction”).
  - `find_shift_cues(think)` returns `(cue_names, first_pos)` for prefiltering.
  - `find_cue_hits(think)` returns just the cue names.

- `core/prompts.py` – centralized prompt templates for shift detection:
  - `SHIFT_JUDGE_SYSTEM_PROMPT` and `SHIFT_JUDGE_USER_TEMPLATE` define the
    strict “shift in reasoning” judge used by the shift annotator.
  - `SHIFT_PROMPT` is a more concise/legacy variant used in some sandbox tools.

- `core/clean_core.py` – helpers to **remove** fallback FALSE labels:
  - Walks a `results_root` and rewrites `*.jsonl` files in place.
  - For each record whose `pass1.shift_rationale_gpt` begins with one of:
    - `"Model call failed; defaulting to FALSE."`
    - `"Unparseable response; default FALSE."`
    it removes all `shift_*` fields so you can re‑run the annotator cleanly.

### Infra modules

- `infra/config.py` – environment/YAML loaders for Azure / Sandbox settings:
  - `load_azure_config(yaml_path=None)` reads from env (`AZURE_OPENAI_*`) and
    an optional `configs/azure.yml` (gitignored) to produce:
    `{endpoint, api_key, deployment, api_version, use_v1}`.
  - `load_sandbox_config()` is a convenience loader for the Princeton sandbox.

- `infra/llm_client.py` – thin wrappers around `openai` clients:
  - `build_preferred_client(endpoint, api_key, api_version, use_v1=True)` tries
    to use the v1 **Responses API** (`client.responses`) when possible, else
    falls back to `AzureOpenAI` chat completions.
  - `build_chat_client(endpoint, api_key, api_version)` returns a simple
    `AzureOpenAI` chat client (no Responses probing).

### CLI entrypoints

- `cli/shift_cli.py` – CLI entrypoint that wraps `shift_core.py`:
  - Handles argument parsing, logging configuration, and optional
    `--clean_failed_first` integration with the cleaning helpers.

- `cli/clean_cli.py` – thin CLI wrapper around `clean_core.py`
  (backcompat: `backcompat/clean_failed_shift_labels.py`).

### Backcompat wrappers

- `backcompat/gpt_eval.py` – **backwards-compatible wrapper** that re-exports the
  functionality from `shift_core.py` / `shift_cli.py` so existing imports
  and tests using `src.annotate.gpt_eval` keep working.

- `backcompat/tasks/shifts.py` – legacy wrapper that re-exports the shift annotator
  API (from `shift_core.py` / `shift_cli.py`) so older imports of
  `src.annotate.tasks.shifts` continue to work.

---

## Running the annotator

From the repo root, after you have inference results under
`artifacts/results/...`:

```bash
conda activate ./openr1  # or your env

# Example: annotate pass-1 shifts on test split only
python -m src.annotate.cli.shift_cli \
  artifacts/results/GRPO-1.5B-math-temp-0.05-3 \
  --split test \
  --passes pass1 \
  --backend azure \
  --endpoint https://<your-azure-res>.openai.azure.com/ \
  --deployment gpt-4o \
  --api_version 2024-12-01-preview \
  --use_v1 1
```

Key flags:
- `results_root` – root directory containing `stepXXXX_<split>.jsonl` files
  (e.g., `artifacts/results/GRPO-1.5B-math-temp-0.05-3`).
- `--split` – optional substring filter on filenames (e.g., `"test"`).
- `--passes` – comma‑separated pass keys to annotate; defaults to `"pass1"`.
- `--max_calls` – optional cap on the number of LLM calls (for dry runs or
  budget‑limited experiments).
- `--dry_run` – log which records would be annotated without calling the
  model or writing changes.
- `--backend` – `"azure"` (default) or `"portkey"` (Princeton AI Sandbox via
  `portkey-ai`); see `shift_core.py` for details.

Azure‑specific settings can be provided via CLI flags or environment variables
(`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, etc.), optionally backed by
a private `configs/azure.yml`.

---

## Cleaning fallback labels

If you previously ran the shift annotator and some examples were marked FALSE solely
because the judge failed (e.g., network issues), you can clear those labels:

```bash
python -m src.annotate.cli.clean_cli \
  artifacts/results/GRPO-1.5B-math-temp-0.05-3
```

Or, if you installed console scripts:

```bash
clean-shift-fallbacks \
  artifacts/results/GRPO-1.5B-math-temp-0.05-3
```

You can also let the shift annotator do this automatically before annotation via:

```bash
python -m src.annotate.cli.shift_cli \
  artifacts/results/... \
  --clean_failed_first
```

After cleaning, re‑run the shift annotator to regenerate `shift_in_reasoning_v1`
labels under your updated configuration.
