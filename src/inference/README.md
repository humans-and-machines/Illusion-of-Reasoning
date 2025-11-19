# Inference package overview

The `src/inference` package contains all of the runtime code used to run
models for evaluation and data generation. It is organized around three
problem domains:

- math word problems (MATH-500 style)
- Rush Hour / car-park puzzles
- cryptic crosswords

and supports multiple backends:

- HuggingFace causal LMs (via `transformers`)
- DeepSpeed ZeRO-3 checkpoints for Llama-style models
- Remote “gateway” APIs (Azure OpenAI, OpenRouter, Portkey, etc.)

This document explains how the pieces fit together and how to use them.

---

## High-level architecture

At a high level the inference stack is:

- **Core task loops**
  - `math_core.py` – two-pass MATH inference using HF models.
  - `carpark_core.py` – two-pass Rush Hour inference.
  - `crossword_core.py` – cryptic-crossword inference with reconsideration logic.

- **Shared plumbing / utilities**
  - `common.py` – core generation utilities (tokenization, entropy, batch helpers).
  - `gateway_utils.py` – dataset / JSONL / CLI / gateway helpers.
  - `math_pass_utils.py` – math-specific tagging, entropy and reconsideration helpers.
  - `text_utils.py` – small text-only helpers (no torch / transformers).
  - `backends.py` – local HF backend and Azure backend abstraction.

- **Entry points and runners**
  - `inference.py` – simple batch inference script for a Qwen-style model using the `OPENR1` prompt.
  - `math_deepseek_azure.py`, `math_openrouter_deepseek.py`, `math_portkey.py` – single-pass “gateway” runners for math.
  - `math_llama_core.py` – DeepSpeed ZeRO-3 backed math runner that reuses `math_core` logic.
  - `unified_math_runner.py`, `unified_carpark_runner.py`, `unified_crossword_runner.py` – thin CLIs that route into the core task loops with a unified interface.
  - `unified_runner_base.py` – shared argument parsing and wiring used by the unified runners.

- **Task configuration / registry**
  - `task_registry.py` – defines system prompts and dataset specs for named tasks and gateway scripts.

- **Summaries and analysis**
  - `summarize_inference_core.py`, `summarize_inference.py` – helpers to aggregate JSONL outputs into CSV/summary stats.

---

## Core generation utilities (`common.py`)

`common.py` is the main place where low-level generation and entropy logic lives.
Key pieces:

- **Tokenization / device placement**
  - `move_inputs_to_device(inputs, device=None)` – moves HF input dict to CUDA (or a provided device) and returns `(inputs, input_lengths)`.
  - `tokenize_prefixes_for_generate(tokenizer, prefixes, max_length=4096, device=None)` – tokenizes a list of prefixes and moves them to the right device.

- **Generate kwargs helpers**
  - `build_generate_kwargs(...)` – builds `model.generate` kwargs given caps, EOS ids and sampling settings (temperature / top_p).
  - `make_generate_kwargs_for_cap(...)` – convenience wrapper that derives `pad_token_id` from a tokenizer.

- **Batch generation and entropy**
  - `GenerationLimits` – shared container for `batch_size`, `num_samples`, `think_cap`, `answer_cap`.
  - `GenerateBatchParams`, `GenerateBatchRuntime` – structured params for running a single batch.
  - `generate_and_score_batch(params, runtime)` – shared generate + decode + entropy computation used by math / carpark / crossword cores.
  - `run_generate_batch(...)` – convenience wrapper around `GenerateBatchParams` + `GenerateBatchRuntime` that many cores call directly.
  - `decode_generated_row(...)`, `decode_and_score_batch(...)` – per-row decode and entropy computation helpers.
  - `entropy_from_start_index(...)`, `first_eos_any(...)` – lower-level entropy / EOS helpers used by the decoders.
  - `StopOnSubstrings`, `classify_stop_reason(...)` – stopping criteria and stop-reason classification.

- **Misc helpers**
  - `repeat_for_samples(values, num_samples)` – row-major repetition of prefixes (used by multi-sample inference).
  - `SamplingConfigBase` – shared base for sampling configuration (temperature, top_p, entropy mode, EOS IDs).
  - `build_math_inference_config_kwargs(...)` and `build_math_inference_config_kwargs_from_args(...)` – pack standard math-style inference settings into a config kwargs dict.
  - `require_torch(caller)`, `require_transformers(caller)` – import-and-validate helpers that raise user-friendly errors if dependencies are missing.

`common.py` also **re-exports** a number of helpers from the more specific modules so callers can just depend on `src.inference.common`:

- From `gateway_utils.py`:
  - Dataset / JSONL helpers (`require_datasets`, `load_local_json_dataset`, `iter_jsonl_objects`, `append_jsonl_row`, `scan_existing_pass1_results`, `scan_existing_problem_samples`).
  - CLI helpers (`build_math_gateway_arg_parser`, `build_math_gateway_messages`, `build_math_gateway_row_base`, `build_usage_dict`, `build_two_pass_row_base`, `prepare_math_gateway_dataset`, `prepare_math_gateway_dataset_from_args`, `build_eos_ids_from_tokenizer`, `configure_tokenizer_and_eos`, `setup_hf_cache_dir_env`, `setup_script_logger`, `call_with_gateway_retries`, `call_with_retries`, `parse_openai_chat_response`, `iter_math_gateway_samples`, `limit_dataset_examples`, `load_remote_dataset_default`).
  - Prompt template: `OPENR1_PROMPT_TEMPLATE`.

- From `math_pass_utils.py`:
  - Math tagging and correctness helpers (`canon_math`, `contains_canon`, `extract_blocks`, `valid_tag_structure`).
  - Per-pass result builders (`build_math_pass_meta`, `pack_math_pass_result`, `build_entropy_pass_base`, `add_token_and_tag_fields`, `finite_mean`).
  - Reconsideration patterns (`RECONSIDER_PATTERNS`) used by math and crossword inference.

In most new code you should import from `src.inference.common` where possible; only reach directly into `gateway_utils` / `math_pass_utils` if you need something very specific that isn’t re-exported.

---

## Math inference core (`math_core.py`)

`math_core.py` implements the two-pass math inference loop for HF models:

- **Configuration**
  - `MathSamplingConfig(SamplingConfigBase)` – adds `batch_size`, `num_samples` on top of shared sampling fields.
  - `MathTwoPassConfig` – controls the optional second pass (`enabled`, cue `phrase`, which sample to reuse, and caps).
  - `MathInferenceConfig` – high-level runtime config combining:
    - dataset split (`split_name`), `output_dir`, `step`
    - `sampling` (MathSamplingConfig)
    - `two_pass_cfg` (MathTwoPassConfig)
  - It exposes a flat property surface (`batch_size`, `num_samples`, `temperature`, `top_p`, `entropy_mode`, `eos_ids`, `two_pass`, `think_cap`, `answer_cap`, etc.) so older code doesn’t need to know about the nested dataclasses.

- **Context**
  - `MathInferenceContext` – bundles `tokenizer`, `model`, and `config` for generation helpers.
  - `BatchLayout` – maps rows back to work items and sample indices.
  - `ExistingPassState`, `TwoPassBatchOutputs`, `BatchWriteContext`, `SecondPassInputs` – helper containers to keep function signatures small and keep the resume/fill logic readable.

- **Generation**
  - `_gen_batch(batch_spec, context)` – calls `run_generate_batch` to generate a set of continuations and per-token entropies.
  - `_build_work_items_for_slice(...)`, `_build_pass1_prefixes(...)` – create prefixes and work-item layouts from a dataset slice, respecting resume/fill.
  - `_build_second_pass_base_prompts(...)` + `build_second_pass_think_prefixes(...)` (from `common.py`) – construct pass-2 `<think>` prefixes aligned with pass-1 rows.

- **Results & correctness**
  - Uses `extract_problem_and_answer`, `canon_math`, `contains_canon`, `build_math_pass_meta`, `pack_math_pass_result` (via `common`) to canonicalize answers, compute entropies and reconsideration markers, and produce a per-pass row dict.
  - Writes JSONL rows with full `<think>...</think><answer>...</answer>` contents plus entropy series and reconsideration metadata.

- **Public surface**
  - `load_math500(cache_dir, split, seed, dataset_path=None)` – load the MATH-500 dataset from disk.
  - `run_inference_on_split(examples, tokenizer, model, config)` – drive a full two-pass inference run over a dataset and write JSONL outputs.

When you write new math-style runners, prefer to call `MathInferenceConfig` + `run_inference_on_split` rather than duplicating the outer loop.

---

## Carpark (Rush Hour) inference (`carpark_core.py`, `carpark_rush_utils.py`)

The car-park (Rush Hour) pipeline mirrors the math pipeline but with different
canonicalization and scoring:

- **`carpark_rush_utils.py`**
  - Implements Rush Hour move canonicalization and a soft-match reward:
    - `_canon_rush_generic`, `_canon_rush_gold` – normalize predicted and gold sequences.
    - `_toklist`, `_lcs_len`, `_multiset_overlap_ratio` – token-level utilities used for scoring.
    - `_score_rush_pair(...)` – metrics for prefix, positional exact matches, LCS, bag overlap, piece/direction match, and step closeness.
    - `rush_soft_match_reward(pred_answer_text, gold_answer_any, ...)` – returns a score in `[0,1]` plus a breakdown of components and chosen gold sequence.

- **`carpark_core.py`**
  - Defines `CarparkInferenceConfig` (I/O + column names + `GenerationLimits` + sampling config + second-pass settings).
  - `InferenceContext` – parallel to `MathInferenceContext`, bundling tokenizer, model, config.
  - `_gen_batch(...)` – wraps `run_generate_batch` with the car-park config.
  - `run_inference_on_split(...)` – two-pass Rush Hour loop:
    - Pass 1: generate `<think>` and `<answer>` blocks for each example/sample.
    - Pass 2 (optional): reconsideration pass using chosen samples from pass 1.
  - `_pack_pass_result(...)` – uses `build_entropy_pass_base`, `add_token_and_tag_fields`, and Rush Hour correctness metrics.
  - `run_carpark_main(...)` – the unified runner entry point (invoked from `unified_carpark_runner.py`).

If you need to add a new Rush Hour runner that uses a different backend, you can usually reuse `run_carpark_main` and just change how the backend is created.

---

## Crossword inference (`crossword_core.py`)

`crossword_core.py` implements inference for cryptic crosswords:

- **Config**
  - `CrosswordCapsConfig` – `batch_size`, `num_samples`, `think_cap`, `answer_cap`.
  - `CrosswordSamplingConfig(SamplingConfigBase)` – sampling configuration for crossword runs.
  - `CrosswordTwoPassConfig` – controls reconsideration behavior and cue phrase.
  - `CrosswordInferenceConfig` – ties everything together along with `eos_ids`.

- **Canonicalization & scoring**
  - `_canon_cross(text)` – canonicalize answers by case-folding and stripping punctuation / spaces.
  - `_compute_entropy_info(ent_think, ent_answer)` – summarise entropy series and token counts.
  - `_find_reconsider_info(...)` – uses `RECONSIDER_PATTERNS` (from `math_pass_utils`) and `find_markers_and_context` (from `text_utils`) to detect reconsideration markers and context.

- **Generation**
  - `BatchGenerationContext(tokenizer, model, config)` – small wrapper passed into `_gen_batch(...)`.
  - `_gen_batch(prefixes, cap, stop_strs, generation)` – uses `run_generate_batch` to generate and score a batch.
  - `_repeat_for_samples` is handled via `repeat_for_samples` imported from `common.py`.

- **Results & runners**
  - `_pack_pass_result(...)` – assembles a per-pass result dict including reconsideration markers and entropy info, using `build_entropy_pass_base` and `add_token_and_tag_fields`.
  - `run_inference_on_split(examples, tokenizer, model, config)` – main crossword loop.
  - `unified_crossword_runner.py` / `run_crossword_main(...)` – CLI wrapper to run crossword inference with HF models, defined in `unified_runner_base.py`.

---

## Gateway helpers and remote math runners

`gateway_utils.py` hosts helpers shared by several “gateway” scripts that talk
to remote APIs instead of local HF models:

- **Datasets & JSONL**
  - `require_datasets()`, `load_local_json_dataset(path)`.
  - `append_jsonl_row(path, row)`, `iter_jsonl_objects(path)`.
  - Resume helpers: `scan_existing_problem_samples`, `scan_existing_pass1_results`.
  - `limit_dataset_examples`, `prepare_math_gateway_dataset(...)`, `prepare_math_gateway_dataset_from_args(...)`.

- **CLI & backend wiring**
  - `add_basic_runner_args`, `add_model_and_output_args`, `add_math_gateway_dataset_args`, `add_two_pass_args`.
  - `build_math_gateway_arg_parser(...)`.
  - `setup_hf_cache_dir_env(...)` – sets HF-related env vars and returns a cache dir.
  - `build_eos_ids_from_tokenizer(...)`, `configure_tokenizer_and_eos(...)`.
  - `init_unified_backend_and_eos(...)` – create an `HFBackend` and EOS ID list for unified runners.

- **Math gateway helpers**
  - `OPENR1_PROMPT_TEMPLATE` – generic prompt template for MATH-style problems.
  - `build_math_gateway_messages(...)` – simple `[system, user]` chat for gateways.
  - `iter_math_gateway_samples(...)` – yield only the samples that still need generation.
  - `parse_openai_chat_response(...)` – extract `(text, finish_reason, usage)` from OpenAI/OpenRouter/Portkey-style responses.
  - `add_math_gateway_sampling_args(...)`.
  - `call_with_retries(...)`, `call_with_gateway_retries(...)` – simple retry logic parameterized by CLI args.

These helpers are used by:

- `math_deepseek_azure.py` – Azure DeepSeek-style math gateway.
- `math_openrouter_deepseek.py` – OpenRouter DeepSeek-style math gateway.
- `math_portkey.py` – Portkey-based math gateway.

Each of these scripts:

- Parses CLI args (dataset, sampling, output path, etc.).
- Uses `prepare_math_gateway_dataset_from_args` + dataset loader (`load_math500` or remote).
- Iterates over `(problem, gold_answer, sample_idx)` from `iter_math_gateway_samples`.
- Calls the remote API with retries and writes JSONL rows using `build_math_gateway_row_base` and friends.

---

## Unified runners

To keep the CLI surface small and consistent, the repository provides unified
runners under `src/inference`:

- `unified_math_runner.py` – entry point for math HF inference.
- `unified_carpark_runner.py` – entry point for carpark HF inference.
- `unified_crossword_runner.py` – entry point for crossword HF inference.

All three use the helpers in `unified_runner_base.py`:

- `parse_math_args`, `parse_carpark_args`, `parse_crossword_args` – construct and parse task-specific CLI arguments (dataset, model, sampling, budgets, output).
- `run_math_main(backend_cls, argv=None)` – run math_core on a dataset using the provided backend class (usually `HFBackend`).
- `run_carpark_main(load_module, backend_cls, argv=None)` – run Rush Hour inference using a dynamically loaded carpark module.
- `run_crossword_main(load_module, backend_cls, argv=None)` – run crossword inference with HF models.

These helpers:

- Call `setup_hf_cache_dir_env` to standardize cache dirs.
- Use `init_unified_backend_and_eos` to construct a backend and EOS ID list.
- Load datasets (`datasets` or local JSONL) and optionally cap the number of examples.
- Build the appropriate config (`MathInferenceConfig`, `CarparkInferenceConfig`, `CrosswordInferenceConfig`) and invoke the core’s `run_inference_on_split`.

For most experiments you should prefer these unified runners over calling the lower-level cores directly.

---

## Backends (`backends.py`)

`backends.py` wraps model plumbing so that task-specific scripts don’t need to
know about the exact HF or Azure APIs:

- `_load_torch_and_transformers()` – lazy import of `torch` and key `transformers` classes to avoid heavy import-time dependencies.
- `HFBackend` – wrapper around `AutoModelForCausalLM` + tokenizer:
  - `HFBackend.from_pretrained(...)` – load a model and tokenizer with the requested dtype, device map and attention implementation.
  - `generate(prompts, stop_strings=None, max_length=None, **gen_kwargs)` – run `model.generate` and return:
    - `texts`: decoded continuations
    - `stop_reasons`: high-level stop reasons
    - `sequences`: tensor of generated token IDs
    - `output`: full HF generate output
- Azure helpers:
  - `load_azure_config`, `build_preferred_client` are imported from the annotate package if available.

The gateway scripts typically rely on these helpers indirectly via `gateway_utils`.

---

## Task registry (`task_registry.py`)

`task_registry.py` defines:

- `DatasetSpec` – how to load a dataset for a task (loader path, default ID, columns, split).
- `TaskSpec` – configuration for a logical task, including:
  - `system_prompt`
  - stop strings (`stop_think`, `stop_answer`)
  - caps (`think_cap`, `answer_cap`, `max_output_tokens`)
  - canonicalization callables (`canonicalize_pred`, `canonicalize_gold`)

It also exposes named system prompts:

- `MATH_SYSTEM_PROMPT`
- `CROSSWORD_SYSTEM_PROMPT`
- `CARPARK_SYSTEM_PROMPT`

and a `TASK_REGISTRY` mapping from task names (e.g. `"math-azure"`) to `TaskSpec`
instances used by the gateway scripts.

---

## Summaries and post-processing

After running inference, you will typically have JSONL files (e.g. under
`artifacts/` or a configured output directory). The summarization helpers:

- `summarize_inference_core.py`
- `summarize_inference.py`

provide utilities to:

- Scan result files (`scan_files`, `iter_jsonl_objects`).
- Accumulate per-step and per-prompt metrics (`StepAgg`, `accumulate_prompt_variants`, etc.).
- Apply filters (per-problem / global caps on prompt variants).
- Emit CSV summaries for plotting or downstream analysis.

See `summarize_inference.py` for CLI options and usage examples.

---

## Typical usage patterns

### 1. Unified math runner (HF model)

Run MATH-500 inference with a local HF checkpoint:

```bash
python -m src.inference.unified_math_runner \\
  --model_name_or_path path/to/qwen2.5-7b \\
  --output_dir artifacts/math_run \\
  --dataset_id MATH-500 \\
  --split test \\
  --batch_size 8 \\
  --num_samples 1 \\
  --temperature 0.0 \\
  --top_p 0.95 \\
  --think_cap 750 \\
  --answer_cap 50
```

### 2. Unified carpark runner (HF model)

```bash
python -m src.inference.unified_carpark_runner \\
  --model_name_or_path path/to/model \\
  --output_dir artifacts/carpark_run \\
  --dataset_id od2961/rush4-5-6-balanced \\
  --batch_size 8 \\
  --num_samples 1
```

### 3. Unified crossword runner

```bash
python -m src.inference.unified_crossword_runner \\
  --model_name_or_path path/to/model \\
  --output_dir artifacts/crossword_run \\
  --dataset_id CROSSWORD-LOCAL \\
  --dataset_path data/crossword/local.jsonl \\
  --batch_size 8 \\
  --num_samples 1
```

### 4. Math Azure / OpenRouter / Portkey gateway scripts

Each gateway script has its own CLI, but the pattern is:

- Use `--dataset_id` / `--dataset_path` to select MATH-500 or a remote HF dataset.
- Provide gateway-specific credentials / endpoint / model flags.
- Choose sampling / budget options (`--temperature`, `--top_p`, `--max_output_tokens`, etc.).

For details, see:

- `src/inference/math_deepseek_azure.py`
- `src/inference/math_openrouter_deepseek.py`
- `src/inference/math_portkey.py`

---

## Extending the inference stack

When adding new functionality, a few guidelines:

- Prefer to reuse `common.py` helpers (`run_generate_batch`, `GenerationLimits`, `SamplingConfigBase`) rather than re-implementing generation loops.
- For new tasks:
  - Add a `TaskSpec` entry to `task_registry.py` if you need a new system prompt / dataset wiring.
  - Factor out domain-specific scoring / canonicalization into a small utility module (similar to `math_pass_utils.py` or `carpark_rush_utils.py`).
- For new backends:
  - Consider adding a small wrapper to `backends.py` and reusing the same interface (`generate`, `HFBackend`-style).
  - Use `gateway_utils` for JSONL and CLI plumbing instead of duplicating it.

This keeps most logic centralized and makes it easier to maintain the inference stack across different tasks and backends.

