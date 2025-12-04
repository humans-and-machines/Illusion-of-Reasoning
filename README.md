# The Illusion of Insight in Reasoning Models

Do reasoning models have ''Aha!'' moments? Prior work suggests that models like DeepSeek-R1-Zero undergo sudden realizations that lead to accurate outputs. Yet it remains unclear whether such shifts in reasoning strategy actually improve performance. In this paper, we formalize intrinsic ''Aha!'' events and instrument training runs (GRPO only) to detect them across multiple tasks. We find that reasoning shifts are rare, do not become more frequent with training, and seldom improve accuracy. However, their effect varies according to model uncertainty. Building on this finding, we show that artificially triggering shifts under high entropy improves accuracy. Overall, our results challenge the perception that reasoning models' problem-solving capabilities stem from mid-reasoning shifts, although these shifts can be exploited to improve performance.

## Quick Start

- **Keep caches local (optional):** `source tools/local_env.sh` pins conda/pip/HF/tmp/W&B caches inside the repo.
- **Create env (conda, local to repo):** `conda env create -f configs/environment.yml -p ./openr1 || conda env update -f configs/environment.yml -p ./openr1`, then `conda activate ./openr1`, then `make install` (installs dev extras in the same env; caches live under `.pip_cache/.tmp/.hf_cache/.conda_pkgs`).
- **Install only runtime deps:** `pip install -e .`; dev extras: `pip install -e .[dev]`.
- **Authenticate for gated assets:** `huggingface-cli login` or `export HF_TOKEN=<token>`.
- **Run jobs:** training launchers live under `scripts/training/` (e.g., `bash scripts/training/training-math-grpo.slurm`); inference launchers live under `scripts/inference/` (e.g., `bash scripts/inference/math-inference.slurm`).

This repository demonstrates GRPO fine-tuning of a base Qwen 2.5-1.5B-Instruct model on the OpenR1 Math 220k dataset (plus crossword and Rush Hour generators). Traces of chain-of-thought reasoning are logged and saved at fixed intervals.

The `src/inference/` package contains the main evaluation stack used throughout the project:

- **Math (MATH-500 / OpenR1-Math-220k)**  
  - `src/inference/domains/math/math_core.py`: two-pass math inference core with resume/fill logic and entropy/marker logging.
  - `src/inference/domains/math/math_llama_core.py`: DeepSpeed/ZeRO-backed Llama runner reusing the math core.
  - `src/inference/cli/unified_math.py`: unified CLI entrypoint for HF math models (via `unified_runner_base`).
  - `src/inference/runners/openr1_math_runner.py`: legacy single-pass batch runner for Qwen checkpoints.
  - `src/inference/gateways/providers/{azure,openrouter,portkey}.py`: gateway-based clients for Azure/OpenRouter/Portkey endpoints.
- **Rush Hour (car-park)**  
  - `src/inference/domains/carpark/carpark_core.py`: two-pass Rush Hour inference and scoring with soft-match rewards.
  - `src/inference/cli/unified_carpark.py`: unified CLI wrapper over the carpark core.
- **Crossword (cryptic clues)**  
  - `src/inference/domains/crossword/crossword_core.py`: crossword inference core with reconsideration markers and entropy summaries.
  - `src/inference/cli/unified_crossword.py`: unified CLI wrapper over the crossword core.
- **Shared utilities**  
  - `src/inference/backends/`: HF and Azure backends that abstract model plumbing.
  - `src/inference/utils/common.py`, `src/inference/utils/gateway_utils.py`, `src/inference/utils/math_pass_utils.py`, `src/inference/utils/text_utils.py`: shared sampling, entropy, canonicalization, and retry helpers.
  - `src/inference/domains/summarize/summarize_inference_core.py`, `src/inference/runners/summarize_inference_runner.py`: step-level aggregation/CSV export for pass-1/pass-2 runs.
  - `src/inference/utils/task_registry.py`: central registry for task prompts, caps, and dataset loaders.

All public functions and classes in `src/inference/` now have Sphinx-style docstrings (with `:param:` and `:returns:`) so that API documentation can be generated automatically via `sphinx.ext.autodoc` if desired.

---

## Table of Contents

1. [Quick Start](#quick-start)  
2. [Repository Structure](#repository-structure)  
3. [Prerequisites](#prerequisites)  
4. [Data](#data)  
5. [Training](#training)  
   1. [Model Arguments (Common)](#model-arguments-common)  
6. [Citation](#citation)  
7. [License](#license)  

---


## Repository Structure

```text
Illusion-of-Reasoning/
├── configs/         # env templates (conda, accelerate), linting, dotenv examples
├── recipes/         # task/model YAMLs for GRPO (MaxEnt-GRPO variants)
├── data/            # task data (car_park, crossword) + human assessment prompts
├── src/training/    # GRPO training + generation entrypoints (grpo.py, generate.py, rewards*, runtime/, utils/)
├── src/inference/   # inference cores, unified runners, backends, and helpers
├── scripts/         # Slurm launchers for training (`scripts/training/`) and inference (`scripts/inference/`)
├── artifacts/models/        # trained checkpoints (by base model/run)
├── artifacts/results/       # inference logs, analysis outputs, and caches
├── tools/           # local cache helpers and conda hook installers
├── Makefile         # installs dev env, lint/format/test, eval helper
└── README.md, LICENSE, setup.py, setup.cfg
```

---

## Prerequisites

### Hardware

- GPU-equipped machine with CUDA 12.4 (or higher) for both training and inference.
- At least 16 GB of VRAM is recommended for Qwen 2.5-1.5B.
- Disk space to cache model weights (≥ 50 GB).

### Software & Libraries

- Python 3.11 (tested)
- PyTorch 2.6.0
- DeepSpeed 0.9+ (for GRPO training)
- Transformers 4.\*
- Datasets 2.\*
- vLLM 0.8.5.post1
- FlashAttention 2
- Accelerate (optional)
- Additional packages:
  - python-dotenv
  - numpy
  - tqdm
  - packaging
  - deepspeed
  - wandb

### Hugging Face Authentication

An authenticated Hugging Face token with write permissions to push to the hubs:

- e.g., `od2961/Qwen2.5-1.5B-Open-R1-GRPO-math-v1`

## Data

We leverage the **OpenR1-Math 220k** dataset (HF: `open-r1/OpenR1-Math-220k`) plus task-specific sources for **crosswords** and **car park** reasoning. Raw inputs live in `data/` (see `data/README.md`); generated artifacts and checkpoints are kept under `artifacts/` (including `artifacts/models/`) to keep the data directory clean.

## Training

All training experiments were conducted using the base model:  
`Qwen/Qwen2.5-1.5B-Instruct (revision: main)`  
with modifications to enable **bfloat16**, **flash_attention_2**, and appropriate gradient settings.

### Model Arguments (Common)

These arguments are shared across GRPO training runs:

```yaml
model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct
model_revision: main
torch_dtype: bfloat16
attn_implementation: flash_attention_2
dataset_name: open-r1/OpenR1-Math-220k
dataset_prompt_column: problem
system_prompt: >
  "You are a helpful AI Assistant that provides well-reasoned and detailed responses.
   You first think about the reasoning process as an internal monologue and then provide
   the user with the answer. Respond in the following format:
   <think>\n...\n</think>\n<answer>\n...\n</answer>"
seed: 42
warmup_ratio: 0.05
bf16: true
use_vllm: true
gradient_accumulation_steps: 4
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false
```

Detailed training configurations live in `recipes/` (per-model + per-task). Helpful pointers:

- Core entrypoints: `src/training/grpo.py` (GRPO training) and `src/training/generate.py` (distilabel/vLLM-based generation).
- Ready-to-run launchers and SLURM specs: `scripts/training/` (GRPO training) and `scripts/inference/` (evaluation runs); see `scripts/README.md` for layout.
- Evaluation helpers and plotting live under `src/analysis/` (see `src/analysis/README.md`).

## Citation

If you use this work or the methodology in your own research, please cite as follows:

Liv G. d’Aliberti and Manoel Horta Ribeiro, “The Illusion of Insight in Reasoning Models,” unpublished workshop, June 2025.

```bibtex
@misc{daliberti2025cot,
  author       = {Liv G. d’Aliberti and Manoel Horta Ribeiro},
  title        = {The Illusion of Insight in Reasoning Models},
  year         = {2025},
  month        = jun,
  note         = {Unpublished workshop. \url{https://github.com/liv-daliberti/Chain-of-Thought-Traces}}
}
```

## License

This project is released under the MIT License. See \texttt{LICENSE} for details.
