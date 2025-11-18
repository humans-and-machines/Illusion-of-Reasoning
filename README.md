# The Illusion of Insight in Reasoning Models

Do reasoning models have ''Aha!'' moments? Prior work suggests that models like DeepSeek-R1-Zero undergo sudden realizations that lead to accurate outputs. Yet it remains unclear whether such shifts in reasoning strategy actually improve performance. In this paper, we formalize intrinsic ''Aha!'' events and instrument training runs to detect them across multiple tasks. We find that reasoning shifts are rare, do not become more frequent with training, and seldom improve accuracy. However, their effect varies according to model uncertainty. Building on this finding, we show that artificially triggering shifts under high entropy improves accuracy. Overall, our results challenge the perception that reasoning models' problem-solving capabilities stem from mid-reasoning shifts, although these shifts can be exploited to improve performance.

## Quick Start

- **Keep caches local (optional):** `source tools/local_env.sh` pins conda/pip/HF/tmp/W&B caches inside the repo.
- **Create env:** `make install` (uses `uv` to build `./openr1` with Python 3.11 and install `.[dev]`), then `source openr1/bin/activate`.
- **Conda alternative:** `conda env create -f configs/environment.yml && conda activate openr1`.
- **Install only runtime deps:** `pip install -e .`; dev extras: `pip install -e .[dev]`.
- **Authenticate for gated assets:** `huggingface-cli login` or `export HF_TOKEN=<token>`.
- **Run jobs:** ready-to-use launchers live under `scripts/inference/` (e.g., `bash scripts/inference/training-math-grpo.sh`); cluster specs live in `scripts/slurm/`.

A project demonstrating fine-tuning of a base Qwen 2.5-1.5B-Instruct model on the OpenR1 Math 220k dataset using two methodologies—Guided Reinforcement Preference Optimization (GRPO) and Supervised Fine-Tuning (SFT). Traces of chain-of-thought reasoning are logged and saved at fixed intervals. This repository also contains inference scripts to evaluate model performance on a subset of 500 Math 220k problems.

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
├── recipes/         # task/model YAMLs for GRPO, MaxEnt-GRPO, and SFT
├── data/            # task data (car_park, crossword) + human assessment prompts
├── src/open_r1/     # training/generation entrypoints (grpo.py, sft.py, generate.py, rlif.py, rewards*)
├── scripts -> src/scripts/  # launchers for inference/training, annotation, analysis, viz, utils, slurm
├── results/, artifacts/, models/  # experiment outputs and checkpoints
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

- `od2961/Qwen2.5-1.5B-Instruct-SFT`
- `od2961/Qwen2.5-1.5B-Instruct-GRPO`

## Data

We leverage the **OpenR1-Math 220k** dataset (HF: `open-r1/OpenR1-Math-220k`) plus task-specific sources for **crosswords** and **car park** reasoning. Raw inputs live in `data/` (see `data/README.md`); generated artifacts and checkpoints are kept under `results/`, `models/`, or `artifacts/` to keep the data directory clean.

## Training

All training experiments were conducted using the base model:  
`Qwen/Qwen2.5-1.5B-Instruct (revision: main)`  
with modifications to enable **bfloat16**, **flash_attention_2**, and appropriate gradient settings.

### Model Arguments (Common)

These arguments are shared between GRPO and SFT:

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

- Core entrypoints: `src/open_r1/grpo.py`, `src/open_r1/sft.py`, `src/open_r1/generate.py`, `src/open_r1/rlif.py`.
- Ready-to-run launchers and SLURM specs: `scripts/inference/` and `scripts/slurm/` (see `scripts/README.md` for layout).
- Evaluation helpers and plotting live under `scripts/analysis/` and `scripts/viz/`.

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
