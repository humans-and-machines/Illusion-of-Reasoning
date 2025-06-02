# Chain-of-Thought-Traces

**Owner:** liv-daliberti  
**Author:** Liv G. d’Aliberti, Manoel Horta Ribeiro  
**Date:** June 2025  

A project demonstrating fine-tuning of a base Qwen 2.5-1.5B-Instruct model on the OpenR1 Math 220k dataset using two methodologies—Guided Reinforcement Preference Optimization (GRPO) and Supervised Fine-Tuning (SFT). Traces of chain-of-thought reasoning are logged and saved at fixed intervals. This repository also contains inference scripts to evaluate model performance on a subset of 500 Math 220k problems.

---

## Table of Contents

1. [Repository Structure](#repository-structure)  
2. [Prerequisites](#prerequisites)  
3. [Data](#data)  
4. [Training](#training)  
   1. [Model Arguments (Common)](#model-arguments-common)  
5. [Citation](#citation)  
6. [License](#license)  

---


## Repository Structure

```text
Chain-of-Thought-Traces/
├── Math220k/
│   ├── GRPO/                          # Contains checkpoint inference for GRPO fine-tuning
│   └── SFT/                           # Contains checkpoint inference for SFT fine-tuning
├── README.md                          # ← This file
├── LICENSE                            # License info
├── inference.py                       # Script to run inference across saved checkpoints
├── judgement.py                       # Script to run evaluate model accuracy during training via gpt-4o
└── yaml/                              # (Optional) Folder for YAML configuration files
```

- **Math220k/GRPO/**  
  Output directories and model checkpoints are stored every 50 steps under the Hugging Face hub (`Qwen2.5-1.5B-Instruct-GRPO`).

- **Math220k/SFT/**  
  Output directories and model checkpoints are stored every 50 steps under the Hugging Face hub (`Qwen2.5-1.5B-Instruct-SFT`).

- **inference.py**  
  Python script that loads each saved revision (SHA) and runs inference on a fixed subset of 500 Math 220k examples, logging chain-of-thought traces into JSONL files.

- **yaml/** (optional)  
  Folder containing YAML files for GRPO and SFT configuration (trainer arguments, hyperparameters, etc.).

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

We leverage the **OpenR1-Math 220k** dataset, which comprises 220,000 math problems with chain-of-thought annotations. For training and inference, we use the Hugging Face `open-r1/OpenR1-Math-220k` repository and its default configuration.

- **Train Split:**  
  Only the first 500 selected for inference, consistent across SFT and GRPO

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

Detailed training configurations are available within the yaml folder. 

## Citation

If you use this work or the methodology in your own research, please cite as follows:

Liv G. d’Aliberti and Manoel Horta Ribeiro, “Chain-of-Thought Traces: Fine-Tuning Qwen 2.5-1.5B with GRPO \& SFT on Math 220k,” unpublished workshop, June 2025.

```bibtex
@misc{daliberti2025cot,
  author       = {Liv G. d’Aliberti and Manoel Horta Ribeiro},
  title        = {Chain-of-Thought Traces: Fine-Tuning Qwen 2.5-1.5B with GRPO \& SFT on Math 220k},
  year         = {2025},
  month        = jun,
  note         = {Unpublished workshop. \url{https://github.com/liv-daliberti/Chain-of-Thought-Traces}}
}
```

## License

This project is released under the MIT License. See \texttt{LICENSE} for details.