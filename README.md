# Chain-of-Thought-Traces

**Owner:** liv-daliberti  
**Author:** Olivia G. d’Aliberti, Manoel Horta Ribeiro  
**Date:** June 2025  

A project demonstrating fine-tuning of a base Qwen 2.5-1.5B-Instruct model on the OpenR1 Math 220k dataset using two methodologies—Guided Reinforcement Preference Optimization (GRPO) and Supervised Fine-Tuning (SFT). Traces of chain-of-thought reasoning are logged and saved at fixed intervals. This repository also contains inference scripts to evaluate model performance on a held-out subset of 500 Math 220k problems.

---

## Table of Contents

1. [Repository Structure](#repository-structure)  
2. [Prerequisites](#prerequisites)  
3. [Setup & Installation](#setup--installation)  
4. [Data](#data)  
5. [Training](#training)  
   1. [Model Arguments (Common)](#model-arguments-common)  
   2. [GRPO Trainer Configuration](#grpo-trainer-configuration)  
   3. [SFT Trainer Configuration](#sft-trainer-configuration)  
   4. [Running GRPO](#running-grpo)  
   5. [Running SFT](#running-sft)  
6. [Inference](#inference)  
   1. [Inference Script Overview](#inference-script-overview)  
   2. [How to Run Inference](#how-to-run-inference)  
7. [Model Checkpoints & Hugging Face Links](#model-checkpoints--hugging-face-links)  
8. [Configuration Files (YAML)](#configuration-files-yaml)  
9. [Notes & Troubleshooting](#notes--troubleshooting)  
10. [Citation](#citation)  
11. [License](#license)  

---


## Repository Structure

```text
Chain-of-Thought-Traces/
├── Math220k/
│   ├── GRPO/                           # Contains logs and checkpoints for GRPO fine-tuning
│   └── SFT/                            # Contains logs and checkpoints for SFT fine-tuning
├── README.md                          # ← This file
├── inference.py                       # Script to run inference across saved checkpoints
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