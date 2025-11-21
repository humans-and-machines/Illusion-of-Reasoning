"""Shared runtime imports and environment patches for GRPO training.

This module centralises side-effectful configuration (torch/ZeRO pickling,
NLTK data path, etc.) so that the higher-level training modules can remain
small and focused.
"""

from __future__ import annotations

import functools
import importlib
import os
from pathlib import Path

datasets = importlib.import_module("datasets")
torch = importlib.import_module("torch")
dist = importlib.import_module("torch.distributed")
torch_serialization = importlib.import_module("torch.serialization")
setattr(torch, "serialization", torch_serialization)
transformers = importlib.import_module("transformers")
wandb = importlib.import_module("wandb")
torch_utils_data = importlib.import_module("torch.utils.data")
DataLoader = getattr(torch_utils_data, "DataLoader")
RandomSampler = getattr(torch_utils_data, "RandomSampler")
accelerate_state = importlib.import_module("accelerate.state")
AcceleratorState = getattr(accelerate_state, "AcceleratorState")
deepspeed_zero_config = importlib.import_module("deepspeed.runtime.zero.config")
ZeroStageEnum = getattr(deepspeed_zero_config, "ZeroStageEnum")
deepspeed_zero_pp = importlib.import_module("deepspeed.runtime.zero.partition_parameters")
ZeroParamStatus = getattr(deepspeed_zero_pp, "ZeroParamStatus")
transformers_trainer_utils = importlib.import_module("transformers.trainer_utils")
get_last_checkpoint = getattr(transformers_trainer_utils, "get_last_checkpoint")
trl_module = importlib.import_module("trl")
get_peft_config = getattr(trl_module, "get_peft_config")
trl_grpo_trainer_module = importlib.import_module("trl.trainer.grpo_trainer")
GRPOTrainer = getattr(trl_grpo_trainer_module, "GRPOTrainer")
TrainerCallback = getattr(transformers, "TrainerCallback")
set_seed = getattr(transformers, "set_seed")

# ────────────────── ZeRO pickle patch (torch-2.6) ─────────────────────────

_TORCH_DEFAULT_WEIGHTS_ONLY_ATTR = "_default_weights_only"
setattr(torch.serialization, _TORCH_DEFAULT_WEIGHTS_ONLY_ATTR, False)

torch.serialization.add_safe_globals(
    {
        ("deepspeed.runtime.zero.config", "ZeroStageEnum"): ZeroStageEnum,
        (
            "deepspeed.runtime.zero.partition_parameters",
            "ZeroParamStatus",
        ): ZeroParamStatus,
    }
)

_orig_load = torch.load
torch.load = functools.partial(_orig_load, weights_only=False)  # type: ignore[arg-type]

# ─────────────────── NLTK data path (WordNet) ───────────────────────────
NLTK_DATA_DEFAULT = Path(__file__).resolve().parents[2] / ".nltk_data"
os.environ.setdefault("NLTK_DATA", str(NLTK_DATA_DEFAULT))


__all__ = [
    "datasets",
    "torch",
    "dist",
    "transformers",
    "wandb",
    "DataLoader",
    "RandomSampler",
    "AcceleratorState",
    "ZeroStageEnum",
    "ZeroParamStatus",
    "get_last_checkpoint",
    "get_peft_config",
    "GRPOTrainer",
    "TrainerCallback",
    "set_seed",
]
