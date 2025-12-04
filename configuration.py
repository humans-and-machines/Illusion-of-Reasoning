"""
Stub module to satisfy static analysis of ``__all__`` when optional deps are absent.

Pylint/astroid complains if names listed in ``__all__`` are not defined. These
placeholders mirror the lazy exports in ``src.training.runtime.env`` and keep
linters quiet in environments where the real dependencies (torch, datasets, etc.)
are not installed.
"""

# The export names intentionally mirror the underlying libraries, so keep their
# original casing to satisfy imports that expect these symbols.
# Pylint sometimes struggles to see these stubbed names in ``__all__`` when
# running in synthetic / analysis-only modes, so silence that check here.
# pylint: disable=invalid-name,duplicate-code,undefined-all-variable
from typing import Any

datasets: Any = None
torch: Any = None
dist: Any = None
transformers: Any = None
wandb: Any = None
DataLoader: Any = None
RandomSampler: Any = None
AcceleratorState: Any = None
ZeroStageEnum: Any = None
ZeroParamStatus: Any = None
get_last_checkpoint: Any = None
get_peft_config: Any = None
GRPOTrainer: Any = None
TrainerCallback: Any = None
set_seed: Any = None


def _resolve_dependency(_instance: Any, _name: str, default: Any) -> Any:
    """
    Stub dependency resolver used when optional training deps are missing.

    Returns the provided default to mirror the runtime helper signature.
    """
    return default
resolve_dependency: Any = _resolve_dependency

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
    "resolve_dependency",
]
