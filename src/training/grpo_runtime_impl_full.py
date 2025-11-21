"""Compatibility shim for the legacy GRPO runtime implementation.

The original :mod:`training.grpo_runtime_impl_full` module used to contain the
entire GRPO training runtime in a single, very large file. To satisfy linting
limits and keep the codebase maintainable, the implementation has been split
into smaller, focused modules:

* :mod:`training.grpo_runtime_env` – shared imports and runtime patches
* :mod:`training.grpo_trainer_replay_support` – helper dataclasses and utilities
* :mod:`training.grpo_trainer_replay_impl` – replay-enabled GRPO trainer class
* :mod:`training.grpo_runtime_main` – top-level ``main`` entry point

This module re-exports the public API so existing imports of
``training.grpo_runtime_impl_full`` continue to work unchanged.
"""

from __future__ import annotations

from .grpo_runtime_main import main
from .grpo_trainer_replay_impl import GRPOTrainerReplay
from .grpo_trainer_replay_support import (
    ReplaySettings,
    TemperatureSchedule,
    MixSettings,
    RuntimeState,
    LossLoggingCallback,
)

__all__ = [
    "ReplaySettings",
    "TemperatureSchedule",
    "MixSettings",
    "RuntimeState",
    "LossLoggingCallback",
    "GRPOTrainerReplay",
    "main",
]
