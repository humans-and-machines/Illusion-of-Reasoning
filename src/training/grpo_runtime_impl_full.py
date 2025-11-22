"""Legacy module that re-exports the full GRPO runtime implementation."""

from src.training.grpo_runtime_main import main
from src.training.grpo_trainer_replay_impl import GRPOTrainerReplay
from src.training.grpo_trainer_replay_support import (
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
