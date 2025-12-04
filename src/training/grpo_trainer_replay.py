"""Compatibility wrapper exposing the GRPO replay trainer stack.

This module re-exports the replay-enabled trainer and related helper
dataclasses defined in :mod:`src.training.grpo_runtime_impl_full` so that
callers can import them from a smaller, focused namespace.
"""

from __future__ import annotations

from .grpo_runtime_impl_full import (  # type: ignore[attr-defined]
    GRPOTrainerReplay,
    LossLoggingCallback,
    MixSettings,
    ReplaySettings,
    RuntimeState,
    TemperatureSchedule,
)


__all__ = [
    name.__name__
    for name in (
        ReplaySettings,
        TemperatureSchedule,
        MixSettings,
        RuntimeState,
        LossLoggingCallback,
        GRPOTrainerReplay,
    )
]
