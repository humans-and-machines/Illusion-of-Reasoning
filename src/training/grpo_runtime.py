"""Thin wrapper around the full GRPO runtime implementation.

This module keeps the public entrypoint name ``training.grpo_runtime`` small
while delegating the heavy implementation to :mod:`training.grpo_runtime_impl`.
"""

from __future__ import annotations

from .grpo_runtime_impl import main  # re-export for backwards compatibility


__all__ = ["main"]
