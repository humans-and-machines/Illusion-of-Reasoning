"""Compatibility wrapper around the full GRPO runtime implementation.

The heavy implementation lives in :mod:`src.training.grpo_runtime_impl_full`.
This module exists so external callers can continue to import
``training.grpo_runtime_impl`` while keeping this file small enough to satisfy
linting limits.
"""

from __future__ import annotations

from .grpo_runtime_impl_full import main


__all__ = ["main"]

