"""Compatibility wrapper around the full GRPO implementation.

The heavy implementation lives in :mod:`src.training.grpo_runtime`. This module
exists so external callers can continue to import ``training.grpo_impl`` while
keeping this file small enough to satisfy linting limits.
"""

from __future__ import annotations

from .grpo_runtime import main  # re-export for backwards compatibility


__all__ = ["main"]
