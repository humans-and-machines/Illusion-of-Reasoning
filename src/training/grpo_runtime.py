"""Backward-compatible alias for the relocated GRPO runtime entrypoint."""

from __future__ import annotations

from .runtime.main import main


__all__ = ["main"]
