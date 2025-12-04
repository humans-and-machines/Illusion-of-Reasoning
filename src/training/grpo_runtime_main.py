"""Compatibility stub for the relocated GRPO runtime entry point."""

from __future__ import annotations

from .runtime import main as _runtime_main


__all__ = [name for name in dir(_runtime_main) if not name.startswith("_")]
globals().update({name: getattr(_runtime_main, name) for name in __all__})
