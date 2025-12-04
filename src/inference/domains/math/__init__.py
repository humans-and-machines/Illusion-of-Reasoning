"""
Math-specific inference domain modules.

This subpackage now hosts the two-pass core, ZeRO/LLM variants, and the
shared runner helpers that previously lived in the legacy ``core`` package.
"""

from __future__ import annotations

from importlib import import_module


__all__ = ["math_core", "math_core_runner", "math_llama_core"]


def __getattr__(name):
    if name in __all__:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
