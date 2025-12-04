"""
Backwards-compatible shim for the shift annotation CLI.

Canonical implementation now lives in :mod:`src.annotate.cli.shift_cli`, but we
resolve that module dynamically so test stubs and future refactors are honored.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable


def _load_impl() -> Any:
    """Load the canonical CLI module (honors sys.modules monkeypatches)."""
    return importlib.import_module("src.annotate.cli.shift_cli")


def __getattr__(name: str) -> Any:
    """Dynamically proxy attributes to the canonical CLI implementation."""
    if name in {"main", "build_argparser"}:
        return getattr(_load_impl(), name)
    raise AttributeError(name)


build_argparser: Callable[..., Any]
main: Callable[..., Any]

# Public API (resolved lazily via __getattr__)
__all__ = ["build_argparser", "main"]


if __name__ == "__main__":
    _load_impl().main()
