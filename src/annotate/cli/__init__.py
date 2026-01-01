"""
CLI entrypoints for annotation utilities.

These modules are primarily intended to be executed as scripts.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable


__all__ = ["shift_build_argparser", "shift_main", "clean_failed_main"]


def __getattr__(name: str) -> Any:
    """
    Lazily resolve CLI callables to avoid importing script modules at package import time.

    This prevents `python -m src.annotate.cli.shift_cli` from triggering
    `RuntimeWarning: ... found in sys.modules ...` due to eager imports.
    """
    if name == "clean_failed_main":
        return importlib.import_module("src.annotate.cli.clean_cli").main
    if name == "shift_build_argparser":
        return importlib.import_module("src.annotate.cli.shift_cli").build_argparser
    if name == "shift_main":
        return importlib.import_module("src.annotate.cli.shift_cli").main
    raise AttributeError(name)


clean_failed_main: Callable[..., Any]
shift_build_argparser: Callable[..., Any]
shift_main: Callable[..., Any]
