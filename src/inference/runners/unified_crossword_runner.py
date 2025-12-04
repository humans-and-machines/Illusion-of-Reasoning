"""
Library helpers for the unified crossword runner.

CLI wiring now lives in :mod:`src.inference.cli.unified_crossword`.
"""

from __future__ import annotations

from importlib import import_module


__all__ = ["load_crossword_module"]


def load_crossword_module():
    """
    Resolve the crossword inference module.

    :returns: Imported :mod:`src.inference.domains.crossword.crossword_core` module.
    """
    return import_module("src.inference.domains.crossword.crossword_core")
