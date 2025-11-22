"""
Command-line entrypoints for inference tasks.

Modules in this package are thin wrappers that wire argument parsing and
backends together while deferring the heavy lifting to ``src.inference.runners``.
"""

from __future__ import annotations

__all__ = ["unified_math", "unified_carpark", "unified_crossword"]
