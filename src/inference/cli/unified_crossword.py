#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified cryptic crossword inference CLI.
"""

from __future__ import annotations

from typing import Optional, Sequence

from src.inference.runners.unified_crossword_runner import load_crossword_module
from src.inference.runners.unified_runner_base import run_crossword_main
from src.inference.backends import HFBackend

__all__ = [
    "HFBackend",
    "load_crossword_module",
    "run_crossword_main",
    "main",
]


def _load_crossword_module():
    """Wrapper so tests may monkeypatch the loader."""
    return load_crossword_module()


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point for crossword inference CLI."""
    run_crossword_main(load_module=_load_crossword_module, backend_cls=HFBackend, argv=argv)


if __name__ == "__main__":
    main()
