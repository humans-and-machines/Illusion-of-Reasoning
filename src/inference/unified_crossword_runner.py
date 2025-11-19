#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified launcher for the cryptic crossword inference loop using crossword_core.
Thin wrapper over src.inference.unified_runner_base.
"""
from __future__ import annotations

from src.inference import crossword_core
from src.inference.backends import HFBackend
from src.inference.unified_runner_base import run_crossword_main


def _load_crossword_module():
    """Indirection for tests to monkeypatch the crossword_core module."""
    return crossword_core


def main() -> None:
    """Run cryptic crossword inference with a unified HF backend."""
    run_crossword_main(load_module=_load_crossword_module, backend_cls=HFBackend)


__all__ = ["main"]


if __name__ == "__main__":
    main()
