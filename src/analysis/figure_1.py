#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aha Ratios (problem-level) command-line interface.

Delegates to :mod:`src.analysis.core.figure_1_components` so that this module
can be invoked with ``python -m src.analysis.figure_1``.
"""

from __future__ import annotations


try:  # pragma: no cover - prefer absolute import when available
    from src.analysis.core.figure_1_components import main as _figure1_main
except ImportError:  # pragma: no cover - fallback for direct execution
    from core.figure_1_components import main as _figure1_main  # type: ignore[import]


def main() -> None:
    """Execute the Figure 1 CLI entry point."""
    _figure1_main()


if __name__ == "__main__":
    main()
