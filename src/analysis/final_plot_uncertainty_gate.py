#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Legacy-compatible wrapper for the uncertainty-gated reconsideration figure.

Canonical usage is now::

  python -m src.analysis.final_plot_uncertainty_gate ...

Internally this forwards to :mod:`src.analysis.final_plot`, which contains the
actual implementation of the figure.
"""

from __future__ import annotations

from src.analysis import final_plot


def main() -> None:
    """
    Thin wrapper that dispatches to :func:`src.analysis.final_plot.main`.
    """
    final_plot.main()


if __name__ == "__main__":
    main()
