#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backwards-compat wrapper re-exporting Figure 2 plotting helpers under
``src.analysis.core`` for older import paths used in tests.
"""

from src.analysis.figure_2_plotting_base import (  # noqa: F401
    FigureSaveConfig,
    Line2D,
    add_lower_center_legend,
    plt,
    save_figure_outputs,
)


__all__ = [
    "FigureSaveConfig",
    "add_lower_center_legend",
    "save_figure_outputs",
    "plt",
    "Line2D",
]
