#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared plotting utilities for Figure 2 helpers.
"""

from dataclasses import dataclass
from typing import Iterable, Tuple

from src.analysis.plotting import a4_size_inches

try:  # pragma: no cover - optional dependency
    import matplotlib as _mpl  # type: ignore[import]
    import matplotlib.pyplot as _plt  # type: ignore[import]
    from matplotlib.lines import Line2D as _Line2D  # type: ignore[import]
except ImportError:  # pragma: no cover - optional dependency
    plt = None  # type: ignore[assignment]  # pylint: disable=invalid-name
    Line2D = None  # type: ignore[assignment]  # pylint: disable=invalid-name
else:  # pragma: no cover - optional dependency
    _mpl.use("Agg")
    plt = _plt
    Line2D = _Line2D


@dataclass
class FigureSaveConfig:
    """
    Shared subset of plot config fields for saving PNG/PDF variants.
    """

    out_png: str
    out_pdf: str
    title_suffix: str
    a4_pdf: bool
    a4_orientation: str


def save_figure_outputs(
    figure,
    config: FigureSaveConfig,
    *,
    dpi: int = 150,
    tight_layout: bool = True,
) -> None:
    """
    Persist ``figure`` as PNG/PDF using the metadata in ``config``.
    """
    if tight_layout and hasattr(figure, "tight_layout"):
        figure.tight_layout()
    figure.savefig(config.out_png, dpi=dpi)
    if config.a4_pdf:
        width, height = a4_size_inches(config.a4_orientation)
        figure.set_size_inches(width, height)
    figure.savefig(config.out_pdf, dpi=dpi)
    plt.close(figure)


def add_lower_center_legend(
    figure,
    handles: Iterable[Line2D],
    *,
    bbox: Tuple[float, float] = (0.5, 0.02),
    columns: int = 2,
) -> None:
    """
    Attach a lower-centered legend shared across several Figure 2 plots.
    """
    figure.legend(
        handles,
        loc="lower center",
        ncol=int(columns),
        frameon=False,
        bbox_to_anchor=bbox,
    )
