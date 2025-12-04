#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Common Matplotlib styling and tiny plotting utilities for analysis scripts.

This module is deliberately minimal; higher-level plotting code should stay in
the figure/graph scripts themselves.
"""

from __future__ import annotations

import os
from typing import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Global style helpers
# ---------------------------------------------------------------------------

A4_PORTRAIT: tuple[float, float] = (8.27, 11.69)
A4_LANDSCAPE: tuple[float, float] = (11.69, 8.27)


DEFAULT_STYLE: Mapping[str, object] = {
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "font.size": 14,
    "legend.fontsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "mathtext.rm": "serif",
    "text.usetex": False,
    "pdf.use14corefonts": False,
}


def apply_default_style(extra: Mapping[str, object] | None = None) -> None:
    """
    Apply a Times-like serif style consistent with the main paper figures.

    Optionally, pass a small dict of overrides via extra.
    """
    mpl.rcdefaults()
    params = dict(DEFAULT_STYLE)
    if extra:
        params.update(extra)
    mpl.rcParams.update(params)
    mpl.rcParams["axes.titlesize"] = params.get(
        "axes.titlesize",
        mpl.rcParams.get("axes.titlesize"),
    )
    if extra:
        # Apply extras a second time to guarantee they override any downstream defaults.
        mpl.rcParams.update(extra)
        if "axes.titlesize" in extra:
            mpl.rcParams["axes.titlesize"] = extra["axes.titlesize"]


def apply_entropy_plot_style(extra: Mapping[str, object] | None = None) -> None:
    """
    Apply a Times-like style tuned for entropy/uncertainty plots (graph_3/graph_4).
    """
    base_extra: dict[str, object] = {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    if extra:
        base_extra.update(extra)
    apply_default_style(base_extra)


def apply_paper_font_style(
    font_family: str = "Times New Roman",
    font_size: int = 12,
    mathtext_fontset: str | None = "dejavuserif",
) -> None:
    """
    Style tuned for A4-style paper figures and PDFs (H1/H3 summaries,
    figure_1/figure_2), shared to avoid duplicated rcParams blocks.
    """
    params: dict[str, object] = {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": ["serif"],
        "font.serif": [
            font_family,
            "Times",
            "DejaVu Serif",
            "Computer Modern Roman",
        ],
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "figure.titlesize": font_size,
    }
    if mathtext_fontset:
        params["mathtext.fontset"] = mathtext_fontset
    mpl.rcParams.update(params)
    # Ensure font.family is treated as a list so rcParams indexing is stable in tests.
    mpl.rcParams["font.family"] = ["serif"]


def a4_size_inches(orientation: str = "landscape") -> tuple[float, float]:
    """
    Return (width, height) in inches for an A4 page.

    ``orientation`` may be ``\"landscape\"`` or ``\"portrait\"`` (case-insensitive).
    """
    if str(orientation).lower().startswith("p"):
        return A4_PORTRAIT
    return A4_LANDSCAPE


# ---------------------------------------------------------------------------
# Small convenience helpers
# ---------------------------------------------------------------------------


def save_figure(
    fig: plt.Figure,
    outbase: str,
    dpi: int = 300,
    formats: Sequence[str] = ("png", "pdf"),
    tight: bool = True,
) -> None:
    """
    Save a Matplotlib figure to multiple formats, creating parent dirs.

    outbase should be a path without extension; typical usage:
        save_figure(fig, \"out/figure1\")
    which will create out/figure1.png and out/figure1.pdf.
    """
    outdir = os.path.dirname(outbase)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    bbox = "tight" if tight else None
    for fmt in formats:
        path = f"{outbase}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches=bbox)
    plt.close(fig)
