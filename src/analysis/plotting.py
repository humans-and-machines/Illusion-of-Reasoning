#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Common Matplotlib styling and tiny plotting utilities for analysis scripts.

This module is deliberately minimal; higher-level plotting code should stay in
the figure/graph scripts themselves.
"""

from __future__ import annotations

import os
from typing import Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Global style helpers
# ---------------------------------------------------------------------------

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
    params = dict(DEFAULT_STYLE)
    if extra:
        params.update(extra)
    mpl.rcParams.update(params)


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


