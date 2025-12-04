#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared plotting styles and utilities used by the H2/H3 figures.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib
from matplotlib import cm


class _ColormapRegistryFallback(dict):
    """Dict-like shim that defers to ``cm.get_cmap``."""

    def __getitem__(self, name):
        return cm.get_cmap(name)


def _resolve_colormap_registry():
    """
    Backfill ``matplotlib.colormaps`` for environments with older matplotlib installs.
    Uses the live ``matplotlib.colormaps`` attribute so monkeypatches are honored.
    """
    return getattr(matplotlib, "colormaps", None) or _ColormapRegistryFallback()


matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "Nimbus Roman",
            "TeX Gyre Termes",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 12,
    }
)

METRIC_LABELS: Dict[str, str] = {
    "sample": "Per-draw accuracy",
    "cluster_mean": "Per-problem mean of 8",
    "cluster_any": "Per-problem any-correct",
}

DEFAULT_COLORS: Dict[str, str] = {
    "bar_primary": "#2C7BB6",
    "bar_secondary": "#FDAE61",
    "bar_tertiary": "#ABD9E9",
    "gain": "#1B7837",
    "loss": "#D73027",
    "stable_pos": "#4D4D4D",
    "stable_neg": "#BDBDBD",
}


def parse_color_overrides(color_string: Optional[str]) -> Dict[str, str]:
    """Parse ``name:hex`` overrides from a CLI string."""
    if not color_string:
        return {}
    overrides: Dict[str, str] = {}
    for part in color_string.split(","):
        if ":" in part:
            key, value = part.split(":", 1)
            overrides[key.strip()] = value.strip()
    return overrides


def cmap_colors(name: str) -> List[Tuple[float, float, float]]:
    """
    Access a qualitative matplotlib colormap and return RGB tuples.
    """
    try:
        seq = _resolve_colormap_registry()[name].colors
    except (KeyError, AttributeError, TypeError):
        seq = cm.get_cmap(name).colors
    rgb_list: List[Tuple[float, float, float]] = []
    for color in seq:
        red, green, blue = color[:3]
        rgb_list.append((float(red), float(green), float(blue)))
    return rgb_list


def darken_colors(
    colors: List[Tuple[float, float, float]],
    factor: float = 0.8,
) -> List[Tuple[float, float, float]]:
    """Scale RGB tuples by ``factor`` to darken/lighten the palette."""
    factor = float(factor)
    return [
        (
            max(0.0, red * factor),
            max(0.0, green * factor),
            max(0.0, blue * factor),
        )
        for red, green, blue in colors
    ]
