"""Shared plotting/CLI helpers used across figure scripts."""

from __future__ import annotations

import argparse
from typing import Any, Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ..plotting import a4_size_inches, apply_paper_font_style


def _safe_attr(line: Any, getter_name: str, fallback_attrs: Sequence[str], default: Any) -> Any:
    """
    Best-effort retrieval for matplotlib line attributes across backends/stubs.
    """
    getter = getattr(line, getter_name, None)
    value = None
    if callable(getter):
        try:
            value = getter()
        except (TypeError, ValueError, AttributeError):  # pragma: no cover - safety net
            value = None
    if value is None:
        for attr in fallback_attrs:
            value = getattr(line, attr, None)
            if value is not None:
                break
    return default if value is None else value


def set_global_fonts(font_family: str = "Times New Roman", font_size: int = 12) -> None:
    """
    Apply the default paper font style (Times, 12pt) for all figure elements.
    """
    apply_paper_font_style(
        font_family=font_family,
        font_size=font_size,
        mathtext_fontset="dejavuserif",
    )


def compute_effective_max_step(
    args: argparse.Namespace,
    hard_max_step: int,
) -> int:
    """
    Shared helper for enforcing a hard cap on ``max_step`` in CLI scripts.
    """
    if args.max_step is None:
        effective = hard_max_step
        print(
            f"[info] Capping max_step to {effective} (hard cap = {hard_max_step}).",
        )
        return effective

    if args.max_step > hard_max_step:
        effective = hard_max_step
        print(
            f"[info] Capping max_step to {effective} (hard cap = {hard_max_step}).",
        )
        return effective

    return args.max_step


def create_a4_figure(
    orientation: str = "portrait",
    dpi: int = 300,
) -> plt.Figure:
    """
    Create a Matplotlib figure sized to an A4 page in inches.
    """
    page_size = a4_size_inches(orientation)
    return plt.figure(figsize=page_size, dpi=dpi)


def aha_histogram_legend_handles(axis_right: Axes) -> Sequence[Line2D | Patch]:
    """
    Build legend handles for the uncertainty histogram + Aha counts figure.
    """
    bars_proxy = Patch(facecolor="#CCCCCC", label="Total (bin count)")
    handles: list[Line2D | Patch] = [bars_proxy]
    for line in axis_right.lines:
        # Some matplotlib line objects (or test doubles) may not expose the full
        # getter API, so pull attributes defensively with sensible fallbacks.
        color = _safe_attr(line, "get_color", ["color", "_color"], "C0")
        label = _safe_attr(line, "get_label", ["label"], "")
        marker = _safe_attr(line, "get_marker", ["marker"], "o")
        linewidth = _safe_attr(line, "get_linewidth", ["linewidth"], 1.8)

        handle = Line2D(
            [0],
            [0],
            color=color,
            lw=linewidth,
            marker=marker,
            label=label,
        )
        handles.append(handle)
    return handles
