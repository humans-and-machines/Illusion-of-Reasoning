#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Accuracy overlay helpers for the uncertainty/correctness figures.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.analysis.figure_2_data import wilson_ci
from src.analysis.figure_2_plotting_base import FigureSaveConfig, plt, save_figure_outputs
from src.analysis.plotting import a4_size_inches


@dataclass
class AccuracySeriesSpec:
    """Mask/color spec for an accuracy overlay series."""

    label: str
    mask: np.ndarray
    color: str


def _build_accuracy_rows(
    *,
    label: str,
    centers: np.ndarray,
    accuracy: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    num_total: np.ndarray,
    num_correct: np.ndarray,
    half_width: float,
) -> List[Dict[str, Any]]:
    """Build per-bin summary rows for a single accuracy series."""
    rows: List[Dict[str, Any]] = []
    for center, value, lower, upper, correct, total in zip(
        centers,
        accuracy,
        lower_bounds,
        upper_bounds,
        num_correct,
        num_total,
    ):
        rows.append(
            {
                "variant": label,
                "bin_center": float(center),
                "bin_left": float(center - half_width),
                "bin_right": float(center + half_width),
                "n": int(total),
                "k": int(correct),
                "acc": float(value),
                "lo": float(lower),
                "hi": float(upper),
            },
        )
    return rows


def _overlay_accuracy_series(
    *,
    samples: pd.DataFrame,
    edges: np.ndarray,
    spec: AccuracySeriesSpec,
    axis,
    half_width: float,
) -> List[Dict[str, Any]]:
    """Plot a single accuracy series and return the rows contributed by it."""
    subset = samples.loc[spec.mask]
    centers, accuracy, lower_bounds, upper_bounds, num_correct, num_total = per_bin_accuracy(
        subset["uncertainty_std"].to_numpy(),
        subset["correct"].to_numpy().astype(int),
        edges,
    )
    axis.plot(
        centers,
        accuracy,
        lw=2.0,
        color=spec.color,
        label=spec.label,
    )
    axis.fill_between(
        centers,
        lower_bounds,
        upper_bounds,
        color=spec.color,
        alpha=0.15,
    )
    return _build_accuracy_rows(
        label=spec.label,
        centers=centers,
        accuracy=accuracy,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        num_total=num_total,
        num_correct=num_correct,
        half_width=half_width,
    )


def per_bin_accuracy(
    values_std: np.ndarray,
    correct: np.ndarray,
    edges: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute accuracy and Wilson CIs per uncertainty bin."""
    counts_total = np.histogram(values_std, bins=edges)[0]
    counts_correct = np.histogram(
        values_std,
        bins=edges,
        weights=correct.astype(float),
    )[0]
    acc = np.divide(
        counts_correct,
        counts_total,
        out=np.full_like(counts_correct, np.nan, dtype=float),
        where=counts_total > 0,
    )
    lower_bounds = np.full_like(acc, np.nan, dtype=float)
    upper_bounds = np.full_like(acc, np.nan, dtype=float)
    for idx, (correct_count, total_count) in enumerate(
        zip(counts_correct.astype(int), counts_total.astype(int))
    ):
        if total_count > 0:
            lower_bounds[idx], upper_bounds[idx] = wilson_ci(correct_count, total_count)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return (
        centers,
        acc,
        lower_bounds,
        upper_bounds,
        counts_correct.astype(int),
        counts_total.astype(int),
    )


@dataclass
class AccuracyPlotConfig(FigureSaveConfig):
    """Configuration for per-bin accuracy overlay plots."""


def plot_accuracy_by_bin_overlay(
    all_samples: pd.DataFrame,
    edges: np.ndarray,
    config: AccuracyPlotConfig,
) -> str:
    """
    Plot per-bin accuracy with Wilson confidence intervals for multiple strata.
    """
    has_formal = "aha_formal" in all_samples.columns
    series_specs = [
        AccuracySeriesSpec("All", np.ones(len(all_samples), dtype=bool), "#666666"),
        AccuracySeriesSpec("Words", all_samples["aha_words"] == 1, "#1f77b4"),
        AccuracySeriesSpec("LLM", all_samples["aha_gpt"] == 1, "#ff7f0e"),
        AccuracySeriesSpec(
            "Formal",
            all_samples["aha_formal"] == 1
            if has_formal
            else np.zeros(len(all_samples), dtype=bool),
            "#2ca02c",
        ),
    ]

    fig_size = (
        a4_size_inches(config.a4_orientation)
        if config.a4_pdf
        else (12.5, 4.8)
    )
    fig, axis = plt.subplots(figsize=fig_size, dpi=150)
    rows: List[Dict[str, Any]] = []
    half_width = (edges[1] - edges[0]) / 2.0

    for spec in series_specs:
        rows.extend(
            _overlay_accuracy_series(
                samples=all_samples,
                edges=edges,
                spec=spec,
                axis=axis,
                half_width=half_width,
            )
        )

    axis.set_title(f"Per-bin accuracy with Wilson 95% CIs\n{config.title_suffix}")
    axis.set_xlabel("uncertainty_std")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0.0, 1.0)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best", frameon=False)

    save_figure_outputs(fig, config, dpi=150)

    out_csv = config.out_png.replace(".png", ".csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv
