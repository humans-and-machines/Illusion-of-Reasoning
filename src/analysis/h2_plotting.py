#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plotting helpers for the H2 analysis.

This module focuses on lightweight figures that live next to the GLMs.
More expensive bucket + histogram helpers now live in
``src.analysis.core.h2_uncertainty_helpers`` so the definitions can be
shared with ``h2_analysis.py`` with little duplication.
"""

from __future__ import annotations

import os
from typing import Any, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.core.h2_uncertainty_helpers import (
    make_all3_uncertainty_buckets_figure,
    plot_uncertainty_hist_100bins,
    style_ame_axis,
)


def lineplot(
    x_values: Any,
    y_values: Any,
    labels: Tuple[str, str, str, str],
) -> None:
    """
    Small helper to draw a single line plot.

    :param x_values: Values for the x-axis.
    :param y_values: Values for the y-axis.
    :param labels: Tuple ``(xlabel, ylabel, title, path)``.
    """
    xlabel, ylabel, title, path = labels
    figure, axis = plt.subplots(figsize=(7.5, 4.5), dpi=140)
    axis.plot(x_values, y_values, marker="o")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def plot_diag_panel(step_df: pd.DataFrame, out_dir: str) -> None:
    """Plot (1) Uncertainty vs Aha, (2) Uncertainty vs step, (3) Aha vs step."""
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), dpi=140)

    # (1) Uncertainty vs Aha
    uncertainty_no_aha = step_df.loc[
        step_df["aha"] == 0,
        "uncertainty_std",
    ].values
    uncertainty_with_aha = step_df.loc[
        step_df["aha"] == 1,
        "uncertainty_std",
    ].values
    try:
        axes[0].boxplot(
            [uncertainty_no_aha, uncertainty_with_aha],
            tick_labels=["aha=0", "aha=1"],
            showfliers=False,
        )
    except TypeError:
        axes[0].boxplot(
            [uncertainty_no_aha, uncertainty_with_aha],
            labels=["aha=0", "aha=1"],
            showfliers=False,
        )
    axes[0].set_title("Uncertainty vs Aha")
    axes[0].set_ylabel("uncertainty_std")

    # (2) Mean uncertainty vs step
    mean_uncertainty_by_step = step_df.groupby(
        "step",
        as_index=False,
    ).agg(
        mu=("uncertainty_std", "mean"),
    )
    axes[1].plot(
        mean_uncertainty_by_step["step"],
        mean_uncertainty_by_step["mu"],
        marker="o",
    )
    axes[1].set_title("Uncertainty vs step")
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("mean uncertainty_std")
    axes[1].grid(True, alpha=0.3)

    # (3) Aha ratio vs step
    aha_ratio_by_step = step_df.groupby(
        "step",
        as_index=False,
    ).agg(
        r=("aha", "mean"),
    )
    axes[2].plot(
        aha_ratio_by_step["step"],
        aha_ratio_by_step["r"],
        marker="o",
    )
    axes[2].set_title("Aha vs step")
    axes[2].set_xlabel("Training step")
    axes[2].set_ylabel("P(aha=1)")
    axes[2].grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(os.path.join(out_dir, "h2_diag_panel.png"))
    plt.close(figure)


def plot_ame_with_ci(regression_df: pd.DataFrame, out_dir: str) -> None:
    """Plot AME(aha) with optional bootstrap confidence intervals."""
    required_columns = {"aha_ame", "aha_ame_lo", "aha_ame_hi"}
    if not required_columns.issubset(regression_df.columns):
        return
    regression_df = regression_df.dropna(subset=["aha_ame"])
    if regression_df.empty:
        return
    figure, axis = plt.subplots(figsize=(7.8, 4.6), dpi=140)
    axis.plot(
        regression_df["step"],
        regression_df["aha_ame"],
        marker="o",
        label="AME(aha)",
    )
    if regression_df[["aha_ame_lo", "aha_ame_hi"]].notna().all().all():
        axis.fill_between(
            regression_df["step"],
            regression_df["aha_ame_lo"],
            regression_df["aha_ame_hi"],
            alpha=0.2,
            label="95% CI",
        )
    style_ame_axis(axis)
    figure.tight_layout()
    figure.savefig(os.path.join(out_dir, "aha_ame_with_ci.png"))
    plt.close(figure)


__all__ = [
    "lineplot",
    "plot_diag_panel",
    "plot_ame_with_ci",
    "make_all3_uncertainty_buckets_figure",
    "plot_uncertainty_hist_100bins",
]
