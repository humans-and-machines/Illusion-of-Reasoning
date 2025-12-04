#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Density and histogram plotting helpers for the uncertainty/correctness figures.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.analysis.figure_2_data import _standardize_uncertainty, compute_correct_hist, density_from_hist
from src.analysis.figure_2_plotting_base import (
    FigureSaveConfig,
    Line2D,
    add_lower_center_legend,
    plt,
    save_figure_outputs,
)
from src.analysis.plotting import a4_size_inches


@dataclass
class DensityPlotConfig(FigureSaveConfig):
    """Configuration for uncertainty density plots."""

    smooth_bins: int


@dataclass
class FourHistConfig(FigureSaveConfig):
    """
    Configuration for the 4-panel count histogram figure.
    """

    edges: np.ndarray


def _ensure_aha_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``data`` with missing aha indicator columns filled in."""
    augmented = data.copy()
    for col in ("aha_words", "aha_gpt", "aha_formal"):
        if col not in augmented.columns:
            augmented[col] = 0
    return augmented


def _density_curves_for_correct(d_all: pd.DataFrame, edges: np.ndarray, smooth_bins: int) -> List[Dict[str, Any]]:
    """Compute density curves for correct answers by aha type."""

    d_std = _ensure_aha_columns(_standardize_uncertainty(d_all))
    base_mask = d_std["correct"] == 1
    if "aha_formal" in d_std.columns:
        formal_mask = d_std["aha_formal"] == 1
    else:
        formal_mask = np.zeros(len(d_std), dtype=bool)

    variants = [
        ("All (correct)", base_mask, "#666666"),
        ("Words (correct∧words)", base_mask & (d_std["aha_words"] == 1), "#1f77b4"),
        ("LLM (correct∧gpt)", base_mask & (d_std["aha_gpt"] == 1), "#ff7f0e"),
        ("Formal (correct∧formal)", base_mask & formal_mask, "#2ca02c"),
    ]

    curves: List[Dict[str, Any]] = []
    smooth_k = int(max(0, smooth_bins))
    for label, mask, color in variants:
        vals = d_std.loc[mask, "uncertainty_std"].to_numpy()
        x_vals, y_vals = density_from_hist(vals, edges, smooth_k=smooth_k)
        curves.append(
            {
                "label": label,
                "x": x_vals,
                "y": y_vals,
                "color": color,
                "n": int(vals.size),
            }
        )
    return curves


def plot_four_correct_hists(
    all_samples: pd.DataFrame,
    config: FourHistConfig,
) -> None:
    """
    Plot four count histograms of correct answers across uncertainty bins.
    """
    centers = 0.5 * (config.edges[:-1] + config.edges[1:])
    all_samples = _ensure_aha_columns(_standardize_uncertainty(all_samples))

    panels = [
        ("All samples", all_samples),
        ('Words of "Aha!"', all_samples[all_samples["aha_words"] == 1]),
        ('LLM-Detected "Aha!"', all_samples[all_samples["aha_gpt"] == 1]),
        (
            'Formal "Aha!"',
            all_samples[all_samples["aha_formal"] == 1]
            if "aha_formal" in all_samples.columns
            else all_samples.iloc[0:0],
        ),
    ]

    fig_size = a4_size_inches(config.a4_orientation) if config.a4_pdf else (16.5, 4.8)
    fig, axes = plt.subplots(1, 4, figsize=fig_size, dpi=150, sharey=True)
    width = (config.edges[1] - config.edges[0]) * 0.95

    for axis, (title, panel_df) in zip(axes, panels):
        if panel_df.empty:
            counts = np.zeros(len(config.edges) - 1, dtype=int)
        else:
            counts = compute_correct_hist(
                panel_df["uncertainty_std"].to_numpy(),
                panel_df["correct"].to_numpy(),
                config.edges,
            )
        axis.bar(centers, counts, width=width, color="#7f9cd8", edgecolor="none")
        axis.set_title(f"{title}\n{config.title_suffix}")
        axis.set_xlabel("uncertainty_std")
        axis.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Count of CORRECT")

    try:
        fig.tight_layout(rect=[0, 0.02, 1, 1])
    except (TypeError, AttributeError):
        # Minimal matplotlib stubs may not accept rect kwarg; swallow the error.
        pass
    save_figure_outputs(fig, config, dpi=150, tight_layout=False)


def plot_overlaid_densities(d_all: pd.DataFrame, edges: np.ndarray, plot_cfg: DensityPlotConfig) -> str:
    """Plot overlaid densities of standardized uncertainty for correct answers."""

    curves = _density_curves_for_correct(d_all, edges, plot_cfg.smooth_bins)
    fig_size = a4_size_inches(plot_cfg.a4_orientation) if plot_cfg.a4_pdf else (12.0, 4.8)
    fig, axis = plt.subplots(figsize=fig_size, dpi=150)

    for curve in curves:
        label = curve["label"]
        axis.plot(
            curve["x"],
            curve["y"],
            lw=2.2,
            label=f"{label}  (n={curve['n']})",
            color=curve["color"],
        )

    axis.set_title(f"Area-normalized density of uncertainty_std (CORRECT only)\n{plot_cfg.title_suffix}")
    axis.set_xlabel("uncertainty_std")
    axis.set_ylabel("Density")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best", frameon=False)

    save_figure_outputs(fig, plot_cfg, dpi=150)

    centers = 0.5 * (edges[:-1] + edges[1:])
    out_csv = plot_cfg.out_png.replace(".png", ".csv")
    df_out = pd.DataFrame({"bin_center": centers})
    for curve in curves:
        key = re.sub(r"[^A-Za-z0-9_]+", "_", curve["label"].lower())
        df_out[key] = curve["y"]
    df_out.to_csv(out_csv, index=False)
    return out_csv


def _correct_incorrect_density_for_mask(
    d_all: pd.DataFrame,
    edges: np.ndarray,
    mask,
    smooth_bins: int,
) -> Dict[str, Any]:
    """Return densities for correct vs incorrect for a single panel mask."""

    subset = d_all if mask is None else d_all[mask]
    if subset.empty:
        centers = 0.5 * (edges[:-1] + edges[1:])
        zeros = np.zeros_like(centers)
        return {
            "x": centers,
            "y_correct": zeros,
            "y_incorrect": zeros,
            "n_correct": 0,
            "n_incorrect": 0,
        }

    standardized = _standardize_uncertainty(subset)
    vals_corr = standardized.loc[standardized["correct"] == 1, "uncertainty_std"].to_numpy()
    vals_inc = standardized.loc[standardized["correct"] == 0, "uncertainty_std"].to_numpy()
    smooth_k = int(max(0, smooth_bins))
    x_vals, y_corr = density_from_hist(vals_corr, edges, smooth_k=smooth_k)
    y_inc = density_from_hist(vals_inc, edges, smooth_k=smooth_k)[1]
    return {
        "x": x_vals,
        "y_correct": y_corr,
        "y_incorrect": y_inc,
        "n_correct": int(vals_corr.size),
        "n_incorrect": int(vals_inc.size),
    }


def _rows_for_density(panel_label: str, density: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build CSV rows from a panel's density dictionary."""

    rows: List[Dict[str, Any]] = []
    for x_center, yi_corr, yi_inc in zip(density["x"], density["y_correct"], density["y_incorrect"]):
        rows.append(
            {
                "panel": panel_label,
                "bin_center": x_center,
                "density_correct": yi_corr,
                "density_incorrect": yi_inc,
            }
        )
    return rows


def plot_correct_incorrect_by_type(d_all: pd.DataFrame, edges: np.ndarray, plot_cfg: DensityPlotConfig) -> str:
    """Plot 2×2 densities of standardized uncertainty for correct vs incorrect answers."""

    d_all = _ensure_aha_columns(d_all)
    panels = [
        ("All samples", None),
        ('Words of "Aha!"', d_all["aha_words"] == 1),
        ('LLM-Detected "Aha!"', d_all["aha_gpt"] == 1),
        (
            'Formal "Aha!"',
            (d_all["aha_formal"] == 1) if "aha_formal" in d_all.columns else np.zeros(len(d_all), dtype=bool),
        ),
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(a4_size_inches(plot_cfg.a4_orientation) if plot_cfg.a4_pdf else (12.0, 7.6)),
        dpi=150,
        sharex=True,
        sharey=True,
    )
    axes = np.array(axes).reshape(2, 2)
    colors = {"correct": "#1f77b4", "incorrect": "#d62728"}
    rows: List[Dict[str, Any]] = []

    for axis, (label, mask) in zip(axes.flat, panels):
        density = _correct_incorrect_density_for_mask(d_all, edges, mask, plot_cfg.smooth_bins)
        axis.plot(
            density["x"],
            density["y_correct"],
            lw=2.2,
            color=colors["correct"],
            label=f"Correct (n={density['n_correct']})",
        )
        axis.plot(
            density["x"],
            density["y_incorrect"],
            lw=2.2,
            color=colors["incorrect"],
            label=f"Incorrect (n={density['n_incorrect']})",
            ls="--",
        )
        axis.set_title(f"{label}\n{plot_cfg.title_suffix}")
        axis.set_xlabel("uncertainty_std")
        axis.set_ylabel("Density")
        axis.grid(True, alpha=0.3)
        rows.extend(_rows_for_density(label, density))

    legend_handles = [
        Line2D([0], [0], color=colors["correct"], lw=2.2, label="Correct"),
        Line2D([0], [0], color=colors["incorrect"], lw=2.2, ls="--", label="Incorrect"),
    ]
    add_lower_center_legend(fig, legend_handles)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save_figure_outputs(fig, plot_cfg, dpi=150, tight_layout=False)

    out_csv = plot_cfg.out_png.replace(".png", ".csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv
