"""
Plotting utilities for the H3 uncertainty bucket analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.plotting import apply_paper_font_style


@dataclass(frozen=True)
class BarSeries:
    """Single series for a paired bar plot with confidence intervals."""

    values: Sequence[float]
    lower: Sequence[float]
    upper: Sequence[float]
    label: str


@dataclass(frozen=True)
class PairedBarPlot:
    """Configuration for drawing paired bar plots with CIs."""

    axis: plt.Axes
    labels: Sequence[str]
    series: Sequence[BarSeries]
    title: str
    ylabel: str


def _plot_bars_with_ci(config: PairedBarPlot) -> None:
    """Render paired bar charts with asymmetric confidence intervals."""

    def _nan_to_zero(array: np.ndarray) -> np.ndarray:
        return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

    width = 0.35
    positions = np.arange(len(config.labels))
    start_offset = -width * (len(config.series) - 1) / 2.0
    for offset, series in enumerate(config.series):
        values = _nan_to_zero(np.asarray(series.values, dtype=float))
        lower = _nan_to_zero(np.asarray(series.lower, dtype=float))
        upper = _nan_to_zero(np.asarray(series.upper, dtype=float))
        error = np.vstack([values - lower, upper - values])
        shift = start_offset + offset * width
        config.axis.bar(
            positions + shift,
            values,
            width,
            label=series.label,
            yerr=error,
        )
    config.axis.set_xticks(positions)
    config.axis.set_xticklabels(config.labels, rotation=45, ha="right")
    config.axis.set_ylabel(config.ylabel)
    config.axis.set_title(config.title)
    config.axis.set_ylim(0.0, 1.0)
    config.axis.legend()


def plot_question_overall_ci(
    q_overall: pd.DataFrame,
    out_png: str,
    also_pdf: bool,
) -> None:
    """Plot question-level any-correct share by group with 95% CIs."""
    if q_overall.empty:
        return
    apply_paper_font_style()
    matplotlib.rcParams["axes.facecolor"] = "white"
    fig = plt.figure(figsize=(6.0, 4.0))
    axis = fig.add_subplot(111)
    _plot_bars_with_ci(PairedBarPlot(
        axis=axis,
        labels=q_overall["group"].astype(str),
        series=(
            BarSeries(
                values=q_overall["any_pass1"],
                lower=q_overall["any_pass1_lo"],
                upper=q_overall["any_pass1_hi"],
                label="Pass1",
            ),
            BarSeries(
                values=q_overall["any_pass2"],
                lower=q_overall["any_pass2_lo"],
                upper=q_overall["any_pass2_hi"],
                label="Pass2",
            ),
        ),
        title="Question-level re-asking (overall)",
        ylabel="Any correct (share)",
    ))
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    if also_pdf:
        fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_question_by_step_ci(
    q_step: pd.DataFrame,
    out_png: str,
    also_pdf: bool,
) -> None:
    """Plot question-level any-correct share by training step."""
    if q_step.empty:
        return
    apply_paper_font_style()
    matplotlib.rcParams["axes.facecolor"] = "white"
    fig = plt.figure(figsize=(6.0, 4.0))
    axis = fig.add_subplot(111)
    for group, subset in q_step.groupby("group"):
        axis.errorbar(
            subset["step"],
            subset["any_pass1"],
            yerr=[
                subset["any_pass1"] - subset["any_pass1_lo"],
                subset["any_pass1_hi"] - subset["any_pass1"],
            ],
            label=f"{group} Pass1",
            marker="o",
        )
        axis.errorbar(
            subset["step"],
            subset["any_pass2"],
            yerr=[
                subset["any_pass2"] - subset["any_pass2_lo"],
                subset["any_pass2_hi"] - subset["any_pass2"],
            ],
            label=f"{group} Pass2",
            marker="s",
        )
    axis.set_xlabel("Step")
    axis.set_ylabel("Any correct")
    axis.set_title("Question-level re-asking by step")
    axis.set_ylim(0.0, 1.0)
    axis.legend()
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    if also_pdf:
        fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_question_by_bucket_ci(
    q_bucket: pd.DataFrame,
    out_png: str,
    also_pdf: bool,
) -> None:
    """Plot question-level any-correct share by perplexity bucket."""
    if q_bucket.empty:
        return
    apply_paper_font_style()
    matplotlib.rcParams["axes.facecolor"] = "white"
    fig = plt.figure(figsize=(6.0, 4.0))
    axis = fig.add_subplot(111)
    for group, subset in q_bucket.groupby("group"):
        axis.errorbar(
            subset["perplexity_bucket"],
            subset["any_pass1"],
            yerr=[
                subset["any_pass1"] - subset["any_pass1_lo"],
                subset["any_pass1_hi"] - subset["any_pass1"],
            ],
            label=f"{group} Pass1",
            marker="o",
        )
        axis.errorbar(
            subset["perplexity_bucket"],
            subset["any_pass2"],
            yerr=[
                subset["any_pass2"] - subset["any_pass2_lo"],
                subset["any_pass2_hi"] - subset["any_pass2"],
            ],
            label=f"{group} Pass2",
            marker="s",
        )
    axis.set_xlabel("Perplexity bucket")
    axis.set_ylabel("Any correct")
    axis.set_title("Question-level re-asking by bucket")
    axis.set_ylim(0.0, 1.0)
    axis.legend()
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    if also_pdf:
        fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_prompt_overall_ci(
    p_overall: pd.DataFrame,
    out_png: str,
    also_pdf: bool,
) -> None:
    """Plot prompt-level accuracy aggregates with confidence intervals."""
    if p_overall.empty:
        return
    apply_paper_font_style()
    matplotlib.rcParams["axes.facecolor"] = "white"
    fig = plt.figure(figsize=(6.0, 4.0))
    axis = fig.add_subplot(111)
    _plot_bars_with_ci(PairedBarPlot(
        axis=axis,
        labels=p_overall["group"].astype(str),
        series=(
            BarSeries(
                values=p_overall["acc_pass1"],
                lower=p_overall["acc_pass1_lo"],
                upper=p_overall["acc_pass1_hi"],
                label="Pass1",
            ),
            BarSeries(
                values=p_overall["acc_pass2"],
                lower=p_overall["acc_pass2_lo"],
                upper=p_overall["acc_pass2_hi"],
                label="Pass2",
            ),
        ),
        title="Prompt-level re-asking (overall)",
        ylabel="Accuracy",
    ))
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    if also_pdf:
        fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_prompt_by_step_ci(
    p_step: pd.DataFrame,
    out_png: str,
    also_pdf: bool,
) -> None:
    """Plot prompt-level accuracy by training step with error bars."""
    if p_step.empty:
        return
    apply_paper_font_style()
    matplotlib.rcParams["axes.facecolor"] = "white"
    fig = plt.figure(figsize=(6.0, 4.0))
    axis = fig.add_subplot(111)
    for group, subset in p_step.groupby("group"):
        axis.errorbar(
            subset["step"],
            subset["acc_pass1"],
            yerr=[
                subset["acc_pass1"] - subset["acc_pass1_lo"],
                subset["acc_pass1_hi"] - subset["acc_pass1"],
            ],
            label=f"{group} Pass1",
            marker="o",
        )
        axis.errorbar(
            subset["step"],
            subset["acc_pass2"],
            yerr=[
                subset["acc_pass2"] - subset["acc_pass2_lo"],
                subset["acc_pass2_hi"] - subset["acc_pass2"],
            ],
            label=f"{group} Pass2",
            marker="s",
        )
    axis.set_xlabel("Step")
    axis.set_ylabel("Accuracy")
    axis.set_title("Prompt-level re-asking by step")
    axis.set_ylim(0.0, 1.0)
    axis.legend()
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    if also_pdf:
        fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_prompt_by_bucket_ci(
    p_bucket: pd.DataFrame,
    out_png: str,
    also_pdf: bool,
) -> None:
    """Plot prompt-level accuracy by perplexity bucket."""
    if p_bucket.empty:
        return
    apply_paper_font_style()
    matplotlib.rcParams["axes.facecolor"] = "white"
    fig = plt.figure(figsize=(6.0, 4.0))
    axis = fig.add_subplot(111)
    for group, subset in p_bucket.groupby("group"):
        axis.errorbar(
            subset["perplexity_bucket"],
            subset["acc_pass1"],
            yerr=[
                subset["acc_pass1"] - subset["acc_pass1_lo"],
                subset["acc_pass1_hi"] - subset["acc_pass1"],
            ],
            label=f"{group} Pass1",
            marker="o",
        )
        axis.errorbar(
            subset["perplexity_bucket"],
            subset["acc_pass2"],
            yerr=[
                subset["acc_pass2"] - subset["acc_pass2_lo"],
                subset["acc_pass2_hi"] - subset["acc_pass2"],
            ],
            label=f"{group} Pass2",
            marker="s",
        )
    axis.set_xlabel("Perplexity bucket")
    axis.set_ylabel("Accuracy")
    axis.set_title("Prompt-level re-asking by bucket")
    axis.set_ylim(0.0, 1.0)
    axis.legend()
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    if also_pdf:
        fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


DELTA_LABELS = ["Δ=-1", "Δ=0", "Δ=+1"]
DELTA_ORDER = (-1, 0, 1)


def _delta_counts(delta_values: pd.Series) -> list[int]:
    """Return ordered histogram counts for Δ=-1, 0, +1."""
    numeric = delta_values.to_numpy(dtype=int, copy=False)
    unique_values, counts = np.unique(numeric, return_counts=True)
    mapping = {int(value): int(count) for value, count in zip(unique_values, counts)}
    return [mapping.get(value, 0) for value in DELTA_ORDER]


def _grouped_delta_counts(pairs_df: pd.DataFrame) -> list[tuple[Any, list[int]]]:
    """Return per-forced counts for each delta bucket."""
    groups: list[tuple[Any, list[int]]] = []
    for forced_value, subgroup in pairs_df.groupby("forced_insight"):
        groups.append((forced_value, _delta_counts(subgroup["delta"])))
    return groups


def _plot_forced_delta_bars(
    axis: plt.Axes,
    grouped_counts: Sequence[tuple[Any, Sequence[int]]],
) -> None:
    """Render grouped bars for per-forced delta counts."""
    if not grouped_counts:
        axis.set_xticks(np.arange(len(DELTA_LABELS)))
        axis.set_xticklabels(DELTA_LABELS)
        return
    base_positions = np.arange(len(DELTA_LABELS))
    width = 0.35
    for index, (forced_value, counts) in enumerate(grouped_counts):
        offset = base_positions + index * width
        axis.bar(offset, counts, width, label=f"forced={forced_value}")
    tick_offset = width * (len(grouped_counts) - 1) / 2.0
    axis.set_xticks(base_positions + tick_offset)
    axis.set_xticklabels(DELTA_LABELS)


def plot_prompt_level_deltas(
    pairs_df: pd.DataFrame,
    out_png: str,
    by_forced: bool = True,
    also_pdf: bool = False,
) -> None:
    """Plot histogram of prompt-level delta outcomes, optionally grouped by forced Aha."""
    if pairs_df.empty:
        return
    fig = plt.figure()
    axis = fig.add_subplot(111)
    show_grouped = by_forced and "forced_insight" in pairs_df.columns
    if show_grouped:
        grouped_counts = _grouped_delta_counts(pairs_df)
        _plot_forced_delta_bars(axis, grouped_counts)
        if grouped_counts:
            axis.legend()
    else:
        axis.bar(DELTA_LABELS, _delta_counts(pairs_df["delta"]))
    axis.set_ylabel("Count of prompts")
    axis.set_title("Prompt-level re-asking Δ (correct2 - correct1)")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    if also_pdf:
        fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
