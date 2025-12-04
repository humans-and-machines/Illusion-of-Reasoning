#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plotting utilities shared by the forced Aha! effect CLI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.forced_aha_shared import summarize_cluster_any, summarize_cluster_mean, summarize_sample_level
from src.analysis.plotting_styles import DEFAULT_COLORS, METRIC_LABELS, cmap_colors, darken_colors


BAR_SEED_OFFSETS = {"sample": 0, "cluster_any": 1, "cluster_mean": 2}
BAR_METRICS = ("sample", "cluster_mean", "cluster_any")


def savefig(figure: plt.Figure, out_base: str) -> None:
    """
    Save a Matplotlib figure to ``out_base`` as both PNG and PDF, creating
    parent directories as needed.
    """
    output_dir = os.path.dirname(out_base)
    os.makedirs(output_dir, exist_ok=True)
    figure.savefig(out_base + ".png", bbox_inches="tight", dpi=300)
    figure.savefig(out_base + ".pdf", bbox_inches="tight")
    plt.close(figure)


@dataclass
class SeriesPlotConfig:
    """Shared configuration for uncertainty/stepwise plots."""

    out_dir: str
    n_boot: int
    seed: int
    palette_name: str = "Pastel1"
    darken: float = 0.80


@dataclass(frozen=True)
class BucketSeries:
    """Bootstrap summary for entropy buckets."""

    delta: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


@dataclass(frozen=True)
class StepwiseSeries:
    """Series data for plotting stepwise deltas."""

    steps: np.ndarray
    delta: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


@dataclass(frozen=True)
class StepwiseColors:
    """Color palette for sample/mean/any series."""

    sample: str
    mean: str
    any_correct: str


@dataclass(frozen=True)
class OverallDeltaInputs:
    """Aggregated summary rows and source DataFrames for overview bars."""

    summary_rows: List[Dict[str, Any]]
    pairs_df: pd.DataFrame
    clusters_df: pd.DataFrame


@dataclass(frozen=True)
class OverallDeltaConfig:
    """Reusable bootstrap/palette configuration for delta overview plots."""

    n_boot: int
    seed: int
    palette: Dict[str, str]


@dataclass(frozen=True)
class OverallDeltaBar:
    """Prepared bar metadata with bootstrap errors and optional p-values."""

    label: str
    height: float
    error_lower: float
    error_upper: float
    color: str
    p_value: Optional[float]


@dataclass(frozen=True)
class ConversionSegment:
    """Waterfall segment describing a transition bucket."""

    label: str
    count: int
    color: str


def _confidence_interval_for_metric(
    inputs: OverallDeltaInputs,
    config: OverallDeltaConfig,
    metric_name: str,
) -> Tuple[float, float]:
    """Return bootstrap CIs for the requested metric."""
    frame = inputs.pairs_df if metric_name == "sample" else inputs.clusters_df
    seed_offset = BAR_SEED_OFFSETS[metric_name]
    return _bootstrap_delta(
        frame,
        metric_name,
        config.n_boot,
        config.seed + seed_offset,
    )


def _metric_p_value(metric_row: Dict[str, Any], metric_name: str) -> Optional[float]:
    """Return the correct p-value column for the given metric."""
    if metric_name in ("sample", "cluster_any"):
        return metric_row.get("p_mcnemar")
    return metric_row.get("p_ttest")


def _build_overall_delta_bars(
    inputs: OverallDeltaInputs,
    config: OverallDeltaConfig,
) -> List[OverallDeltaBar]:
    """Construct bar specifications for the overall delta plot."""
    by_metric = {row["metric"]: row for row in inputs.summary_rows}
    color_cycle = [
        config.palette["bar_primary"],
        config.palette["bar_secondary"],
        config.palette["bar_tertiary"],
    ]
    bars: List[OverallDeltaBar] = []
    for idx, metric_name in enumerate(BAR_METRICS):
        metric_row = by_metric.get(metric_name)
        if metric_row is None:
            continue
        ci_low, ci_high = _confidence_interval_for_metric(inputs, config, metric_name)
        delta_pp = float(metric_row["delta_pp"])
        bars.append(
            OverallDeltaBar(
                label=METRIC_LABELS.get(metric_name, metric_name),
                height=delta_pp,
                error_lower=delta_pp - ci_low,
                error_upper=ci_high - delta_pp,
                color=color_cycle[idx % len(color_cycle)],
                p_value=_metric_p_value(metric_row, metric_name),
            ),
        )
    return bars


def _draw_overall_delta_bars(out_dir: str, bars: List[OverallDeltaBar]) -> None:
    """Render and save the overall delta bar chart."""
    figure = plt.figure(figsize=(7.0, 4.2))
    axis = figure.add_axes([0.12, 0.16, 0.83, 0.74])
    positions = np.arange(len(bars))
    error_array = np.array(
        [
            [spec.error_lower for spec in bars],
            [spec.error_upper for spec in bars],
        ]
    )
    axis.bar(
        positions,
        [spec.height for spec in bars],
        color=[spec.color for spec in bars],
        yerr=error_array,
        capsize=4,
    )
    axis.axhline(0, linewidth=0.8, color="#444")
    axis.set_xticks(positions, [spec.label for spec in bars])
    axis.set_ylabel("Δ accuracy (pp)")
    axis.set_title("Forced Aha — Overall Δ (95% bootstrap CI)")
    for pos, spec in enumerate(bars):
        p_value = spec.p_value
        if p_value is None or not np.isfinite(p_value):
            continue
        label_text = "p≈0" if (p_value == 0 or p_value < 1e-300) else f"p={p_value:.1e}"
        axis.text(
            positions[pos],
            spec.height,
            label_text,
            ha="center",
            va="bottom",
            fontsize=10,
        )
    savefig(figure, os.path.join(out_dir, "figures", "overall_deltas"))


def plot_overall_deltas(
    out_dir: str,
    inputs: OverallDeltaInputs,
    config: OverallDeltaConfig,
) -> None:
    """
    Plot per-metric second-pass delta accuracy with bootstrap confidence bars.
    """
    bars = _build_overall_delta_bars(inputs, config)
    if not bars:
        return
    _draw_overall_delta_bars(out_dir, bars)


def plot_conversion_waterfall(
    out_dir: str,
    clusters_df: pd.DataFrame,
    palette: Dict[str, str],
) -> None:
    """
    Plot a waterfall-style bar chart of any-correct conversions.
    """
    segments = [
        ConversionSegment(
            label="Incorrect → Correct",
            count=int(((clusters_df["any_p1"] == 0) & (clusters_df["any_p2"] == 1)).sum()),
            color=palette["gain"],
        ),
        ConversionSegment(
            label="Correct → Incorrect",
            count=int(((clusters_df["any_p1"] == 1) & (clusters_df["any_p2"] == 0)).sum()),
            color=palette["loss"],
        ),
        ConversionSegment(
            label="Stayed Correct",
            count=int(((clusters_df["any_p1"] == 1) & (clusters_df["any_p2"] == 1)).sum()),
            color=palette["stable_pos"],
        ),
        ConversionSegment(
            label="Stayed Incorrect",
            count=int(((clusters_df["any_p1"] == 0) & (clusters_df["any_p2"] == 0)).sum()),
            color=palette["stable_neg"],
        ),
    ]
    total = max(1, sum(segment.count for segment in segments))

    figure = plt.figure(figsize=(7.2, 4.2))
    axis = figure.add_axes([0.28, 0.16, 0.68, 0.78])
    y_positions = np.arange(len(segments))
    axis.barh(
        y_positions,
        [segment.count for segment in segments],
        color=[segment.color for segment in segments],
    )
    axis.set_yticks(y_positions, [segment.label for segment in segments])
    axis.set_xlabel("Count (problems × steps)")
    axis.set_title("Any-correct conversions (overall)")
    for index, segment in enumerate(segments):
        percent = 100.0 * segment.count / total
        axis.text(
            segment.count,
            index,
            f" {segment.count}  ({percent:.1f}%)",
            va="center",
            ha="left",
        )
    axis.xaxis.grid(True, linestyle=":", alpha=0.4)
    savefig(figure, os.path.join(out_dir, "figures", "any_conversion_waterfall"))


def plot_headroom_scatter(out_dir: str, step_df: pd.DataFrame) -> None:
    """
    Plot Δ(any-correct) vs baseline any-correct per step (headroom plot).
    """
    subset = step_df[step_df["metric"] == "cluster_any"].copy().sort_values("step")
    x_values = subset["acc_pass1"].to_numpy(float)
    y_values = subset["delta_pp"].to_numpy(float)
    figure = plt.figure(figsize=(6.0, 4.5))
    axis = figure.add_axes([0.14, 0.14, 0.82, 0.82])
    axis.scatter(x_values, y_values, s=30)
    if len(x_values) >= 2 and np.isfinite(x_values).all() and np.isfinite(y_values).all():
        slope, intercept = np.polyfit(x_values, y_values, 1)
        x_line = np.linspace(min(x_values), max(x_values), 100)
        axis.plot(x_line, slope * x_line + intercept, ls="--")
    axis.axhline(0, lw=0.8)
    axis.set_xlabel("Baseline any-correct (pass-1)")
    axis.set_ylabel("Δ any-correct (pp)")
    axis.set_title("Headroom plot: Δ(any) vs baseline any (per step)")
    savefig(figure, os.path.join(out_dir, "figures", "any_headroom_scatter"))


def _select_series_colors(config: SeriesPlotConfig) -> StepwiseColors:
    """Return consistent colors for series plots."""
    palette_colors = darken_colors(
        cmap_colors(config.palette_name),
        config.darken,
    )
    return StepwiseColors(
        sample=palette_colors[1],
        mean=palette_colors[2],
        any_correct=palette_colors[4],
    )


def _aggregate_sample_delta(group: pd.DataFrame) -> float:
    """Return delta accuracy (pp) for sample-level rows."""
    return (group["correct2"].mean() - group["correct1"].mean()) * 100.0


def _aggregate_mean_delta(group: pd.DataFrame) -> float:
    """Return delta accuracy (pp) for per-problem mean-of-8 rows."""
    return (group["acc_p2"].mean() - group["acc_p1"].mean()) * 100.0


def _bootstrap_bucket_groups(
    groups: List[pd.DataFrame],
    aggregate_fn,
    n_boot: int,
    seed: int,
) -> BucketSeries:
    """Bootstrap delta distributions for each entropy bucket."""
    rng = np.random.default_rng(seed)
    delta_values: List[float] = []
    lower_bounds: List[float] = []
    upper_bounds: List[float] = []
    for group in groups:
        if group.empty:
            delta_values.append(np.nan)
            lower_bounds.append(np.nan)
            upper_bounds.append(np.nan)
            continue
        indices = np.arange(len(group))
        bootstrap_stats: List[float] = []
        for _ in range(n_boot):
            sampled = group.iloc[rng.choice(indices, size=len(indices), replace=True)]
            bootstrap_stats.append(aggregate_fn(sampled))
        lower_percentile, upper_percentile = np.percentile(
            bootstrap_stats,
            [2.5, 97.5],
        )
        delta_values.append(aggregate_fn(group))
        lower_bounds.append(float(lower_percentile))
        upper_bounds.append(float(upper_percentile))
    return BucketSeries(
        delta=np.asarray(delta_values, dtype=float),
        lower=np.asarray(lower_bounds, dtype=float),
        upper=np.asarray(upper_bounds, dtype=float),
    )


def _compute_uncertainty_bucket_data(
    pairs_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> Optional[Dict[str, Any]]:
    """
    Compute per-quintile delta-accuracy values and bootstrap CIs for entropy buckets.
    """
    df_s = pairs_df.dropna(subset=["entropy_p1"]).copy()
    df_c = clusters_df.dropna(subset=["entropy_p1_cluster"]).copy()
    if df_s.empty or df_c.empty:
        return None
    df_s["bucket"] = pd.qcut(df_s["entropy_p1"], q=5, labels=False)
    df_c["bucket"] = pd.qcut(df_c["entropy_p1_cluster"], q=5, labels=False)

    sample_groups = [df_s[df_s["bucket"] == bucket_index] for bucket_index in range(5)]
    cluster_groups = [df_c[df_c["bucket"] == bucket_index] for bucket_index in range(5)]
    return {
        "x": np.arange(5),
        "sample": _bootstrap_bucket_groups(
            sample_groups,
            _aggregate_sample_delta,
            n_boot,
            seed + 20,
        ),
        "cluster": _bootstrap_bucket_groups(
            cluster_groups,
            _aggregate_mean_delta,
            n_boot,
            seed + 25,
        ),
    }


def plot_uncertainty_buckets(
    pairs_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    config: SeriesPlotConfig,
) -> None:
    """
    Plot delta-accuracy vs entropy quintiles with bootstrap confidence intervals.
    """
    data = _compute_uncertainty_bucket_data(
        pairs_df,
        clusters_df,
        config.n_boot,
        config.seed,
    )
    if data is None:
        print("[warn] No entropy_p1 available; skipping uncertainty_buckets.")
        return

    figure = plt.figure(figsize=(7.2, 4.6))
    axis = figure.add_axes([0.10, 0.15, 0.86, 0.75])

    _plot_uncertainty_bucket_panel(
        axis=axis,
        bucket_data=data,
        colors=_select_series_colors(config),
    )
    savefig(
        figure,
        os.path.join(config.out_dir, "figures", "uncertainty_buckets"),
    )


def _step_groups_for_boot(
    pairs_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
) -> Tuple[
    List[int],
    Dict[int, pd.DataFrame],
    Dict[int, pd.DataFrame],
    Dict[int, pd.DataFrame],
]:
    """
    Build step-indexed DataFrame groups for bootstrapping.
    """
    steps = sorted(
        set(pairs_df["step"].unique()).union(set(clusters_df["step"].unique())),
    )
    sample_step = {step: pairs_df[pairs_df["step"] == step] for step in steps}
    any_step = {step: clusters_df[clusters_df["step"] == step] for step in steps}
    mean_step = any_step
    return steps, sample_step, any_step, mean_step


def _compute_stepwise_ci_by_metric(
    pairs_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    config: SeriesPlotConfig,
) -> Dict[str, Dict[int, Tuple[float, float]]]:
    """Return bootstrap confidence intervals for each metric/step."""
    _, sample_step, any_step, mean_step = _step_groups_for_boot(pairs_df, clusters_df)
    return {
        "sample": _bootstrap_delta_by_step(
            sample_step,
            "sample",
            config.n_boot,
            config.seed,
        ),
        "cluster_any": _bootstrap_delta_by_step(
            any_step,
            "cluster_any",
            config.n_boot,
            config.seed + 5,
        ),
        "cluster_mean": _bootstrap_delta_by_step(
            mean_step,
            "cluster_mean",
            config.n_boot,
            config.seed + 10,
        ),
    }


def _build_stepwise_series(
    step_df: pd.DataFrame,
    ci_by_metric: Dict[str, Dict[int, Tuple[float, float]]],
) -> Dict[str, StepwiseSeries]:
    """
    Build step-wise (x, y, lower, upper) series for each metric.
    """

    def _series_for(
        metric: str,
    ) -> StepwiseSeries:
        subset = step_df[step_df["metric"] == metric].sort_values("step")
        x_values = subset["step"].to_numpy(int)
        y_values = subset["delta_pp"].to_numpy(float)
        intervals = [ci_by_metric[metric].get(int(step), (np.nan, np.nan)) for step in x_values]
        if intervals:
            lower_array = np.array([pair[0] for pair in intervals], dtype=float)
            upper_array = np.array([pair[1] for pair in intervals], dtype=float)
        else:
            lower_array = np.array([], dtype=float)
            upper_array = np.array([], dtype=float)
        return StepwiseSeries(
            steps=x_values,
            delta=y_values,
            lower=lower_array,
            upper=upper_array,
        )

    return {
        "sample": _series_for(metric="sample"),
        "cluster_mean": _series_for(metric="cluster_mean"),
        "cluster_any": _series_for(metric="cluster_any"),
    }


def _plot_stepwise_series(
    axis: plt.Axes,
    series_by_metric: Dict[str, StepwiseSeries],
    colors: StepwiseColors,
) -> None:
    """Plot sample/mean/any series with fills."""
    style_specs = [
        ("sample", "Per-draw (single completion)", colors.sample, "o", 0.22),
        ("cluster_mean", "Per-problem mean of 8 (avg of 8)", colors.mean, "s", 0.22),
        ("cluster_any", "Per-problem any-correct", colors.any_correct, "^", 0.18),
    ]
    for metric, label, color, marker, alpha in style_specs:
        series = series_by_metric[metric]
        axis.plot(
            series.steps,
            series.delta,
            marker=marker,
            color=color,
            label=label,
        )
        axis.fill_between(
            series.steps,
            series.lower,
            series.upper,
            color=color,
            alpha=alpha,
            linewidth=0,
        )


def _plot_uncertainty_bucket_panel(
    axis: plt.Axes,
    bucket_data: Dict[str, Any],
    colors: StepwiseColors,
) -> None:
    """Plot entropy bucket deltas with confidence intervals."""
    sample_series: BucketSeries = bucket_data["sample"]
    cluster_series: BucketSeries = bucket_data["cluster"]
    axis.errorbar(
        bucket_data["x"] - 0.05,
        sample_series.delta,
        yerr=[
            sample_series.delta - sample_series.lower,
            sample_series.upper - sample_series.delta,
        ],
        fmt="o-",
        capsize=4,
        label="Per-draw (single completion)",
        color=colors.sample,
    )
    axis.errorbar(
        bucket_data["x"] + 0.05,
        cluster_series.delta,
        yerr=[
            cluster_series.delta - cluster_series.lower,
            cluster_series.upper - cluster_series.delta,
        ],
        fmt="s-",
        capsize=4,
        label="Per-problem mean of 8 (avg of 8 completions)",
        color=colors.mean,
    )
    axis.axhline(0, linewidth=0.8)
    axis.set_xticks(bucket_data["x"])
    axis.set_xticklabels([f"Q{bucket_index + 1}" for bucket_index in range(5)])
    axis.set_xlabel("Baseline pass-1 answer entropy (quintiles)")
    axis.set_ylabel("Delta Accuracy (pp) for second pass")
    axis.set_title("Uncertainty buckets: Forced Aha gains by entropy")
    axis.legend()


def plot_stepwise_overlay(
    step_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    config: SeriesPlotConfig,
) -> None:
    """
    Plot step-wise delta accuracy for sample/cluster metrics with bootstrap CIs.
    """
    ci_by_metric = _compute_stepwise_ci_by_metric(pairs_df, clusters_df, config)
    series_by_metric = _build_stepwise_series(step_df, ci_by_metric)

    figure = plt.figure(figsize=(8.6, 4.3))
    axis = figure.add_axes([0.10, 0.15, 0.86, 0.75])

    _plot_stepwise_series(
        axis,
        series_by_metric,
        colors=_select_series_colors(config),
    )

    axis.axhline(0, lw=0.8)
    axis.set_xlabel("Training step")
    axis.set_ylabel("Delta Accuracy (pp) for second pass")
    axis.set_title(
        "Stepwise Δ — overlay of per-draw / mean-of-8 / any-correct (95% bootstrap CIs)",
    )
    axis.legend(ncol=1)
    savefig(
        figure,
        os.path.join(config.out_dir, "figures", "stepwise_overlay"),
    )


def plot_uncertainty_and_stepwise(
    step_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    config: SeriesPlotConfig,
) -> None:
    """
    Plot combined figure with entropy buckets (left) and stepwise overlay (right).
    """
    data = _compute_uncertainty_bucket_data(
        pairs_df,
        clusters_df,
        config.n_boot,
        config.seed,
    )
    if data is None:
        print(
            "[warn] No entropy available; skipping combined uncertainty_and_stepwise.",
        )
        return

    ci_by_metric = _compute_stepwise_ci_by_metric(pairs_df, clusters_df, config)
    series_by_metric = _build_stepwise_series(step_df, ci_by_metric)
    colors = _select_series_colors(config)

    figure = plt.figure(figsize=(12.4, 4.8))
    grid_spec = figure.add_gridspec(
        1,
        2,
        left=0.08,
        right=0.98,
        top=0.92,
        bottom=0.18,
        wspace=0.26,
    )
    axis_left = figure.add_subplot(grid_spec[0, 0])
    axis_right = figure.add_subplot(grid_spec[0, 1])

    _plot_uncertainty_bucket_panel(axis_left, data, colors)
    _plot_stepwise_series(axis_right, series_by_metric, colors)
    axis_right.axhline(0, lw=0.8)
    axis_right.set_xlabel("Training step")
    axis_right.set_ylabel("Delta Accuracy (pp) for second pass")
    axis_right.set_title("Stepwise Δ — overlay (95% bootstrap CIs)")
    axis_right.legend()

    savefig(
        figure,
        os.path.join(config.out_dir, "figures", "uncertainty_and_stepwise"),
    )


def _bar_specs_with_stats(
    pairs_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Return ordered (metric_key, label, stats) tuples for overview bars."""
    specs = [
        ("cluster_mean", "Per-problem\nmean of 8", summarize_cluster_mean(clusters_df)),
        ("cluster_any", "Per-problem\nany-correct", summarize_cluster_any(clusters_df)),
    ]
    if not pairs_df.empty:
        specs.insert(
            0,
            ("sample", "Per-draw\naccuracy", summarize_sample_level(pairs_df)),
        )
    return specs


def _build_overview_bar_data(
    pairs_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    style: SeriesPlotConfig,
) -> Dict[str, Any]:
    """
    Build labels, heights, confidence intervals, and colors for overview bars.
    """
    bar_specs = _bar_specs_with_stats(pairs_df, clusters_df)
    labels = [label for _, label, _ in bar_specs]
    heights = [stat["delta_pp"] for _, _, stat in bar_specs]

    def _ci_offsets(metric_key: str, delta_point: float) -> Tuple[float, float]:
        data_source = pairs_df if metric_key == "sample" else clusters_df
        metric_name = "sample" if metric_key == "sample" else metric_key
        seed_offset = BAR_SEED_OFFSETS[metric_key]
        lower_bound, upper_bound = _bootstrap_delta(
            data_source,
            metric_name,
            style.n_boot,
            style.seed + seed_offset,
        )
        return delta_point - lower_bound, upper_bound - delta_point

    ci_pairs = [_ci_offsets(metric_key, delta_value) for (metric_key, _, _), delta_value in zip(bar_specs, heights)]
    if ci_pairs:
        ci_lowers, ci_uppers = map(list, zip(*ci_pairs))
    else:
        ci_lowers, ci_uppers = [], []

    palette_colors = darken_colors(
        cmap_colors(style.palette_name),
        style.darken,
    )
    bar_colors = [palette_colors[i % len(palette_colors)] for i in (1, 2, 4)][: len(labels)]

    return {
        "labels": labels,
        "heights": heights,
        "ci_lowers": ci_lowers,
        "ci_uppers": ci_uppers,
        "colors": bar_colors,
    }


def _compute_waterfall_data(
    clusters_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compute counts, labels, colors, and percentages for the waterfall plot.
    """
    num_incorrect_incorrect = int(
        ((clusters_df["any_p1"] == 0) & (clusters_df["any_p2"] == 0)).sum(),
    )
    num_incorrect_correct = int(
        ((clusters_df["any_p1"] == 0) & (clusters_df["any_p2"] == 1)).sum(),
    )
    num_correct_incorrect = int(
        ((clusters_df["any_p1"] == 1) & (clusters_df["any_p2"] == 0)).sum(),
    )
    num_correct_correct = int(
        ((clusters_df["any_p1"] == 1) & (clusters_df["any_p2"] == 1)).sum(),
    )
    total = max(
        num_incorrect_incorrect + num_incorrect_correct + num_correct_incorrect + num_correct_correct,
        1,
    )

    waterfall_values = [
        num_incorrect_correct,
        num_correct_incorrect,
        num_correct_correct,
        num_incorrect_incorrect,
    ]
    waterfall_labels = [
        "Incorrect →\nCorrect",
        "Correct →\nIncorrect",
        "Stayed\nCorrect",
        "Stayed\nIncorrect",
    ]
    waterfall_colors = [
        DEFAULT_COLORS["gain"],
        DEFAULT_COLORS["loss"],
        DEFAULT_COLORS["stable_pos"],
        DEFAULT_COLORS["stable_neg"],
    ]
    waterfall_percentages = [100.0 * value / total for value in waterfall_values]

    return {
        "values": waterfall_values,
        "labels": waterfall_labels,
        "colors": waterfall_colors,
        "percentages": waterfall_percentages,
        "total": total,
    }


def _create_overview_axes() -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
    """
    Create figure and axes for the overview side-by-side layout.
    """
    figure = plt.figure(figsize=(12.2, 4.2))
    grid_spec = figure.add_gridspec(
        nrows=1,
        ncols=2,
        left=0.07,
        right=0.99,
        top=0.92,
        bottom=0.24,
        wspace=0.28,
    )
    axis_left = figure.add_subplot(grid_spec[0, 0])
    axis_right = figure.add_subplot(grid_spec[0, 1])
    return figure, axis_left, axis_right


def plot_overview_side_by_side(config: Dict[str, Any]) -> None:
    """
    Side-by-side overview with the SAME inputs as plot_uncertainty_and_stepwise:
      (left) Overall Δ bars (bootstrap CIs), no p-values shown.
      (right) Any-correct conversions waterfall (Incorrect→Correct first).
    Left bars use a darkened qualitative palette; right uses semantic colors.
    """
    style = SeriesPlotConfig(
        out_dir=str(config["out_dir"]),
        n_boot=int(config["n_boot"]),
        seed=int(config["seed"]),
        palette_name=str(config.get("palette_name", "Pastel1")),
        darken=float(config.get("darken", 0.80)),
    )
    bar_data = _build_overview_bar_data(
        pairs_df=config["pairs_df"],
        clusters_df=config["clusters_df"],
        style=style,
    )
    waterfall_data = _compute_waterfall_data(config["clusters_df"])

    figure, axis_left, axis_right = _create_overview_axes()

    x_positions = np.arange(len(bar_data["labels"]))
    axis_left.bar(
        x_positions,
        bar_data["heights"],
        color=bar_data["colors"],
        yerr=[bar_data["ci_lowers"], bar_data["ci_uppers"]],
        capsize=4,
    )
    axis_left.axhline(0, linewidth=0.8, color="#444444")
    axis_left.set_xticks(x_positions)
    axis_left.set_xticklabels(bar_data["labels"])
    axis_left.set_ylabel("Δ accuracy (pp)\nsecond pass (forced Aha)")
    axis_left.set_title("Forced Aha — Overall Δ (95% bootstrap CI)")

    y_positions = np.arange(len(waterfall_data["values"]))
    axis_right.barh(
        y_positions,
        waterfall_data["values"],
        color=waterfall_data["colors"],
    )
    axis_right.set_yticks(y_positions)
    axis_right.set_yticklabels(waterfall_data["labels"])
    axis_right.set_xlabel("Count\n(problems × steps)")
    axis_right.set_title("Any-correct conversions (overall)")
    axis_right.xaxis.grid(True, linestyle=":", alpha=0.35)

    for index, (value, percentage) in enumerate(
        zip(waterfall_data["values"], waterfall_data["percentages"]),
    ):
        axis_right.text(
            value + max(waterfall_data["total"] * 0.004, 8),
            index,
            f"{value:,}  ({percentage:.1f}%)",
            va="center",
            ha="left",
        )

    savefig(
        figure,
        os.path.join(
            str(config["out_dir"]),
            "figures",
            "overview_deltas_and_waterfall",
        ),
    )


# Helper functions that rely on bootstrap logic defined in the main CLI.
def _bootstrap_delta(
    pass_pairs: pd.DataFrame,
    metric: str,
    n_boot: int,
    seed: int,
) -> Tuple[float, float]:
    """Bootstrap delta accuracy for the requested metric."""
    rng = np.random.default_rng(seed)
    deltas: List[float] = []
    idx = np.arange(len(pass_pairs))
    if metric == "sample":
        for _ in range(n_boot):
            bootstrap_sample = pass_pairs.iloc[rng.choice(idx, size=len(idx), replace=True)]
            deltas.append(
                (bootstrap_sample["correct2"].mean() - bootstrap_sample["correct1"].mean()) * 100.0,
            )
    elif metric == "cluster_any":
        for _ in range(n_boot):
            bootstrap_sample = pass_pairs.iloc[rng.choice(idx, size=len(idx), replace=True)]
            deltas.append(
                (bootstrap_sample["any_p2"].mean() - bootstrap_sample["any_p1"].mean()) * 100.0,
            )
    elif metric == "cluster_mean":
        for _ in range(n_boot):
            bootstrap_sample = pass_pairs.iloc[rng.choice(idx, size=len(idx), replace=True)]
            deltas.append(
                (bootstrap_sample["acc_p2"].mean() - bootstrap_sample["acc_p1"].mean()) * 100.0,
            )
    if not deltas:
        return (np.nan, np.nan)
    low, high = np.percentile(deltas, [2.5, 97.5])
    return float(low), float(high)


def _bootstrap_delta_by_step(
    step_to_frame: Dict[int, pd.DataFrame],
    metric: str,
    n_boot: int,
    seed: int,
) -> Dict[int, Tuple[float, float]]:
    """Bootstrap delta accuracy per step for the requested metric."""
    conf_intervals: Dict[int, Tuple[float, float]] = {}
    for step_value, frame in step_to_frame.items():
        if len(frame) == 0:
            conf_intervals[step_value] = (np.nan, np.nan)
            continue
        low, high = _bootstrap_delta(
            frame,
            metric,
            n_boot,
            int(seed) + int(step_value),
        )
        conf_intervals[step_value] = (low, high)
    return conf_intervals
