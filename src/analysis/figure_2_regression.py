#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Regression helpers for Figure 2.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.analysis.figure_2_plotting_base import (
    FigureSaveConfig,
    Line2D,
    add_lower_center_legend,
    plt,
    save_figure_outputs,
)
from src.analysis.metrics import lazy_import_statsmodels, predict_formula
from src.analysis.plotting import a4_size_inches


def fit_glm_interaction(
    data_frame: pd.DataFrame,
    aha_column: str,
    bucket_column: str,
    ridge_alpha: float = 0.5,
):
    """
    Fit a logistic GLM with an Aha×bucket interaction, falling back to ridge.
    """
    statsmodels_mod, statsmodels_formula = lazy_import_statsmodels()
    model = statsmodels_formula.glm(
        f"correct ~ C(problem) + C(step) + {aha_column}:C({bucket_column})",
        data=data_frame,
        family=statsmodels_mod.families.Binomial(),
    )
    used_covariance = "none"
    try:
        res = model.fit(cov_type="HC1")
        if not np.isfinite(res.params).all():
            raise RuntimeError("Unstable MLE; switching to ridge.")
    except (ValueError, np.linalg.LinAlgError, RuntimeError):
        res = model.fit_regularized(alpha=float(ridge_alpha), L1_wt=0.0)
        used_covariance = "ridge"
    return res, model, used_covariance


def _bootstrap_bucket_means(
    *,
    context: "BucketRegressionContext",
    level_value: Any,
    aha_value: int,
    num_bootstrap_samples: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    base_frame = context.base_df.copy()
    base_frame[context.aha_column] = int(aha_value)
    base_frame[context.bucket_column] = pd.Categorical(
        [level_value] * len(base_frame),
        categories=context.bucket_levels,
        ordered=True,
    )

    mean_value = float(
        np.mean(
            predict_formula(
                context.res,
                context.model,
                base_frame,
            ),
        ),
    )

    if num_bootstrap_samples <= 0:
        return mean_value, np.nan, np.nan

    num_rows = len(base_frame)
    bootstrap_means = np.empty(num_bootstrap_samples, dtype=float)
    for sample_index in range(num_bootstrap_samples):
        sampled_indices = rng.choice(num_rows, size=num_rows, replace=True)
        bootstrap_means[sample_index] = float(
            np.mean(
                predict_formula(
                    context.res,
                    context.model,
                    base_frame.iloc[sampled_indices],
                ),
            ),
        )

    lower, upper = np.nanpercentile(bootstrap_means, [2.5, 97.5])
    return mean_value, float(lower), float(upper)


def predict_margins_by_bucket(
    context: "BucketRegressionContext",
    aha_value: int,
    num_bootstrap_samples: int = 500,
    rng_seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict bucket-level marginal accuracies and bootstrap confidence intervals.
    """
    rng = np.random.default_rng(rng_seed)
    means: List[float] = []
    lower_bounds: List[float] = []
    upper_bounds: List[float] = []

    for level_value in context.bucket_levels:
        mean_value, lower, upper = _bootstrap_bucket_means(
            context=context,
            level_value=level_value,
            aha_value=aha_value,
            num_bootstrap_samples=num_bootstrap_samples,
            rng=rng,
        )
        means.append(mean_value)
        lower_bounds.append(lower)
        upper_bounds.append(upper)

    return (
        np.asarray(means, dtype=float),
        np.asarray(lower_bounds, dtype=float),
        np.asarray(upper_bounds, dtype=float),
    )


@dataclass
class BucketRegressionContext:
    """Context for predicting bucket-wise margins from a fitted GLM."""

    res: Any
    model: Any
    base_df: pd.DataFrame
    aha_column: str
    bucket_column: str
    bucket_levels: List[Any]


@dataclass
class RegressionOutputConfig(FigureSaveConfig):
    """Output-specific configuration for regression figures."""


@dataclass
class RegressionPlotConfig:
    """Configuration for GLM-based regression curves."""

    frame: pd.DataFrame
    ppx_bucket_column: str
    dataset: str
    model_name: str
    num_bootstrap_samples: int
    output: RegressionOutputConfig


@dataclass
class VariantPlotContext:
    """State shared across regression variant plots."""

    prepared_frame: pd.DataFrame
    bucket_levels: List[Any]
    bucket_labels: List[str]
    config: RegressionPlotConfig
    rows_for_csv: List[Dict[str, Any]]


REGRESSION_VARIANTS: Tuple[Tuple[str, str], ...] = (
    ("aha_words", "Words"),
    ("aha_gpt", "LLM"),
    ("aha_formal", "Formal"),
)
REGRESSION_COLORS: Dict[int, str] = {0: "#999999", 1: "#2f5597"}


@dataclass
class RegressionPanelMeta:
    """Metadata needed to draw one regression panel."""

    bucket_levels: List[Any]
    bucket_labels: List[str]
    colors: Dict[int, str]
    title: str
    dataset: str
    model_name: str


def _prepare_bucket_frame(
    frame: pd.DataFrame,
    ppx_bucket_column: str,
) -> Tuple[pd.DataFrame, List[Any], List[str]]:
    """
    Normalize the perplexity bucket column and return frame + level metadata.
    """
    data = frame.copy()
    data[ppx_bucket_column] = pd.Categorical(data[ppx_bucket_column])
    bucket_levels = list(data[ppx_bucket_column].cat.categories)
    bucket_labels = [str(level) for level in bucket_levels]
    return data, bucket_levels, bucket_labels


def _build_regression_context(
    base_df: pd.DataFrame,
    aha_column: str,
    bucket_column: str,
    bucket_levels: List[Any],
) -> BucketRegressionContext:
    """
    Fit the GLM and wrap it in a :class:`BucketRegressionContext`.
    """
    res, model, _ = fit_glm_interaction(
        base_df,
        aha_column,
        bucket_column,
        ridge_alpha=0.5,
    )
    data = base_df.copy()
    data[aha_column] = data[aha_column].astype(int)
    return BucketRegressionContext(
        res=res,
        model=model,
        base_df=data,
        aha_column=aha_column,
        bucket_column=bucket_column,
        bucket_levels=bucket_levels,
    )


def _predict_curves_for_context(
    context: BucketRegressionContext,
    num_bootstrap_samples: int,
) -> Dict[str, np.ndarray]:
    """
    Compute predicted curves and CIs for aha=0 and aha=1 for all buckets.
    """
    pred_noaha, low_noaha, high_noaha = predict_margins_by_bucket(
        context,
        aha_value=0,
        num_bootstrap_samples=num_bootstrap_samples,
        rng_seed=0,
    )
    pred_aha, low_aha, high_aha = predict_margins_by_bucket(
        context,
        aha_value=1,
        num_bootstrap_samples=num_bootstrap_samples,
        rng_seed=1,
    )
    return {
        "pred_noaha": pred_noaha,
        "low_noaha": low_noaha,
        "high_noaha": high_noaha,
        "pred_aha": pred_aha,
        "low_aha": low_aha,
        "high_aha": high_aha,
    }


def _draw_curves_for_variant(
    axis: Any,
    curves: Dict[str, np.ndarray],
    meta: RegressionPanelMeta,
) -> None:
    pred_noaha = curves["pred_noaha"]
    low_noaha = curves["low_noaha"]
    high_noaha = curves["high_noaha"]
    pred_aha = curves["pred_aha"]
    low_aha = curves["low_aha"]
    high_aha = curves["high_aha"]

    x_positions = np.arange(len(meta.bucket_levels))
    axis.plot(
        x_positions,
        pred_noaha,
        marker="o",
        color=meta.colors[0],
        label=f"{meta.title}: aha=0",
    )
    axis.fill_between(
        x_positions,
        low_noaha,
        high_noaha,
        color=meta.colors[0],
        alpha=0.15,
    )
    axis.plot(
        x_positions,
        pred_aha,
        marker="o",
        color=meta.colors[1],
        label=f"{meta.title}: aha=1",
    )
    axis.fill_between(
        x_positions,
        low_aha,
        high_aha,
        color=meta.colors[1],
        alpha=0.15,
    )

    axis.set_xticks(x_positions)
    axis.set_xticklabels(meta.bucket_labels, rotation=25, ha="right")
    axis.set_xlabel("perplexity bucket")
    axis.set_title(f"{meta.title}\n{meta.dataset}, {meta.model_name}")
    axis.grid(True, alpha=0.3)


def _extend_rows_for_csv(
    rows_for_csv: List[Dict[str, Any]],
    bucket_levels: List[Any],
    title: str,
    curves: Dict[str, np.ndarray],
) -> None:
    for level, mean0, low0, high0 in zip(
        bucket_levels,
        curves["pred_noaha"],
        curves["low_noaha"],
        curves["high_noaha"],
    ):
        rows_for_csv.append(
            {
                "variant": title,
                "bucket": str(level),
                "aha": 0,
                "mean": mean0,
                "lo": low0,
                "hi": high0,
            },
        )
    for level, mean1, low1, high1 in zip(
        bucket_levels,
        curves["pred_aha"],
        curves["low_aha"],
        curves["high_aha"],
    ):
        rows_for_csv.append(
            {
                "variant": title,
                "bucket": str(level),
                "aha": 1,
                "mean": mean1,
                "lo": low1,
                "hi": high1,
            },
    )


def _plot_variant_axis(
    axis,
    aha_column: str,
    title: str,
    plot_context: VariantPlotContext,
) -> None:
    if aha_column not in plot_context.prepared_frame.columns:
        axis.set_visible(False)
        return
    regression_context = _build_regression_context(
        plot_context.prepared_frame,
        aha_column=aha_column,
        bucket_column=plot_context.config.ppx_bucket_column,
        bucket_levels=plot_context.bucket_levels,
    )
    curves = _predict_curves_for_context(
        regression_context,
        num_bootstrap_samples=plot_context.config.num_bootstrap_samples,
    )
    _draw_curves_for_variant(
        axis,
        curves=curves,
        meta=RegressionPanelMeta(
            bucket_levels=plot_context.bucket_levels,
            bucket_labels=plot_context.bucket_labels,
            colors=REGRESSION_COLORS,
            title=title,
            dataset=plot_context.config.dataset,
            model_name=plot_context.config.model_name,
        ),
    )
    _extend_rows_for_csv(
        plot_context.rows_for_csv,
        bucket_levels=plot_context.bucket_levels,
        title=title,
        curves=curves,
    )


def plot_regression_curves(
    config: RegressionPlotConfig,
) -> str:
    """
    Plot Figure-2-style regression curves and write an accompanying CSV.
    """
    prepared_frame, bucket_levels, bucket_labels = _prepare_bucket_frame(
        config.frame,
        config.ppx_bucket_column,
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            a4_size_inches(config.output.a4_orientation)
            if config.output.a4_pdf
            else (16.5, 4.8)
        ),
        dpi=150,
        sharey=True,
    )
    rows_for_csv: List[Dict[str, Any]] = []
    variant_context = VariantPlotContext(
        prepared_frame=prepared_frame,
        bucket_levels=bucket_levels,
        bucket_labels=bucket_labels,
        config=config,
        rows_for_csv=rows_for_csv,
    )

    for axis, (aha_column, title) in zip(axes, REGRESSION_VARIANTS):
        _plot_variant_axis(
            axis,
            aha_column=aha_column,
            title=title,
            plot_context=variant_context,
        )

    axes[0].set_ylabel("Predicted accuracy")
    legend_handles = [
        Line2D([0], [0], color=REGRESSION_COLORS[0], marker="o", lw=2, label="aha=0"),
        Line2D([0], [0], color=REGRESSION_COLORS[1], marker="o", lw=2, label="aha=1"),
    ]
    add_lower_center_legend(fig, legend_handles)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save_figure_outputs(fig, config.output, dpi=150, tight_layout=False)

    out_csv = config.output.out_png.replace(".png", ".csv")
    pd.DataFrame(rows_for_csv).to_csv(out_csv, index=False)
    return out_csv
