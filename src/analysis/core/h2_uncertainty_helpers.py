"""Shared helpers for the H2 uncertainty bucket + histogram views."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.analysis.common.problem_utils import resolve_problem_identifier
from src.analysis.common.uncertainty import standardize_uncertainty
from src.analysis.core import (
    build_formal_thresholds_from_args,
    build_problem_step_from_samples,
    mark_formal_pairs_with_gain,
)
from src.analysis.core.plotting_helpers import aha_histogram_legend_handles
from src.analysis.io import iter_records_from_file
from src.analysis.labels import aha_words
from src.analysis.metrics import wilson_ci
from src.analysis.utils import choose_uncertainty, coerce_bool, get_aha_gpt_flag, nat_step_from_path


def _aha_gpt_eff(
    pass1_dict: Dict[str, Any],
    rec: Dict[str, Any],
    gate_by_words: bool = False,
) -> Tuple[int, int]:
    """Return (aha_gpt_eff, aha_words) tuple for a record."""
    gpt = get_aha_gpt_flag(pass1_dict, rec)
    words = aha_words(pass1_dict)
    gpt = 0 if gpt is None else int(gpt)
    words = int(words)
    return int((gpt and words) if gate_by_words else gpt), words


def _build_bucket_row(
    step_from_name: Optional[int],
    rec: Dict[str, Any],
    unc_field: str,
    gate_gpt_by_words: bool,
) -> Optional[Dict[str, Any]]:
    """Build a single row for the buckets/histogram views."""
    pass1_dict = rec.get("pass1") or {}
    if not pass1_dict:
        return None

    problem = resolve_problem_identifier(rec)

    step_raw = rec.get(
        "step",
        step_from_name if step_from_name is not None else None,
    )
    if step_raw is None:
        return None
    step_value = int(step_raw)

    corr_raw = coerce_bool(pass1_dict.get("is_correct_pred"))
    if corr_raw is None:
        return None
    correct = int(corr_raw)

    uncertainty = choose_uncertainty(pass1_dict, unc_field)
    if uncertainty is None:
        return None

    gpt_eff, words = _aha_gpt_eff(
        pass1_dict,
        rec,
        gate_by_words=gate_gpt_by_words,
    )

    return {
        "problem": str(problem),
        "step": step_value,
        "correct": correct,
        "uncertainty": float(uncertainty),
        "aha_gpt": int(gpt_eff),
        "aha_words": int(words),
    }


def load_all_defs_for_buckets(
    files: List[str],
    unc_field: str,
    gate_gpt_by_words: bool,
) -> pd.DataFrame:
    """
    Load all samples needed for the uncertainty buckets and histogram views.

    Returns a DataFrame with problem, step, correctness, uncertainty, and
    both GPT and words-based Aha labels.
    """
    rows: List[Dict[str, Any]] = []
    for path in files:
        step_from_name = nat_step_from_path(path)
        for rec in iter_records_from_file(path):
            row = _build_bucket_row(
                step_from_name,
                rec,
                unc_field,
                gate_gpt_by_words,
            )
            if row is not None:
                rows.append(row)

    data = pd.DataFrame(rows)
    if data.empty:
        raise RuntimeError("No rows for buckets/hist figure.")
    return data


mark_formal_pairs = mark_formal_pairs_with_gain


def label_formal_samples(
    samples_df: pd.DataFrame,
    formal_step_df: pd.DataFrame,
) -> pd.DataFrame:
    """Sample-level Formal=1 when (problem, step) is formal_pair and GPT shift == 1."""
    mask_formal = formal_step_df["aha_formal_pair"] == 1
    formal_step_keys = {
        (str(row["problem"]), int(row["step"]))
        for _, row in formal_step_df.loc[
            mask_formal,
            ["problem", "step"],
        ].iterrows()
    }
    sample_keys = list(
        zip(
            samples_df["problem"].astype(str),
            samples_df["step"].astype(int),
        ),
    )
    labeled_df = samples_df.copy()
    aha_gpt_flags = labeled_df["aha_gpt"].astype(int)
    labeled_df["aha_formal"] = [
        int((key in formal_step_keys) and (gpt_flag == 1)) for key, gpt_flag in zip(sample_keys, aha_gpt_flags)
    ]
    return labeled_df


def _make_uncertainty_buckets(
    samples_df: pd.DataFrame,
    n_buckets: int,
) -> pd.DataFrame:
    """Standardize uncertainty and assign quantile buckets."""
    bucket_df = standardize_uncertainty(samples_df)
    bucket_count = int(max(3, n_buckets))
    bucket_count = min(bucket_count, len(bucket_df))

    # Use ranks to avoid duplicate bin edges on small/degenerate inputs.
    ranks = bucket_df["uncertainty_std"].rank(method="first")
    buckets = pd.qcut(ranks, q=bucket_count, duplicates="drop")
    if buckets.cat.categories.size < bucket_count:
        # Fallback to equally spaced bins if qcut collapsed categories.
        buckets = pd.cut(ranks, bins=bucket_count, duplicates="drop")

    bucket_df["unc_bucket"] = buckets
    bucket_df["bucket_id"] = bucket_df["unc_bucket"].cat.codes
    bucket_df["bucket_label"] = bucket_df["unc_bucket"].astype(str)
    return bucket_df


def _aggregate_buckets(
    bucket_df: pd.DataFrame,
    label_col: str,
) -> pd.DataFrame:
    """Aggregate Aha counts and Wilson CIs per uncertainty bucket."""
    grouped = bucket_df.groupby(
        ["bucket_id", "bucket_label"],
        as_index=False,
    ).agg(
        n=("uncertainty", "size"),
        k_aha=(label_col, "sum"),
    )
    grouped["aha_ratio"] = grouped["k_aha"] / grouped["n"]
    lower_bounds: List[float] = []
    upper_bounds: List[float] = []
    for _, row in grouped.iterrows():
        lower, upper = wilson_ci(
            int(row["k_aha"]),
            int(row["n"]),
        )
        lower_bounds.append(lower)
        upper_bounds.append(upper)
    grouped["lo"] = lower_bounds
    grouped["hi"] = upper_bounds
    return grouped.sort_values("bucket_id").reset_index(drop=True)


def plot_uncertainty_buckets_three(
    d_words: pd.DataFrame,
    d_gpt: pd.DataFrame,
    d_formal: pd.DataFrame,
    out_path: str,
    title_suffix: str = "",
) -> None:
    """Plot three-panel uncertainty buckets figure for Words, GPT, and Formal Aha."""
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), dpi=150, sharey=True)
    items = [
        ('Words of "Aha!"', _aggregate_buckets(d_words, "aha_words")),
        ('LLM-Detected "Aha!"', _aggregate_buckets(d_gpt, "aha_gpt")),
        ('Formal "Aha!"', _aggregate_buckets(d_formal, "aha_formal")),
    ]
    for axis, (title, table) in zip(axes, items):
        axis.plot(
            table["bucket_id"],
            table["aha_ratio"],
            marker="o",
            label="Aha ratio",
        )
        axis.fill_between(
            table["bucket_id"],
            table["lo"],
            table["hi"],
            alpha=0.2,
            label="95% CI",
        )
        axis.set_xticks(table["bucket_id"])
        axis.set_xticklabels(table["bucket_label"], rotation=25, ha="right")
        axis.set_xlabel("uncertainty_std quantile bucket")
        axis.set_ylabel("Aha ratio")
        axis.set_title(f"{title}\n{title_suffix}")
        axis.grid(True, alpha=0.35)
    line = Line2D([0], [0], color="C0", marker="o", label="Aha ratio")
    band = Patch(facecolor="C0", alpha=0.2, label="95% CI")
    figure.legend(
        handles=[line, band],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
    )
    figure.tight_layout(rect=[0, 0.08, 1, 1])
    figure.savefig(out_path)
    figure.savefig(out_path.replace(".png", ".pdf"))
    plt.close(figure)


def _build_bucket_series_table(
    d_words: pd.DataFrame,
    d_gpt: pd.DataFrame,
    d_formal: pd.DataFrame,
) -> pd.DataFrame:
    """Construct a long-form bucket table for all three Aha definitions."""
    series_frames: List[pd.DataFrame] = []
    for name, table in [
        ("words", _aggregate_buckets(d_words, "aha_words")),
        ("gpt", _aggregate_buckets(d_gpt, "aha_gpt")),
        ("formal", _aggregate_buckets(d_formal, "aha_formal")),
    ]:
        table_with_series = table.copy()
        table_with_series["series"] = name
        series_frames.append(
            table_with_series[
                [
                    "bucket_id",
                    "bucket_label",
                    "n",
                    "k_aha",
                    "aha_ratio",
                    "lo",
                    "hi",
                    "series",
                ]
            ],
        )
    return pd.concat(series_frames, ignore_index=True)


@dataclass(frozen=True)
class UncertaintyHistCounts:
    """Bin edges, centers, and per-Aha counts for the uncertainty histogram."""

    edges: np.ndarray
    centers: np.ndarray
    n_total: np.ndarray
    n_words: np.ndarray
    n_gpt: np.ndarray
    n_formal: np.ndarray


def _compute_uncertainty_hist_counts(
    standardized_df: pd.DataFrame,
    problem_step_df: pd.DataFrame,
    num_bins: int,
) -> UncertaintyHistCounts:
    """Compute histogram edges, centers, and Aha counts for the uncertainty histogram."""
    hist_range = (
        standardized_df["uncertainty_std"].min(),
        standardized_df["uncertainty_std"].max(),
    )
    edges = np.linspace(hist_range[0], hist_range[1], num_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    n_total, _ = np.histogram(
        standardized_df["uncertainty_std"].values,
        bins=edges,
    )
    n_words, _ = np.histogram(
        standardized_df.loc[
            standardized_df["aha_words"] == 1,
            "uncertainty_std",
        ].values,
        bins=edges,
    )
    n_gpt, _ = np.histogram(
        standardized_df.loc[
            standardized_df["aha_gpt"] == 1,
            "uncertainty_std",
        ].values,
        bins=edges,
    )

    if "aha_formal" not in standardized_df.columns:
        formal_mask = problem_step_df["aha_formal_pair"] == 1
        formal_keys = {
            (str(row["problem"]), int(row["step"]))
            for _, row in problem_step_df.loc[
                formal_mask,
                ["problem", "step"],
            ].iterrows()
        }
        keys = list(
            zip(
                standardized_df["problem"].astype(str),
                standardized_df["step"].astype(int),
            ),
        )
        standardized_df["aha_formal"] = [
            int((key in formal_keys) and (gpt_flag == 1))
            for key, gpt_flag in zip(
                keys,
                standardized_df["aha_gpt"].astype(int),
            )
        ]

    n_formal, _ = np.histogram(
        standardized_df.loc[
            standardized_df["aha_formal"] == 1,
            "uncertainty_std",
        ].values,
        bins=edges,
    )
    return UncertaintyHistCounts(
        edges=edges,
        centers=centers,
        n_total=n_total,
        n_words=n_words,
        n_gpt=n_gpt,
        n_formal=n_formal,
    )


def _write_uncertainty_hist_csv(
    out_csv: str,
    counts: UncertaintyHistCounts,
) -> None:
    """Write histogram bin statistics to CSV."""
    pd.DataFrame(
        {
            "bin_left": counts.edges[:-1],
            "bin_right": counts.edges[1:],
            "bin_center": counts.centers,
            "n_total": counts.n_total.astype(int),
            "n_words": counts.n_words.astype(int),
            "n_gpt": counts.n_gpt.astype(int),
            "n_formal": counts.n_formal.astype(int),
        },
    ).to_csv(out_csv, index=False)


def _plot_uncertainty_hist_figure(
    out_dir: str,
    slug: str,
    counts: UncertaintyHistCounts,
    num_bins: int,
    title_suffix: str,
) -> str:
    """Plot the uncertainty histogram figure and return the PNG path."""
    figure, axis_left = plt.subplots(figsize=(12.0, 4.8), dpi=150)
    width = counts.edges[1] - counts.edges[0]
    axis_left.bar(
        counts.centers,
        counts.n_total,
        width=width * 0.95,
        color="#CCCCCC",
        edgecolor="none",
        label="Total (bin count)",
    )

    axis_right = axis_left.twinx()
    axis_right.plot(
        counts.centers,
        counts.n_words,
        marker="o",
        lw=1.8,
        label="Words Aha (count)",
    )
    axis_right.plot(
        counts.centers,
        counts.n_gpt,
        marker="o",
        lw=1.8,
        label="LLM Aha (count)",
    )
    axis_right.plot(
        counts.centers,
        counts.n_formal,
        marker="o",
        lw=1.8,
        label="Formal Aha (count)",
    )

    axis_left.set_xlabel("uncertainty_std")
    axis_left.set_ylabel("Total samples per bin")
    axis_right.set_ylabel("Aha count per bin")
    axis_left.grid(True, alpha=0.3)
    axis_left.set_title(
        f"Uncertainty histogram (bins={num_bins}) with Aha counts\n{title_suffix}",
    )

    axis_left.legend(
        handles=aha_histogram_legend_handles(axis_right),
        loc="upper left",
        frameon=False,
    )
    figure.tight_layout()
    out_png = os.path.join(
        out_dir,
        f"h2_uncertainty_hist_100bins__{slug}.png",
    )
    figure.savefig(out_png)
    figure.savefig(out_png.replace(".png", ".pdf"))
    plt.close(figure)
    return out_png


def make_all3_uncertainty_buckets_figure(
    files: List[str],
    out_dir: str,
    args: argparse.Namespace,
) -> tuple[str, str, pd.DataFrame, pd.DataFrame]:
    """
    Build the three-way uncertainty buckets figure and associated CSV tables.

    Returns the figure path, CSV path, the annotated sample DataFrame, and the
    problem-step table used for Formal Aha definitions.
    """
    d_all = load_all_defs_for_buckets(
        files,
        args.unc_field,
        bool(args.gpt_gate_by_words),
    )
    problem_step_df = build_problem_step_from_samples(
        d_all,
        include_native=True,
        native_col="aha_words",
    )
    formal_thresholds = build_formal_thresholds_from_args(
        delta1=float(args.delta1),
        delta2=float(args.delta2),
        min_prior_steps=int(args.min_prior_steps),
        delta3=None if args.delta3 is None else float(args.delta3),
    )
    problem_step_df = mark_formal_pairs(problem_step_df, formal_thresholds)
    d_all = label_formal_samples(d_all, problem_step_df)

    try:
        d_all.to_csv(
            os.path.join(out_dir, "h2_all3_pass1_samples.csv"),
            index=False,
        )
    except OSError:
        pass
    try:
        problem_step_df.to_csv(
            os.path.join(out_dir, "h2_all3_problem_step.csv"),
            index=False,
        )
    except OSError:
        pass

    d_all = _make_uncertainty_buckets(d_all, n_buckets=int(args.unc_buckets))
    d_words = d_all.copy()
    d_gpt = d_all.copy()
    d_formal = d_all.copy()

    dataset_slug = args.dataset_name.replace(" ", "_")
    model_slug = args.model_name.replace(" ", "_")
    slug = f"{dataset_slug}__{model_slug}"
    out_png = os.path.join(
        out_dir,
        f"h2_aha_vs_uncertainty_buckets__{slug}.png",
    )
    title_suffix = f"{args.dataset_name}, {args.model_name}"
    plot_uncertainty_buckets_three(
        d_words,
        d_gpt,
        d_formal,
        out_png,
        title_suffix=title_suffix,
    )

    out_csv = os.path.join(out_dir, "h2_aha_vs_uncertainty_buckets.csv")
    _build_bucket_series_table(d_words, d_gpt, d_formal).to_csv(
        out_csv,
        index=False,
    )
    return out_png, out_csv, d_all, problem_step_df


def plot_uncertainty_hist_100bins(
    d_all: pd.DataFrame,
    problem_step_df: pd.DataFrame,
    out_dir: str,
    args: argparse.Namespace,
) -> tuple[str, str]:
    """
    Histogram of uncertainty_std with total counts and Aha COUNTS per bin
    (Words, GPT, Formal). Writes a PNG/PDF and a CSV with bin stats.
    """
    standardized_df = standardize_uncertainty(d_all)
    num_bins = int(max(10, args.hist_bins))
    counts = _compute_uncertainty_hist_counts(
        standardized_df,
        problem_step_df,
        num_bins,
    )

    dataset_slug = args.dataset_name.replace(" ", "_")
    model_slug = args.model_name.replace(" ", "_")
    slug = f"{dataset_slug}__{model_slug}"
    title_suffix = f"{args.dataset_name}, {args.model_name}"

    out_csv = os.path.join(out_dir, "h2_uncertainty_hist_100bins.csv")
    _write_uncertainty_hist_csv(out_csv, counts)
    out_png = _plot_uncertainty_hist_figure(
        out_dir,
        slug,
        counts,
        num_bins,
        title_suffix,
    )
    return out_png, out_csv


def style_ame_axis(axis) -> None:
    """Apply consistent styling to AME plots."""
    axis.set_xlabel("Training step")
    axis.set_ylabel("AME(aha)")
    axis.set_title("Aha AME with bootstrap CI")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="lower right")


__all__ = [
    "load_all_defs_for_buckets",
    "mark_formal_pairs",
    "label_formal_samples",
    "plot_uncertainty_buckets_three",
    "make_all3_uncertainty_buckets_figure",
    "plot_uncertainty_hist_100bins",
    "style_ame_axis",
    "get_aha_gpt_flag",
]
