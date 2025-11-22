#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Forced Aha Effect — analysis + publication figures.

Adds:
  • Pastel1 palette for uncertainty & stepwise figures
  • Clearer labels (y: "Delta Accuracy (pp) for second pass")
  • Series names: "Per-draw (single completion)", "Per-problem mean of 8 (avg of 8 completions)"
  • New combined figure: uncertainty buckets (left) + stepwise overlay of all three metrics (right)

Figures (saved under <out_dir>/figures):
  - overall_deltas.{png,pdf}
  - stepwise_delta_sample.{png,pdf}
  - stepwise_delta_cluster_any.{png,pdf}
  - stepwise_delta_cluster_mean.{png,pdf}
  - stepwise_overlay.{png,pdf}                  <-- NEW (all three deltas in one subplot)
  - any_conversion_waterfall.{png,pdf}
  - any_headroom_scatter.{png,pdf}
  - uncertainty_buckets.{png,pdf}               <-- updated labels & Pastel1 by default
  - uncertainty_and_stepwise.{png,pdf}          <-- NEW two-subfigure figure
  - overview_deltas_and_waterfall.{png,pdf}
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd

from src.analysis.forced_aha_plotting import (
    OverallDeltaConfig,
    OverallDeltaInputs,
    SeriesPlotConfig,
    plot_conversion_waterfall,
    plot_headroom_scatter,
    plot_overall_deltas,
    plot_overview_side_by_side,
    plot_stepwise_overlay,
    plot_uncertainty_and_stepwise,
    plot_uncertainty_buckets,
)
from src.analysis.forced_aha_sampling import prepare_forced_aha_samples
from src.analysis.forced_aha_shared import (
    PASS1_KEYS,
    PASS2_KEYS,
    extract_correct_flag,
    extract_entropy,
    extract_sample_idx,
    first_nonempty,
    mcnemar_from_pairs,
    paired_t_and_wilcoxon,
    pass_with_correctness,
    summarize_cluster_any,
    summarize_cluster_mean,
    summarize_sample_level,
)
from src.analysis.io import iter_records_from_file, scan_jsonl_files
from src.analysis.plotting_styles import (
    DEFAULT_COLORS,
    METRIC_LABELS,
    parse_color_overrides,
)
from src.analysis.utils import nat_step_from_path

# ===================== Matplotlib (Times everywhere) =====================
# Force Times / Times New Roman across all figures
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,            # embed TrueType (editable text in PDF)
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"],
    "mathtext.fontset": "stix",                        # Times-like math without usetex
    "font.size": 12,
    "axes.titlesize": 12, "axes.labelsize": 12,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 12, "figure.titlesize": 12,
})

# ===================== loaders =====================


def _maybe_int(value: Any, default: int = -1) -> int:
    """Attempt to cast ``value`` to ``int``, returning ``default`` on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def _common_fields(rec: Dict[str, Any], step_from_name: Optional[int]) -> Dict[str, Any]:
    dataset = rec.get("dataset")
    model = rec.get("model")
    problem = (
        rec.get("problem")
        or rec.get("question")
        or rec.get("row_key")
        or "unknown"
    )
    step = rec.get("step", step_from_name if step_from_name is not None else None)
    split = rec.get("split")
    return {
        "dataset": dataset,
        "model": model,
        "problem": str(problem),
        "step": _maybe_int(step, -1),
        "split": split,
    }


@dataclass(frozen=True)
class SampleRowContext:
    """Context for constructing sample rows with consistent metadata."""

    variant: str
    entropy_field: str
    step_from_name: Optional[int]


def _build_sample_row(
    record: Dict[str, Any],
    pass_obj: Dict[str, Any],
    correct_flag: int,
    context: SampleRowContext,
) -> Dict[str, Any]:
    """
    Construct a single sample-level row with consistent metadata.
    """
    row = {
        **_common_fields(record, context.step_from_name),
        "sample_idx": extract_sample_idx(record, pass_obj),
        "correct": int(correct_flag),
    }
    if context.variant == "pass1":
        row["entropy_p1"] = extract_entropy(
            pass_obj,
            preferred=context.entropy_field,
        )
    return row

def load_samples_from_root(
    root: str,
    split_value: Optional[str],
    variant: str,
    entropy_field: str,
    pass2_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load sample-level rows from a single root for either PASS-1 or PASS-2.

    When ``variant == 'pass1'`` the function populates ``entropy_p1`` from the
    requested ``entropy_field``; for PASS-2 it only records correctness.
    """
    rows: List[Dict[str, Any]] = []
    for path in scan_jsonl_files(root, split_value):
        step_from_name = nat_step_from_path(path)
        row_context = SampleRowContext(
            variant=variant,
            entropy_field=entropy_field,
            step_from_name=step_from_name,
        )
        for rec in iter_records_from_file(path):
            if (
                split_value is not None
                and str(rec.get("split", "")).lower()
                != str(split_value).lower()
            ):
                continue
            pass_result = pass_with_correctness(
                rec,
                variant=variant,
                pass2_key=pass2_key,
            )
            if pass_result is None:
                continue
            pass_obj, corr = pass_result
            rows.append(
                _build_sample_row(
                    rec,
                    pass_obj,
                    corr,
                    row_context,
                ),
            )
    return pd.DataFrame(rows)

def load_samples_from_single_root_with_both(
    root: str,
    split_value: Optional[str],
    entropy_field: str,
    pass2_key: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load paired PASS-1 and PASS-2 samples from a single root directory.

    Returns two DataFrames, one for pass1 and one for pass2, sharing common
    metadata columns (dataset/model/problem/step/split/sample_idx).
    """
    files = scan_jsonl_files(root, split_value)
    rows1: List[Dict[str, Any]] = []
    rows2: List[Dict[str, Any]] = []
    for path in files:
        step_from_name = nat_step_from_path(path)
        for rec in iter_records_from_file(path):
            if (
                split_value is not None
                and str(rec.get("split", "")).lower()
                != str(split_value).lower()
            ):
                continue
            meta = _common_fields(rec, step_from_name)
            pass1_obj = first_nonempty(rec, PASS1_KEYS)
            if pass1_obj:
                correct_pass1 = extract_correct_flag(pass1_obj)
                if correct_pass1 is not None:
                    rows1.append({
                        **meta,
                        "sample_idx": extract_sample_idx(rec, pass1_obj),
                        "correct": int(correct_pass1),
                        "entropy_p1": extract_entropy(pass1_obj, preferred=entropy_field),
                    })
            if pass2_key:
                pass2_obj = rec.get(pass2_key) or {}
            else:
                pass2_obj = first_nonempty(rec, PASS2_KEYS)
            if pass2_obj:
                correct_pass2 = extract_correct_flag(pass2_obj)
                if correct_pass2 is not None:
                    rows2.append({
                        **meta,
                        "sample_idx": extract_sample_idx(rec, pass2_obj),
                        "correct": int(correct_pass2),
                    })
    return pd.DataFrame(rows1), pd.DataFrame(rows2)

# ===================== pairing helpers =====================

def _choose_merge_keys(df_left: pd.DataFrame, df_right: pd.DataFrame) -> List[str]:
    keys: List[str] = []
    must = ["problem"]
    optional = ["step", "dataset", "model", "split"]
    for k in must:
        if (k in df_left.columns) and (k in df_right.columns):
            keys.append(k)
    for k in optional:
        if (k in df_left.columns) and (k in df_right.columns):
            if df_left[k].notna().any() and df_right[k].notna().any():
                keys.append(k)
    if not keys:
        raise SystemExit("No common non-null keys to merge on. Need at least 'problem'.")
    return keys

def _fill_missing_id_cols(
    frame: pd.DataFrame,
    cols: List[str],
) -> None:
    """
    Fill missing values in identifier columns with a sentinel string.

    Operates in-place on the provided DataFrame.
    """
    for col in cols:
        if col in frame.columns:
            frame[col] = frame[col].fillna("(missing)")

# ===================== pairing =====================

def pair_samples(
    df_pass1: pd.DataFrame,
    df_pass2: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Pair sample-level rows from two passes on shared identifiers (problem/step/etc.).
    """
    _fill_missing_id_cols(df_pass1, ["dataset", "model", "split"])
    _fill_missing_id_cols(df_pass2, ["dataset", "model", "split"])
    keys = _choose_merge_keys(df_pass1, df_pass2)
    if (
        ("sample_idx" in df_pass1.columns)
        and ("sample_idx" in df_pass2.columns)
        and df_pass1["sample_idx"].notna().any()
        and df_pass2["sample_idx"].notna().any()
    ):
        keys = keys + ["sample_idx"]
    left = df_pass1.rename(columns={"correct": "correct1"})
    right = df_pass2.rename(columns={"correct": "correct2"})
    pairs = left.merge(right, on=keys, how="inner")
    print(f"[info] Sample-level pairing on keys: {keys}  (pairs={len(pairs)})")
    return pairs, keys

def build_clusters(samples_df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Aggregate sample-level rows into per-problem clusters for one pass.

    The ``label`` is used as a suffix for count/accuracy fields (for example, ``p1`` or ``p2``).
    """
    samples_df = samples_df.copy()
    keys = [
        col
        for col in ["dataset", "model", "split", "problem", "step"]
        if col in samples_df.columns
    ]
    if "problem" not in keys:
        raise SystemExit("Cluster build requires at least 'problem' in records.")
    aggregations = {
        f"n_{label}": ("correct", "size"),
        f"k_{label}": ("correct", "sum"),
    }
    if label == "p1" and "entropy_p1" in samples_df.columns:
        aggregations["entropy_p1_cluster"] = ("entropy_p1", "mean")
    clustered = samples_df.groupby(keys, as_index=False).agg(**aggregations)
    clustered[f"acc_{label}"] = clustered[f"k_{label}"] / clustered[f"n_{label}"]
    clustered[f"any_{label}"] = (clustered[f"k_{label}"] > 0).astype(int)
    return clustered

def pair_clusters(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Build and pair per-problem clusters from two sample-level DataFrames.

    The input frames are aggregated into clusters (per problem/step/etc.) and then
    inner-joined on the shared identifier columns.
    """
    clusters_pass1 = build_clusters(df1, "p1")
    clusters_pass2 = build_clusters(df2, "p2")
    _fill_missing_id_cols(clusters_pass1, ["dataset", "model", "split"])
    _fill_missing_id_cols(clusters_pass2, ["dataset", "model", "split"])
    keys = _choose_merge_keys(clusters_pass1, clusters_pass2)
    merged = clusters_pass1.merge(clusters_pass2, on=keys, how="inner")
    print(f"[info] Cluster-level pairing on keys: {keys}  (clusters={len(merged)})")
    return merged

# ===================== bootstrapping helpers =====================

def rng(seed: int) -> None:
    """
    Seed both the Python ``random`` module and NumPy with a robust integer seed.
    """
    try:
        normalized_seed = int(seed)
    except (TypeError, ValueError):
        normalized_seed = 0
    random.seed(normalized_seed)
    np.random.seed(normalized_seed)


# ===================== NEW/UPDATED: uncertainty buckets + stepwise overlay =====================


@dataclass
class PlotArtifacts:
    """Bundles data required to render plots."""

    out_dir: str
    summary_rows: List[Dict[str, Any]]
    pairs: pd.DataFrame
    clusters: pd.DataFrame
    step_df: pd.DataFrame


# ===================== main helpers =====================

@dataclass
class McNemarCounts:
    """Counts and p-value for McNemar tests."""

    both_wrong: int
    wins_pass2: int
    wins_pass1: int
    both_correct: int
    p_value: float


def _compute_mcnemar_counts(
    input_df: pd.DataFrame,
    col1: str,
    col2: str,
) -> McNemarCounts:
    """
    Run McNemar's test on two boolean columns and return labeled counts.
    """
    left_wrong, wins_pass2, wins_pass1, both_correct, p_value = mcnemar_from_pairs(
        input_df,
        col1,
        col2,
    )
    return McNemarCounts(
        both_wrong=int(left_wrong),
        wins_pass2=int(wins_pass2),
        wins_pass1=int(wins_pass1),
        both_correct=int(both_correct),
        p_value=float(p_value),
    )


def _build_stepwise_effect_table(
    pairs: pd.DataFrame,
    clusters: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a combined per-step table for sample and cluster-level forced Aha effects.
    """
    step_rows: List[Dict[str, Any]] = []

    if not pairs.empty:
        for step, sub_frame in pairs.groupby("step"):
            counts = _compute_mcnemar_counts(sub_frame, "correct1", "correct2")
            num_units = int(len(sub_frame))
            acc_pass1 = float(sub_frame["correct1"].mean()) if num_units else np.nan
            acc_pass2 = float(sub_frame["correct2"].mean()) if num_units else np.nan
            delta_pp = (acc_pass2 - acc_pass1) * 100.0 if num_units else np.nan
            step_rows.append(
                {
                    "metric": "sample",
                    "step": int(step),
                    "n_units": num_units,
                    "acc_pass1": acc_pass1,
                    "acc_pass2": acc_pass2,
                    "delta_pp": delta_pp,
                    "p_mcnemar": counts.p_value,
                    "p_ttest": None,
                    "p_wilcoxon": None,
                    "wins_pass2": counts.wins_pass2,
                    "wins_pass1": counts.wins_pass1,
                    "both_correct": counts.both_correct,
                    "both_wrong": counts.both_wrong,
                },
            )

    for step, sub_frame in clusters.groupby("step"):
        counts = _compute_mcnemar_counts(sub_frame, "any_p1", "any_p2")
        num_units = int(len(sub_frame))
        acc_pass1 = float(sub_frame["any_p1"].mean()) if num_units else np.nan
        acc_pass2 = float(sub_frame["any_p2"].mean()) if num_units else np.nan
        delta_pp = (acc_pass2 - acc_pass1) * 100.0 if num_units else np.nan
        step_rows.append(
            {
                "metric": "cluster_any",
                "step": int(step),
                "n_units": num_units,
                "acc_pass1": acc_pass1,
                "acc_pass2": acc_pass2,
                "delta_pp": delta_pp,
                "p_mcnemar": counts.p_value,
                "p_ttest": None,
                "p_wilcoxon": None,
                "wins_pass2": counts.wins_pass2,
                "wins_pass1": counts.wins_pass1,
                "both_correct": counts.both_correct,
                "both_wrong": counts.both_wrong,
            },
        )

    for step, sub_frame in clusters.groupby("step"):
        num_units = int(len(sub_frame))
        acc_pass1 = float(sub_frame["acc_p1"].mean()) if num_units else np.nan
        acc_pass2 = float(sub_frame["acc_p2"].mean()) if num_units else np.nan
        diffs = (sub_frame["acc_p2"] - sub_frame["acc_p1"]).to_numpy(float)
        p_ttest, p_wilcoxon = paired_t_and_wilcoxon(diffs)
        delta_pp = (acc_pass2 - acc_pass1) * 100.0 if num_units else np.nan
        step_rows.append(
            {
                "metric": "cluster_mean",
                "step": int(step),
                "n_units": num_units,
                "acc_pass1": acc_pass1,
                "acc_pass2": acc_pass2,
                "delta_pp": delta_pp,
                "p_mcnemar": None,
                "p_ttest": p_ttest,
                "p_wilcoxon": p_wilcoxon,
                "wins_pass2": None,
                "wins_pass1": None,
                "both_correct": None,
                "both_wrong": None,
            },
        )

    step_df = pd.DataFrame(step_rows).sort_values(["metric", "step"]).reset_index(
        drop=True,
    )
    return step_df


def _verdict_for_binary_metric(result_row: Dict[str, Any], label: str) -> str:
    """Return verdict string for sample/any metrics."""
    acc_pass1 = result_row["acc_pass1"]
    acc_pass2 = result_row["acc_pass2"]
    delta_pp = result_row["delta_pp"]
    num_units = result_row["n_units"]
    p_mcnemar = result_row.get("p_mcnemar")
    has_effect = (
        np.isfinite(delta_pp)
        and delta_pp > 0
        and (p_mcnemar is not None)
        and p_mcnemar < 0.05
    )
    maybe_effect = np.isfinite(delta_pp) and delta_pp > 0
    verdict = "YES" if has_effect else ("MAYBE" if maybe_effect else "NO")
    p_text = f"{p_mcnemar:.4g}" if p_mcnemar is not None else "nan"
    return (
        f"[{label}] N={num_units} acc1={acc_pass1:.4f} acc2={acc_pass2:.4f} "
        f"Δ={delta_pp:+.2f}pp  McNemar p={p_text} -> {verdict}"
    )


def _verdict_for_mean_metric(result_row: Dict[str, Any], label: str) -> str:
    """Return verdict string for per-problem mean metrics."""
    acc_pass1 = result_row["acc_pass1"]
    acc_pass2 = result_row["acc_pass2"]
    delta_pp = result_row["delta_pp"]
    num_units = result_row["n_units"]
    p_ttest = result_row.get("p_ttest")
    p_wilcoxon = result_row.get("p_wilcoxon")
    is_significant = (
        (p_ttest is not None and p_ttest < 0.05)
        or (p_wilcoxon is not None and p_wilcoxon < 0.05)
    )
    has_effect = np.isfinite(delta_pp) and delta_pp > 0 and is_significant
    maybe_effect = np.isfinite(delta_pp) and delta_pp > 0
    verdict = "YES" if has_effect else ("MAYBE" if maybe_effect else "NO")
    t_str = f"{p_ttest:.4g}" if p_ttest is not None else "nan"
    w_str = f"{p_wilcoxon:.4g}" if p_wilcoxon is not None else "nan"
    return (
        f"[{label}] N={num_units} mean-acc1={acc_pass1:.4f} mean-acc2={acc_pass2:.4f} "
        f"Δ={delta_pp:+.2f}pp t_p={t_str} W={w_str} -> {verdict}"
    )


def verdict_line(result_row: Dict[str, Any]) -> str:
    """
    Build a human-readable verdict line summarizing forced Aha effects.
    """
    metric = result_row["metric"]
    label = METRIC_LABELS.get(metric, metric)
    if metric in ("sample", "cluster_any"):
        return _verdict_for_binary_metric(result_row, label)
    return _verdict_for_mean_metric(result_row, label)


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser for the forced Aha effect CLI.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("root1")
    parser.add_argument("root2", nargs="?", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument(
        "--pass2_key",
        default=None,
        help=(
            "Optional specific second-pass key to analyze "
            "(e.g., 'pass2', 'pass2a', 'pass2b', 'pass2c'). "
            "Defaults to the first available in PASS2_KEYS."
        ),
    )

    parser.add_argument("--make_plots", action="store_true")
    parser.add_argument("--n_boot", type=int, default=800)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--colors",
        default=None,
        help=(
            "Overrides for overview/waterfall palette, e.g. "
            "'gain:#009E73,loss:#D55E00'"
        ),
    )

    parser.add_argument(
        "--entropy_field",
        default="entropy_answer",
        help=(
            'Entropy source for buckets (default: "entropy_answer"; '
            'fallback to "entropy").'
        ),
    )
    parser.add_argument(
        "--series_palette",
        default="Dark2",
        help=(
            "Qualitative colormap for uncertainty/stepwise series "
            "(default: Pastel1)."
        ),
    )
    parser.add_argument(
        "--darken",
        type=float,
        default=0.80,
        help=(
            "Darken factor for series colors (0.0–1.0, lower = darker, "
            "default 0.80)."
        ),
    )
    return parser


def _run_plots_if_requested(
    args: argparse.Namespace,
    plots: PlotArtifacts,
) -> None:
    """
    Generate all figures when --make_plots is enabled.
    """
    if not args.make_plots:
        return

    print("[info] Generating figures...")
    palette = {**DEFAULT_COLORS, **parse_color_overrides(args.colors)}
    plot_overall_deltas(
        plots.out_dir,
        OverallDeltaInputs(
            summary_rows=plots.summary_rows,
            pairs_df=plots.pairs,
            clusters_df=plots.clusters,
        ),
        OverallDeltaConfig(
            n_boot=args.n_boot,
            seed=args.seed,
            palette=palette,
        ),
    )
    plot_conversion_waterfall(plots.out_dir, plots.clusters, palette=palette)
    plot_headroom_scatter(plots.out_dir, plots.step_df)

    series_config = SeriesPlotConfig(
        out_dir=plots.out_dir,
        n_boot=args.n_boot,
        seed=args.seed,
        palette_name=args.series_palette,
        darken=args.darken,
    )
    plot_uncertainty_buckets(
        plots.pairs,
        plots.clusters,
        series_config,
    )
    plot_stepwise_overlay(
        plots.step_df,
        plots.pairs,
        plots.clusters,
        series_config,
    )
    plot_uncertainty_and_stepwise(
        plots.step_df,
        plots.pairs,
        plots.clusters,
        series_config,
    )
    plot_overview_side_by_side(
        {
            "out_dir": plots.out_dir,
            "pairs_df": plots.pairs,
            "clusters_df": plots.clusters,
            "n_boot": args.n_boot,
            "seed": args.seed,
            "palette_name": args.series_palette,
            "darken": args.darken,
        },
    )
    print("[info] Figures saved to:", os.path.join(plots.out_dir, "figures"))


# ===================== main =====================

def main() -> None:
    """
    CLI entry point for forced Aha effect analysis and plotting.
    """
    parser = build_arg_parser()
    args = parser.parse_args()

    out_dir, df1, df2 = prepare_forced_aha_samples(
        args,
        load_samples_from_root,
        load_samples_from_single_root_with_both,
    )
    if df1.empty:
        raise SystemExit("No sample-level pass-1 rows found (after filtering).")
    if df2.empty:
        raise SystemExit("No sample-level pass-2 rows found (after filtering).")

    print(f"[info] Loaded: pass1 N={len(df1)} rows; pass2 N={len(df2)} rows")

    pairs, key_cols = pair_samples(df1, df2)
    clusters = pair_clusters(df1, df2)
    if clusters.empty:
        raise SystemExit(
            "No overlapping clusters between pass-1 and pass-2 after adaptive keying.",
        )

    summary_rows: List[Dict[str, Any]] = []
    if not pairs.empty:
        summary_rows.append(summarize_sample_level(pairs))
    summary_rows.append(summarize_cluster_any(clusters))
    summary_rows.append(summarize_cluster_mean(clusters))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, "forced_aha_summary.csv"), index=False)

    step_df = _build_stepwise_effect_table(pairs, clusters)
    step_df.to_csv(os.path.join(out_dir, "forced_aha_by_step.csv"), index=False)

    print("\n=== Forced Aha Effect — Overall ===")
    for row in summary_rows:
        print(verdict_line(row))

    print("\nWrote:", os.path.join(out_dir, "forced_aha_summary.csv"))
    print("By-step:", os.path.join(out_dir, "forced_aha_by_step.csv"))

    plot_artifacts = PlotArtifacts(
        out_dir=out_dir,
        summary_rows=summary_rows,
        pairs=pairs,
        clusters=clusters,
        step_df=step_df,
    )
    _run_plots_if_requested(args, plot_artifacts)

    if not pairs.empty:
        print("Sample-level paired on keys:", ", ".join(key_cols))
    else:
        print("Sample-level pairing not available; reported only cluster metrics.")

if __name__ == "__main__":
    main()
