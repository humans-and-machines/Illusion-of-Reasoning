#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
H2: Are Aha! Moments Important During Different Stages of Training?
-------------------------------------------------------------------

This module now focuses on wiring the CLI, plotting, and reporting layers.
Data loading and GLM helpers live in ``h2_analysis_loader.py`` and
``h2_analysis_glm.py`` so each piece can evolve independently.
"""

from __future__ import annotations

import argparse
import importlib
import os
from typing import List, Optional

import pandas as pd

from src.analysis.common.uncertainty import standardize_uncertainty_with_stats
from src.analysis.h2_analysis_glm import StepwiseGlmConfig, compute_pooled_step_effects, fit_stepwise_glms
from src.analysis.h2_analysis_loader import load_pass1_rows
from src.analysis.io import scan_jsonl_files


def _configure_matplotlib() -> None:
    """Ensure matplotlib uses the Agg backend before importing pyplot modules."""
    matplotlib_module = importlib.import_module("matplotlib")
    matplotlib_module.use("Agg")


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", help="Root containing step*/.../*.jsonl")
    parser.add_argument(
        "--split",
        default=None,
        help="Filter filenames by substring (e.g., 'test').",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output dir (default: <results_root>/h2_analysis)",
    )
    parser.add_argument("--min_step", type=int, default=None)
    parser.add_argument("--max_step", type=int, default=None)
    parser.add_argument(
        "--unc_field",
        choices=["answer", "overall", "think"],
        default="answer",
        help="Which entropy field to use as uncertainty (default: answer entropy).",
    )
    parser.add_argument(
        "--aha_source",
        choices=["gpt", "native"],
        default="gpt",
        help="Prefer GPT-labeled shift (default) or native reconsider cue.",
    )
    parser.add_argument(
        "--interaction",
        action="store_true",
        help="Include aha×uncertainty_std interaction.",
    )
    parser.add_argument(
        "--compare_native",
        action="store_true",
        help="Also fit and plot using native aha labels.",
    )
    parser.add_argument(
        "--penalty",
        choices=["none", "ridge", "firth"],
        default="ridge",
        help='Penalty for step-wise GLMs; "firth" currently falls back to ridge.',
    )
    parser.add_argument(
        "--ridge_l2",
        "--l2",
        dest="ridge_l2",
        type=float,
        default=1.0,
        help="Ridge strength for stabilized GLMs (alias --l2 kept for compatibility).",
    )
    parser.add_argument(
        "--bootstrap_ame",
        type=int,
        default=200,
        help="Bootstrap reps for AME CIs (per step).",
    )
    parser.add_argument(
        "--ame_grid",
        type=int,
        default=9,
        help="Number of u grid points in [-2,2] for AME(u).",
    )
    parser.add_argument(
        "--fdr_alpha",
        type=float,
        default=0.05,
        help="BH/FDR alpha for step-wise aha p-values.",
    )
    parser.add_argument(
        "--unc_buckets",
        type=int,
        default=6,
        help="Number of quantile buckets in the all-3 Aha ratio figure.",
    )
    parser.add_argument(
        "--hist_bins",
        type=int,
        default=100,
        help="Number of bins for uncertainty histogram with Aha counts.",
    )
    parser.add_argument(
        "--delta1",
        type=float,
        default=0.13,
        help="Formal δ1 prior-failure threshold.",
    )
    parser.add_argument(
        "--delta2",
        type=float,
        default=0.13,
        help="Formal δ2 prior-stability threshold.",
    )
    parser.add_argument(
        "--delta3",
        type=float,
        default=None,
        help="Optional Formal gain-at-shift; require P(correct|shift)-P(correct) > δ3.",
    )
    parser.add_argument(
        "--min_prior_steps",
        type=int,
        default=2,
        help="Formal: require at least this many prior steps.",
    )
    parser.add_argument(
        "--gpt_gate_by_words",
        action="store_true",
        help="Gate GPT shifts by Words cue (so LLM<=Words).",
    )
    parser.add_argument("--dataset_name", default="MATH-500")
    parser.add_argument("--model_name", default="Qwen2.5-1.5B")
    return parser


def _subset_by_step(
    pass1_df: pd.DataFrame,
    min_step: Optional[int],
    max_step: Optional[int],
) -> pd.DataFrame:
    """Filter rows by ``min_step``/``max_step`` when provided."""
    if min_step is not None:
        pass1_df = pass1_df[pass1_df["step"] >= min_step]
    if max_step is not None:
        pass1_df = pass1_df[pass1_df["step"] <= max_step]
    return pass1_df


def _diagnostic_plots(pass1_df: pd.DataFrame, reg_df: pd.DataFrame, out_dir: str) -> None:
    plotting = importlib.import_module("src.analysis.h2_plotting")

    plotting.plot_diag_panel(pass1_df, out_dir)
    if reg_df.empty:
        return
    plotting.lineplot(
        reg_df["step"],
        reg_df["aha_coef"],
        (
            "Training step",
            "β(aha)",
            "Aha coefficient vs. step",
            os.path.join(out_dir, "aha_coef_vs_step.png"),
        ),
    )
    plotting.lineplot(
        reg_df["step"],
        reg_df["aha_ame"],
        (
            "Training step",
            "AME(aha)",
            "Aha average marginal effect vs. step",
            os.path.join(out_dir, "aha_ame_vs_step.png"),
        ),
    )
    plotting.lineplot(
        reg_df["step"],
        reg_df["unc_coef"],
        (
            "Training step",
            "β(uncertainty_std)",
            "Uncertainty coefficient vs. step",
            os.path.join(out_dir, "uncertainty_coef_vs_step.png"),
        ),
    )
    plotting.lineplot(
        reg_df["step"],
        reg_df["naive_delta"],
        (
            "Training step",
            "Δ accuracy (aha=1 − aha=0)",
            "Naïve Δaccuracy vs. step",
            os.path.join(out_dir, "naive_delta_vs_step.png"),
        ),
    )
    plotting.plot_ame_with_ci(reg_df, out_dir)


def _run_uncertainty_figures(files: List[str], out_dir: str, args: argparse.Namespace) -> None:
    """Render bucket and histogram figures (all Aha variants)."""
    helpers = importlib.import_module("src.analysis.core.h2_uncertainty_helpers")

    buckets_png, buckets_csv, d_all, ps_formal = helpers.make_all3_uncertainty_buckets_figure(
        files=files,
        out_dir=out_dir,
        args=args,
    )
    print("Buckets figure:", buckets_png)
    print("Buckets CSV   :", buckets_csv)
    hist_png, hist_csv = helpers.plot_uncertainty_hist_100bins(
        d_all=d_all,
        problem_step_df=ps_formal,
        out_dir=out_dir,
        args=args,
    )
    print("Histogram (100 bins):", hist_png)
    print("Histogram CSV       :", hist_csv)


def _plot_pooled_effects(pooled_df: pd.DataFrame, out_dir: str) -> None:
    """Persist pooled aha effects to CSV + a simple line plot."""
    if pooled_df.empty:
        return

    pooled_csv = os.path.join(out_dir, "h2_pooled_aha_by_step.csv")
    pooled_df.to_csv(pooled_csv, index=False)

    _configure_matplotlib()
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    fig, axis = plt.subplots(figsize=(7.8, 4.6), dpi=140)
    axis.plot(pooled_df["step"], pooled_df["aha_effect"], marker="o")
    axis.set_xlabel("Training step")
    axis.set_ylabel("Pooled effect of aha (log-odds)")
    axis.set_title("Pooled GLM (step FE): per-step aha effect (ridge)")
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "h2_pooled_aha_by_step.png"))
    plt.close(fig)


def _summarize_outputs(out_dir: str, samples_csv: str) -> None:
    """Print a quick recap of the primary outputs."""
    print(f"Wrote samples CSV: {samples_csv}")
    print(f"Wrote step regression CSV: {os.path.join(out_dir, 'h2_step_regression.csv')}")
    print("Plots written:")
    print("  h2_diag_panel.png")
    print("  aha_coef_vs_step.png, aha_ame_vs_step.png, uncertainty_coef_vs_step.png, naive_delta_vs_step.png")
    print("  aha_ame_with_ci.png, aha_ame_grid.png (if grid used elsewhere)")
    print("  h2_pooled_aha_by_step.png")
    print("  h2_aha_vs_uncertainty_buckets__*.png/.pdf, h2_uncertainty_hist_100bins__*.png/.pdf")
    print("CSVs:")
    print("  h2_balance_by_step.csv, h2_ame_grid.csv, h2_fdr_summary.txt, h2_pooled_aha_by_step.csv")
    print("  h2_aha_vs_uncertainty_buckets.csv, h2_uncertainty_hist_100bins.csv")


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the H2 training-stage analysis."""
    out_dir = args.out_dir or os.path.join(args.results_root, "h2_analysis")
    os.makedirs(out_dir, exist_ok=True)

    files = scan_jsonl_files(args.results_root, args.split)
    if not files:
        raise SystemExit("No JSONL files found. Check path or --split.")

    _configure_matplotlib()

    pass1_df = load_pass1_rows(files, args.unc_field, args.aha_source)
    pass1_df = _subset_by_step(pass1_df, args.min_step, args.max_step)
    if pass1_df.empty:
        raise SystemExit("No rows left after step filtering.")

    pass1_df, _, _ = standardize_uncertainty_with_stats(pass1_df)
    samples_csv = os.path.join(out_dir, "h2_pass1_samples.csv")
    pass1_df.to_csv(samples_csv, index=False)

    glm_config = StepwiseGlmConfig(
        out_dir=out_dir,
        interaction=bool(args.interaction),
        penalty=str(args.penalty),
        ridge_l2=float(args.ridge_l2),
        bootstrap_ame=int(args.bootstrap_ame),
        ame_grid=int(args.ame_grid),
        fdr_alpha=float(args.fdr_alpha),
    )
    reg_df = fit_stepwise_glms(pass1_df, glm_config)
    _diagnostic_plots(pass1_df, reg_df, out_dir)
    _run_uncertainty_figures(files, out_dir, args)
    pooled_df = compute_pooled_step_effects(pass1_df, ridge_l2=float(args.ridge_l2))
    _plot_pooled_effects(pooled_df, out_dir)

    if args.compare_native:
        native_df = load_pass1_rows(files, args.unc_field, "native")
        native_df = _subset_by_step(native_df, args.min_step, args.max_step)
        native_df, _, _ = standardize_uncertainty_with_stats(native_df)
        native_csv = os.path.join(out_dir, "h2_pass1_samples_native.csv")
        native_df.to_csv(native_csv, index=False)

    _summarize_outputs(out_dir, samples_csv)


def main() -> None:
    """CLI entrypoint."""
    parser = build_arg_parser()
    run_pipeline(parser.parse_args())


if __name__ == "__main__":
    main()
