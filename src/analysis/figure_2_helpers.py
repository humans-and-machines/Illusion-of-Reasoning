#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Uncertainty → Correctness: Counts, Densities, Accuracy, and Regression (All in One)
Times (12 pt) + optional exact A4 PDFs
-------------------------------------------------------------------------------

Figures written:
  1) unc_vs_correct_4hists__<DS>__<MODEL>.png/.pdf
     # 4-panel COUNT(CORRECT) histograms
  2) unc_vs_correct_overlaid__<DS>__<MODEL>.png/.pdf
     # overlaid densities (CORRECT only)
  3) unc_vs_corr_incorr_by_type__<DS>__<MODEL>.png/.pdf
     # 2×2 CORRECT vs INCORRECT (All, Words, LLM, Formal)
  4) unc_accuracy_by_bin__<DS>__<MODEL>.png/.pdf
     # per-bin accuracy with Wilson 95% CIs
  5) acc_vs_uncertainty_regression__<DS>__<MODEL>.png/.pdf
     # GLM: correct ~ C(problem)+C(step)+aha:C(perplexity_bucket)

CSVs:
  - unc_vs_correct_4hists.csv
  - unc_vs_correct_overlaid.csv
  - unc_vs_corr_incorr_by_type.csv
  - unc_accuracy_by_bin.csv
  - acc_vs_uncertainty_regression.csv

Use --a4_pdf to save exact A4 PDFs; text is Times/Times New Roman (12 pt).
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.analysis.common.parser_helpers import standard_results_parser
from src.analysis.core.plotting_helpers import set_global_fonts
from src.analysis.figure_2_accuracy import AccuracyPlotConfig, plot_accuracy_by_bin_overlay
from src.analysis.figure_2_data import _load_all_samples, _standardize_uncertainty, make_edges_from_std
from src.analysis.figure_2_density import (
    DensityPlotConfig,
    FourHistConfig,
    plot_correct_incorrect_by_type,
    plot_four_correct_hists,
    plot_overlaid_densities,
)
from src.analysis.figure_2_regression import RegressionOutputConfig, RegressionPlotConfig, plot_regression_curves
from src.analysis.utils import (
    add_formal_threshold_args,
    add_gpt_mode_arg,
    add_run_rq2_flag,
    add_uncertainty_field_arg,
    slugify,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct and return the CLI argument parser."""
    parser = standard_results_parser()

    add_uncertainty_field_arg(parser)
    add_gpt_mode_arg(parser)
    parser.add_argument(
        "--gpt_gate_by_words",
        action="store_true",
        help="Gate GPT by Words (LLM≤Words).",
    )

    add_formal_threshold_args(parser)
    parser.add_argument("--delta3", type=float, default=None)

    parser.add_argument("--hist_bins", type=int, default=10)
    parser.add_argument("--density_bins", type=int, default=10)
    parser.add_argument("--acc_bins", type=int, default=10)
    parser.add_argument(
        "--ppx_buckets",
        type=int,
        default=8,
        help="Perplexity quantile buckets for regression.",
    )
    parser.add_argument(
        "--smooth_bins",
        type=int,
        default=5,
        help="Moving-average window for densities (0=off).",
    )
    parser.add_argument(
        "--xlim_std",
        nargs=2,
        type=float,
        default=None,
        help="Explicit x-limits for uncertainty_std (e.g., --xlim_std -1.0 3.0)",
    )

    parser.add_argument("--font_family", default="Times New Roman")
    parser.add_argument("--font_size", type=int, default=12)
    parser.add_argument(
        "--a4_pdf",
        action="store_true",
        help="Save PDFs at exact A4 page size.",
    )
    parser.add_argument(
        "--a4_orientation",
        choices=["landscape", "portrait"],
        default="landscape",
    )

    parser.add_argument("--B_ci", type=int, default=500)

    add_run_rq2_flag(
        parser,
        "for this results_root before building these uncertainty figures.",
    )
    parser.add_argument(
        "--rq2_dir",
        default=None,
        help=(
            "Optional override for the directory containing H2/RQ2 outputs. "
            "If not set, defaults to <results_root>/rq2/h2_analysis."
        ),
    )
    return parser


def run_uncertainty_figures(args: argparse.Namespace) -> None:
    """Run the full uncertainty/correctness figure pipeline."""
    set_global_fonts(args.font_family, args.font_size)

    out_dir = args.out_dir or os.path.join(args.results_root, "unc_correct_all")
    os.makedirs(out_dir, exist_ok=True)
    slug = f"{slugify(args.dataset_name)}__{slugify(args.model_name)}"
    title_suffix = f"{args.dataset_name}, {args.model_name}"

    d_all = _load_all_samples(args)

    d_all = _standardize_uncertainty(d_all)
    edges_hist = make_edges_from_std(
        d_all["uncertainty_std"].to_numpy(),
        bins=int(args.hist_bins),
        xlim=args.xlim_std,
    )
    edges_den = make_edges_from_std(
        d_all["uncertainty_std"].to_numpy(),
        bins=int(args.density_bins),
        xlim=args.xlim_std,
    )
    edges_acc = make_edges_from_std(
        d_all["uncertainty_std"].to_numpy(),
        bins=int(args.acc_bins),
        xlim=args.xlim_std,
    )

    results_summary: Dict[str, Tuple[str, str, Optional[str]]] = {}

    out_png = os.path.join(out_dir, f"unc_vs_correct_4hists__{slug}.png")
    out_pdf = os.path.join(out_dir, f"unc_vs_correct_4hists__{slug}.pdf")
    plot_four_correct_hists(
        d_all,
        FourHistConfig(
            edges=edges_hist,
            out_png=out_png,
            out_pdf=out_pdf,
            title_suffix=title_suffix,
            a4_pdf=bool(args.a4_pdf),
            a4_orientation=args.a4_orientation,
        ),
    )
    results_summary["counts"] = (out_png, out_pdf, None)

    out_png = os.path.join(out_dir, f"unc_vs_correct_overlaid__{slug}.png")
    out_pdf = os.path.join(out_dir, f"unc_vs_correct_overlaid__{slug}.pdf")
    csv_path = plot_overlaid_densities(
        d_all,
        edges_den,
        DensityPlotConfig(
            out_png=out_png,
            out_pdf=out_pdf,
            title_suffix=title_suffix,
            smooth_bins=int(args.smooth_bins),
            a4_pdf=bool(args.a4_pdf),
            a4_orientation=args.a4_orientation,
        ),
    )
    results_summary["density"] = (out_png, out_pdf, csv_path)

    out_png = os.path.join(out_dir, f"unc_vs_corr_incorr_by_type__{slug}.png")
    out_pdf = os.path.join(out_dir, f"unc_vs_corr_incorr_by_type__{slug}.pdf")
    csv_path = plot_correct_incorrect_by_type(
        d_all,
        edges_den,
        DensityPlotConfig(
            out_png=out_png,
            out_pdf=out_pdf,
            title_suffix=title_suffix,
            smooth_bins=int(args.smooth_bins),
            a4_pdf=bool(args.a4_pdf),
            a4_orientation=args.a4_orientation,
        ),
    )
    results_summary["correct_incorrect"] = (out_png, out_pdf, csv_path)

    out_png = os.path.join(out_dir, f"unc_accuracy_by_bin__{slug}.png")
    out_pdf = os.path.join(out_dir, f"unc_accuracy_by_bin__{slug}.pdf")
    csv_path = plot_accuracy_by_bin_overlay(
        d_all,
        edges_acc,
        AccuracyPlotConfig(
            out_png=out_png,
            out_pdf=out_pdf,
            title_suffix=title_suffix,
            a4_pdf=bool(args.a4_pdf),
            a4_orientation=args.a4_orientation,
        ),
    )
    results_summary["accuracy"] = (out_png, out_pdf, csv_path)

    d_all["perplexity"] = np.exp(d_all["uncertainty"].astype(float))
    d_all["perplexity_bucket"] = pd.qcut(
        d_all["perplexity"],
        q=int(max(3, args.ppx_buckets)),
        duplicates="drop",
    ).astype("category")

    out_png = os.path.join(out_dir, f"acc_vs_uncertainty_regression__{slug}.png")
    out_pdf = os.path.join(out_dir, f"acc_vs_uncertainty_regression__{slug}.pdf")
    csv_path = plot_regression_curves(
        RegressionPlotConfig(
            frame=d_all,
            ppx_bucket_column="perplexity_bucket",
            dataset=args.dataset_name,
            model_name=args.model_name,
            num_bootstrap_samples=int(args.B_ci),
            output=RegressionOutputConfig(
                out_png=out_png,
                out_pdf=out_pdf,
                title_suffix=title_suffix,
                a4_pdf=bool(args.a4_pdf),
                a4_orientation=args.a4_orientation,
            ),
        ),
    )
    results_summary["regression"] = (out_png, out_pdf, csv_path)

    print("WROTE:")
    print("  (1) Panels (counts):", *results_summary["counts"][:2])
    print(
        "  (2) Overlaid density:",
        *results_summary["density"][:2],
        " CSV:",
        results_summary["density"][2],
    )
    print(
        "  (3) Correct vs Incorrect (2×2):",
        *results_summary["correct_incorrect"][:2],
        " CSV:",
        results_summary["correct_incorrect"][2],
    )
    print(
        "  (4) Accuracy by bin:",
        *results_summary["accuracy"][:2],
        " CSV:",
        results_summary["accuracy"][2],
    )
    print(
        "  (5) Regression curves:",
        *results_summary["regression"][:2],
        " CSV:",
        results_summary["regression"][2],
    )
