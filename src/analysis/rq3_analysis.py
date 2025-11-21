#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RQ3: Uncertainty and Intervention Effects
=========================================

This module centralizes the analysis code for RQ3:

  • Do second-pass / reconsideration steps help more when the model is
    uncertain?  (H3; GLMs over PASS-1 entropy buckets)
      → src/analysis/h3-analysis.py

  • How do uncertainty-bucket interventions behave and how can we export
    per-cue variants for custom plots?
      → src/analysis/uncertainty_bucket_effects.py
      → src/analysis/entropy_bin_regression.py
      → src/analysis/export_cue_variants.py

Rather than duplicating logic, this file provides a single entrypoint that
invokes the core H3 analysis and, optionally, exports a flat CSV for
per-cue analysis in pandas/R.

Typical usage (per domain/root)
-------------------------------

  python -m src.analysis.rq3_analysis \\
      artifacts/results/GRPO-1.5B-math-temp-0.05-3 \\
      --split test

This will populate:
  • <results_root>/rq3/h3_analysis   (H3 uncertainty×phase GLMs + plots)
  • <results_root>/rq3/cue_variants*.csv (if --export_cues is set)
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from src.analysis import export_cue_variants, h3_analysis


def _run_module_main_with_argv(module_main, argv: List[str], prog: str) -> None:
    """
    Invoke a module-style ``main`` with a synthetic ``sys.argv``, preserving
    the caller's arguments. This lets us reuse existing CLIs without
    modifying their internals.
    """
    old_argv = list(sys.argv)
    sys.argv = [prog] + argv
    try:
        module_main()
    finally:
        sys.argv = old_argv


def _run_h3_analysis(
    results_root: str,
    split: Optional[str],
    out_dir: str,
    uncertainty_field: str,
    num_buckets: int,
) -> None:
    """
    Run the H3 pass1/pass2×uncertainty GLMs and bucket plots.
    """
    h3_out = os.path.join(out_dir, "h3_analysis")
    os.makedirs(h3_out, exist_ok=True)

    argv: List[str] = [results_root]
    if split:
        argv += ["--split", split]
    argv += [
        "--out_dir",
        h3_out,
        "--uncertainty_field",
        uncertainty_field,
        "--num_buckets",
        str(int(num_buckets)),
    ]
    _run_module_main_with_argv(h3_analysis.main, argv, prog="h3_analysis.py")


def _export_cue_variants(
    results_root: str,
    split: Optional[str],
    out_dir: str,
    passes: List[str],
) -> None:
    """
    Use export_cue_variants to produce a flat CSV with one row per
    (sample, cue_variant) to support custom RQ3 plotting.
    """
    suffix = f"_{split}" if split else ""
    out_csv = os.path.join(out_dir, f"cue_variants{suffix}.csv")
    export_cue_variants.export_cue_variants(
        results_root=results_root,
        split_substr=split,
        out_csv=out_csv,
        passes=passes,
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "RQ3: Analyze how uncertainty and forced reconsideration interact "
            "with accuracy (H3 + cue-variant export)."
        ),
    )
    parser.add_argument(
        "results_root",
        help="Root containing step*/.../*.jsonl for a two-pass run.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Optional substring filter on filenames (e.g., 'test').",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Base output directory (default: <results_root>/rq3).",
    )
    parser.add_argument(
        "--uncertainty_field",
        default="entropy",
        choices=["entropy", "entropy_answer", "entropy_think"],
        help="Field to treat as PASS-1 uncertainty moderator (default: entropy).",
    )
    parser.add_argument(
        "--num_buckets",
        type=int,
        default=4,
        help="Number of equal-count uncertainty buckets (default: 4).",
    )
    parser.add_argument(
        "--no_h3",
        action="store_true",
        help="Skip the H3 GLM/bucket analysis.",
    )
    parser.add_argument(
        "--export_cues",
        action="store_true",
        help=(
            "Export a flat CSV with one row per (sample, cue_variant) using "
            "export_cue_variants."
        ),
    )
    parser.add_argument(
        "--passes",
        default="pass1,pass2,pass2a,pass2b,pass2c",
        help=(
            "Comma-separated pass keys to include when exporting cue variants "
            "(default: pass1,pass2,pass2a,pass2b,pass2c)."
        ),
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    out_base = args.out_dir or os.path.join(args.results_root, "rq3")
    os.makedirs(out_base, exist_ok=True)

    if not args.no_h3:
        _run_h3_analysis(
            results_root=args.results_root,
            split=args.split,
            out_dir=out_base,
            uncertainty_field=args.uncertainty_field,
            num_buckets=int(args.num_buckets),
        )

    if args.export_cues:
        passes = [
            p.strip()
            for p in (args.passes or "").split(",")
            if p.strip()
        ]
        if not passes:
            raise SystemExit("Must specify at least one pass key via --passes.")
        _export_cue_variants(
            results_root=args.results_root,
            split=args.split,
            out_dir=out_base,
            passes=passes,
        )


if __name__ == "__main__":
    main()
