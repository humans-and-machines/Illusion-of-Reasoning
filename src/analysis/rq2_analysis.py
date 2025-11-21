#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RQ2: Training Stage and Temperature Effects
==========================================

This module provides a small orchestration layer for the analyses that
underpin RQ2 in the paper:

  • Training-stage effects of shifts (per-step GLMs, uncertainty buckets):
      → src/analysis/h2-analysis.py
  • Temperature-wise effects of shifts (raw deltas + GLMs):
      → src/analysis/temperature_effects.py

Rather than duplicating modeling code, this file gives a single, RQ2-branded
entrypoint that forwards to the existing scripts with a consistent set of
arguments.

Typical usage (per domain/root)
-------------------------------

  python -m src.analysis.rq2_analysis \\
      artifacts/results/GRPO-1.5B-math-temp-0.05-3 \\
      --split test

By default this will:
  1) run h2-analysis.py with its default settings, writing into
     <results_root>/rq2/h2_analysis
  2) (optionally) invoke temperature_effects.py if you pass --temp_root
     pointing at a directory containing per-temperature runs.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from src.analysis import h2_analysis, temperature_effects


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


def _run_h2_analysis(
    results_root: str,
    split: Optional[str],
    out_dir: str,
) -> None:
    """
    Run the H2 per-step GLMs and uncertainty-bucket plots.
    """
    h2_out = os.path.join(out_dir, "h2_analysis")
    os.makedirs(h2_out, exist_ok=True)

    argv: List[str] = [results_root]
    if split:
        argv += ["--split", split]
    argv += ["--out_dir", h2_out]
    _run_module_main_with_argv(h2_analysis.main, argv, prog="h2_analysis.py")


def _run_temperature_effects(
    temp_root: str,
    split: Optional[str],
    out_dir: str,
    low_alias: float,
) -> None:
    """
    Run the temperature_effects.py script on a directory that encodes
    temperatures in its subdirectory names (as used in the paper).

    This is intentionally thin: most configuration (which domains, extra
    Math-2 run, labels) is left to temperature_effects.py itself.
    """
    te_out = os.path.join(out_dir, "temperature_effects")
    os.makedirs(te_out, exist_ok=True)

    argv: List[str] = [temp_root]
    if split:
        argv += ["--split", split]
    argv += ["--out_dir", te_out, "--low_alias", str(low_alias)]
    _run_module_main_with_argv(temperature_effects.main, argv, prog="temperature_effects.py")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "RQ2: Analyze how the effect of reasoning shifts varies with "
            "training stage and decoding temperature."
        ),
    )
    parser.add_argument(
        "results_root",
        help="Root containing step*/.../*.jsonl for a fixed temperature run.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Optional substring filter on filenames (e.g., 'test').",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Base output directory (default: <results_root>/rq2).",
    )
    parser.add_argument(
        "--no_stage",
        action="store_true",
        help="Skip the per-step H2 analysis (training-stage GLMs).",
    )
    parser.add_argument(
        "--temp_root",
        default=None,
        help=(
            "Optional root directory containing per-temperature runs for the "
            "same model/domain (used by temperature_effects.py). "
            "If omitted, the temperature analysis step is skipped."
        ),
    )
    parser.add_argument(
        "--low_alias",
        type=float,
        default=0.0,
        help=(
            "Numerical alias for 'low' temperatures in directory names "
            "(default: 0.0)."
        ),
    )
    parser.add_argument(
        "--no_temp",
        action="store_true",
        help="Skip the temperature_effects analysis even if --temp_root is set.",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    out_base = args.out_dir or os.path.join(args.results_root, "rq2")
    os.makedirs(out_base, exist_ok=True)

    if not args.no_stage:
        _run_h2_analysis(
            results_root=args.results_root,
            split=args.split,
            out_dir=out_base,
        )

    if args.temp_root and not args.no_temp:
        _run_temperature_effects(
            temp_root=args.temp_root,
            split=args.split,
            out_dir=out_base,
            low_alias=float(args.low_alias),
        )


if __name__ == "__main__":
    main()
