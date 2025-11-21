#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RQ1: Do reasoning shifts raise model accuracy?
=============================================

This module provides a thin orchestration layer around the existing H1 / shift
analysis code so that the pipeline for RQ1 lives in a single, clearly named
entrypoint.

It does **not** reimplement the heavy GLM logic; instead it:

- calls ``h1-analysis.py`` to fit the Binomial GLMs
  (``correct ~ C(problem) + step_std + aha``) and write:
    * ``h1_glm_ame_summary.csv``
    * ``h1_group_accuracy.csv``
    * ``h1_group_accuracy_delta.csv``
    * ``h1_group_accuracy_by_step.csv``
- optionally calls ``shift_summary.summarize_root`` to print an accuracy×shift
  2×2 table over the same results.

Typical usage (from the repo root)
----------------------------------

Math example:

  python -m src.analysis.rq1_analysis \\
      artifacts/results/GRPO-1.5B-math-temp-0.05-3 \\
      --split test \\
      --dataset_name MATH-500 \\
      --model_name Qwen2.5-1.5B

Crossword example:

  python -m src.analysis.rq1_analysis \\
      artifacts/results/cryptic-xword-run \\
      --split test \\
      --dataset_name Xword \\
      --model_name Qwen2.5-1.5B

This will create a ``rq1/`` subfolder under the results root and place the H1
GLM outputs under ``rq1/h1_glm``.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

from src.analysis import h1_analysis, shift_summary


def _run_module_main_with_argv(module_main, argv: List[str]) -> None:
    """
    Invoke a module-style ``main`` with a synthetic ``sys.argv``, preserving
    the caller's arguments. This lets us reuse the existing CLI in
    ``h1_analysis`` without modifying its internals.
    """
    old_argv = list(sys.argv)
    sys.argv = ["h1_analysis.py"] + argv
    try:
        module_main()
    finally:
        sys.argv = old_argv


def _run_h1_glm(
    results_root: str,
    split: str | None,
    out_dir: str,
    dataset_name: str,
    model_name: str,
) -> None:
    """
    Delegate to the H1 GLM script (now ``h1_analysis.py``) with a controlled
    argument set.
    """
    h1_out = os.path.join(out_dir, "h1_glm")
    os.makedirs(h1_out, exist_ok=True)

    argv: List[str] = [results_root]
    if split:
        argv += ["--split", split]
    argv += [
        "--out_dir",
        h1_out,
        "--dataset_name",
        dataset_name,
        "--model_name",
        model_name,
    ]
    _run_module_main_with_argv(h1_analysis.main, argv)


def _run_shift_summary(
    results_root: str,
    split: str | None,
    max_examples: int | None,
) -> None:
    """
    Call the lightweight 2×2 correctness×shift summary on pass1.
    """
    root_abs = os.path.abspath(results_root)
    shift_summary.summarize_root(root_abs, split=split, max_examples=max_examples)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "RQ1: Do reasoning shifts raise model accuracy? "
            "Runs the H1 GLM pipeline and, optionally, the simple shift summary."
        ),
    )
    parser.add_argument(
        "results_root",
        help="Root directory containing step*/.../*.jsonl (e.g., artifacts/results/GRPO-1.5B-math-temp-0.05-3).",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Optional substring filter on filenames (e.g., 'test').",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Base output directory (default: <results_root>/rq1).",
    )
    parser.add_argument(
        "--dataset_name",
        default="MATH-500",
        help="Dataset label for GLM outputs (default: MATH-500).",
    )
    parser.add_argument(
        "--model_name",
        default="Qwen2.5-1.5B",
        help="Model label for GLM outputs (default: Qwen2.5-1.5B).",
    )
    parser.add_argument(
        "--no_h1_glm",
        action="store_true",
        help="Skip the H1 GLM run (only run shift_summary).",
    )
    parser.add_argument(
        "--no_shift_summary",
        action="store_true",
        help="Skip the shift_summary 2×2 table (only run H1 GLM).",
    )
    parser.add_argument(
        "--shift_max_examples",
        type=int,
        default=0,
        help=(
            "Maximum number of individual shift examples to print in shift_summary "
            "(default: 0, i.e., summary only)."
        ),
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    out_base = args.out_dir or os.path.join(args.results_root, "rq1")
    os.makedirs(out_base, exist_ok=True)

    if not args.no_h1_glm:
        _run_h1_glm(
            results_root=args.results_root,
            split=args.split,
            out_dir=out_base,
            dataset_name=args.dataset_name,
            model_name=args.model_name,
        )

    if not args.no_shift_summary:
        _run_shift_summary(
            results_root=args.results_root,
            split=args.split,
            max_examples=args.shift_max_examples,
        )


if __name__ == "__main__":
    main()
