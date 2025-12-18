#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistic 1: pooled shift rate and conditional accuracies.

Given an ``aha_conditionals__*.csv`` file produced by ``table_1.py``,
compute the across-domain:

  - shift_rate = N_shift / N_total
  - P(correct | shift)
  - P(correct | no shift)

This is a small wrapper to reproduce the aggregated “across all domains”
numbers used in the paper.
"""

from __future__ import annotations

import argparse
import os
import traceback
from typing import List, Dict

import pandas as pd

from src.analysis.io import scan_jsonl_files
from src.analysis.metrics import lazy_import_statsmodels, make_carpark_success_fn
from src.analysis.script_utils import ensure_script_context
from src.analysis.table_1 import GptAhaConfig, LoadRowsConfig, load_rows
from src.analysis.utils import gpt_keys_for_mode


ensure_script_context()


def build_argparser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser.

    Usage:
      python -m src.analysis.statistic_1 path/to/aha_conditionals__*.csv
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compute pooled shift rate and conditional accuracies from a "
            "table_1 aha_conditionals CSV."
        ),
    )
    parser.add_argument(
        "csv_path",
        help="Path to aha_conditionals__<dataset>__<model>.csv produced by table_1.py.",
    )
    parser.add_argument(
        "--root_math",
        default=None,
        help="Optional Math results root for pooled logistic regression (e.g., GRPO-1.5B-math-temp-0.05).",
    )
    parser.add_argument(
        "--root_crossword",
        default=None,
        help="Optional Crossword results root for pooled logistic regression.",
    )
    parser.add_argument(
        "--root_carpark",
        default=None,
        help="Optional Carpark results root for pooled logistic regression.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Substring filter for JSONL filenames when running the pooled logistic regression (default: 'test').",
    )
    return parser


def _print_pooled_counts(csv_path: str) -> None:
    """
    Load an aha_conditionals CSV, aggregate across domains, and print pooled counts.
    """
    df = pd.read_csv(csv_path)
    n_total = int(df["n_total"].sum())
    n_shift = int(df["n_shift"].sum())
    k_correct_shift = int(df["k_correct_shift"].sum())
    k_correct_noshift = int(df["k_correct_noshift"].sum())

    n_noshift = n_total - n_shift

    shift_rate = (n_shift / n_total) if n_total > 0 else float("nan")
    p_shift = (k_correct_shift / n_shift) if n_shift > 0 else float("nan")
    p_noshift = (k_correct_noshift / n_noshift) if n_noshift > 0 else float("nan")

    print(f"N_total = {n_total}")
    print(f"N_shift = {n_shift} (shift_rate = {shift_rate:.6f})")
    print(f"P(correct | shift)    = {p_shift:.6f}")
    print(f"P(correct | no shift) = {p_noshift:.6f}")


def _run_pooled_logistic(args: argparse.Namespace) -> None:
    """
    Run a single pooled logistic regression over all provided roots:

        correct ~ shift

    where ``shift`` is the GPT-labeled reasoning shift used in ``table_1``.
    This is the logistic analogue of the pooled shift vs no-shift accuracy
    comparison (2.6% vs 17.2% in the text).
    """
    # Build per-domain file lists using the same conventions as table_1.
    domain_roots: Dict[str, str] = {}
    if args.root_crossword:
        domain_roots["Crossword"] = args.root_crossword
    if args.root_math:
        domain_roots["Math"] = args.root_math
    if args.root_carpark:
        domain_roots["Carpark"] = args.root_carpark

    if not domain_roots:
        return

    print("\n[statistic_1] Running pooled logistic regression over roots:")
    for dom, root in domain_roots.items():
        print(f"  - {dom}: {os.path.abspath(root)}")

    files_by_domain: Dict[str, List[str]] = {}
    total_files = 0
    for dom, root in domain_roots.items():
        files = scan_jsonl_files(root, split_substr=args.split)
        if files:
            files_by_domain[dom] = files
            total_files += len(files)

    if not files_by_domain:
        print("[statistic_1] No JSONL files found for pooled logistic regression.")
        return
    print(f"[statistic_1] Found {total_files} JSONL files across {len(files_by_domain)} domains (split={args.split!r}).")

    # Match table_1 defaults: canonical GPT labels, native cue gating, standard
    # Carpark success policy, steps <= 1000.
    gpt_config = GptAhaConfig(
        gpt_keys=gpt_keys_for_mode("canonical"),
        gpt_subset_native=True,
        allow_judge_cues_non_xword=False,
        broad_counts_marker_lists=False,
    )
    load_config = LoadRowsConfig(
        gpt_config=gpt_config,
        min_step=None,
        max_step=1000,
        carpark_success_fn=make_carpark_success_fn("gt", 0.1),
        debug=False,
    )

    rows_df = load_rows(files_by_domain, load_config)
    print(f"[statistic_1] Loaded {len(rows_df)} rows for pooled GLM.")

    # Quick sanity check: pooled shift prevalence and conditional accuracies.
    n_total = int(len(rows_df))
    n_shift = int((rows_df["shift"] == 1).sum())
    n_noshift = n_total - n_shift
    k_shift = int(rows_df.loc[rows_df["shift"] == 1, "correct"].sum()) if n_shift > 0 else 0
    k_noshift = int(rows_df.loc[rows_df["shift"] == 0, "correct"].sum()) if n_noshift > 0 else 0
    if n_shift > 0 and n_noshift > 0:
        p_shift = k_shift / n_shift
        p_noshift = k_noshift / n_noshift
        print(
            f"[statistic_1] Sanity: shift_rate={n_shift/n_total:.6f}, "
            f"P(correct|shift)={p_shift:.6f}, P(correct|no shift)={p_noshift:.6f}",
        )

    # Fit Binomial GLM: correct ~ shift
    print("[statistic_1] Fitting pooled GLM correct ~ shift (this may take a moment)...")
    try:
        sm, smf = lazy_import_statsmodels()
        model = smf.glm(
            "correct ~ shift",
            data=rows_df,
            family=sm.families.Binomial(),
        )
        result = model.fit()
    except Exception as exc:  # pragma: no cover - defensive logging
        print(
            "[statistic_1] WARNING: pooled logistic regression failed; "
            "environment may lack a compatible statsmodels/SciPy stack.",
        )
        print(f"[statistic_1] Exception: {exc!r}")
        traceback.print_exc()
        return
    print("[statistic_1] Finished fitting pooled GLM.")

    coef = float(result.params.get("shift", float("nan")))
    se = float(result.bse.get("shift", float("nan")))
    p_value = float(result.pvalues.get("shift", float("nan")))

    print("\nPooled logistic regression (all domains):")
    print("  Model: correct ~ shift")
    print(f"  coef(shift) = {coef:.6f}")
    print(f"  se(shift)   = {se:.6f}")
    print(f"  p-value     = {p_value:.3g}")


def main() -> None:
    """
    CLI entry point: aggregate counts from the CSV and, optionally,
    run a pooled logistic regression over Math/Xword/Carpark roots.
    """
    parser = build_argparser()
    args = parser.parse_args()

    _print_pooled_counts(args.csv_path)
    _run_pooled_logistic(args)


if __name__ == "__main__":
    main()
