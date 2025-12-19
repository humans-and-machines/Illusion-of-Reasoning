#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pooled logistic regression (correct ~ shift) for Qwen2.5-1.5B.

This script fits a Binomial GLM:

  correct ~ shift

over the same pooled subset used in qwen15b_pooled_shift_stats.py:
  - Domains: Crossword, Math, Rush Hour (Carpark).
  - Temperatures: 0, 0.05, 0.3, 0.7.
  - Steps: [0, 950] by default.
  - Split: test.
  - Shift definition: GPT-labeled shift (canonical or broad),
    optionally gated by words-based cues.

Example:

  python tools/qwen15b_pooled_shift_glm.py
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Iterable

import pandas as pd

# Ensure the repo root (containing the ``src`` package) is on sys.path when
# this script is invoked as ``python tools/qwen15b_pooled_shift_glm.py``.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.analysis.io import iter_records_from_file, scan_jsonl_files
from src.analysis.labels import aha_gpt
from src.analysis.metrics import extract_correct, lazy_import_statsmodels
from src.analysis.utils import extract_pass1_and_step, nat_step_from_path


QWEN15B_DOMAIN_ROOTS: Dict[str, Iterable[str]] = {
    "Crossword": (
        "artifacts/results/GRPO-1.5B-xword-temp-0",
        "artifacts/results/GRPO-1.5B-xword-temp-0.05",
        "artifacts/results/GRPO-1.5B-xword-temp-0.3",
        "artifacts/results/GRPO-1.5B-xword-temp-0.7",
    ),
    "Math": (
        "artifacts/results/GRPO-1.5B-math-temp-0.0",
        "artifacts/results/GRPO-1.5B-math-temp-0.05",
        "artifacts/results/GRPO-1.5B-math-temp-0.3",
        "artifacts/results/GRPO-1.5B-math-temp-0.7",
    ),
    "Rush Hour": (
        "artifacts/results/GRPO-1.5B-carpark-temp-0",
        "artifacts/results/GRPO-1.5B-carpark-temp-0.05",
        "artifacts/results/GRPO-1.5B-carpark-temp-0.3",
        "artifacts/results/GRPO-1.5B-carpark-temp-0.7",
    ),
}


def build_argparser() -> argparse.ArgumentParser:
    """
    Construct an argument parser for the pooled GLM script.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Fit a pooled logistic regression correct ~ shift over "
            "Qwen2.5-1.5B across Crossword, Math, and Rush Hour."
        ),
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Record-level split filter (default: 'test').",
    )
    parser.add_argument(
        "--min_step",
        type=int,
        default=0,
        help="Minimum training step (inclusive, default: 0).",
    )
    parser.add_argument(
        "--max_step",
        type=int,
        default=950,
        help="Maximum training step (inclusive, default: 950).",
    )
    parser.add_argument(
        "--gpt_mode",
        choices=["canonical", "broad"],
        default="canonical",
        help="GPT shift key set (default: canonical).",
    )
    parser.add_argument(
        "--gate_by_words",
        action="store_true",
        help="If set, require GPT shifts to be gated by words-based cues.",
    )
    return parser


def _load_pooled_dataframe(
    split_filter: str,
    min_step: int,
    max_step: int,
    gpt_mode: str,
    gate_by_words: bool,
) -> pd.DataFrame:
    """
    Load pooled (correct, shift) rows into a DataFrame.
    """
    rows = []
    for domain, roots in QWEN15B_DOMAIN_ROOTS.items():
        for root in roots:
            files = scan_jsonl_files(root, split_substr=None)
            for path in files:
                step_hint = nat_step_from_path(path)
                for record in iter_records_from_file(path):
                    if split_filter and str(record.get("split", "")).lower() != split_filter:
                        continue
                    pass1, step = extract_pass1_and_step(record, step_hint)
                    if not pass1 or step is None:
                        continue
                    if step < min_step or step > max_step:
                        continue

                    correct_flag = extract_correct(pass1, record)
                    if correct_flag is None:
                        continue

                    shift_flag = aha_gpt(
                        pass1,
                        record,
                        mode=gpt_mode,
                        gate_by_words=gate_by_words,
                        domain=domain,
                    )
                    rows.append(
                        {
                            "correct": int(correct_flag),
                            "shift": int(shift_flag),
                        },
                    )
    if not rows:
        raise SystemExit("No rows found for pooled GLM. Check roots/split/step filters.")
    return pd.DataFrame(rows)


def main() -> None:
    """
    Fit correct ~ shift and print the GLM summary and key statistics.
    """
    parser = build_argparser()
    args = parser.parse_args()

    df = _load_pooled_dataframe(
        split_filter=str(args.split or ""),
        min_step=args.min_step,
        max_step=args.max_step,
        gpt_mode=args.gpt_mode,
        gate_by_words=bool(args.gate_by_words),
    )

    print(f"N_total = {len(df)}  N_shift = {int(df['shift'].sum())}")

    sm, smf = lazy_import_statsmodels()
    model = smf.glm("correct ~ shift", data=df, family=sm.families.Binomial())
    res = model.fit()

    print("\nGLM summary (correct ~ shift):")
    print(res.summary())

    coef = float(res.params.get("shift", float("nan")))
    se = float(res.bse.get("shift", float("nan")))
    p_value = float(res.pvalues.get("shift", float("nan")))

    print("\ncoef(shift) =", coef)
    print("se(shift)   =", se)
    print("p(shift)    =", p_value)


if __name__ == "__main__":
    main()
