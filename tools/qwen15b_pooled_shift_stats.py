#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pooled shift prevalence and conditional accuracy for Qwen2.5-1.5B.

This script reproduces the aggregate statistics used in the paper text:

  - Overall shift rate (share of samples with a GPT-labeled shift).
  - P(correct | shift) and P(correct | no shift).

By default it aggregates across:
  - Domains: Crossword, Math, Rush Hour (Carpark).
  - Temperatures: 0, 0.05, 0.3, 0.7.
  - Steps: [0, 950].
  - Split: test.
  - Shift definition: canonical GPT shift (change_way_of_thinking OR
    shift_in_reasoning_v1), without words-based gating.

Example:

  python tools/qwen15b_pooled_shift_stats.py
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Iterable, Tuple

# Ensure the repo root (containing the ``src`` package) is on sys.path when
# this script is invoked as ``python tools/qwen15b_pooled_shift_stats.py``.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.analysis.io import iter_records_from_file, scan_jsonl_files
from src.analysis.labels import aha_gpt
from src.analysis.metrics import extract_correct
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
    Construct an argument parser for the pooled Qwen2.5-1.5B stats script.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compute pooled shift prevalence and conditional accuracy for "
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


def _iter_qwen15b_rows(
    split_filter: str,
    min_step: int,
    max_step: int,
    gpt_mode: str,
    gate_by_words: bool,
) -> Iterable[Tuple[int, int]]:
    """
    Yield (correct, shift_flag) rows across all Qwen2.5-1.5B roots.
    """
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
                    yield int(correct_flag), int(shift_flag)


def main() -> None:
    """
    Compute and print pooled shift prevalence and conditional accuracy.
    """
    parser = build_argparser()
    args = parser.parse_args()

    N_total = 0
    N_shift = 0
    N_noshift = 0
    correct_shift = 0
    correct_noshift = 0

    for correct, shift_flag in _iter_qwen15b_rows(
        split_filter=str(args.split or ""),
        min_step=args.min_step,
        max_step=args.max_step,
        gpt_mode=args.gpt_mode,
        gate_by_words=bool(args.gate_by_words),
    ):
        N_total += 1
        if shift_flag:
            N_shift += 1
            correct_shift += int(correct)
        else:
            N_noshift += 1
            correct_noshift += int(correct)

    shift_rate = (N_shift / N_total) if N_total else 0.0
    p_shift = (correct_shift / N_shift) if N_shift else 0.0
    p_noshift = (correct_noshift / N_noshift) if N_noshift else 0.0

    print(f"N_total = {N_total}")
    print(f"N_shift = {N_shift}")
    print(f"shift_rate = {shift_rate:.6f} ({shift_rate * 100:.3f}%)")
    print(f"P(correct | shift)    = {p_shift:.6f} ({p_shift * 100:.3f}%)")
    print(f"P(correct | no shift) = {p_noshift:.6f} ({p_noshift * 100:.3f}%)")


if __name__ == "__main__":
    main()
