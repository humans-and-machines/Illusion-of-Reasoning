#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize MATH-500 accuracy stability across system-prompt paraphrases.

This script is intended for the prompt-reliability sweep run via
``scripts/inference/math_prompt_reliability_t0.slurm``, which writes:

  artifacts/results/prompt_reliability/epoch{0..3}/variant{1..5}/step{STEP}/stepXXXX_test.jsonl

For each epoch (0 = pre, 1 = ckpt 500, 2 = ckpt 1000, 3 = final) and each
prompt variant (1..5), we compute:

  - Example-level accuracy: fraction of problems with at least one correct
    PASS-1 sample in that (epoch, variant) run.

We then aggregate across the 5 prompt variants to report:

  - Mean accuracy across prompts
  - Std. dev. across prompts
  - Range (min, max) across prompts

By default, each variant’s accuracy is computed over the problems that
actually appear in its JSONL. Optionally, ``--restrict_to_common`` can be
used to restrict to the intersection of problems observed across all
variants for a given epoch.

Typical usage (from repo root):

  python tools/math_prompt_reliability_summary.py \\
    --root artifacts/results/prompt_reliability

This prints a CSV-style summary plus LaTeX-ready table rows for
Table~\\ref{tab:judge-reliability-ckpt}.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple

import numpy as np

# Ensure the repo root (containing the ``src`` package) is on sys.path when
# this script is invoked as ``python tools/math_prompt_reliability_summary.py``.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.analysis.io import iter_records_from_file  # type: ignore[import]
from src.analysis.metrics import extract_correct  # type: ignore[import]
from src.analysis.utils import (  # type: ignore[import]
    extract_pass1_and_step,
    problem_key_from_record,
)


@dataclass
class VariantStats:
    """Per-variant accuracy bookkeeping."""

    problems: Set[str]
    correct_by_problem: Mapping[str, bool]
    num_samples: int
    num_correct_samples: int


EPOCH_STEPS: Dict[int, int] = {
    0: 0,
    1: 500,
    2: 1000,
    3: 1500,
}

EPOCH_LABELS: Dict[int, str] = {
    0: "0 (pre)",
    1: "1 (ckpt 500)",
    2: "2 (ckpt 1000)",
    3: "3 (final)",
}


def _load_variant_stats(
    jsonl_path: Path,
    split_filter: Optional[str],
) -> VariantStats:
    """
    Load a single prompt-variant JSONL and compute per-problem correctness.

    Correctness is computed from PASS-1 using :func:`extract_correct`, and
    example-level correctness is defined as "at least one correct sample
    for the problem".
    """
    problems: Set[str] = set()
    correct_by_problem: Dict[str, bool] = {}
    num_samples = 0
    num_correct_samples = 0

    # ``iter_records_from_file`` already skips blank lines and handles JSON errors.
    for record in iter_records_from_file(str(jsonl_path)):
        if split_filter is not None:
            rec_split = str(record.get("split", "")).lower()
            if rec_split != split_filter.lower():
                continue

        # For math runs, PASS-1 holds the primary correctness flag.
        pass1, _step = extract_pass1_and_step(record, None)
        if not pass1:
            continue

        correct_flag = extract_correct(pass1, record)
        if correct_flag is None:
            continue

        problem = problem_key_from_record(record, missing_default="unknown")
        problem_str = str(problem)
        problems.add(problem_str)

        num_samples += 1
        if correct_flag:
            num_correct_samples += 1
            # Example-level: any correct sample ⇒ problem marked correct.
            correct_by_problem[problem_str] = True
        else:
            # Only set False if we have not already seen a correct sample.
            correct_by_problem.setdefault(problem_str, False)

    return VariantStats(
        problems=problems,
        correct_by_problem=correct_by_problem,
        num_samples=num_samples,
        num_correct_samples=num_correct_samples,
    )


def _example_accuracy(
    stats: VariantStats,
    problem_subset: Optional[Iterable[str]] = None,
) -> Tuple[int, float]:
    """
    Return (num_problems, example_accuracy) for a variant.

    If ``problem_subset`` is provided, restricts to those problems; otherwise,
    uses all problems seen for this variant.
    """
    if problem_subset is None:
        problems = stats.problems
    else:
        problems = set(problem_subset) & stats.problems

    if not problems:
        return 0, float("nan")

    num_correct = sum(
        1 for prob in problems if bool(stats.correct_by_problem.get(prob, False))
    )
    acc = num_correct / float(len(problems))
    return len(problems), acc


def build_argparser() -> argparse.ArgumentParser:
    """Construct an argument parser for the prompt-reliability summary script."""
    parser = argparse.ArgumentParser(
        description=(
            "Summarize MATH-500 accuracy stability across system-prompt variants "
            "for Qwen2.5-1.5B (prompt_reliability sweep)."
        ),
    )
    parser.add_argument(
        "--root",
        default="artifacts/results/prompt_reliability",
        help=(
            "Root directory containing epoch*/variant*/step*/stepXXXX_test.jsonl "
            "(default: artifacts/results/prompt_reliability)."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="*",
        default=[0, 1, 2, 3],
        help="Epoch indices to summarize (default: 0 1 2 3).",
    )
    parser.add_argument(
        "--variants",
        type=int,
        nargs="*",
        default=[1, 2, 3, 4, 5],
        help="Prompt variant IDs (default: 1 2 3 4 5).",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Split filter (value of rec['split']; default: 'test').",
    )
    parser.add_argument(
        "--restrict_to_common",
        action="store_true",
        help=(
            "If set, compute example accuracy using only problems that appear "
            "in all variants for a given epoch."
        ),
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root directory not found: {root}")

    print(
        "epoch,step,used_problems,variants_used,mean_acc,std_acc,min_acc,max_acc",
    )

    epoch_summaries: List[Tuple[int, int, int, int, float, float, float, float]] = []

    for epoch in sorted(set(args.epochs)):
        step_value = EPOCH_STEPS.get(epoch)
        if step_value is None:
            # Skip unknown epochs silently; keeps the CLI flexible.
            continue

        epoch_dir = root / f"epoch{epoch}"
        if not epoch_dir.is_dir():
            continue

        variant_stats: Dict[int, VariantStats] = {}
        for variant in sorted(set(args.variants)):
            step_dir = epoch_dir / f"variant{variant}" / f"step{step_value}"
            jsonl_path = step_dir / f"step{step_value:04d}_test.jsonl"
            if not jsonl_path.exists():
                continue
            stats = _load_variant_stats(jsonl_path, split_filter=args.split)
            if stats.num_samples == 0:
                continue
            variant_stats[variant] = stats

        if not variant_stats:
            continue

        # Optional intersection over problems for all variants in this epoch.
        problem_subset: Optional[Set[str]] = None
        if args.restrict_to_common:
            problem_sets = [vs.problems for vs in variant_stats.values()]
            problem_subset = set.intersection(*problem_sets) if problem_sets else set()

        accuracies: List[float] = []
        num_problems_used = 0
        for variant, stats in sorted(variant_stats.items()):
            n_problems, acc = _example_accuracy(stats, problem_subset)
            if n_problems == 0 or not np.isfinite(acc):
                continue
            num_problems_used = n_problems  # same for all when restrict_to_common=True
            accuracies.append(acc)

        if not accuracies:
            continue

        acc_array = np.asarray(accuracies, dtype=float)
        mean_acc = float(acc_array.mean())
        std_acc = float(acc_array.std(ddof=0))
        min_acc = float(acc_array.min())
        max_acc = float(acc_array.max())

        epoch_summaries.append(
            (
                epoch,
                step_value,
                num_problems_used,
                len(accuracies),
                mean_acc,
                std_acc,
                min_acc,
                max_acc,
            ),
        )

        print(
            f"{epoch},{step_value},{num_problems_used},{len(accuracies)},"
            f"{mean_acc:.6f},{std_acc:.6f},{min_acc:.6f},{max_acc:.6f}",
        )

    if not epoch_summaries:
        return

    print()
    print("% LaTeX-ready rows for Table~\\ref{tab:judge-reliability-ckpt}")
    for epoch, step_value, n_probs, n_vars, mean_acc, std_acc, min_acc, max_acc in epoch_summaries:
        label = EPOCH_LABELS.get(epoch, str(epoch))
        mean_pct = mean_acc * 100.0
        std_pct = std_acc * 100.0
        min_pct = min_acc * 100.0
        max_pct = max_acc * 100.0
        print(
            f"% epoch={epoch} step={step_value} problems={n_probs} variants={n_vars}",
        )
        print(
            f"{label} & {mean_pct:.1f} & {std_pct:.1f} & "
            f"({min_pct:.1f}, {max_pct:.1f}) \\\\",
        )


if __name__ == "__main__":
    main()

