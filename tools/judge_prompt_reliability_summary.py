#!/usr/bin/env python3
"""
Summarize inter-prompt agreement for GPT shift judges on MATH-500.

This script mirrors the structure used in tools/math_prompt_reliability_summary.py
but operates on shift labels produced by annotating the same inference outputs
with five judge-prompt variants (v1..v5).

Expected directory layout (one annotated copy per variant, per epoch):
  <root>/epoch{0..3}/variant{1..5}/step{STEP}/stepXXXX_test.jsonl

Epoch → step mapping follows the Qwen-1.5B training checkpoints:
  0 -> step0, 1 -> step500, 2 -> step1000, 3 -> step1500

Usage (from repo root):
  python tools/judge_prompt_reliability_summary.py \
    --root artifacts/results/judge_prompt_reliability

Outputs:
  - CSV-style rows: epoch,step,judged_N,variants_used,mean_PO,mean_kappa,ci_low,ci_high
  - LaTeX-ready rows for Table~\\ref{tab:judge-reliability-ckpt}
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Ensure the repo root (containing the ``src`` package) is on sys.path when
# this script is invoked as ``python tools/judge_prompt_reliability_summary.py``.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.analysis.io import iter_records_from_file  # type: ignore[import]
from src.analysis.utils import problem_key_from_record  # type: ignore[import]

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


def cohen_kappa(labels_a: Sequence[bool], labels_b: Sequence[bool]) -> float:
    """Unweighted Cohen's kappa for two Boolean label sequences."""
    if len(labels_a) != len(labels_b):
        raise ValueError("Label sequences must have the same length.")
    n = len(labels_a)
    agree = sum(int(a == b) for a, b in zip(labels_a, labels_b))
    p_yes_a = sum(int(x) for x in labels_a) / float(n)
    p_yes_b = sum(int(x) for x in labels_b) / float(n)
    po = agree / float(n)
    pe = p_yes_a * p_yes_b + (1.0 - p_yes_a) * (1.0 - p_yes_b)
    return (po - pe) / (1.0 - pe) if pe != 1.0 else 0.0


def percent_agreement(labels_a: Sequence[bool], labels_b: Sequence[bool]) -> float:
    """Simple percent agreement between two Boolean label sequences."""
    if len(labels_a) != len(labels_b):
        raise ValueError("Label sequences must have the same length.")
    return sum(int(a == b) for a, b in zip(labels_a, labels_b)) / float(len(labels_a))


def load_variant_labels(
    jsonl_path: Path,
    *,
    split_filter: Optional[str],
    pass_key: str,
    require_prompt_variant: bool,
) -> Mapping[str, bool]:
    """
    Load a single variant JSONL and return {problem -> shift_label}.

    Only rows with a boolean ``shift_in_reasoning_v1`` under ``pass_key`` are kept.
    """
    labels: Dict[str, bool] = {}
    for record in iter_records_from_file(str(jsonl_path)):
        if split_filter is not None:
            rec_split = str(record.get("split", "")).lower()
            if rec_split != split_filter.lower():
                continue

        section = record.get(pass_key) or {}
        if not isinstance(section, dict):
            continue
        shift_flag = section.get("shift_in_reasoning_v1")
        if shift_flag is None:
            continue
        if require_prompt_variant and "shift_judge_prompt_variant" not in section:
            continue

        problem = str(problem_key_from_record(record, "unknown"))
        labels[problem] = bool(shift_flag)
    return labels


def mean_pairwise_metrics(
    label_sets: Mapping[int, Mapping[str, bool]],
    problems: Sequence[str],
) -> Tuple[float, float]:
    """
    Return (mean_PO, mean_kappa) over all variant pairs on the given problems.
    """
    pos: List[float] = []
    kappas: List[float] = []
    for a, b in combinations(sorted(label_sets.keys()), 2):
        la = [label_sets[a][p] for p in problems]
        lb = [label_sets[b][p] for p in problems]
        pos.append(percent_agreement(la, lb))
        kappas.append(cohen_kappa(la, lb))
    return float(np.mean(pos)), float(np.mean(kappas))


def bootstrap_kappa_ci(
    label_sets: Mapping[int, Mapping[str, bool]],
    problems: Sequence[str],
    *,
    num_samples: int,
    rng: random.Random,
) -> Tuple[float, float]:
    """
    Bootstrap mean pairwise kappa over problems; return (ci_low, ci_high).
    """
    if not problems:
        return float("nan"), float("nan")
    estimates: List[float] = []
    for _ in range(num_samples):
        sample = [rng.choice(problems) for _ in range(len(problems))]
        _, kappa = mean_pairwise_metrics(label_sets, sample)
        estimates.append(kappa)
    arr = np.asarray(estimates, dtype=float)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize inter-prompt agreement for shift judges (Qwen-1.5B, MATH-500).",
    )
    parser.add_argument(
        "--root",
        default="artifacts/results/judge_prompt_reliability",
        help=(
            "Root directory containing epoch*/variant*/step*/stepXXXX_test.jsonl "
            "(default: artifacts/results/judge_prompt_reliability)."
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
        help="Judge prompt variant IDs (default: 1 2 3 4 5).",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Split filter (value of rec['split']; default: 'test').",
    )
    parser.add_argument(
        "--pass_key",
        default="pass1",
        help="Pass key to read shift labels from (default: pass1).",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
        help="Number of bootstrap samples for kappa CI (default: 2000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for bootstrap sampling (default: 1234).",
    )
    parser.add_argument(
        "--require_prompt_variant",
        action="store_true",
        default=True,
        help=(
            "When set (default), only count rows that include shift_judge_prompt_variant "
            "to reflect actually judged examples."
        ),
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    rng = random.Random(args.seed)
    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root directory not found: {root}")

    print("epoch,step,judged_N,variants_used,mean_PO,mean_kappa,ci_low,ci_high")

    epoch_rows: List[Tuple[int, int, int, int, float, float, float, float]] = []

    for epoch in sorted(set(args.epochs)):
        step_value = EPOCH_STEPS.get(epoch)
        if step_value is None:
            continue

        epoch_dir = root / f"epoch{epoch}"
        if not epoch_dir.is_dir():
            continue

        variant_labels: Dict[int, Mapping[str, bool]] = {}
        for variant in sorted(set(args.variants)):
            step_dir = epoch_dir / f"variant{variant}" / f"step{step_value}"
            jsonl_path = step_dir / f"step{step_value:04d}_test.jsonl"
            if not jsonl_path.exists():
                continue
            labels = load_variant_labels(
                jsonl_path,
                split_filter=args.split,
                pass_key=args.pass_key,
                require_prompt_variant=args.require_prompt_variant,
            )
            if labels:
                variant_labels[variant] = labels

        if len(variant_labels) < 2:
            continue

        # Intersection of problems across all included variants for this epoch.
        problem_sets = [set(lbls.keys()) for lbls in variant_labels.values()]
        common_problems = sorted(set.intersection(*problem_sets)) if problem_sets else []
        if not common_problems:
            continue

        mean_po, mean_kappa = mean_pairwise_metrics(variant_labels, common_problems)
        ci_low, ci_high = bootstrap_kappa_ci(
            variant_labels,
            common_problems,
            num_samples=args.bootstrap,
            rng=rng,
        )

        epoch_rows.append(
            (
                epoch,
                step_value,
                len(common_problems),
                len(variant_labels),
                mean_po,
                mean_kappa,
                ci_low,
                ci_high,
            ),
        )
        print(
            f"{epoch},{step_value},{len(common_problems)},{len(variant_labels)},"
            f"{mean_po:.3f},{mean_kappa:.3f},{ci_low:.3f},{ci_high:.3f}",
        )

    if not epoch_rows:
        return

    print()
    print("% LaTeX-ready rows for Table~\\ref{tab:judge-reliability-ckpt}")
    for epoch, step_value, n, n_vars, mean_po, mean_kappa, ci_low, ci_high in epoch_rows:
        label = EPOCH_LABELS.get(epoch, str(epoch))
        print(f"% epoch={epoch} step={step_value} problems={n} variants={n_vars}")
        print(f"{label} & {n:3d} & {mean_po:.3f} & {mean_kappa:.3f} \\; [{ci_low:.3f}, {ci_high:.3f}] \\\\")


if __name__ == "__main__":
    main()
