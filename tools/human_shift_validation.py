#!/usr/bin/env python3
"""
Compute human vs. GPT shift-label agreement for the 20-item validation set.

This script expects the anonymized Google Form export at
`data/Change_in_Thinking_Form_Responses_anon.csv`, where:
  - Column 0 holds a score for human raters and is blank for the GPT row.
  - Columns 1..20 hold Yes/No answers for each item.

By default, ties in the human majority vote are dropped from the comparison
(mirroring the paper table), and the same item mask is applied to the
pairwise kappa calculations.

Usage (from repo root):
  python tools/human_shift_validation.py
"""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

LabelSeq = Sequence[str]


def normalize_label(label: str) -> str:
    """Return a normalized yes/no label; raises on anything else."""
    val = label.strip().lower()
    if val.startswith("y"):
        return "yes"
    if val.startswith("n"):
        return "no"
    raise ValueError(f"Unexpected label (wanted Yes/No): {label!r}")


def load_responses(csv_path: Path) -> Tuple[List[LabelSeq], LabelSeq]:
    """
    Return (human_responses, gpt_responses) from the anonymized CSV.

    Humans are rows with a non-empty score in column 0; the GPT row has an
    empty score. All rows must contain at least 21 columns (score + 20 items).
    """
    humans: List[LabelSeq] = []
    gpt_rows: List[LabelSeq] = []

    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"{csv_path} is empty")

        for row in reader:
            if not any(cell.strip() for cell in row):
                continue
            if len(row) < 21:
                row = row + [""] * (21 - len(row))
            score = row[0].strip()
            labels = [cell.strip() for cell in row[1:21]]
            if score:
                humans.append(labels)
            else:
                gpt_rows.append(labels)

    if not humans:
        raise ValueError("No human rows found in the CSV.")
    if not gpt_rows:
        raise ValueError("No GPT/model row found in the CSV (blank score).")
    if len(gpt_rows) > 1:
        raise ValueError("Expected a single GPT/model row; found multiple.")

    return humans, gpt_rows[0]


def cohen_kappa(a: LabelSeq, b: LabelSeq) -> float:
    """Unweighted Cohen's kappa for two yes/no label sequences."""
    if len(a) != len(b):
        raise ValueError("Label sequences must have the same length.")

    n = len(a)
    agree = 0
    a_yes = 0
    b_yes = 0
    for la, lb in zip(a, b):
        na, nb = normalize_label(la), normalize_label(lb)
        agree += int(na == nb)
        a_yes += int(na == "yes")
        b_yes += int(nb == "yes")

    po = agree / float(n)
    pa_yes = a_yes / float(n)
    pb_yes = b_yes / float(n)
    pe = pa_yes * pb_yes + (1.0 - pa_yes) * (1.0 - pb_yes)
    return (po - pe) / (1.0 - pe) if pe != 1.0 else 0.0


def majority_labels(
    human_labels: Sequence[LabelSeq],
    tie_strategy: str,
    gpt_labels: LabelSeq | None = None,
) -> Tuple[List[str], List[int]]:
    """
    Compute majority labels and keep indices according to the tie strategy.

    Returns (majority, kept_indices). tie_strategy choices:
      - drop: skip items with a 3/3 split (default)
      - yes:  treat ties as 'yes'
      - no:   treat ties as 'no'
      - gpt:  break ties using the GPT label (requires gpt_labels)
    """
    majority: List[str] = []
    kept: List[int] = []
    num_items = len(human_labels[0])

    for idx in range(num_items):
        votes = [normalize_label(r[idx]) for r in human_labels]
        yes = votes.count("yes")
        no = votes.count("no")
        if yes == no:
            if tie_strategy == "drop":
                continue
            if tie_strategy == "yes":
                winner = "yes"
            elif tie_strategy == "no":
                winner = "no"
            elif tie_strategy == "gpt":
                if gpt_labels is None:
                    raise ValueError("tie_strategy='gpt' requires GPT labels.")
                winner = normalize_label(gpt_labels[idx])
            else:
                raise ValueError(f"Unknown tie_strategy: {tie_strategy}")
        else:
            winner = "yes" if yes > no else "no"

        majority.append(winner)
        kept.append(idx)

    return majority, kept


def subset_labels(labels: LabelSeq, indices: Iterable[int]) -> List[str]:
    """Return labels restricted to the provided indices."""
    return [labels[i] for i in indices]


def mean_pairwise_kappa(raters: Sequence[LabelSeq]) -> float:
    """Mean pairwise kappa across raters."""
    kappas: List[float] = []
    for i, j in combinations(range(len(raters)), 2):
        kappas.append(cohen_kappa(raters[i], raters[j]))
    return sum(kappas) / len(kappas)


def mean_gpt_human_kappa(
    humans: Sequence[LabelSeq], gpt_labels: LabelSeq
) -> float:
    """Mean kappa between GPT labels and each human rater."""
    kappas = [cohen_kappa(human, gpt_labels) for human in humans]
    return sum(kappas) / len(kappas)


def percent_agreement(a: LabelSeq, b: LabelSeq) -> float:
    """Simple percent agreement between two label sequences."""
    if len(a) != len(b):
        raise ValueError("Label sequences must have the same length.")
    matches = sum(
        normalize_label(x) == normalize_label(y) for x, y in zip(a, b)
    )
    return matches / float(len(a))


def mean_pairwise_po(raters: Sequence[LabelSeq]) -> float:
    """Mean pairwise percent agreement across raters."""
    pos: List[float] = []
    for i, j in combinations(range(len(raters)), 2):
        pos.append(percent_agreement(raters[i], raters[j]))
    return sum(pos) / len(pos)


def mean_gpt_human_po(
    humans: Sequence[LabelSeq], gpt_labels: LabelSeq
) -> float:
    """Mean percent agreement between GPT labels and each human rater."""
    pos = [percent_agreement(human, gpt_labels) for human in humans]
    return sum(pos) / len(pos)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize human vs. GPT shift-label agreement."
    )
    parser.add_argument(
        "--csv",
        default="data/Change_in_Thinking_Form_Responses_anon.csv",
        type=Path,
        help="Path to the anonymized form responses CSV (default: %(default)s).",
    )
    parser.add_argument(
        "--tie-strategy",
        choices=["drop", "yes", "no", "gpt"],
        default="drop",
        help="How to resolve 3/3 human ties (default: drop tied items).",
    )
    args = parser.parse_args()

    humans, gpt = load_responses(args.csv)

    majority, kept_indices = majority_labels(
        humans, tie_strategy=args.tie_strategy, gpt_labels=gpt
    )
    gpt_kept = subset_labels(gpt, kept_indices)
    human_kept = [subset_labels(r, kept_indices) for r in humans]

    po = percent_agreement(majority, gpt_kept)
    kappa = cohen_kappa(majority, gpt_kept)

    human_mean = mean_pairwise_kappa(human_kept)
    gpt_human_mean = mean_gpt_human_kappa(human_kept, gpt_kept)
    human_mean_po = mean_pairwise_po(human_kept)
    gpt_human_mean_po = mean_gpt_human_po(human_kept, gpt_kept)

    original_n = len(humans[0])
    n_used = len(kept_indices)
    dropped = original_n - n_used

    print(f"Loaded {len(humans)} human raters and 1 GPT row from {args.csv}")
    print(f"Tie strategy: {args.tie_strategy} (dropped {dropped} of {original_n} items)")
    print(f"N used: {n_used}")
    print(f"GPT vs. human majority: PO={po:.3f}, kappa={kappa:.3f}")
    print(f"Mean human–human (pairwise kappa): {human_mean:.3f}")
    print(f"Mean GPT–human (pairwise kappa): {gpt_human_mean:.3f}")
    print(f"Mean human–human (pairwise PO): {human_mean_po:.3f}")
    print(f"Mean GPT–human (pairwise PO): {gpt_human_mean_po:.3f}")

    print("\nLaTeX rows:")
    print(
        f"GPT\\mbox{{-}}4o vs. human majority vote & {n_used} & {po:.3f} & {kappa:.3f} \\\\"
    )
    print(
        f"Mean human--human (pairwise)          & {n_used} & {human_mean_po:.3f} & {human_mean:.2f} \\\\"
    )
    print(
        f"Mean LLM--human (pairwise)            & {n_used} & {gpt_human_mean_po:.3f} & {gpt_human_mean:.2f} \\\\"
    )


if __name__ == "__main__":
    main()
