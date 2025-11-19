#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
summarize_per_checkpoint_recursive.py

For each immediate checkpoint subdirectory under --results_dir, this script:
  • Recursively finds every file matching "*_{split}.jsonl" under that checkpoint
  • For each such file, computes:
      - #P    = total number of unique problems in that file
      - ACC₋  = fraction of those problems with ≥1 correct "before" sample
      - ACC₊  = fraction of those problems with ≥1 correct "after" sample
      - ENT₋  = mean uncertainty_before across all samples in that file
      - ENT₊  = mean uncertainty_after  across all samples in that file

Usage:
    python summarize_per_checkpoint_recursive.py \
        --results_dir path/to/artifacts/results/.../1.5B \
        --split test
"""

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass
from typing import List, Tuple

from tabulate import tabulate


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the checkpoint summarizer."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        required=True,
        help="Directory containing checkpoint-*/checkpoint-* subfolders",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Suffix to look for in JSONL filenames (default: test)",
    )
    return parser.parse_args()


def parse_step(name: str) -> int:
    """Extract the first integer in a directory name, or 0 if none."""
    match = re.search(r"(\\d+)", name)
    return int(match.group(1)) if match else 0


@dataclass
class FileStats:
    """Aggregate statistics for a single JSONL result file."""

    unique_problems: List[str]
    solved_before: List[str]
    solved_after: List[str]
    sum_uncertainty_before: float = 0.0
    sum_uncertainty_after: float = 0.0
    total_samples: int = 0


def compute_file_stats(path: str) -> Tuple[int, float, float, float, float]:
    """Compute (#problems, acc_before, acc_after, ent_before, ent_after) for a file."""
    stats = FileStats(unique_problems=[], solved_before=[], solved_after=[])

    with open(path, "r", encoding="utf-8") as file_handle:
        for line in file_handle:
            row = json.loads(line)
            problem = str(row.get("problem", ""))
            if problem not in stats.unique_problems:
                stats.unique_problems.append(problem)

            if float(row.get("accuracy_before", 0.0)) > 0:
                if problem not in stats.solved_before:
                    stats.solved_before.append(problem)
            if float(row.get("accuracy_after", 0.0)) > 0:
                if problem not in stats.solved_after:
                    stats.solved_after.append(problem)

            stats.sum_uncertainty_before += float(row.get("uncertainty_before", 0.0))
            stats.sum_uncertainty_after += float(row.get("uncertainty_after", 0.0))
            stats.total_samples += 1

    num_problems = len(stats.unique_problems)
    if num_problems == 0 or stats.total_samples == 0:
        return num_problems, 0.0, 0.0, 0.0, 0.0

    acc_before = len(stats.solved_before) / num_problems
    acc_after = len(stats.solved_after) / num_problems
    ent_before = stats.sum_uncertainty_before / stats.total_samples
    ent_after = stats.sum_uncertainty_after / stats.total_samples

    return num_problems, acc_before, acc_after, ent_before, ent_after


def main() -> None:
    """Entry point for summarizing per-checkpoint JSONL results."""
    args = parse_args()

    # Gather and sort checkpoint directories
    checkpoint_dirs = [
        checkpoint
        for checkpoint in os.listdir(args.results_dir)
        if os.path.isdir(os.path.join(args.results_dir, checkpoint))
    ]
    checkpoint_dirs.sort(key=parse_step)

    # Table rows to collect
    table: List[List[object]] = []

    for checkpoint in checkpoint_dirs:
        ck_path = os.path.join(args.results_dir, checkpoint)
        pattern = os.path.join(ck_path, "**", f"*_{args.split}.jsonl")
        files = glob.glob(pattern, recursive=True)
        if not files:
            table.append([checkpoint, "[no file found]", "", "", "", "", ""])
            continue

        for path in sorted(files):
            (
                num_problems,
                acc_before,
                acc_after,
                ent_before,
                ent_after,
            ) = compute_file_stats(path)
            rel = os.path.relpath(path, args.results_dir)
            table.append(
                [
                    checkpoint,
                    rel,
                    num_problems,
                    f"{acc_before:.3f}",
                    f"{acc_after:.3f}",
                    f"{ent_before:.3f}",
                    f"{ent_after:.3f}",
                ]
            )

    # Pretty print the table
    headers = ["CKPT", "FILE", "#P", "ACC₋", "ACC₊", "ENT₋", "ENT₊"]
    print(tabulate(table, headers=headers, tablefmt="github"))


if __name__ == "__main__":
    main()
