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
        --results_dir path/to/results/.../1.5B \
        --split test
"""

import os
import re
import glob
import json
import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results_dir", required=True,
        help="Directory containing checkpoint-*/checkpoint-* subfolders"
    )
    p.add_argument(
        "--split", default="test",
        help="Suffix to look for in JSONL filenames (default: test)"
    )
    return p.parse_args()

def parse_step(name: str) -> int:
    """Extract the first integer in a directory name, or 0 if none."""
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else 0

def compute_file_stats(path: str):
    unique_probs = set()
    solved_b = set()
    solved_a = set()
    sum_unc_b = 0.0
    sum_unc_a = 0.0
    total_samples = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            prob = row.get("problem", "")
            unique_probs.add(prob)

            if float(row.get("accuracy_before", 0.0)) > 0:
                solved_b.add(prob)
            if float(row.get("accuracy_after",  0.0)) > 0:
                solved_a.add(prob)

            sum_unc_b += float(row.get("uncertainty_before", 0.0))
            sum_unc_a += float(row.get("uncertainty_after",  0.0))
            total_samples += 1

    Q = len(unique_probs)
    if Q == 0 or total_samples == 0:
        return Q, 0.0, 0.0, 0.0, 0.0

    acc_b = len(solved_b) / Q
    acc_a = len(solved_a) / Q
    ent_b = sum_unc_b / total_samples
    ent_a = sum_unc_a / total_samples

    return Q, acc_b, acc_a, ent_b, ent_a

from tabulate import tabulate

def main():
    args = parse_args()

    # Gather and sort checkpoint directories
    ckpts = [
        d for d in os.listdir(args.results_dir)
        if os.path.isdir(os.path.join(args.results_dir, d))
    ]
    ckpts.sort(key=parse_step)

    # Table rows to collect
    table = []

    for ck in ckpts:
        ck_path = os.path.join(args.results_dir, ck)
        pattern = os.path.join(ck_path, "**", f"*_{args.split}.jsonl")
        files = glob.glob(pattern, recursive=True)
        if not files:
            table.append([ck, "[no file found]", "", "", "", "", ""])
            continue

        for path in sorted(files):
            Q, acc_b, acc_a, ent_b, ent_a = compute_file_stats(path)
            rel = os.path.relpath(path, args.results_dir)
            table.append([ck, rel, Q, f"{acc_b:.3f}", f"{acc_a:.3f}", f"{ent_b:.3f}", f"{ent_a:.3f}"])

    # Pretty print the table
    headers = ["CKPT", "FILE", "#P", "ACC₋", "ACC₊", "ENT₋", "ENT₊"]
    print(tabulate(table, headers=headers, tablefmt="github"))
    

if __name__ == "__main__":
    main()
