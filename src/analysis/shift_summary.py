#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quick summary of shift-in-reasoning and correctness for JSONL results.

Given a root directory containing step*/.../*.jsonl (e.g.,
  artifacts/results/deepseek-r1-openrouter
  artifacts/results/gpt4o-math-portkey
)
this script:
  1) Computes total examples, #shifts, and #correct (using the shared
     analysis helpers).
  2) Prints out every example that contains a GPT-labeled shift
     (canonical shift_in_reasoning_v1 / change_way_of_thinking).

Usage
-----
python -m src.analysis.shift_summary artifacts/results/deepseek-r1-openrouter --split test
python -m src.analysis.shift_summary artifacts/results/gpt4o-math-portkey --split test
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Tuple

from src.analysis.io import scan_jsonl_files, iter_records_from_file
from src.analysis.labels import aha_gpt_canonical
from src.analysis.metrics import extract_correct


def summarize_root(results_root: str, split: str | None, max_examples: int | None) -> None:
    """
    Compute aggregate counts and an accuracy×shift table.

    By default, this prints only summary stats and a 2×2 table over:
        correct ∈ {0,1}, shift ∈ {0,1}.
    Set --max_examples>0 to additionally print individual shift examples.
    """
    files = scan_jsonl_files(results_root, split_substr=split)
    if not files:
        print(f"[shift_summary] No JSONL files under {results_root!r} (split={split!r}).")
        return

    total = 0
    shifts = 0
    correct = 0
    # joint_counts[(correct, shift)] = count
    joint_counts: Dict[Tuple[int, int], int] = {}
    shift_examples: List[Tuple[str, Dict[str, Any]]] = []

    for path in files:
        for rec in iter_records_from_file(path):
            if not isinstance(rec, dict):
                continue
            total += 1
            pass1 = rec.get("pass1") or {}

            # Shift flag (canonical: change_way_of_thinking OR shift_in_reasoning_v1)
            shift_flag = 1 if aha_gpt_canonical(pass1, rec) == 1 else 0
            if shift_flag == 1:
                shifts += 1
                shift_examples.append((path, rec))

            # Correctness flag (robust helper)
            c = extract_correct(pass1, rec)
            if c is not None:
                correct += c
                key = (int(c), shift_flag)
                joint_counts[key] = joint_counts.get(key, 0) + 1

    frac_shifts = shifts / total if total else 0.0
    frac_correct = correct / total if total else 0.0
    print(
        f"[shift_summary] root={results_root} split={split!r} "
        f"N={total} shifts={shifts} ({frac_shifts:.2%}) "
        f"correct={correct} ({frac_correct:.2%})"
    )

    # 2×2 table over correctness × shift flags.
    c0_s0 = joint_counts.get((0, 0), 0)
    c0_s1 = joint_counts.get((0, 1), 0)
    c1_s0 = joint_counts.get((1, 0), 0)
    c1_s1 = joint_counts.get((1, 1), 0)
    row0 = c0_s0 + c0_s1
    row1 = c1_s0 + c1_s1
    col0 = c0_s0 + c1_s0
    col1 = c0_s1 + c1_s1
    table_total = row0 + row1

    print("\n[shift_summary] Correctness × shift table (counts)")
    print("                shift=0    shift=1    total")
    print(f"correct=0    {c0_s0:8d} {c0_s1:9d} {row0:8d}")
    print(f"correct=1    {c1_s0:8d} {c1_s1:9d} {row1:8d}")
    print(f"total        {col0:8d} {col1:9d} {table_total:8d}")

    if not shift_examples:
        print("\n[shift_summary] No shift examples found under this root.")
        return

    # Print all (or up to max_examples) shift examples for inspection.
    if max_examples is None or max_examples <= 0:
        # Examples suppressed unless max_examples > 0.
        return

    print("\n[shift_summary] Shift examples:")
    limit = max_examples
    for idx, (path, rec) in enumerate(shift_examples):
        if idx >= limit:
            remaining = len(shift_examples) - limit
            if remaining > 0:
                print(f"\n[shift_summary] ... truncated {remaining} additional shift examples.")
            break

        p1 = rec.get("pass1") or {}
        problem = rec.get("problem") or rec.get("clue") or ""
        endpoint = rec.get("endpoint")
        deployment = rec.get("deployment")
        step = rec.get("step")
        sample_idx = rec.get("sample_idx")
        c = extract_correct(p1, rec)
        shift_flag = p1.get("shift_in_reasoning_v1")

        output = p1.get("output") or ""
        # Show a reasonably sized snippet of the reasoning text.
        snippet = output.strip()
        if len(snippet) > 800:
            snippet = snippet[:800] + " ...[truncated]..."

        print("=" * 80)
        print(f"file: {path}")
        print(f"step={step} sample_idx={sample_idx} endpoint={endpoint} deployment={deployment}")
        print(f"is_correct={c} shift_in_reasoning_v1={shift_flag}")
        if problem:
            p_snippet = str(problem).strip()
            if len(p_snippet) > 400:
                p_snippet = p_snippet[:400] + " ...[truncated]..."
            print("\nProblem:")
            print(p_snippet)
        if snippet:
            print("\nPass-1 output (reasoning excerpt):")
            print(snippet)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize shift_in_reasoning_v1 and correctness for JSONL results, "
        "and print all examples with canonical GPT shifts."
    )
    parser.add_argument(
        "results_root",
        help="Root directory containing step*/.../*.jsonl (e.g., artifacts/results/deepseek-r1-openrouter).",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Optional substring filter on filenames (e.g., 'test').",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum number of shift examples to print (default: all).",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    root = os.path.abspath(args.results_root)
    summarize_root(root, split=args.split, max_examples=args.max_examples)


if __name__ == "__main__":
    main()
