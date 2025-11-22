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
from src.analysis.utils import add_results_root_and_split_args


def _compute_shift_counts(
    files: List[str],
) -> Tuple[int, int, int, Dict[Tuple[int, int], int], List[Tuple[str, Dict[str, Any]]]]:
    """
    Scan JSONL files and compute aggregate counts and shift examples.

    :returns: Tuple of
        ``(total_examples, total_shifts, total_correct, joint_counts, shift_examples)``.
    """
    total_examples = 0
    total_shifts = 0
    total_correct = 0
    joint_counts: Dict[Tuple[int, int], int] = {}
    shift_examples: List[Tuple[str, Dict[str, Any]]] = []

    for path in files:
        for record in iter_records_from_file(path):
            if not isinstance(record, dict):
                continue
            total_examples += 1
            pass1 = record.get("pass1") or {}

            shift_flag = 1 if aha_gpt_canonical(pass1, record) == 1 else 0
            if shift_flag == 1:
                total_shifts += 1
                shift_examples.append((path, record))

            is_correct = extract_correct(pass1, record)
            if is_correct is None:
                continue
            total_correct += is_correct
            key = (int(is_correct), shift_flag)
            joint_counts[key] = joint_counts.get(key, 0) + 1

    return total_examples, total_shifts, total_correct, joint_counts, shift_examples


def _print_summary_header(
    results_root: str,
    split: str | None,
    total_examples: int,
    total_shifts: int,
    total_correct: int,
) -> None:
    """
    Print the top-level summary line with overall counts and fractions.
    """
    frac_shifts = total_shifts / total_examples if total_examples else 0.0
    frac_correct = total_correct / total_examples if total_examples else 0.0
    print(
        f"[shift_summary] root={results_root} split={split!r} "
        f"N={total_examples} shifts={total_shifts} ({frac_shifts:.2%}) "
        f"correct={total_correct} ({frac_correct:.2%})",
    )


def _print_joint_table(joint_counts: Dict[Tuple[int, int], int]) -> None:
    """
    Print a 2×2 table over correctness × shift flags.
    """
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


def _format_text_snippet(text: str, max_length: int) -> str:
    """
    Return a trimmed text snippet with an optional truncation marker.
    """
    snippet = text.strip()
    if len(snippet) > max_length:
        return snippet[:max_length] + " ...[truncated]..."
    return snippet


def _build_example_metadata(
    record: Dict[str, Any],
    pass1: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract metadata and text snippets for a single shift example.
    """
    problem_raw = record.get("problem") or record.get("clue") or ""
    endpoint = record.get("endpoint")
    deployment = record.get("deployment")
    step = record.get("step")
    sample_idx = record.get("sample_idx")
    is_correct = extract_correct(pass1, record)
    shift_flag = pass1.get("shift_in_reasoning_v1")

    problem_snippet = (
        _format_text_snippet(str(problem_raw), 400) if problem_raw else ""
    )
    output_text = pass1.get("output") or ""
    reasoning_snippet = (
        _format_text_snippet(str(output_text), 800) if output_text else ""
    )

    return {
        "problem_snippet": problem_snippet,
        "reasoning_snippet": reasoning_snippet,
        "endpoint": endpoint,
        "deployment": deployment,
        "step": step,
        "sample_idx": sample_idx,
        "is_correct": is_correct,
        "shift_flag": shift_flag,
    }


def _print_shift_examples(
    shift_examples: List[Tuple[str, Dict[str, Any]]],
    max_examples: int,
) -> None:
    """
    Print up to ``max_examples`` individual shift examples for inspection.
    """
    print("\n[shift_summary] Shift examples:")
    for idx, (path, record) in enumerate(shift_examples):
        if idx >= max_examples:
            remaining = len(shift_examples) - max_examples
            if remaining > 0:
                print(
                    f"\n[shift_summary] ... truncated "
                    f"{remaining} additional shift examples.",
                )
            break

        pass1 = record.get("pass1") or {}
        meta = _build_example_metadata(record, pass1)

        print("=" * 80)
        print(f"file: {path}")
        print(
            f"step={meta['step']} sample_idx={meta['sample_idx']} "
            f"endpoint={meta['endpoint']} deployment={meta['deployment']}",
        )
        print(
            f"is_correct={meta['is_correct']} "
            f"shift_in_reasoning_v1={meta['shift_flag']}",
        )
        if meta["problem_snippet"]:
            print("\nProblem:")
            print(meta["problem_snippet"])
        if meta["reasoning_snippet"]:
            print("\nPass-1 output (reasoning excerpt):")
            print(meta["reasoning_snippet"])


def summarize_root(
    results_root: str,
    split: str | None,
    max_examples: int | None,
) -> None:
    """
    Compute aggregate counts and an accuracy×shift table for a results root.

    By default, this prints only summary stats and a 2×2 table over:
    ``correct ∈ {0,1}``, ``shift ∈ {0,1}``. When ``max_examples>0``,
    individual shift examples are printed as well.
    """
    files = scan_jsonl_files(results_root, split_substr=split)
    if not files:
        print(
            f"[shift_summary] No JSONL files under {results_root!r} "
            f"(split={split!r}).",
        )
        return

    (
        total_examples,
        total_shifts,
        total_correct,
        joint_counts,
        shift_examples,
    ) = _compute_shift_counts(files)

    _print_summary_header(
        results_root=results_root,
        split=split,
        total_examples=total_examples,
        total_shifts=total_shifts,
        total_correct=total_correct,
    )
    _print_joint_table(joint_counts)

    if not shift_examples:
        print("\n[shift_summary] No shift examples found under this root.")
        return

    if max_examples is None or max_examples <= 0:
        # Examples suppressed unless max_examples > 0.
        return

    _print_shift_examples(shift_examples, max_examples)


def build_argparser() -> argparse.ArgumentParser:
    """
    Build and return the CLI argument parser for shift summary.
    """
    parser = argparse.ArgumentParser(
        description="Summarize shift_in_reasoning_v1 and correctness for JSONL results, "
        "and print all examples with canonical GPT shifts."
    )
    add_results_root_and_split_args(parser)
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum number of shift examples to print (default: all).",
    )
    return parser


def main() -> None:
    """
    Parse command-line arguments and run shift summary over the results root.
    """
    parser = build_argparser()
    args = parser.parse_args()

    root = os.path.abspath(args.results_root)
    summarize_root(root, split=args.split, max_examples=args.max_examples)


if __name__ == "__main__":
    main()
