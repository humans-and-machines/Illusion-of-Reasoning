#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute GPT-labeled reasoning-shift rates.

This helper has two modes:

1) Aggregate mode (manual counts)
   --------------------------------
   python tools/shift_rate_calc.py \\
     --group qwen15b_math 320000 833 \\
     --group qwen15b_xword 83200 29

2) Root mode (compute N and shifts from JSONL roots)
   -------------------------------------------------
   python tools/shift_rate_calc.py \\
     --root-group qwen15b_math Math \\
       artifacts/results/GRPO-1.5B-math-temp-0.0 \\
       artifacts/results/GRPO-1.5B-math-temp-0.05 \\
       artifacts/results/GRPO-1.5B-math-temp-0.3 \\
       artifacts/results/GRPO-1.5B-math-temp-0.7 \\
     --root-group qwen15b_xword Crossword \\
       artifacts/results/GRPO-1.5B-xword-temp-0 \\
       artifacts/results/GRPO-1.5B-xword-temp-0.05 \\
       artifacts/results/GRPO-1.5B-xword-temp-0.3 \\
       artifacts/results/GRPO-1.5B-xword-temp-0.7 \\
     --root-group qwen15b_rush \"Rush Hour\" \\
       artifacts/results/GRPO-1.5B-carpark-temp-0 \\
       artifacts/results/GRPO-1.5B-carpark-temp-0.05 \\
       artifacts/results/GRPO-1.5B-carpark-temp-0.3 \\
       artifacts/results/GRPO-1.5B-carpark-temp-0.7

By default this mirrors the filtering logic used in
``src.analysis.reasoning_shift_entropy``:

- split == \"test\"
- step in [0, 950]
- pass-1 entropy present (mode=\"combined\")
- canonical GPT shifts, gated by reconsideration cues
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple


# Ensure the repo root (containing the ``src`` package) is on sys.path when
# this script is invoked as ``python tools/shift_rate_calc.py``.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.analysis.io import iter_records_from_file, scan_jsonl_files
from src.analysis.labels import aha_gpt
from src.analysis.utils import extract_pass1_and_step, entropy_from_pass1, nat_step_from_path


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-group and overall GPT-labeled shift rates either "
            "from manual counts or directly from JSONL result roots."
        ),
    )
    parser.add_argument(
        "--group",
        nargs=3,
        metavar=("LABEL", "N", "SHIFTS"),
        action="append",
        help=(
            "Add a group with label, total count N, and shift count. "
            "Example: --group qwen15b_math 320000 833"
        ),
    )
    parser.add_argument(
        "--root-group",
        nargs="+",
        metavar="ARGS",
        action="append",
        help=(
            "Add a group defined by LABEL, DOMAIN, and one or more ROOT "
            "directories containing step*/.../*.jsonl. N and shifts are "
            "computed directly from the JSONL records."
        ),
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Split filter for records (default: 'test').",
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
        "--step-bounds",
        nargs=3,
        metavar=("LABEL", "MIN_STEP", "MAX_STEP"),
        action="append",
        help=(
            "Override step bounds for a specific LABEL (matching a --root-group or --group label). "
            "MIN_STEP and MAX_STEP are inclusive integer bounds, e.g.: "
            "--step-bounds qwen15b_math 0 950 --step-bounds qwen7b_math 0 450"
        ),
    )
    parser.add_argument(
        "--entropy_mode",
        default="combined",
        choices=["combined", "sum", "think", "answer"],
        help="Pass-1 entropy aggregation (default: 'combined').",
    )
    parser.add_argument(
        "--gpt_mode",
        default="canonical",
        choices=["canonical", "broad"],
        help="GPT shift key set (default: canonical).",
    )
    parser.add_argument(
        "--no_words_gate",
        action="store_true",
        help="Disable gating GPT shifts by native reconsideration cues.",
    )
    return parser.parse_args(argv)


def _parse_groups(raw_groups: List[List[str]]) -> List[Tuple[str, int, int]]:
    groups: List[Tuple[str, int, int]] = []
    for label, n_str, s_str in raw_groups:
        try:
            n = int(n_str)
            s = int(s_str)
        except ValueError as exc:
            raise SystemExit(f"Invalid numeric values for group {label!r}: {n_str!r}, {s_str!r}") from exc
        if n < 0 or s < 0:
            raise SystemExit(f"N and shifts must be non-negative for group {label!r}.")
        groups.append((label, n, s))
    return groups


def _parse_step_bounds(raw_bounds: List[List[str]]) -> dict[str, tuple[int, int]]:
    """
    Parse per-label step bounds from CLI arguments.

    Each entry is (LABEL, MIN_STEP, MAX_STEP) with inclusive integer bounds.
    """
    bounds: dict[str, tuple[int, int]] = {}
    for label, min_str, max_str in raw_bounds:
        try:
            min_step = int(min_str)
            max_step = int(max_str)
        except ValueError as exc:
            raise SystemExit(
                f"Invalid step bounds for {label!r}: {min_str!r}, {max_str!r} "
                "(expected integer MIN_STEP MAX_STEP).",
            ) from exc
        if min_step < 0 or max_step < 0:
            raise SystemExit(f"Step bounds must be non-negative for label {label!r}.")
        if min_step > max_step:
            raise SystemExit(
                f"MIN_STEP must be <= MAX_STEP for label {label!r}: "
                f"{min_step} > {max_step}",
            )
        bounds[label] = (min_step, max_step)
    return bounds


def _parse_root_groups(raw_root_groups: List[List[str]]) -> List[Tuple[str, str, List[str]]]:
    groups: List[Tuple[str, str, List[str]]] = []
    for parts in raw_root_groups:
        if len(parts) < 3:
            raise SystemExit(
                f"--root-group expects at least LABEL DOMAIN ROOT, got: {parts!r}",
            )
        label = parts[0]
        domain = parts[1]
        roots = parts[2:]
        groups.append((label, domain, roots))
    return groups


def iter_shift_counts_for_root(
    root: str,
    *,
    domain: str,
    split: str,
    min_step: int,
    max_step: int,
    entropy_mode: str,
    gpt_mode: str,
    gate_by_words: bool,
) -> Tuple[int, int]:
    """
    Scan a results root and return (N, shifts) using the same filters as
    the reasoning_shift_entropy analysis.
    """
    files = scan_jsonl_files(root, split_substr=None)
    total_n = 0
    total_shifts = 0
    for file_path in files:
        step_hint = nat_step_from_path(file_path)
        for record in iter_records_from_file(file_path):
            if split and str(record.get("split", "")).lower() != split.lower():
                continue
            pass1, step = extract_pass1_and_step(record, step_hint)
            if not pass1 or step is None:
                continue
            if step < min_step or step > max_step:
                continue
            entropy = entropy_from_pass1(pass1, entropy_mode)
            if entropy is None:
                continue
            shift_flag = aha_gpt(
                pass1,
                record,
                mode=gpt_mode,
                gate_by_words=gate_by_words,
                domain=domain,
            )
            total_n += 1
            total_shifts += int(shift_flag)
    return total_n, total_shifts


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    raw_groups = args.group or []
    raw_root_groups = args.root_group or []
    step_bounds = _parse_step_bounds(args.step_bounds or [])

    if not raw_groups and not raw_root_groups:
        raise SystemExit(
            "No groups provided. Use --group LABEL N SHIFTS and/or "
            "--root-group LABEL DOMAIN ROOT [...].",
        )

    groups: List[Tuple[str, int, int]] = []

    # Manual-count groups
    if raw_groups:
        groups.extend(_parse_groups(raw_groups))

    # Root-based groups: compute N and shifts directly from JSONL files.
    if raw_root_groups:
        root_groups = _parse_root_groups(raw_root_groups)
        gate_by_words = not args.no_words_gate
        for label, domain, roots in root_groups:
            # Per-label step bounds (fallback to global min/max if not overridden)
            min_step = args.min_step
            max_step = args.max_step
            if label in step_bounds:
                min_step, max_step = step_bounds[label]
                print(
                    f"[info] Using custom step bounds for {label}: "
                    f"min_step={min_step}, max_step={max_step}",
                )

            print(f"=== {label} (domain={domain}) ===")
            label_n = 0
            label_shifts = 0
            for root in roots:
                n_root, s_root = iter_shift_counts_for_root(
                    root,
                    domain=domain,
                    split=args.split,
                    min_step=min_step,
                    max_step=max_step,
                    entropy_mode=args.entropy_mode,
                    gpt_mode=args.gpt_mode,
                    gate_by_words=gate_by_words,
                )
                rate_root = (s_root / n_root) if n_root else 0.0
                print(
                    f"root={root}: N={n_root}, shifts={s_root}, "
                    f"share_shift={rate_root:.6f} ({rate_root * 100:.3f}%)",
                )
                label_n += n_root
                label_shifts += s_root
            rate_label = (label_shifts / label_n) if label_n else 0.0
            print(
                f"TOTAL {label}: N={label_n}, shifts={label_shifts}, "
                f"share_shift={rate_label:.6f} ({rate_label * 100:.3f}%)\n",
            )
            groups.append((label, label_n, label_shifts))

    # Final summary across all groups (manual + root-based).
    total_n = sum(n for _, n, _ in groups)
    total_shifts = sum(s for _, _, s in groups)

    for label, n, s in groups:
        rate = (s / n) if n else 0.0
        print(f"{label}: N={n}, shifts={s}, share_shift={rate:.6f} ({rate * 100:.3f}%)")

    overall = (total_shifts / total_n) if total_n else 0.0
    print(
        f"\nOVERALL: N={total_n}, shifts={total_shifts}, "
        f"share_shift={overall:.6f} ({overall * 100:.3f}%)",
    )


if __name__ == "__main__":
    main()
