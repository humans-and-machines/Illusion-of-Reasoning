#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helpers to report shift-annotation coverage for multi-pass JSONL files.

The :func:`count_progress` helper counts rows whose second-pass sections
already contain GPT-4o rationale metadata so operators can monitor long-
running annotation jobs without leaving the :mod:`src.annotate` package.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple


def _iter_records(path: Path) -> Iterable[dict]:
    """Yield JSON-decoded records from a line-delimited JSONL file."""
    with path.open("r", encoding="utf-8") as file_handle:
        for line in file_handle:
            stripped = line.strip()
            if not stripped:
                continue
            yield json.loads(stripped)


def _has_judge_tag(record: dict) -> bool:
    """Return True if any pass2 section carries a GPT shift rationale tag."""
    for key in ("pass2a", "pass2b", "pass2c"):
        section = record.get(key)
        if section and section.get("shift_rationale_gpt_model"):
            return True
    return False


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for reporting annotation coverage."""
    parser = argparse.ArgumentParser(
        description="Show annotation progress for GPT-4o shift labels.",
    )
    parser.add_argument(
        "jsonl",
        type=Path,
        help="Multi-pass results JSONL (e.g., step0950_test.jsonl).",
    )
    parser.add_argument(
        "--require-shift-flag",
        action="store_true",
        help="Only count rows whose pass2 sections have shift_in_reasoning_v1.",
    )
    return parser.parse_args()


def count_progress(path: Path, require_shift_flag: bool = False) -> Tuple[int, int]:
    """
    Count total rows and rows with completed GPT-4o annotation.

    When ``require_shift_flag`` is True, only rows where any pass2 section
    already has ``shift_in_reasoning_v1`` are counted as annotated.
    """
    total = 0
    seen = 0
    for rec in _iter_records(path):
        total += 1
        if require_shift_flag:
            if any(rec.get(f"pass2{k}", {}).get("shift_in_reasoning_v1") for k in ("a", "b", "c")):
                seen += 1
        else:
            if _has_judge_tag(rec):
                seen += 1
    return total, seen


def main() -> None:
    """Entry point: print total, processed, and pending row counts."""
    args = parse_args()
    total, seen = count_progress(args.jsonl, require_shift_flag=args.require_shift_flag)
    print(f"total rows: {total}")
    print(f"processed by GPT-4o judge: {seen}")
    print(f"pending: {total - seen}")


__all__ = ["count_progress", "main", "parse_args"]


if __name__ == "__main__":
    main()
