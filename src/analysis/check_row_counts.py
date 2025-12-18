#!/usr/bin/env python3
"""Audit inference logs for missing rows per checkpoint.

This utility counts rows in each ``step*_test.jsonl`` under the
predefined results roots and reports any files that fall short of the
expected number of samples. The default configuration matches the
paper's checkpoint selection (steps ≤950 for 1.5B runs, ≤500 for the
7B/8B math runs).
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class DatasetSpec:
    """Description of a dataset family to audit."""

    label: str
    target_rows: int
    roots: List[str]
    step_regex: re.Pattern[str]


DATASETS: List[DatasetSpec] = [
    DatasetSpec(
        label="Llama-8B math (≤500)",
        target_rows=4000,
        roots=[
            "artifacts/results/GRPO-Llama8B-math-temp-0",
            "artifacts/results/GRPO-Llama8B-math-temp-0.05",
            "artifacts/results/GRPO-Llama8B-math-temp-0.3",
            "artifacts/results/GRPO-Llama8B-math-temp-0.7",
        ],
        step_regex=re.compile(
            r".*/step-?0?[0-4]\d\d(?:/step-?0?[0-4]\d\d)?_test\.jsonl"
        ),
    ),
    DatasetSpec(
        label="Qwen-7B math (≤500)",
        target_rows=4000,
        roots=[
            "artifacts/results/GRPO-7B-math-temp-0",
            "artifacts/results/GRPO-7B-math-temp-0.05",
            "artifacts/results/GRPO-7B-math-temp-0.3",
            "artifacts/results/GRPO-7B-math-temp-0.7",
        ],
        step_regex=re.compile(
            r".*/step-?0?[0-4]\d\d(?:/step-?0?[0-4]\d\d)?_test\.jsonl"
        ),
    ),
    DatasetSpec(
        label="Qwen-1.5B math (≤950)",
        target_rows=4000,
        roots=[
            "artifacts/results/GRPO-1.5B-math-temp-0.0",
            "artifacts/results/GRPO-1.5B-math-temp-0.05",
            "artifacts/results/GRPO-1.5B-math-temp-0.3",
            "artifacts/results/GRPO-1.5B-math-temp-0.7",
        ],
        step_regex=re.compile(
            r".*/step-?0?[0-9]{3}(?:/step-?0?[0-9]{3})?_test\.jsonl"
        ),
    ),
    DatasetSpec(
        label="Crossword (≤950)",
        target_rows=1040,
        roots=[
            "artifacts/results/GRPO-1.5B-xword-temp-0",
            "artifacts/results/GRPO-1.5B-xword-temp-0.05",
            "artifacts/results/GRPO-1.5B-xword-temp-0.3",
            "artifacts/results/GRPO-1.5B-xword-temp-0.7",
        ],
        step_regex=re.compile(
            r".*/step-?0?[0-9]{3}(?:/step-?0?[0-9]{3})?_test\.jsonl"
        ),
    ),
    DatasetSpec(
        label="Carpark (≤950)",
        target_rows=4000,
        roots=[
            "artifacts/results/GRPO-1.5B-carpark-temp-0",
            "artifacts/results/GRPO-1.5B-carpark-temp-0.05",
            "artifacts/results/GRPO-1.5B-carpark-temp-0.3",
            "artifacts/results/GRPO-1.5B-carpark-temp-0.7",
        ],
        step_regex=re.compile(
            r".*/step-?0?[0-9]{3}(?:/step-?0?[0-9]{3})?_test\.jsonl"
        ),
    ),
]


def count_rows(path: Path) -> int:
    """Return the line count for a JSONL file."""

    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def iter_step_files(root: Path) -> Iterable[Path]:
    """Yield all checkpoint JSONL paths under ``root``.

    Handles both ``step-XXX/step0XXX_test.jsonl`` and flattened
    ``step0XXX_test.jsonl`` layouts.
    """

    yield from root.glob("step*/step*_test.jsonl")
    yield from root.glob("step*_test.jsonl")


def audit_dataset(spec: DatasetSpec) -> List[tuple[Path, int, int]]:
    """Return a list of (path, rows, missing_rows) tuples for ``spec``."""

    missing: List[tuple[Path, int, int]] = []
    for root_str in spec.roots:
        root = Path(root_str)
        if not root.exists():
            continue
        for path in iter_step_files(root):
            if not spec.step_regex.match(path.as_posix()):
                continue
            try:
                rows = count_rows(path)
            except OSError:
                continue
            deficit = max(spec.target_rows - rows, 0)
            if deficit > 0:
                missing.append((path, rows, deficit))
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show-ok",
        action="store_true",
        help="Also print files that meet the target row count.",
    )
    args = parser.parse_args()

    for spec in DATASETS:
        missing_rows = audit_dataset(spec)
        print(f"=== {spec.label} ===")
        total_missing = 0
        reported = False
        if missing_rows:
            for path, rows, deficit in sorted(missing_rows):
                print(f"  {path}: {rows} rows (missing {deficit})")
                total_missing += deficit
                reported = True
        if args.show_ok:
            # List OK files by subtracting missing paths from all paths.
            missing_set = {path for path, _, _ in missing_rows}
            for root_str in spec.roots:
                root = Path(root_str)
                if not root.exists():
                    continue
                for path in iter_step_files(root):
                    if path in missing_set:
                        continue
                    if not spec.step_regex.match(path.as_posix()):
                        continue
                    rows = count_rows(path)
                    print(f"  {path}: {rows} rows (ok)")
                    reported = True
        if not reported:
            print("  All files at target row count")
        else:
            print(
                "  Total rows missing vs. target:"
                f" {total_missing}\n" if missing_rows else "",
                end="",
            )
            if not missing_rows:
                print()
        if not args.show_ok and not missing_rows:
            # Keep spacing consistent when nothing was printed.
            continue


if __name__ == "__main__":
    main()
