#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility to trim math step*_test.jsonl files so that each problem has at most
``max_samples`` rows (default: 8). This is useful when repeated runs have
appended extra samples and you want to restore exactly 4000 rows
for MATH-500 (500 problems × 8 samples).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def trim_file(path: Path, max_samples: int = 8, dry_run: bool = False) -> None:
    """
    Trim a single JSONL results file in place.

    - Groups rows by ``problem``.
    - Within each group, sorts by ``sample_idx`` (when present).
    - Keeps at most ``max_samples`` rows per problem.
    - Writes the trimmed rows back to ``path``, after saving a ``.bak`` copy.
    """
    original_rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            original_rows.append(obj)

    by_problem: Dict[str, List[Dict]] = defaultdict(list)
    for obj in original_rows:
        prob = obj.get("problem")
        if prob is None:
            # Skip rows that don't look like math samples.
            continue
        by_problem[prob].append(obj)

    trimmed: List[Dict] = []
    for prob, rows in by_problem.items():
        # Sort by sample_idx when available for deterministic ordering.
        if any("sample_idx" in r for r in rows):
            rows.sort(key=lambda r: r.get("sample_idx", 0))
        trimmed.extend(rows[:max_samples])

    # Stable ordering: by problem, then sample_idx.
    trimmed.sort(key=lambda r: (r.get("problem", ""), r.get("sample_idx", 0)))

    print(
        f"{path}: problems={len(by_problem)} "
        f"rows_before={len(original_rows)} rows_after={len(trimmed)}",
    )

    if dry_run:
        return

    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        path.rename(backup)

    with path.open("w", encoding="utf-8") as handle:
        for obj in trimmed:
            json.dump(obj, handle, ensure_ascii=False)
            handle.write("\n")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Trim math JSONL result files so each problem has at most max-samples rows."
        ),
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="One or more step*_test.jsonl files (or globs) to trim.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=8,
        help="Maximum samples to keep per problem (default: 8).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes but do not modify files.",
    )
    args = parser.parse_args(argv)

    paths: List[Path] = []
    for pattern in args.files:
        matches = list(Path(".").glob(pattern))
        if not matches:
            print(f"No files matched pattern: {pattern}")
        paths.extend(matches)

    for path in sorted(set(paths)):
        if path.is_file():
            trim_file(path, max_samples=args.max_samples, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

