#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility to strip fallback shift annotations from JSONL results.

This is useful after running src.annotate.gpt_eval when some examples
were marked with conservative FALSE labels because the LLM judge failed
or returned unparseable output (e.g., "Model call failed; defaulting to FALSE.").

Usage
-----
From the repo root:

  python -m src.annotate.clean_failed_shift_labels artifacts/results/gpt4o-math-portkey-temp005

You can pass any results_root; the script will walk it recursively and
rewrite all *.jsonl files in place, removing shift_* fields from pass1
whenever the rationale clearly indicates a fallback failure.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Tuple


BAD_RATIONALE_PREFIXES = (
    "Model call failed; defaulting to FALSE.",
    "Unparseable response; default FALSE.",
)

SHIFT_FIELDS = [
    "shift_in_reasoning_v1",
    "shift_markers_v1",
    "shift_first_marker_char",
    "shift_before_excerpt",
    "shift_after_excerpt",
    "shift_rationale_gpt",
    "shift_rationale_gpt_model",
    "shift_rationale_gpt_time",
]


def _should_clear(pass1: Dict[str, Any]) -> bool:
    """Return True if this pass1 section has a fallback failure rationale."""
    rationale = pass1.get("shift_rationale_gpt") or ""
    if not isinstance(rationale, str):
        return False
    rationale = rationale.strip()
    return any(rationale.startswith(prefix) for prefix in BAD_RATIONALE_PREFIXES)


def clean_file(path: str) -> Tuple[int, int]:
    """
    Rewrite a single JSONL file in place.

    Returns (n_records, n_cleared).
    """
    total = 0
    cleared = 0
    tmp_path = path + ".tmp"

    with open(path, "r", encoding="utf-8") as fin, open(
        tmp_path,
        "w",
        encoding="utf-8",
    ) as fout:
        for line in fin:
            stripped_line = line.strip()
            if not stripped_line:
                fout.write(line)
                continue
            try:
                rec = json.loads(stripped_line)
            except json.JSONDecodeError:
                fout.write(line)
                continue

            if not isinstance(rec, dict):
                fout.write(line)
                continue

            total += 1
            pass1 = rec.get("pass1") or {}
            if isinstance(pass1, dict) and _should_clear(pass1):
                for field in SHIFT_FIELDS:
                    pass1.pop(field, None)
                rec["pass1"] = pass1
                cleared += 1

            json.dump(rec, fout, ensure_ascii=False)
            fout.write("\n")

    os.replace(tmp_path, path)
    return total, cleared


def clean_root(results_root: str) -> Tuple[int, int, int]:
    """
    Clean all JSONL files under results_root.

    Returns (n_files, n_records, n_cleared).
    """
    root = os.path.abspath(results_root)
    if not os.path.isdir(root):
        raise SystemExit(f"results_root is not a directory: {root}")

    total_files = 0
    total_records = 0
    total_cleared = 0

    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if not filename.endswith(".jsonl"):
                continue
            path = os.path.join(dirpath, filename)
            n_records, n_cleared = clean_file(path)
            total_files += 1
            total_records += n_records
            total_cleared += n_cleared
            print(f"[clean_failed_shift_labels] {path}: records={n_records}, cleared={n_cleared}")

    print(
        f"[clean_failed_shift_labels] Done. files={total_files}, "
        f"records={total_records}, cleared={total_cleared}"
    )
    return total_files, total_records, total_cleared


def main() -> None:
    """CLI entrypoint for cleaning fallback shift labels."""
    parser = argparse.ArgumentParser(
        description=(
            "Strip fallback shift_in_reasoning_v1 annotations from JSONL results "
            "under a given root directory."
        )
    )
    parser.add_argument(
        "results_root",
        help=(
            "Root directory containing step*/.../*.jsonl "
            "(e.g., artifacts/results/gpt4o-math-portkey-temp005)."
        ),
    )
    args = parser.parse_args()

    clean_root(args.results_root)


if __name__ == "__main__":
    main()
