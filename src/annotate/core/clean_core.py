#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core helpers for cleaning fallback shift annotations from JSONL results.

CLI wrapper lives in :mod:`src.annotate.cli.clean_cli`
(backcompat: :mod:`src.annotate.backcompat.clean_failed_shift_labels`).
"""

from __future__ import annotations

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

    with (
        open(path, "r", encoding="utf-8") as fin,
        open(
            tmp_path,
            "w",
            encoding="utf-8",
        ) as fout,
    ):
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
            print(f"[clean_shift_fallbacks] {path}: records={n_records}, cleared={n_cleared}")

    print(
        f"[clean_shift_fallbacks] Done. files={total_files}, records={total_records}, cleared={total_cleared}",
    )
    return total_files, total_records, total_cleared


__all__ = ["BAD_RATIONALE_PREFIXES", "SHIFT_FIELDS", "clean_file", "clean_root"]
