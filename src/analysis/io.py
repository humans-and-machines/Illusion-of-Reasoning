#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File and record loading utilities for analysis scripts.

These helpers centralize the common patterns used across the ad-hoc analysis
scripts (JSON/JSONL/JSONL.GZ reading and "step-aware" directory scanning).
"""

from __future__ import annotations

import gzip
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional

from .utils import nat_step_from_path

# ---------------------------------------------------------------------------
# Scanning patterns
# ---------------------------------------------------------------------------

# Directories that almost never contain user data of interest
SKIP_DIR_DEFAULT = {"compare-1shot", "1shot", "hf_cache", "__pycache__"}

# Directory names like ".../step-01234/"
STEP_DIR_PAT = re.compile(r"(?:^|/)(step[-_]?\d{1,5})(?:/|$)", re.I)

# Filenames like "step100.jsonl", "global_step-100.jsonl", "checkpoint_100.jsonl"
STEP_FILE_PAT = re.compile(r"^(?:step|global[_-]?step|checkpoint)[-_]?\d{1,5}", re.I)


# ---------------------------------------------------------------------------
# Scanning helpers
# ---------------------------------------------------------------------------

def _normalize_skip(skip_substrings: Optional[Iterable[str]]) -> List[str]:
    if skip_substrings is None:
        return sorted(SKIP_DIR_DEFAULT)
    return [str(s).lower() for s in skip_substrings]


def scan_jsonl_files(root: str, split_substr: Optional[str] = None) -> List[str]:
    """
    Recursively collect `.jsonl` files under root, optionally filtering by a
    substring in the filename (used for "train"/"test" splits).
    """
    out: List[str] = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if not fn.endswith(".jsonl"):
                continue
            if split_substr and split_substr not in fn:
                continue
            out.append(os.path.join(dp, fn))
    out.sort()
    return out


def scan_files_step_only(
    root: str,
    split_substr: Optional[str] = None,
    skip_substrings: Optional[Iterable[str]] = None,
) -> List[str]:
    """
    Scan for JSON/JSONL/JSONL.GZ files that are associated with a training
    step, either via the directory name or filename.

    This mirrors the "step-aware" scanning used in several scripts (e.g.,
    temperature_effects, temp_graph, entropy_bin_regression, final-plot).
    """
    skip = _normalize_skip(skip_substrings)
    out: List[str] = []

    for dp, _, fns in os.walk(root):
        dp_norm = dp.replace("\\", "/").lower()
        if any(s in dp_norm for s in skip):
            continue
        dir_has_step = STEP_DIR_PAT.search(dp_norm) is not None
        for fn in fns:
            low = fn.lower()
            if not (low.endswith(".jsonl") or low.endswith(".jsonl.gz") or low.endswith(".json")):
                continue
            if split_substr and split_substr not in fn:
                continue
            file_has_step = STEP_FILE_PAT.search(fn) is not None
            if not (dir_has_step or file_has_step):
                continue
            out.append(os.path.join(dp, fn))

    out.sort()
    return out


def scan_files_with_steps_or_meta(
    root: str,
    split_substr: Optional[str] = None,
    skip_substrings: Optional[Iterable[str]] = None,
) -> List[str]:
    """
    Similar to scan_files_step_only, but also keeps files where we can infer a
    step from record contents (via nat_step_from_path as a last resort).

    This is useful when some logs don't encode the step in filenames/dirs but
    do provide it inside the JSON records.
    """
    files = scan_files_step_only(root, split_substr=split_substr, skip_substrings=skip_substrings)
    if files:
        return files

    # Fallback: be more permissive, still skipping common non-data dirs.
    skip = _normalize_skip(skip_substrings)
    out: List[str] = []
    for dp, _, fns in os.walk(root):
        dp_norm = dp.replace("\\", "/").lower()
        if any(s in dp_norm for s in skip):
            continue
        for fn in fns:
            low = fn.lower()
            if not (low.endswith(".jsonl") or low.endswith(".jsonl.gz") or low.endswith(".json")):
                continue
            if split_substr and split_substr not in fn:
                continue
            full = os.path.join(dp, fn)
            if nat_step_from_path(full) is not None:
                out.append(full)

    out.sort()
    return out


# ---------------------------------------------------------------------------
# Record iteration
# ---------------------------------------------------------------------------

def iter_records_from_file(path: str) -> Iterable[Dict[str, Any]]:
    """
    Yield JSON records from a `.jsonl`, `.jsonl.gz`, or `.json` file.

    This mirrors the most robust variant used in entropy_bin_regression,
    temp_graph, temperature_effects, and final-plot: it tolerates either
    newline-delimited JSON or a single JSON list/dict in `.json` files.
    """
    if path.endswith(".jsonl.gz"):
        opener = gzip.open
        mode = "rt"
    else:
        opener = open
        mode = "r"

    try:
        with opener(path, mode, encoding="utf-8") as f:  # type: ignore[arg-type]
            if path.endswith(".json"):
                text = f.read().strip()
                if not text:
                    return
                try:
                    obj = json.loads(text)
                    if isinstance(obj, list):
                        for rec in obj:
                            if isinstance(rec, dict):
                                yield rec
                    elif isinstance(obj, dict):
                        yield obj
                except Exception:
                    for line in text.splitlines():
                        s = line.strip()
                        if not s:
                            continue
                        try:
                            rec = json.loads(s)
                            if isinstance(rec, dict):
                                yield rec
                        except Exception:
                            continue
            else:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        rec = json.loads(s)
                        if isinstance(rec, dict):
                            yield rec
                    except Exception:
                        continue
    except Exception:
        return


