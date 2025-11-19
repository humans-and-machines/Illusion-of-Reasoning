#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared low-level helpers for analysis scripts.

This module intentionally keeps dependencies light and behavior conservative so
it can be safely imported from many one-off scripts.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Path / naming helpers
# ---------------------------------------------------------------------------

# Flexible step patterns used across scripts (dir names, file names, substrings)
STEP_PATS = [
    re.compile(r"step\s*[-_]?(?P<step>\d+)", re.I),
    re.compile(r"global[_-]?step\s*[-_]?(?P<step>\d+)", re.I),
    re.compile(r"checkpoint\s*[-_]?(?P<step>\d+)", re.I),
    # bare numeric subdirs at the end of the path (e.g., ".../01234/")
    re.compile(r"/(?P<step>\d{2,5})(?=/|$)"),
]

# Common temperature tokens in directory names
TEMP_PATS = [
    re.compile(r"(?:^|[-_])temp[-_](?P<t>low|[0-9]+(?:\.[0-9]+)?)$", re.I),
    re.compile(r"(?:^|[-_])(?P<t>low|[0-9]+(?:\.[0-9]+)?)[-_]temp$", re.I),
    re.compile(r"(?:^|[-_])low[-_]temp$", re.I),
]


def nat_step_from_path(path: str) -> Optional[int]:
    """
    Try to infer an integer training step from a path or filename.

    Uses a collection of regexes that cover the various conventions used in
    analysis scripts (step123, global_step-123, checkpoint_123, ...).
    """
    s = str(path)
    for pat in STEP_PATS:
        m = pat.search(s)
        if not m:
            continue
        g = m.groupdict().get("step") or m.group(1)
        try:
            return int(g)
        except Exception:
            # Fall through to try other patterns
            continue
    return None


def parse_temp_from_dir(dirname: str, low_alias: float) -> Optional[float]:
    """
    Infer temperature from a directory name.

    Supports tokens like:
      temp-0.7, 0.7-temp, temp-low, low-temp
    """
    d = dirname.lower()
    for pat in TEMP_PATS:
        m = pat.search(d)
        if not m:
            continue
        tok = m.groupdict().get("t", "low").lower()
        if tok == "low":
            return float(low_alias)
        try:
            return float(tok)
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# Generic coercion helpers
# ---------------------------------------------------------------------------

def coerce_bool(x: Any) -> Optional[int]:
    """
    Best-effort conversion to {0,1}.

    Returns:
      1/0 for recognized truthy/falsey values, or None if there is no
      reasonable interpretation (e.g., complex structures).
    """
    if x is None:
        return None
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, np.integer)):
        return int(bool(x))
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"1", "true", "t", "yes", "y"}:
            return 1
        if s in {"0", "false", "f", "no", "n"}:
            return 0
    try:
        return int(bool(x))
    except Exception:
        return None


def coerce_float(x: Any) -> Optional[float]:
    """
    Best-effort conversion to float, returning None on failure instead of
    raising.
    """
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def get_problem_id(rec: Dict[str, Any]) -> Optional[str]:
    """
    Heuristic problem identifier that works across math/xword/carpark.

    Prefers explicit problem-like keys; falls back to sample index when
    available.
    """
    for k in ("problem_id", "example_id", "id", "question", "problem", "clue", "title", "uid"):
        v = rec.get(k)
        if v is not None and not isinstance(v, (dict, list)):
            return str(v)
    v = rec.get("sample_idx")
    return None if v is None else f"sample_{v}"


def first_nonempty_str(*xs: Any) -> Optional[str]:
    """
    Return the first non-empty string among the arguments, or None.
    """
    for x in xs:
        if isinstance(x, str):
            s = x.strip()
            if s:
                return s
    return None


def canon_equal(pred_canon: Optional[str], gold: Any) -> Optional[int]:
    """
    Compare a canonical predicted answer to gold.

    - If gold is a string, require exact match after stripping.
    - If gold is a list/tuple/set of strings, treat it as a set.
    - Otherwise, return None.
    """
    if pred_canon is None or gold is None:
        return None
    if isinstance(gold, (list, tuple, set)):
        gold_set = {str(g).strip() for g in gold if isinstance(g, str)}
        return int(pred_canon.strip() in gold_set) if gold_set else None
    if isinstance(gold, str):
        return int(pred_canon.strip() == gold.strip())
    return None


