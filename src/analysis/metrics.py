#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared metrics utilities for analysis scripts.

This currently focuses on correctness extraction and a few common derived
metrics that are reused across uncertainty / shift analyses.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .utils import canon_equal, coerce_bool, coerce_float, first_nonempty_str


# ---------------------------------------------------------------------------
# Correctness extraction
# ---------------------------------------------------------------------------

CORRECT_KEYS = {
    "is_correct",
    "is_correct_pred",
    "correct",
    "pred_correct",
    "y_is_correct",
    "exact_match",
    "em",
    "acc",
    "pass1_is_correct",
    "pass1_correct",
    "answer_correct",
    "label_correct",
}


def find_correct_in_obj(obj: Any) -> Optional[int]:
    """
    Depth-first search for correctness-ish booleans or [0,1] floats inside an
    arbitrary JSON-like object (dicts/lists).
    """
    q = [obj]
    while q:
        cur = q.pop(0)
        if isinstance(cur, dict):
            for k, v in cur.items():
                kl = str(k).lower()
                if any(cand in kl for cand in CORRECT_KEYS):
                    cb = coerce_bool(v)
                    if cb is not None:
                        return int(cb)
                if kl in {"acc", "accuracy", "score"}:
                    fv = coerce_float(v)
                    if fv is not None:
                        if fv in (0.0, 1.0):
                            return int(fv)
                        if 0.0 <= fv <= 1.0:
                            return int(fv >= 0.5)
            q.extend(cur.values())
        elif isinstance(cur, list):
            q.extend(cur)
    return None


def extract_correct(obj_like: Dict[str, Any], rec: Dict[str, Any]) -> Optional[int]:
    """
    Robust correctness from an object (e.g., pass1 or pass2) with rec as
    context for gold answers.

    Merges the more permissive/canonical variants from final-plot and
    entropy_bin_regression.
    """
    # 1) Look for explicit correctness flags anywhere inside obj_like
    cb = find_correct_in_obj(obj_like)
    if cb is not None:
        return cb

    # 2) Canonical answer comparison, allowing gold sets
    pred_canon = first_nonempty_str(
        obj_like.get("pred_answer_canon"),
        rec.get("pred_answer_canon"),
        obj_like.get("final_answer_canon"),
        rec.get("final_answer_canon"),
    )
    gold_canon = rec.get("gold_answer_canon_set") or rec.get("gold_answer_canon")
    ce = canon_equal(pred_canon, gold_canon)
    if ce is not None:
        return ce

    # 3) Raw answer comparison
    pred_raw = first_nonempty_str(
        obj_like.get("pred_answer"),
        rec.get("pred_answer"),
        obj_like.get("final_answer"),
        rec.get("final_answer"),
        obj_like.get("prediction"),
        rec.get("prediction"),
    )
    gold_raw = first_nonempty_str(
        rec.get("gold_answer"),
        rec.get("answer"),
        rec.get("target"),
        rec.get("label"),
    )
    if pred_raw is not None and gold_raw is not None:
        return int(pred_raw.strip() == gold_raw.strip())

    return None


def carpark_success_from_soft_reward(
    rec: Dict[str, Any],
    pass_obj: Dict[str, Any],
    op: str,
    thr: float,
) -> Optional[int]:
    """
    Compute a Carpark-style success flag from a soft_reward field.

    op ∈ {gt, ge, eq}, thr is a float threshold.
    """

    def _cmp(val: Any) -> Optional[int]:
        x = coerce_float(val)
        if x is None:
            return None
        if op == "gt":
            return int(x > thr)
        if op == "ge":
            return int(x >= thr)
        if op == "eq":
            return int(x == thr)
        return int(x > thr)

    sr = rec.get("soft_reward", pass_obj.get("soft_reward"))
    return _cmp(sr)


def wilson_ci(k: int, n: int) -> tuple[float, float]:
    """
    Wilson 95% confidence interval for a binomial proportion k/n.
    """
    if n <= 0:
        return (np.nan, np.nan)

    z = 1.959963984540054
    p = k / n
    denom = 1.0 + (z * z) / n
    centre = p + (z * z) / (2.0 * n)
    adj = z * ((p * (1.0 - p) + (z * z) / (4.0 * n)) / n) ** 0.5
    lo = (centre - adj) / denom
    hi = (centre + adj) / denom
    return (max(0.0, lo), min(1.0, hi))


