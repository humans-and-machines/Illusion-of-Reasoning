#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared helpers for the Forced Aha analyses.

This module centralizes the small utilities and statistical helpers that were
previously duplicated across ``forced_aha_effect.py`` and
``forced_aha_effect_impl.py`` so we can keep the linter happy and avoid drift.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.analysis.metrics import wilson_ci
from src.analysis.utils import coerce_bool

try:  # pragma: no cover - optional dependency
    from statsmodels.stats.contingency_tables import mcnemar as _statsmodels_mcnemar
except ImportError:  # pragma: no cover - optional dependency
    _statsmodels_mcnemar = None

try:  # pragma: no cover - optional dependency
    import scipy.stats as _scipy_stats
except ImportError:  # pragma: no cover - optional dependency
    _scipy_stats = None

# Public keys reused across scripts ------------------------------------------------

PASS1_KEYS: List[str] = ["pass1", "p1", "first_pass"]
# For second-pass style results, prefer explicitly named forced/second-pass fields,
# but also fall back to multi-cue variants when present.
PASS2_KEYS: List[str] = [
    "pass2_forced",
    "pass2",
    "pass2c",
    "pass2b",
    "pass2a",
    "p2",
    "forced",
    "forced_aha",
    "aha_forced",
]


def first_nonempty(mapping: Dict[str, Any], keys: Sequence[str]) -> Optional[Dict[str, Any]]:
    """
    Return the first ``mapping[key]`` that is a non-empty dict, or ``None``.
    """
    for key in keys:
        candidate = mapping.get(key)
        if isinstance(candidate, dict) and candidate:
            return candidate
    return None


def extract_correct_flag(pass_obj: Dict[str, Any]) -> Optional[int]:
    """
    Extract a correctness flag from a pass dict or nested sample/completion dict.
    """
    for key in ("is_correct_pred", "is_correct", "correct"):
        if key in pass_obj:
            return coerce_bool(pass_obj.get(key))
    for container_key in ("sample", "completion"):
        nested = pass_obj.get(container_key)
        if isinstance(nested, dict):
            for key in ("is_correct_pred", "is_correct", "correct"):
                if key in nested:
                    return coerce_bool(nested[key])
    return None


def extract_entropy(pass_obj: Dict[str, Any], preferred: str = "entropy_answer") -> Optional[float]:
    """
    Extract a numeric entropy value from a pass dict, preferring ``preferred``.
    """
    value = pass_obj.get(preferred)
    if value is not None:
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    value = pass_obj.get("entropy")
    if value is not None:
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    return None


def extract_sample_idx(rec: Dict[str, Any], pass_obj: Dict[str, Any]) -> Optional[int]:
    """
    Try a variety of sample index field names across the record + pass dict.
    """
    for key in ("sample_idx", "sample_index", "idx", "i"):
        if key in rec:
            try:
                return int(rec[key])
            except (TypeError, ValueError):
                continue
        if key in pass_obj:
            try:
                return int(pass_obj[key])
            except (TypeError, ValueError):
                continue
    return None


def select_pass_object(
    record: Dict[str, Any],
    variant: str,
    pass2_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Locate the dict corresponding to ``variant`` ("pass1" or "pass2") in ``record``.
    """
    if variant == "pass1":
        return first_nonempty(record, PASS1_KEYS)
    if pass2_key:
        candidate = record.get(pass2_key)
        if isinstance(candidate, dict):
            return candidate
        return candidate or {}
    return first_nonempty(record, PASS2_KEYS)


def pass_with_correctness(
    record: Dict[str, Any],
    variant: str,
    pass2_key: Optional[str] = None,
) -> Optional[Tuple[Dict[str, Any], int]]:
    """
    Return ``(pass_obj, correct_flag)`` for the requested variant, or ``None``.
    """
    pass_obj = select_pass_object(record, variant, pass2_key=pass2_key)
    if not pass_obj:
        return None
    corr = extract_correct_flag(pass_obj)
    if corr is None:
        return None
    return pass_obj, int(corr)


# Statistical helpers -------------------------------------------------------------

def mcnemar_pvalue_table(
    count_00: int,
    count_01: int,
    count_10: int,
    count_11: int,
) -> float:
    """
    Compute a McNemar-style p-value for a 2×2 contingency table.
    """
    if _statsmodels_mcnemar is not None:
        try:
            result = _statsmodels_mcnemar(
                [[count_00, count_01], [count_10, count_11]],
                exact=False,
                correction=True,
            )
            return float(result.pvalue)
        except (TypeError, ValueError):
            pass

    off_diag_b = count_01
    off_diag_c = count_10
    total_off_diag = off_diag_b + off_diag_c
    if total_off_diag == 0:
        return 1.0
    chi2 = (abs(off_diag_b - off_diag_c) - 1) ** 2 / total_off_diag
    approx_p = 0.2
    for threshold, candidate in (
        (10.83, 0.001),
        (6.63, 0.01),
        (3.84, 0.05),
        (2.71, 0.10),
    ):
        if chi2 >= threshold:
            approx_p = candidate
            break
    return approx_p


def mcnemar_from_pairs(
    pairs_df: pd.DataFrame,
    col1: str,
    col2: str,
) -> Tuple[int, int, int, int, float]:
    """Count outcomes and compute a McNemar p-value from paired boolean columns."""
    num_00 = int(((pairs_df[col1] == 0) & (pairs_df[col2] == 0)).sum())
    num_01 = int(((pairs_df[col1] == 0) & (pairs_df[col2] == 1)).sum())
    num_10 = int(((pairs_df[col1] == 1) & (pairs_df[col2] == 0)).sum())
    num_11 = int(((pairs_df[col1] == 1) & (pairs_df[col2] == 1)).sum())
    p_value = mcnemar_pvalue_table(num_00, num_01, num_10, num_11)
    return num_00, num_01, num_10, num_11, p_value


def paired_t_and_wilcoxon(diffs: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute paired t-test and Wilcoxon signed-rank p-values for a vector of differences.
    """
    t_p: Optional[float] = None
    w_p: Optional[float] = None

    if _scipy_stats is None:
        return t_p, w_p
    if diffs.size < 2 or not np.all(np.isfinite(diffs)):
        return t_p, w_p

    t_result = _scipy_stats.ttest_rel(
        diffs,
        np.zeros_like(diffs),
        nan_policy="omit",
    )
    t_p = float(t_result.pvalue)

    non_zero = diffs[np.abs(diffs) > 1e-12]
    if non_zero.size >= 1:
        try:
            w_result = _scipy_stats.wilcoxon(non_zero)
            w_p = float(w_result.pvalue)
        except ValueError:
            w_p = None
    return t_p, w_p


def summarize_cluster_any(merged: pd.DataFrame) -> Dict[str, Any]:
    """
    Summarize problem-level any-correct clusters into accuracy and McNemar stats.
    """
    num_clusters = int(len(merged))
    num_any_pass1 = int(merged["any_p1"].sum())
    num_any_pass2 = int(merged["any_p2"].sum())

    acc_pass1 = num_any_pass1 / num_clusters if num_clusters else float("nan")
    acc_pass2 = num_any_pass2 / num_clusters if num_clusters else float("nan")

    lo_pass1, hi_pass1 = wilson_ci(num_any_pass1, num_clusters)
    lo_pass2, hi_pass2 = wilson_ci(num_any_pass2, num_clusters)

    (
        num_both_wrong,
        num_pass2_wins,
        num_pass1_wins,
        num_both_correct,
        p_mcnemar,
    ) = mcnemar_from_pairs(merged, "any_p1", "any_p2")

    return {
        "metric": "cluster_any",
        "n_units": num_clusters,
        "acc_pass1": acc_pass1,
        "acc_pass1_lo": lo_pass1,
        "acc_pass1_hi": hi_pass1,
        "acc_pass2": acc_pass2,
        "acc_pass2_lo": lo_pass2,
        "acc_pass2_hi": hi_pass2,
        "delta_pp": (acc_pass2 - acc_pass1) * 100.0
        if np.isfinite(acc_pass1) and np.isfinite(acc_pass2)
        else np.nan,
        "wins_pass2": num_pass2_wins,
        "wins_pass1": num_pass1_wins,
        "both_correct": num_both_correct,
        "both_wrong": num_both_wrong,
        "p_mcnemar": p_mcnemar,
        "p_ttest": None,
        "p_wilcoxon": None,
    }

def summarize_cluster_mean(merged: pd.DataFrame) -> Dict[str, Any]:
    """
    Summarize problem-level mean accuracies with paired t-test and Wilcoxon stats.
    """
    acc1 = float(merged["acc_p1"].mean()) if len(merged) else float("nan")
    acc2 = float(merged["acc_p2"].mean()) if len(merged) else float("nan")
    delta_pp = (acc2 - acc1) * 100.0 if np.isfinite(acc1) and np.isfinite(acc2) else np.nan
    diffs = (merged["acc_p2"] - merged["acc_p1"]).to_numpy(dtype=float)
    t_p, w_p = paired_t_and_wilcoxon(diffs)
    return {
        "metric": "cluster_mean",
        "n_units": int(len(merged)),
        "acc_pass1": acc1,
        "acc_pass1_lo": None,
        "acc_pass1_hi": None,
        "acc_pass2": acc2,
        "acc_pass2_lo": None,
        "acc_pass2_hi": None,
        "delta_pp": delta_pp,
        "wins_pass2": None,
        "wins_pass1": None,
        "both_correct": None,
        "both_wrong": None,
        "p_mcnemar": None,
        "p_ttest": t_p,
        "p_wilcoxon": w_p,
    }


def summarize_sample_level(pairs_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Summarize sample-level accuracy with Wilson intervals and McNemar stats.
    """
    num_pairs = int(len(pairs_df))
    num_correct_pass1 = int(pairs_df["correct1"].sum())
    num_correct_pass2 = int(pairs_df["correct2"].sum())

    acc_pass1 = num_correct_pass1 / num_pairs if num_pairs else float("nan")
    acc_pass2 = num_correct_pass2 / num_pairs if num_pairs else float("nan")

    lo_pass1, hi_pass1 = wilson_ci(num_correct_pass1, num_pairs)
    lo_pass2, hi_pass2 = wilson_ci(num_correct_pass2, num_pairs)

    (
        num_both_wrong,
        num_pass2_wins,
        num_pass1_wins,
        num_both_correct,
        p_mcnemar,
    ) = mcnemar_from_pairs(pairs_df, "correct1", "correct2")

    return {
        "metric": "sample",
        "n_units": num_pairs,
        "acc_pass1": acc_pass1,
        "acc_pass1_lo": lo_pass1,
        "acc_pass1_hi": hi_pass1,
        "acc_pass2": acc_pass2,
        "acc_pass2_lo": lo_pass2,
        "acc_pass2_hi": hi_pass2,
        "delta_pp": (acc_pass2 - acc_pass1) * 100.0
        if np.isfinite(acc_pass1) and np.isfinite(acc_pass2)
        else np.nan,
        "wins_pass2": num_pass2_wins,
        "wins_pass1": num_pass1_wins,
        "both_correct": num_both_correct,
        "both_wrong": num_both_wrong,
        "p_mcnemar": p_mcnemar,
        "p_ttest": None,
        "p_wilcoxon": None,
    }


__all__ = [
    "PASS1_KEYS",
    "PASS2_KEYS",
    "first_nonempty",
    "extract_correct_flag",
    "extract_entropy",
    "extract_sample_idx",
    "select_pass_object",
    "pass_with_correctness",
    "mcnemar_pvalue_table",
    "mcnemar_from_pairs",
    "paired_t_and_wilcoxon",
    "summarize_cluster_any",
    "summarize_cluster_mean",
    "summarize_sample_level",
]
