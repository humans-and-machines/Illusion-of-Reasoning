#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared metrics utilities for analysis scripts.

This currently focuses on correctness extraction and a few common derived
metrics that are reused across uncertainty / shift analyses.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

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


def _correct_from_mapping(mapping: Dict[str, Any]) -> Optional[int]:
    """Return a correctness flag from a single mapping, if present."""

    for key, value in mapping.items():
        key_lower = str(key).lower()
        if any(candidate in key_lower for candidate in CORRECT_KEYS):
            coerced_bool = coerce_bool(value)
            if coerced_bool is not None:
                return int(coerced_bool)
        if key_lower in {"acc", "accuracy", "score"}:
            float_value = coerce_float(value)
            if float_value is None:
                continue
            if float_value in (0.0, 1.0):
                return int(float_value)
            if 0.0 <= float_value <= 1.0:
                return int(float_value >= 0.5)
    return None


def _children_for_traversal(obj: Any) -> list[Any]:
    """Return child objects to traverse when searching for correctness."""

    if isinstance(obj, dict):
        return list(obj.values())
    if isinstance(obj, list):
        return list(obj)
    return []


def find_correct_in_obj(obj: Any) -> Optional[int]:
    """
    Depth-first search for correctness-ish booleans or [0,1] floats inside an
    arbitrary JSON-like object (dicts/lists).
    """
    queue: list[Any] = [obj]
    while queue:
        current = queue.pop(0)
        if isinstance(current, dict):
            result = _correct_from_mapping(current)
            if result is not None:
                return result
        queue.extend(_children_for_traversal(current))
    return None


def extract_correct(obj_like: Dict[str, Any], rec: Dict[str, Any]) -> Optional[int]:
    """
    Robust correctness from an object (e.g., pass1 or pass2) with rec as
    context for gold answers.

    Merges the more permissive/canonical variants from final-plot and
    entropy_bin_regression.
    """
    # 1) Look for explicit correctness flags anywhere inside obj_like
    correct_flag = find_correct_in_obj(obj_like)
    if correct_flag is not None:
        return correct_flag

    # 2) Canonical answer comparison, allowing gold sets
    pred_canon = first_nonempty_str(
        obj_like.get("pred_answer_canon"),
        rec.get("pred_answer_canon"),
        obj_like.get("final_answer_canon"),
        rec.get("final_answer_canon"),
    )
    gold_canon = rec.get("gold_answer_canon_set") or rec.get("gold_answer_canon")
    canon_equal_result = canon_equal(pred_canon, gold_canon)
    if canon_equal_result is not None:
        return canon_equal_result

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
    comparison_op: str,
    threshold: float,
) -> Optional[int]:
    """
    Compute a Carpark-style success flag from a soft_reward field.

    comparison_op ∈ {gt, ge, eq}, threshold is a float threshold.
    """

    def _compare(soft_value: Any) -> Optional[int]:
        score_value = coerce_float(soft_value)
        if score_value is None:
            return None
        if comparison_op == "gt":
            return int(score_value > threshold)
        if comparison_op == "ge":
            return int(score_value >= threshold)
        if comparison_op == "eq":
            return int(score_value == threshold)
        return int(score_value > threshold)

    soft_reward_value = rec.get("soft_reward", pass_obj.get("soft_reward"))
    return _compare(soft_reward_value)


def make_carpark_success_fn(
    comparison_op: str,
    threshold: float,
) -> Callable[[Any], Optional[int]]:
    """
    Build a soft-reward → success comparator for Carpark-style tasks.

    ``comparison_op`` is one of ``{\"gt\", \"ge\", \"eq\"}``; ``threshold`` is the cutoff.
    """

    def _compare(soft_value: Any) -> Optional[int]:
        score_value = coerce_float(soft_value)
        if score_value is None:
            return None
        if comparison_op == "gt":
            return int(score_value > threshold)
        if comparison_op == "ge":
            return int(score_value >= threshold)
        if comparison_op == "eq":
            return int(score_value == threshold)
        return int(score_value > threshold)

    return _compare


def shift_conditional_counts(data_frame: pd.DataFrame) -> tuple[int, float, float, float]:
    """
    Compute conditional accuracy and share of shifted examples.

    Returns ``(n_total, share_shift, p_correct_shift, p_correct_noshift)``.
    """
    total_count = int(len(data_frame))
    n_shift = int((data_frame["shift"] == 1).sum())
    n_noshift = total_count - n_shift
    k_shift = (
        int(data_frame.loc[data_frame["shift"] == 1, "correct"].sum())
        if n_shift > 0
        else 0
    )
    k_noshift = (
        int(data_frame.loc[data_frame["shift"] == 0, "correct"].sum())
        if n_noshift > 0
        else 0
    )
    p_shift = (k_shift / n_shift) if n_shift > 0 else np.nan
    p_noshift = (k_noshift / n_noshift) if n_noshift > 0 else np.nan
    share = (n_shift / total_count) if total_count > 0 else np.nan
    return total_count, share, p_shift, p_noshift


def add_step_std_column(
    data_frame: pd.DataFrame,
    step_column: str = "step",
    std_column: str = "step_std",
) -> pd.DataFrame:
    """
    Return a copy of the DataFrame with an added standardized step column.
    """
    result = data_frame.copy()
    step_values = result[step_column].to_numpy(dtype=float)
    step_std = (step_values - step_values.mean()) / (step_values.std(ddof=0) + 1e-8)
    result[std_column] = step_std
    return result


def wilson_ci(num_success: int, num_trials: int) -> tuple[float, float]:
    """
    Wilson 95% confidence interval for a binomial proportion k/n.
    """
    if num_trials <= 0:
        return (np.nan, np.nan)

    z_score = 1.959963984540054
    proportion = num_success / num_trials
    denom = 1.0 + (z_score * z_score) / num_trials
    centre = proportion + (z_score * z_score) / (2.0 * num_trials)
    adj = z_score * (
        (proportion * (1.0 - proportion) + (z_score * z_score) / (4.0 * num_trials))
        / num_trials
    ) ** 0.5
    lower = (centre - adj) / denom
    upper = (centre + adj) / denom
    return (max(0.0, lower), min(1.0, upper))


def write_glm_summary_header(
    out_txt: str,
    res: Any,
    cov_type: str,
    cov_kwds: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Write a standard statsmodels GLM summary header with covariance info.

    This helper is shared across H1/H3-style scripts to avoid duplicate
    boilerplate when persisting GLM summaries.
    """
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as file_handle:
        file_handle.write(res.summary().as_text())
        file_handle.write(f"\nCovariance: {cov_type}")
        if cov_kwds and "groups" in cov_kwds:
            file_handle.write(" (clustered by problem)")


def cond_counts(rows_df: pd.DataFrame) -> Tuple[int, float, float, float]:
    """
    Compute conditional accuracy and share of shifted examples.

    The input DataFrame is expected to have boolean/indicator columns
    ``shift`` and ``correct``. The return value is
    ``(total_count, share_shift, p_correct_shift, p_correct_noshift)``.
    """
    total_count = int(len(rows_df))
    n_shift = int((rows_df["shift"] == 1).sum())
    n_no_shift = total_count - n_shift
    correct_shift = int(
        rows_df.loc[rows_df["shift"] == 1, "correct"].sum(),
    ) if n_shift > 0 else 0
    correct_no_shift = int(
        rows_df.loc[rows_df["shift"] == 0, "correct"].sum(),
    ) if n_no_shift > 0 else 0
    p_shift = (correct_shift / n_shift) if n_shift > 0 else np.nan
    p_no_shift = (correct_no_shift / n_no_shift) if n_no_shift > 0 else np.nan
    share = (n_shift / total_count) if total_count > 0 else np.nan
    return total_count, share, p_shift, p_no_shift


def glm_cov_spec(
    rows_df: pd.DataFrame,
    cluster_by: str,
) -> tuple[str, Optional[Dict[str, Any]]]:
    """
    Shared covariance configuration for statsmodels GLM fits.

    For ``cluster_by == 'problem'`` we use clustered covariance with robust
    corrections; otherwise fall back to HC1.
    """
    if cluster_by == "problem":
        groups = pd.Categorical(rows_df["problem"]).codes
        return "cluster", {
            "groups": groups,
            "use_correction": True,
            "df_correction": True,
        }
    return "HC1", None


def lazy_import_statsmodels():
    """
    Lazily import statsmodels API and formula modules.

    Raises a clear RuntimeError when statsmodels is not installed instead of
    failing at import time for modules that only conditionally depend on it.
    """
    try:
        sm_module = importlib.import_module("statsmodels.api")
        smf_module = importlib.import_module("statsmodels.formula.api")
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("statsmodels is required (pip install statsmodels)") from exc
    return sm_module, smf_module


def glm_fit_with_covariance(
    model,
    data_frame: pd.DataFrame,
    cluster_by: str,
):
    """
    Fit a statsmodels GLM with the shared covariance spec, handling older
    statsmodels versions that reject certain ``cov_kwds``.

    Returns ``(result, cov_type, cov_kwds)``.
    """
    cov_type, cov_kwds = glm_cov_spec(data_frame, cluster_by)
    try:
        result = model.fit(cov_type=cov_type, cov_kwds=(cov_kwds or {}))
    except TypeError:
        minimal_kw = (
            {"groups": cov_kwds.get("groups")}
            if cov_kwds and "groups" in cov_kwds
            else {}
        )
        result = model.fit(cov_type=cov_type, cov_kwds=minimal_kw)
    return result, cov_type, cov_kwds


def predict_formula(
    res: Any,
    model: Any,
    new_data: pd.DataFrame,
) -> np.ndarray:
    """
    Robust prediction helper for formula-based statsmodels GLMs.

    Prefers ``res.predict(df_new)`` but falls back to rebuilding the
    design matrix from the original ``design_info`` when needed, and
    finally to treating ``df_new`` as a numeric exog matrix.
    """
    try:
        return np.asarray(res.predict(new_data))
    except (TypeError, ValueError, AttributeError):
        pass

    try:
        patsy_module = importlib.import_module("patsy")
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("patsy is required for formula-based prediction") from exc

    design_info = getattr(getattr(model, "data", None), "design_info", None)
    if design_info is not None:
        build_design_matrices = getattr(patsy_module, "build_design_matrices")
        design_matrices = build_design_matrices(
            [design_info],
            new_data,
            return_type="dataframe",
        )
        design_matrix = design_matrices[0]
        linear_predictor = np.dot(
            np.asarray(design_matrix),
            np.asarray(res.params),
        )
        return model.family.link.inverse(linear_predictor)

    return np.asarray(model.predict(res.params, new_data))
