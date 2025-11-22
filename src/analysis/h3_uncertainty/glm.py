"""
Statsmodels GLM helpers for the H3 uncertainty bucket analysis.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
except ImportError:  # pragma: no cover - defer failure to runtime
    sm = smf = None


def _cov_spec(data_frame: pd.DataFrame, cluster_by: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Return ``(cov_type, cov_kwargs)`` for statsmodels GLM fits.
    """
    if cluster_by == "problem":
        groups = pd.Categorical(data_frame["problem"]).codes
        return "cluster", {"groups": groups, "use_correction": True, "df_correction": True}
    return "HC1", None


def _build_glm_formula(aha_col: str, strict_interaction_only: bool) -> str:
    """
    Assemble the statsmodels formula string for the bucket GLM.
    """
    if strict_interaction_only:
        return f"correct ~ C(problem) + step_std + {aha_col}:C(perplexity_bucket)"
    return (
        f"correct ~ C(problem) + step_std + {aha_col} + "
        f"C(perplexity_bucket) + {aha_col}:C(perplexity_bucket)"
    )


def _write_glm_summary(
    out_path: str,
    result: Any,
    cov_type: str,
    cov_kwargs: Optional[Dict[str, Any]],
) -> None:
    """
    Persist the GLM summary text alongside covariance metadata.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as file_handle:
        file_handle.write(result.summary().as_text())
        file_handle.write(f"\nCovariance: {cov_type}")
        if cov_kwargs and "groups" in cov_kwargs:
            file_handle.write(" (clustered by problem)")


def _compute_bucket_rows(result: Any, glm_df: pd.DataFrame, aha_col: str) -> List[Dict[str, Any]]:
    """
    Compute per-bucket AME-style deltas from a fitted GLM result.
    """
    bucket_rows: List[Dict[str, Any]] = []
    for bucket in sorted(glm_df["perplexity_bucket"].unique().tolist()):
        base = glm_df.copy()
        alt = glm_df.copy()
        alt[aha_col] = 1
        base[aha_col] = 0
        alt["perplexity_bucket"] = bucket
        base["perplexity_bucket"] = bucket
        ame_bucket = float(np.mean(result.predict(alt) - result.predict(base)))
        subset = glm_df[glm_df["perplexity_bucket"] == bucket]
        bucket_rows.append(
            {
                "bucket": bucket,
                "N": int(len(subset)),
                "share_aha": float(subset[aha_col].mean()),
                "AME_bucket": ame_bucket,
            },
        )
    return bucket_rows


def fit_glm_bucket_interaction(
    data_frame: pd.DataFrame,
    aha_col: str,
    strict_interaction_only: bool,
    cluster_by: str,
    out_txt: str,
) -> Tuple[Dict[str, Any], Any]:
    """
    Fit a GLM that captures accuracy deltas across uncertainty buckets.
    """
    if sm is None or smf is None:
        raise RuntimeError("statsmodels is required (pip install statsmodels)")

    glm_df = data_frame.copy()
    glm_df["step_std"] = (glm_df["step"] - glm_df["step"].mean()) / (
        glm_df["step"].std(ddof=0) + 1e-8
    )
    glm_df = glm_df[~glm_df["perplexity_bucket"].isna()].copy()

    formula = _build_glm_formula(aha_col, strict_interaction_only)
    model = smf.glm(formula, data=glm_df, family=sm.families.Binomial())
    cov_type, cov_kwds = _cov_spec(glm_df, cluster_by)
    fit_kwargs = cov_kwds or {}
    try:
        result = model.fit(cov_type=cov_type, cov_kwds=fit_kwargs)
    except TypeError:
        fallback_kwargs = (
            {"groups": fit_kwargs["groups"]} if "groups" in fit_kwargs else {}
        )
        result = model.fit(cov_type=cov_type, cov_kwds=fallback_kwargs)

    _write_glm_summary(out_txt, result, cov_type, cov_kwds)
    bucket_rows = _compute_bucket_rows(result, glm_df, aha_col)

    summary = {
        "N": int(len(glm_df)),
        "acc_overall": float(glm_df["correct"].mean()),
        "bucket_rows": bucket_rows,
    }
    return summary, result


def bucket_group_accuracy(records: pd.DataFrame, aha_col: str) -> pd.DataFrame:
    """
    Compute per-bucket accuracy for aha/non-aha groups.
    """
    grouped = (
        records.groupby(["perplexity_bucket", aha_col], as_index=False)
        .agg(n=("correct", "size"), k=("correct", "sum"))
    )
    grouped["accuracy"] = grouped["k"] / grouped["n"]
    grouped = grouped.rename(columns={aha_col: "aha"})
    return grouped[["perplexity_bucket", "aha", "n", "k", "accuracy"]].copy()
