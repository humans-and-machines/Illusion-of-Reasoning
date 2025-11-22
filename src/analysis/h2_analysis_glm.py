#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Step-wise GLM helpers for the H2 analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import os

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import multipletests
    from statsmodels.tools.sm_exceptions import PerfectSeparationError
    _STATS_MODELS_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover
    sm = smf = None  # type: ignore[assignment]
    multipletests = None  # type: ignore[assignment]

    class PerfectSeparationError(Exception):  # type: ignore[no-redef]
        """Fallback placeholder when statsmodels isn't installed."""

    _STATS_MODELS_IMPORT_ERROR = exc

try:
    from patsy import build_design_matrices
except ImportError:  # pragma: no cover
    build_design_matrices = None  # type: ignore[assignment]


def _require_statsmodels() -> None:
    """Ensure statsmodels is available before running heavy analyses."""
    if _STATS_MODELS_IMPORT_ERROR is not None:
        raise ImportError(
            "statsmodels is required for H2 analysis utilities. "
            "Install it with `pip install statsmodels` or `conda install statsmodels`."
        ) from _STATS_MODELS_IMPORT_ERROR


def _get_param(res, name: str, default=np.nan) -> float:
    """Robustly fetch a coefficient by name from statsmodels results."""
    try:
        params = getattr(res, "params", None)
        if params is None:
            return default
        if isinstance(params, pd.Series):
            return float(params.get(name, default))
        names = getattr(getattr(res, "model", None), "exog_names", None)
        if names is not None:
            try:
                idx = names.index(name)
                return float(params[idx])
            except ValueError:
                return default
    except (ValueError, AttributeError):
        pass
    return default


def _fetch_series_value(series_like, name: str) -> float:
    """Fetch a value from statsmodels bse/pvalue Series-like objects."""
    if series_like is None:
        return float("nan")
    try:
        if isinstance(series_like, pd.Series):
            return float(series_like.get(name, np.nan))
        if hasattr(series_like, "get"):
            return float(series_like.get(name, np.nan))
    except (TypeError, ValueError):
        return float("nan")
    return float("nan")


def _fit_glm_force_ridge(data_frame: pd.DataFrame, formula: str, ridge_penalty: float):
    """Fit Binomial GLM with ridge (L2) directly to avoid MLE overflows."""
    _require_statsmodels()
    model = smf.glm(formula, data=data_frame, family=sm.families.Binomial())
    res = model.fit_regularized(alpha=float(ridge_penalty), L1_wt=0.0)
    return res, model, "ridge"


def _fit_glm_with_ridge_if_needed(
    data_frame: pd.DataFrame,
    formula: str,
    ridge_penalty: float,
):
    """Try MLE; if unstable, fall back to ridge."""
    _require_statsmodels()
    model = smf.glm(formula, data=data_frame, family=sm.families.Binomial())
    used = "none"
    try:
        res = model.fit(cov_type="HC1")
        aha_unstable = (
            "aha" in res.params.index
            and abs(res.params["aha"]) > 10
        )
        if not np.isfinite(res.params).all() or aha_unstable:
            raise RuntimeError("Unstable MLE; switching to ridge.")
    except (np.linalg.LinAlgError, ValueError, PerfectSeparationError):
        res = model.fit_regularized(alpha=float(ridge_penalty), L1_wt=0.0)
        used = "ridge"
    return res, model, used


def _predict_from_formula(res, model, design_df):
    """Predict P(correct=1) for design_df, rebuilding the design matrix if needed."""
    try:
        return np.asarray(res.predict(design_df))
    except (ValueError, AttributeError):
        pass
    try:
        design_info = getattr(getattr(model, "data", None), "design_info", None)
        if build_design_matrices is None or design_info is None:
            raise ImportError("patsy design_info unavailable.")
        design_matrix = build_design_matrices([design_info], design_df, return_type="dataframe")[0]
        linpred = np.dot(np.asarray(design_matrix), np.asarray(res.params))
        return model.family.link.inverse(linpred)
    except (ValueError, AttributeError, ImportError):
        return np.asarray(model.predict(res.params, design_df))


def _first_finite_value(fetch_fn, keys) -> float:
    """Return the first finite value from ``fetch_fn`` given candidate keys."""
    for key in keys:
        value = fetch_fn(key)
        if np.isfinite(value):
            return float(value)
    return np.nan


def _extract_glm_statistics(result) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Collect coefficient, standard-error, and p-value dictionaries."""

    def coeff_fetch(key: str) -> float:
        return _get_param(result, key, np.nan)

    coeffs = {
        "aha": coeff_fetch("aha"),
        "unc": coeff_fetch("uncertainty_std"),
        "inter": _first_finite_value(
            coeff_fetch,
            ("aha:uncertainty_std", "uncertainty_std:aha"),
        ),
    }
    bse_series = getattr(result, "bse", pd.Series())
    pvalue_series = getattr(result, "pvalues", pd.Series())

    def std_fetch(key: str) -> float:
        return _fetch_series_value(bse_series, key)

    def pval_fetch(key: str) -> float:
        return _fetch_series_value(pvalue_series, key)

    std_errors = {
        "aha": std_fetch("aha"),
        "unc": std_fetch("uncertainty_std"),
        "inter": _first_finite_value(
            std_fetch,
            ("aha:uncertainty_std", "uncertainty_std:aha"),
        ),
    }
    p_values = {
        "aha": pval_fetch("aha"),
        "unc": pval_fetch("uncertainty_std"),
        "inter": _first_finite_value(
            pval_fetch,
            ("aha:uncertainty_std", "uncertainty_std:aha"),
        ),
    }
    return coeffs, std_errors, p_values


@dataclass
class StepwiseGlmConfig:
    """Configuration for per-step GLM fitting."""

    out_dir: str
    interaction: bool = False
    penalty: str = "ridge"
    ridge_l2: float = 1.0
    bootstrap_ame: int = 200
    ame_grid: int = 9
    fdr_alpha: float = 0.05


def _build_formula(include_interaction: bool) -> str:
    if include_interaction:
        return "correct ~ C(problem) + aha + uncertainty_std + aha:uncertainty_std"
    return "correct ~ C(problem) + aha + uncertainty_std"


def _fit_step_model(
    step_df: pd.DataFrame,
    formula: str,
    config: StepwiseGlmConfig,
):
    if config.penalty in ("ridge", "firth"):
        return _fit_glm_force_ridge(step_df, formula, config.ridge_l2)
    if config.penalty == "none":
        res, model, used = _fit_glm_with_ridge_if_needed(step_df, formula, 0.0)
        return res, model, "none" if used == "none" else "ridge"
    return _fit_glm_force_ridge(step_df, formula, config.ridge_l2)


def _balance_row(step_value: int, step_df: pd.DataFrame) -> Dict[str, Any]:
    aha0_df = step_df[step_df["aha"] == 0]
    aha1_df = step_df[step_df["aha"] == 1]
    return {
        "step": step_value,
        "n": len(step_df),
        "n_aha0": int(len(aha0_df)),
        "n_aha1": int(len(aha1_df)),
        "mean_unc_aha0": (
            float(aha0_df["uncertainty"].mean()) if len(aha0_df) else np.nan
        ),
        "mean_unc_aha1": (
            float(aha1_df["uncertainty"].mean()) if len(aha1_df) else np.nan
        ),
        "aha_ratio": float(step_df["aha"].mean()),
    }


def _init_regression_record(
    step_value: int,
    step_df: pd.DataFrame,
    penalty_label: str,
) -> Dict[str, Any]:
    return {
        "step": step_value,
        "n": len(step_df),
        "penalty": penalty_label,
        "aha_coef": np.nan,
        "aha_se": np.nan,
        "aha_z": np.nan,
        "aha_p": np.nan,
        "aha_ame": np.nan,
        "aha_ame_lo": np.nan,
        "aha_ame_hi": np.nan,
        "inter_coef": np.nan,
        "inter_se": np.nan,
        "inter_z": np.nan,
        "inter_p": np.nan,
        "unc_coef": np.nan,
        "unc_se": np.nan,
        "unc_z": np.nan,
        "unc_p": np.nan,
        "acc": step_df["correct"].mean(),
        "aha_ratio": step_df["aha"].mean(),
        "mean_uncertainty": step_df["uncertainty"].mean(),
        "naive_delta": np.nan,
    }


def _ame_at_mean_uncertainty(res, model, step_df: pd.DataFrame) -> float:
    base = step_df.copy()
    mean_uncertainty = float(base["uncertainty_std"].mean())
    base["uncertainty_std"] = mean_uncertainty
    aha_one_df = base.copy()
    aha_zero_df = base.copy()
    aha_one_df["aha"] = 1
    aha_zero_df["aha"] = 0
    prob_one = _predict_from_formula(res, model, aha_one_df)
    prob_zero = _predict_from_formula(res, model, aha_zero_df)
    return float(np.mean(prob_one - prob_zero))


def _bootstrap_ame_interval(
    step_df: pd.DataFrame,
    formula: str,
    config: StepwiseGlmConfig,
) -> Tuple[float, float]:
    if int(config.bootstrap_ame) <= 0 or len(step_df) <= 10:
        return np.nan, np.nan
    rng = np.random.default_rng(0)
    scores = np.empty(int(config.bootstrap_ame), dtype=float)
    indices = np.arange(len(step_df))
    for bootstrap_index in range(int(config.bootstrap_ame)):
        take = rng.choice(indices, size=len(step_df), replace=True)
        bootstrap_df = step_df.iloc[take].copy()
        boot_res, boot_model, _ = _fit_glm_force_ridge(bootstrap_df, formula, config.ridge_l2)
        scores[bootstrap_index] = _ame_at_mean_uncertainty(boot_res, boot_model, bootstrap_df)
    return np.nanpercentile(scores, [2.5, 97.5])


def _regression_row_no_variation(
    step_value: int,
    step_df: pd.DataFrame,
) -> Dict[str, Any]:
    record = _init_regression_record(step_value, step_df, "n/a")
    aha0_df = step_df[step_df["aha"] == 0]
    aha1_df = step_df[step_df["aha"] == 1]
    if len(aha0_df) and len(aha1_df):
        record["naive_delta"] = float(aha1_df["correct"].mean() - aha0_df["correct"].mean())
    return record


def _regression_row_with_model(
    step_value: int,
    step_df: pd.DataFrame,
    config: StepwiseGlmConfig,
) -> Dict[str, Any]:
    formula = _build_formula(config.interaction)
    res, model, used_penalty = _fit_step_model(step_df, formula, config)
    record = _init_regression_record(step_value, step_df, used_penalty)
    coeffs, std_errors, p_values = _extract_glm_statistics(res)

    record.update({
        "aha_coef": float(coeffs["aha"]),
        "unc_coef": float(coeffs["unc"]),
        "inter_coef": float(coeffs["inter"]),
        "aha_se": float(std_errors["aha"]),
        "unc_se": float(std_errors["unc"]),
        "inter_se": float(std_errors["inter"]),
        "aha_z": (
            float(coeffs["aha"] / std_errors["aha"])
            if np.isfinite(std_errors["aha"]) and std_errors["aha"]
            else np.nan
        ),
        "unc_z": (
            float(coeffs["unc"] / std_errors["unc"])
            if np.isfinite(std_errors["unc"]) and std_errors["unc"]
            else np.nan
        ),
        "inter_z": (
            float(coeffs["inter"] / std_errors["inter"])
            if np.isfinite(std_errors["inter"]) and std_errors["inter"]
            else np.nan
        ),
        "aha_p": float(p_values["aha"]),
        "unc_p": float(p_values["unc"]),
        "inter_p": float(p_values["inter"]),
    })

    record["aha_ame"] = _ame_at_mean_uncertainty(res, model, step_df)
    ame_lo, ame_hi = _bootstrap_ame_interval(step_df, formula, config)
    record["aha_ame_lo"] = float(ame_lo)
    record["aha_ame_hi"] = float(ame_hi)
    record["naive_delta"] = float(
        step_df.loc[step_df["aha"] == 1, "correct"].mean()
        - step_df.loc[step_df["aha"] == 0, "correct"].mean()
    )
    return record


def fit_stepwise_glms(
    data_frame: pd.DataFrame,
    config: StepwiseGlmConfig,
) -> pd.DataFrame:
    """Fit per-step GLMs and export regression tables/diagnostics."""
    _require_statsmodels()
    steps = sorted(data_frame["step"].unique().tolist())
    rows, bal_rows = [], []

    for step_value in steps:
        step_df = data_frame[data_frame["step"] == step_value].copy()
        if step_df.empty:
            continue
        bal_rows.append(_balance_row(step_value, step_df))
        if step_df["aha"].nunique() < 2:
            rows.append(_regression_row_no_variation(step_value, step_df))
            continue
        rows.append(_regression_row_with_model(step_value, step_df, config))

    out = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    balance_df = pd.DataFrame(bal_rows).sort_values("step").reset_index(drop=True)
    balance_path = os.path.join(config.out_dir, "h2_balance_by_step.csv")
    balance_df.to_csv(balance_path, index=False)

    if not out.empty and config.fdr_alpha is not None and multipletests is not None:
        pvals = out["aha_p"].fillna(1.0).to_numpy(dtype=float)
        rejected, adj_pvals, _, _ = multipletests(pvals, alpha=config.fdr_alpha, method="fdr_bh")
        out["aha_p_adj"] = adj_pvals
        out["aha_fdr_reject"] = rejected
    out_path = os.path.join(config.out_dir, "h2_step_regression.csv")
    out.to_csv(out_path, index=False)
    return out


def compute_pooled_step_effects(pass1_df: pd.DataFrame, ridge_l2: float) -> pd.DataFrame:
    """Return per-step pooled aha effects (log-odds) from a ridge GLM."""
    try:
        _require_statsmodels()
    except ImportError:
        print("Pooled model skipped (statsmodels not available).")
        return pd.DataFrame()

    try:
        formula = "correct ~ C(problem) + C(step) + aha + uncertainty_std + aha:C(step)"
        model = smf.glm(formula, data=pass1_df, family=sm.families.Binomial())
        res = model.fit_regularized(alpha=max(0.1, ridge_l2), L1_wt=0.0)
    except (np.linalg.LinAlgError, ValueError) as err:
        print(f"Pooled model skipped ({err}).")
        return pd.DataFrame()

    base = _get_param(res, "aha", 0.0)
    rows = []
    for step_value in sorted(pass1_df["step"].unique()):
        aha_term = f"aha:C(step)[T.{step_value}]"
        interaction_term = f"C(step)[T.{step_value}]:aha"
        delta = _get_param(res, aha_term, np.nan)
        delta = _get_param(res, interaction_term, 0.0) if not np.isfinite(delta) else delta
        effect = float(base) + (float(delta) if np.isfinite(delta) else 0.0)
        rows.append({"step": int(step_value), "aha_effect": effect})
    return pd.DataFrame(rows).sort_values("step")


__all__ = [
    "StepwiseGlmConfig",
    "fit_stepwise_glms",
    "compute_pooled_step_effects",
    "_require_statsmodels",
]
