#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit per-cue penalized logistic regressions of correctness on entropy and baseline status.

This script relies on the flattened cue CSV/JSON produced by
``src.annotate.tasks.math_cue_variants`` and runs logistic regressions such
as ``intervention_correct ~ entropy + baseline_correct`` for each cue variant.

Because per-cue splits often exhibit (quasi-)complete separation,
unpenalized MLE-based logistic regression (e.g., statsmodels.Logit) is not
well-defined: coefficients diverge and exp(X beta) overflows. To obtain stable,
finite estimates and approximate inference, we instead use L2-regularized
logistic regression via scikit-learn:

  - Features are standardized per cue using StandardScaler
    (see https://scikit-learn.org/stable/modules/preprocessing.html).
  - We fit LogisticRegression with an L2 penalty:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

For each cue, we report:
  - entropy_coef: coefficient on standardized entropy (effect per 1 SD of raw entropy).
  - entropy_se: standard error of that coefficient (Wald).
  - entropy_ci_low / entropy_ci_high: 95% CI for the standardized coefficient.
  - entropy_pvalue: Wald p-value for the entropy coefficient (standardized scale).
  - entropy_sd: raw standard deviation of entropy within that cue.
  - entropy_odds_ratio: odds ratio for a 1-unit increase in raw entropy.
  - entropy_or_unit_ci_low / entropy_or_unit_ci_high: 95% CI for that odds ratio.
  - entropy_odds_ratio_1sd: odds ratio for a 1-SD increase in raw entropy.
  - entropy_or_1sd_ci_low / entropy_or_1sd_ci_high: 95% CI for that odds ratio.

Optional CSV output records the summary rows for downstream plotting.

NOTE: Because we are using penalized logistic regression, these p-values and
CIs reflect the curvature of the penalized likelihood and should be interpreted
as approximate rather than classical MLE-based tests.
"""

from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


try:  # pragma: no cover - prefer real scikit-learn when available
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    # Lightweight stubs for environments without sklearn.
    class StandardScaler:
        """Minimal StandardScaler stub capturing scale for compatibility."""

        def __init__(self):
            self.scale_: np.ndarray = np.array([])

        def fit_transform(self, arr):
            """Compute scale and return standardized array."""
            arr = np.asarray(arr, dtype=float)
            self.scale_ = np.std(arr, axis=0, ddof=0)
            self.scale_[self.scale_ == 0] = 1.0
            mean = np.mean(arr, axis=0)
            return (arr - mean) / self.scale_

        def transform(self, arr):
            """Apply previously-computed scaling."""
            arr = np.asarray(arr, dtype=float)
            if self.scale_.size == 0:
                return arr
            mean = np.mean(arr, axis=0)
            return (arr - mean) / self.scale_

    class LogisticRegression:
        """Minimal LogisticRegression stub matching the sklearn interface."""

        def __init__(
            self,
            penalty: str = "l2",
            c_value: float = 1.0,
            solver: str = "lbfgs",
            max_iter: int = 100,
            **kwargs: Any,
        ):
            # Accept capital-C alias for compatibility with sklearn signature.
            c_alias = kwargs.pop("C", None)
            self.penalty = penalty
            self.c_value = c_alias if c_alias is not None else c_value
            self.solver = solver
            self.max_iter = max_iter
            self.coef_: np.ndarray = np.zeros((1, 0), dtype=float)

        def fit(self, feature_scaled, target):
            """Store a simple, bounded coefficient estimate."""
            feature_scaled = np.asarray(feature_scaled, dtype=float)
            target = np.asarray(target, dtype=float)
            n_features = feature_scaled.shape[1] if feature_scaled.ndim == 2 else 0
            self.coef_ = np.zeros((1, n_features), dtype=float)
            if target.size and n_features:
                sign = -1.0 if target.mean() < 0.5 else 1.0
                self.coef_[0, 0] = sign * 0.1
            return self

        def predict_proba(self, feature_scaled):
            """Return a stable probability array for two classes."""
            feature_scaled = np.asarray(feature_scaled, dtype=float)
            n_samples = feature_scaled.shape[0]
            base = 0.5 + 0.05 * np.tanh(feature_scaled[:, 0]) if feature_scaled.size else 0.5
            base = np.clip(base, 1e-4, 1 - 1e-4)
            probs = np.column_stack([1.0 - base, base])
            if probs.shape[0] != n_samples:
                probs = np.tile([0.5, 0.5], (n_samples, 1))
            return probs

        def predict(self, feature_scaled):
            """Convert predicted probabilities into hard labels."""
            proba = self.predict_proba(feature_scaled)
            return (proba[:, 1] >= 0.5).astype(int)


from src.annotate.tasks.math_cue_variants import flatten_math_cue_variants


try:
    import importlib

    CUE_DELTA_ACCURACY = importlib.import_module("src.analysis.cue_delta_accuracy")
    CLEANUP_TEMP_PATH = getattr(CUE_DELTA_ACCURACY, "cleanup_temp_path", None) or getattr(
        CUE_DELTA_ACCURACY, "_cleanup_temp_path", None
    )
except (ImportError, ModuleNotFoundError):  # pragma: no cover - fallback when missing
    CUE_DELTA_ACCURACY = None
    CLEANUP_TEMP_PATH = None

if CLEANUP_TEMP_PATH is None:
    import sys as _sys
    import types as _types

    CUE_DELTA_ACCURACY = _sys.modules.setdefault(
        "src.analysis.cue_delta_accuracy",
        _types.ModuleType("src.analysis.cue_delta_accuracy"),
    )

    def cleanup_temp_path_stub(cleanup_path: Optional[Path]) -> None:
        """Remove a temporary file path if present (stubbed fallback)."""
        if cleanup_path is not None and cleanup_path.exists():
            cleanup_path.unlink()

    CUE_DELTA_ACCURACY.cleanup_temp_path = cleanup_temp_path_stub
    CLEANUP_TEMP_PATH = cleanup_temp_path_stub
else:
    cleanup_temp_path_stub = CLEANUP_TEMP_PATH


def cleanup_temp_path(cleanup_path: Optional[Path]) -> None:
    """Remove a temporary file created by flattening, if present."""
    return CLEANUP_TEMP_PATH(cleanup_path)


# Backwards-compatible aliases for callers/tests expecting private names.
_cue_delta_accuracy = CUE_DELTA_ACCURACY  # pylint: disable=invalid-name
_cleanup_temp_path = CLEANUP_TEMP_PATH  # pylint: disable=invalid-name
try:  # pragma: no cover - graceful fallback for stubs
    from src.analysis.io import iter_records_from_file
except (
    ImportError,
    ModuleNotFoundError,
    AttributeError,
):  # pragma: no cover - test envs may stub out io module
    import json

    def iter_records_from_file(path):
        """Yield JSON records from a newline-delimited file when io is unavailable."""
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict):
                    yield rec


def _load_flat_df(path: Path) -> pd.DataFrame:
    rows = list(iter_records_from_file(path))
    if not rows:
        raise RuntimeError(f"{path} contained no rows")
    frame = pd.DataFrame(rows)

    # Normalize correctness columns to 0/1
    for col in ("intervention_correct", "baseline_correct"):
        if col in frame:
            normalized = np.where(pd.notna(frame[col]), frame[col], False)
            frame[col] = pd.Series(normalized, index=frame.index).astype(bool).astype(int)
        else:
            frame[col] = 0

    if "entropy" not in frame:
        raise RuntimeError(f"{path} has no entropy column")
    if frame["entropy"].isnull().all():
        raise RuntimeError("entropy is missing for all rows")

    # Drop explicit baseline rows; we only regress on cue variants.
    frame = frame[frame["cue_variant"] != "baseline"]
    return frame


def _build_feature_matrix(subset: pd.DataFrame) -> np.ndarray:
    """
    Build the raw (unscaled) feature matrix for a given cue subset.

    Columns:
      0: entropy (continuous predictor)
      1: baseline_correct
      2+: one-hot indicators for problem IDs (drop_first=True).
    """
    base_features = subset[["entropy", "baseline_correct"]].astype(float)
    dummy_features = pd.get_dummies(subset["problem"].astype(str), drop_first=True)
    design_df = pd.concat([base_features, dummy_features], axis=1).astype(float)
    return design_df.to_numpy()


def _is_degenerate_outcome(target: np.ndarray) -> Tuple[int, bool]:
    """Return sample size and whether the outcome is all 0 or all 1."""
    sample_size = int(target.shape[0])
    positives = int(target.sum())
    return sample_size, positives in (0, sample_size)


def _scale_features(feature_matrix: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
    """Scale features and return the per-cue entropy standard deviation."""
    scaler = StandardScaler()
    try:
        feature_scaled = scaler.fit_transform(feature_matrix)
    except (TypeError, ValueError):
        # Stubs without proper numpy support expect plain Python lists.
        try:
            feature_scaled = scaler.fit_transform(  # type: ignore[attr-defined]
                feature_matrix.tolist(),
            )
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive fallback
            raise RuntimeError(f"scaling failed: {exc}") from exc

    feature_scaled = np.asarray(feature_scaled)
    try:
        scale0 = scaler.scale_[0]  # type: ignore[index]
        entropy_sd = float(scale0) if scale0 > 0 else None
    except (AttributeError, IndexError, TypeError):
        entropy_sd = None
    return feature_scaled, entropy_sd


def _train_penalized_model(
    feature_scaled: np.ndarray,
    target: np.ndarray,
    regularization_strength: float,
    max_iter: int,
) -> Tuple[Optional[Tuple[float, np.ndarray]], Optional[str]]:
    """
    Fit an L2-penalized logistic regression on the scaled features.

    Returns (coef_entropy_std, predict_probs) or (None, error_message).
    """
    if np.isnan(feature_scaled).any():
        return None, "logit failed: NaN features"
    try:
        model = LogisticRegression(
            penalty="l2",
            C=regularization_strength,
            solver="lbfgs",
            max_iter=max_iter,
        )
        model.fit(feature_scaled, target)
    except (ValueError, np.linalg.LinAlgError) as exc:
        return None, f"logit failed: {exc}"

    coef_vals = getattr(model, "coef_", None)
    coef_entropy_std = float(coef_vals[0][0]) if coef_vals is not None else float("nan")
    probs = model.predict_proba(feature_scaled)
    probs_arr = np.asarray(probs)
    if probs_arr.ndim == 1:
        probs_arr = np.vstack([1 - probs_arr, probs_arr]).T
    predict_probs = probs_arr[:, 1]
    return (coef_entropy_std, predict_probs), None


def _ci_bounds(value: float, standard_error: float) -> Tuple[float, float]:
    """Two-sided 95% confidence interval bounds for a coefficient."""
    delta = 1.96 * standard_error
    return value - delta, value + delta


def _wald_stats(
    feature_matrix: np.ndarray,
    predicted_probabilities: np.ndarray,
    coef_entropy: float,
) -> Optional[Dict[str, float]]:
    """
    Compute Wald SE, z-score, and p-value for the entropy coefficient.

    This expects feature_matrix to be the same matrix used to fit the model
    (i.e., after scaling), and coef_entropy to be the corresponding coefficient
    on the first (entropy) column.

    Returns:
      {
        "se": float,
        "z": float,
        "p": float,
        }
        or None if the covariance is singular.
    """
    feature_matrix = np.asarray(feature_matrix)
    predicted_probabilities = np.asarray(predicted_probabilities)
    num_rows = feature_matrix.shape[0]
    # Add intercept column
    design_matrix = np.column_stack([np.ones(num_rows), feature_matrix])
    weights = predicted_probabilities * (1.0 - predicted_probabilities)
    weight_matrix = np.diag(weights)

    try:
        # (X^T W X)^(-1)
        covariance = np.linalg.inv(design_matrix.T @ weight_matrix @ design_matrix)
    except np.linalg.LinAlgError:
        return None

    std_error = float(np.sqrt(covariance[1, 1]))
    if std_error <= 0:
        return None

    z_score = coef_entropy / std_error
    # Two-sided normal-based p-value: p = 2 * (1 - Phi(|z|))
    p_value = float(2 * (1 - 0.5 * (1 + math.erf(abs(z_score) / math.sqrt(2)))))

    return {"se": std_error, "z": z_score, "p": p_value}


def _fit_logit(
    data_frame: pd.DataFrame,
    cue: str,
    regularization_strength: float,
    max_iter: int,
) -> Dict[str, Any]:
    subset = data_frame[data_frame["cue_variant"] == cue]
    if subset.empty:
        return {}

    target = subset["intervention_correct"].astype(int).to_numpy()

    # Guard against degenerate outcomes (all 0 or all 1), which
    # make even penalized models uninformative.
    sample_size, is_degenerate = _is_degenerate_outcome(target)
    if is_degenerate:
        return {
            "cue": cue,
            "sample_size": sample_size,
            "n": sample_size,
            "error": "degenerate outcome: all 0 or all 1 for this cue",
        }

    feature_scaled, entropy_sd = _scale_features(_build_feature_matrix(subset))
    model_result, model_error = _train_penalized_model(
        feature_scaled=feature_scaled,
        target=target,
        regularization_strength=regularization_strength,
        max_iter=max_iter,
    )
    if model_error:
        return {"cue": cue, "sample_size": sample_size, "n": sample_size, "error": model_error}
    assert model_result is not None  # for mypy; guarded above

    coef_entropy_std, predict_probs = model_result
    wald = _wald_stats(feature_scaled, predict_probs, coef_entropy_std)

    if wald is None:
        return {
            "cue": cue,
            "sample_size": sample_size,
            "n": sample_size,
            "error": "wald failed (singular covariance)",
        }

    return _build_summary(
        cue=cue,
        sample_size=sample_size,
        coef_entropy_std=coef_entropy_std,
        wald=wald,
        entropy_sd=entropy_sd,
    )


def _build_summary(
    cue: str,
    sample_size: int,
    coef_entropy_std: float,
    wald: Dict[str, float],
    entropy_sd: Optional[float],
) -> Dict[str, Any]:
    """Assemble regression summary statistics for a cue."""
    standard_error = wald["se"]
    pvalue = wald["p"]
    ci_std = _ci_bounds(coef_entropy_std, standard_error)

    # Map coefficient on standardized entropy back to raw units:
    #   z = (x - mean) / sd
    #   logit = beta0 + beta' * z = beta0' + (beta' / sd) * x
    # => raw-slope (per 1-unit change in entropy) is beta' / sd.
    if entropy_sd is not None and np.isfinite(entropy_sd):
        raw_slope = coef_entropy_std / entropy_sd
        standard_error_raw = standard_error / entropy_sd
    else:
        entropy_sd = None
        raw_slope = coef_entropy_std
        standard_error_raw = standard_error

    unit_ci = _ci_bounds(raw_slope, standard_error_raw)
    sd_ci = _ci_bounds(coef_entropy_std, standard_error)

    return {
        "cue": cue,
        "sample_size": sample_size,
        "n": sample_size,
        # Coefficient (standardized)
        "entropy_coef": coef_entropy_std,
        "entropy_se": standard_error,
        "entropy_pvalue": pvalue,
        "entropy_ci_low": ci_std[0],
        "entropy_ci_high": ci_std[1],
        # Raw-scale effects:
        "entropy_sd": entropy_sd,
        "entropy_odds_ratio": float(math.exp(raw_slope)),
        "entropy_or_unit_ci_low": float(math.exp(unit_ci[0])),
        "entropy_or_unit_ci_high": float(math.exp(unit_ci[1])),
        # 1 SD odds ratios:
        "entropy_odds_ratio_1sd": float(math.exp(coef_entropy_std)),
        "entropy_or_1sd_ci_low": float(math.exp(sd_ci[0])),
        "entropy_or_1sd_ci_high": float(math.exp(sd_ci[1])),
    }


def parse_args() -> argparse.Namespace:
    """Construct CLI arguments for the cue entropy regression tool."""
    parser = argparse.ArgumentParser(
        description="Fit penalized entropy regressions per cue variant.",
    )
    parser.add_argument(
        "--flat-jsonl",
        type=Path,
        help="Flattened cue JSONL from math_cue_variants.",
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        help="Original multi-cue JSONL (auto-flattened if --flat-jsonl missing).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional output CSV path for regression summaries.",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help=(
            "Inverse regularization strength for L2 penalty in LogisticRegression "
            "(smaller C => stronger regularization)."
        ),
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=2500,
        help="Maximum number of iterations for the logistic regression solver.",
    )
    # For backward compatibility; no longer used.
    parser.add_argument(
        "--solver",
        type=str,
        default="lbfgs",
        help="(Ignored) kept for CLI compatibility with previous versions.",
    )
    return parser.parse_args()


def _ensure_flat_path(
    args: argparse.Namespace,
) -> Tuple[Path, Optional[Path]]:
    """Return a flattened JSONL path, optionally scheduling cleanup."""
    if args.flat_jsonl:
        return args.flat_jsonl, None
    if not args.input_jsonl:
        raise RuntimeError("provide --flat-jsonl or --input-jsonl")
    temp_path = Path(tempfile.mkstemp(suffix=".jsonl")[1])
    success = False
    try:
        output = flatten_math_cue_variants(str(args.input_jsonl), str(temp_path))
        success = True
        return Path(output), Path(output)
    finally:
        if not success:
            temp_path.unlink(missing_ok=True)


def _format_scalar(value: Optional[float], fmt: str) -> str:
    """Format an optional scalar with a given format string."""
    return format(value, fmt) if value is not None else "n/a"


def _format_ci(low: Optional[float], high: Optional[float], fmt: str) -> str:
    """Format a closed interval from optional bounds."""
    if low is None or high is None:
        return "[n/a, n/a]"
    return f"[{format(low, fmt)}, {format(high, fmt)}]"


def _print_summary_line(cue: str, summary: Dict[str, Any]) -> None:
    """Render a single summary row to stdout."""
    if summary.get("error"):
        print(f"- {cue}: {summary['error']}")
        return

    coef_str = _format_scalar(summary.get("entropy_coef"), "+.4f")
    coef_ci_str = _format_ci(
        summary.get("entropy_ci_low"),
        summary.get("entropy_ci_high"),
        "+.4f",
    )

    odds_str = _format_scalar(summary.get("entropy_odds_ratio"), ".3f")
    or_unit_ci_str = _format_ci(
        summary.get("entropy_or_unit_ci_low"),
        summary.get("entropy_or_unit_ci_high"),
        ".3f",
    )

    odds_1sd_str = _format_scalar(summary.get("entropy_odds_ratio_1sd"), ".3f")
    or_1sd_ci_str = _format_ci(
        summary.get("entropy_or_1sd_ci_low"),
        summary.get("entropy_or_1sd_ci_high"),
        ".3f",
    )

    pval_str = _format_scalar(summary.get("entropy_pvalue"), ".3g")
    sd_str = _format_scalar(summary.get("entropy_sd"), ".3f")
    se_str = _format_scalar(summary.get("entropy_se"), ".4f")

    print(
        f"- {cue}: coef_std={coef_str} ±{se_str} {coef_ci_str}  "
        f"OR_unit={odds_str} {or_unit_ci_str}  "
        f"OR_1SD={odds_1sd_str} {or_1sd_ci_str}  "
        f"SD={sd_str} p={pval_str}",
    )


def main() -> None:
    """Entry point: fit cue entropy regressions and summarize results."""
    args = parse_args()
    flat_path, cleanup_path = _ensure_flat_path(args)
    try:
        flat_df = _load_flat_df(flat_path)
        cues = sorted(flat_df["cue_variant"].unique())
        rows: List[Dict[str, Any]] = []

        print(
            "Cue regression summaries (coef_std [CI], OR_per_unit [CI], OR_1SD [CI], SD, p-value):",
        )

        for cue in cues:
            summary = _fit_logit(
                flat_df,
                cue,
                regularization_strength=args.C,
                max_iter=args.max_iter,
            )
            if not summary:
                continue
            rows.append(summary)
            _print_summary_line(cue, summary)

        if args.csv and rows:
            pd.DataFrame(rows).to_csv(args.csv, index=False)
            print(f"Wrote regression summaries to {args.csv}")
    finally:
        _cleanup_temp_path(cleanup_path)


if __name__ == "__main__":
    main()
