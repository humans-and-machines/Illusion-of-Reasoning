"""Data loading and aggregation helpers for Figure 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..io import iter_records_from_file, scan_jsonl_files
from ..labels import aha_gpt_for_rec, aha_words
from ..utils import coerce_bool
from ..utils import nat_step_from_path as utils_nat_step_from_path
from ..utils import slugify as utils_slugify
from . import FORMAL_REQUIRED_COLUMNS


slugify = utils_slugify
nat_step_from_path = utils_nat_step_from_path


def scan_files(root: str, split_substr: Optional[str]) -> List[str]:
    """Collect JSONL files under ``root`` optionally filtered by ``split_substr``."""
    return scan_jsonl_files(root, split_substr)


def _problem_identifier(rec: Dict[str, Any]) -> str:
    """Best-effort problem identifier across datasets."""
    candidate = rec.get("problem") or rec.get("clue") or rec.get("row_key")
    if candidate is not None:
        return str(candidate)
    dataset_index = rec.get("dataset_index")
    return f"idx:{dataset_index}" if dataset_index is not None else "unknown"


def _row_from_pass1_record(
    rec: Dict[str, Any],
    domain: str,
    step_hint: Optional[int],
    gpt_keys: List[str],
    gpt_subset_native: bool,
) -> Optional[Dict[str, Any]]:
    if not isinstance(rec, dict):
        return None
    pass1 = rec.get("pass1") or {}
    if not isinstance(pass1, dict) or not pass1:
        return None

    step_value = rec.get("step", step_hint if step_hint is not None else None)
    if step_value is None:
        return None
    correct = coerce_bool(pass1.get("is_correct_pred"))
    if correct is None:
        return None

    row = {
        "domain": str(domain),
        "step": int(step_value),
        "problem": _problem_identifier(rec),
        "aha_native": int(aha_words(pass1)),
        "aha_gpt": int(aha_gpt_for_rec(pass1, rec, gpt_subset_native, gpt_keys, domain)),
        "correct": int(correct),
    }
    return row


def load_pass1_samples_multi(
    files_by_domain: Dict[str, List[str]],
    gpt_keys: List[str],
    gpt_subset_native: bool,
) -> pd.DataFrame:
    """Load pass-1 records for multiple domains."""
    rows: List[Dict[str, Any]] = []
    for domain, files in files_by_domain.items():
        if not files:
            continue
        for path in files:
            step_from_name = nat_step_from_path(path)
            for record in iter_records_from_file(path):
                parsed_row = _row_from_pass1_record(
                    record,
                    str(domain),
                    step_from_name,
                    gpt_keys,
                    gpt_subset_native,
                )
                if parsed_row:
                    rows.append(parsed_row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit("No PASS-1 rows found across provided roots. Check paths/--split.")
    return frame


def build_problem_step(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate correctness and Aha flags at the (problem, step) level."""
    group_keys = ["domain", "step", "problem"] if "domain" in data_frame.columns else ["step", "problem"]
    base = (
        data_frame.groupby(group_keys, as_index=False)
        .agg(
            n_samples=("correct", "size"),
            freq_correct=("correct", "mean"),
            aha_any_gpt=("aha_gpt", "max"),
            aha_rate_gpt=("aha_gpt", "mean"),
            aha_any_native=("aha_native", "max"),
            aha_rate_native=("aha_native", "mean"),
        )
        .sort_values(group_keys)
        .reset_index(drop=True)
    )

    def _pcs_row(group: pd.DataFrame) -> pd.Series:
        mask = group["aha_gpt"] == 1
        if mask.any():
            return pd.Series({"p_correct_given_shift": float(group.loc[mask, "correct"].mean())})
        return pd.Series({"p_correct_given_shift": np.nan})

    try:
        pcs = data_frame.groupby(group_keys).apply(_pcs_row, include_groups=False)
    except TypeError:  # pragma: no cover - older pandas
        pcs = data_frame.groupby(group_keys).apply(_pcs_row)
    pcs = pcs.reset_index()
    problem_step_df = base.merge(pcs, on=group_keys, how="left")
    for column in ("n_samples", "aha_any_gpt", "aha_any_native"):
        problem_step_df[column] = problem_step_df[column].astype(int)
    for column in ("freq_correct", "aha_rate_gpt", "aha_rate_native", "p_correct_given_shift"):
        problem_step_df[column] = problem_step_df[column].astype(float)
    return problem_step_df


@dataclass(frozen=True)
class FormalCriteria:
    """Thresholds needed to mark Formal Aha pairs."""

    delta1: float
    delta2: float
    min_prior_steps: int
    delta3: Optional[float]


@dataclass(frozen=True)
class FormalSignals:
    """Container for per-step accuracy/aha signals."""

    freq: np.ndarray
    rate: np.ndarray
    shift_flags: np.ndarray
    p_correct_shift: np.ndarray


@dataclass(frozen=True)
class TrendSeries:
    """Arrays used to compute weighted trend diagnostics."""

    step: np.ndarray
    ratio: np.ndarray
    weights: np.ndarray
    ratio_mean: float


def _meets_formal_condition(
    signals: FormalSignals,
    index: int,
    criteria: FormalCriteria,
) -> bool:
    if index < criteria.min_prior_steps:
        return False
    prior_fail = float(np.max(signals.freq[:index])) < criteria.delta1
    prior_stable = float(np.max(signals.rate[:index])) < criteria.delta2
    shift_ok = int(signals.shift_flags[index]) == 1
    gain_ok = True
    if criteria.delta3 is not None:
        value = signals.p_correct_shift[index]
        gain_ok = np.isfinite(value) and (value - signals.freq[index]) > criteria.delta3
    return bool(prior_fail and prior_stable and shift_ok and gain_ok)


def _formal_flags_for_subset(subset: pd.DataFrame, criteria: FormalCriteria) -> np.ndarray:
    """Return 0/1 flags for a single problem (already sorted by step)."""
    signals = FormalSignals(
        freq=subset["freq_correct"].to_numpy(float),
        rate=subset["aha_rate_gpt"].to_numpy(float),
        shift_flags=subset["aha_any_gpt"].to_numpy(int),
        p_correct_shift=subset["p_correct_given_shift"].to_numpy(float),
    )
    local_flags = np.zeros(len(subset), dtype=int)
    for index in range(len(subset)):
        local_flags[index] = int(
            _meets_formal_condition(signals, index, criteria),
        )
    return local_flags


def mark_formal_pairs(
    problem_step_df: pd.DataFrame,
    delta1: float = 0.20,
    delta2: float = 0.20,
    min_prior_steps: int = 2,
    delta3: Optional[float] = None,
) -> pd.DataFrame:
    """Mark rows that satisfy the Formal Aha definition."""
    missing = FORMAL_REQUIRED_COLUMNS - set(problem_step_df.columns)
    if missing:
        raise ValueError(f"mark_formal_pairs: missing columns {missing}")

    group_cols = ["problem"]
    if "domain" in problem_step_df.columns:
        group_cols = ["domain"] + group_cols

    sorted_df = problem_step_df.sort_values(group_cols + ["step"]).reset_index(drop=True).copy()
    flags = np.zeros(len(sorted_df), dtype=int)
    criteria = FormalCriteria(
        delta1=delta1,
        delta2=delta2,
        min_prior_steps=min_prior_steps,
        delta3=delta3,
    )
    for _, subset in sorted_df.groupby(group_cols, sort=False):
        ordered = subset.sort_values("step")
        local_flags = _formal_flags_for_subset(ordered, criteria)
        flags[ordered.index.to_numpy()] = local_flags
    sorted_df["aha_formal"] = flags
    return sorted_df


def _bootstrap_ratio_interval(
    column_values: np.ndarray,
    num_bootstrap: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """Return (lo, hi) bootstrap percentiles for the mean of ``column_values``."""
    sample_count = column_values.size
    if num_bootstrap <= 0 or sample_count <= 1:
        return np.nan, np.nan
    bootstrap_means = np.empty(num_bootstrap, dtype=float)
    for draw_index in range(num_bootstrap):
        take = rng.integers(0, sample_count, sample_count)
        bootstrap_means[draw_index] = float(column_values[take].mean())
    lo_ci, hi_ci = np.percentile(bootstrap_means, [2.5, 97.5])
    return float(lo_ci), float(hi_ci)


def bootstrap_problem_ratio(
    problem_step_df: pd.DataFrame,
    column: str,
    num_bootstrap: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Aggregate per-step accuracy ratios with optional bootstrap CIs."""
    if column not in problem_step_df.columns:
        raise KeyError(f"bootstrap_problem_ratio: column '{column}' not found.")
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    for step_value, subset in problem_step_df.groupby("step"):
        value_array = subset[column].astype(int).to_numpy()
        sample_count = len(value_array)
        if sample_count == 0:
            rows.append(
                {
                    "step": int(step_value),
                    "k": 0,
                    "n": 0,
                    "ratio": np.nan,
                    "lo": np.nan,
                    "hi": np.nan,
                },
            )
            continue
        mean_ratio = float(value_array.mean())
        lo_ci, hi_ci = _bootstrap_ratio_interval(value_array, int(num_bootstrap), rng)
        rows.append(
            {
                "step": int(step_value),
                "k": int(value_array.sum()),
                "n": int(sample_count),
                "ratio": mean_ratio,
                "lo": float(lo_ci),
                "hi": float(hi_ci),
            }
        )
    return pd.DataFrame(rows).sort_values("step")


def fit_trend_wls(
    ratio_df: pd.DataFrame,
) -> Tuple[float, float, float, float, float, np.ndarray, np.ndarray]:
    """Fit a weighted least-squares trend line for per-step ratios."""
    step_values = ratio_df["step"].to_numpy(dtype=float)
    ratio_values = ratio_df["ratio"].to_numpy(dtype=float)
    weights = ratio_df["n"].to_numpy(dtype=float)
    valid_mask = np.isfinite(step_values) & np.isfinite(ratio_values) & np.isfinite(weights) & (weights > 0)
    step_values = step_values[valid_mask]
    ratio_values = ratio_values[valid_mask]
    weights = weights[valid_mask]
    if step_values.size < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan, step_values, ratio_values
    return _weighted_trend_from_arrays(step_values, ratio_values, weights)


def _weighted_trend_from_arrays(
    step_values: np.ndarray,
    ratio_values: np.ndarray,
    weights: np.ndarray,
) -> Tuple[float, float, float, float, float, np.ndarray, np.ndarray]:
    """Return slope/intercept/fit arrays for weighted trend."""
    weighted_step_mean = np.average(step_values, weights=weights)
    weighted_ratio_mean = np.average(ratio_values, weights=weights)
    numerator = np.average(
        (step_values - weighted_step_mean) * (ratio_values - weighted_ratio_mean),
        weights=weights,
    )
    denominator = np.average(
        (step_values - weighted_step_mean) ** 2,
        weights=weights,
    )
    slope = numerator / denominator if denominator > 0 else np.nan
    intercept = weighted_ratio_mean - slope * weighted_step_mean if np.isfinite(slope) else np.nan
    series = TrendSeries(
        step=step_values,
        ratio=ratio_values,
        weights=weights,
        ratio_mean=weighted_ratio_mean,
    )
    metrics = _weighted_trend_metrics(series, slope, intercept)
    x_fit = np.linspace(step_values.min(), step_values.max(), 200)
    y_fit = intercept + slope * x_fit
    return (*metrics, x_fit, y_fit)


def _weighted_trend_metrics(
    series: TrendSeries,
    slope: float,
    intercept: float,
) -> Tuple[float, float, float, float, float]:
    """Compute slope/intercept/range stats separated from yhat arrays."""
    if not np.isfinite(slope) or not np.isfinite(intercept):
        return np.nan, np.nan, np.nan, np.nan, np.nan
    yhat = intercept + slope * series.step
    sse = np.sum(series.weights * (series.ratio - yhat) ** 2)
    sst = np.sum(series.weights * (series.ratio - series.ratio_mean) ** 2)
    r_squared = 1.0 - (sse / sst) if sst > 0 else np.nan
    slope_per_1k = slope * 1000.0
    delta_range = slope * (series.step.max() - series.step.min())
    return float(slope), float(intercept), float(slope_per_1k), float(delta_range), float(r_squared)


def _iter_domain_step_groups(problem_step_df: pd.DataFrame):
    """Yield (domain, step, subset) tuples, injecting an 'All' domain if needed."""
    if "domain" in problem_step_df.columns:
        yield from problem_step_df.groupby(["domain", "step"], sort=False)
        return
    dup = problem_step_df.copy()
    dup["domain"] = "All"
    yield from dup.groupby(["domain", "step"], sort=False)


def _positive_delta_flag(subset: pd.DataFrame) -> bool:
    """Return True when the delta threshold is exceeded for the subset."""
    p_shift = subset["p_correct_given_shift"].to_numpy()
    mask = np.isfinite(p_shift) & (subset["aha_any_gpt"].to_numpy() == 1)
    if mask.any():
        delta = subset.loc[mask, "p_correct_given_shift"].to_numpy() - subset.loc[mask, "freq_correct"].to_numpy()
        return bool(np.nanmean(delta) > 0.12)
    return False


def build_positive_delta_flags(problem_step_df: pd.DataFrame) -> Dict[str, Dict[int, bool]]:
    """Return green-dot flags keyed by domain and step."""
    flags: Dict[str, Dict[int, bool]] = {}
    for (domain, step_value), subset in _iter_domain_step_groups(problem_step_df):
        flags.setdefault(str(domain), {})[int(step_value)] = _positive_delta_flag(subset)
    return flags


def parse_float_list(value: str) -> List[float]:
    """Parse a comma-separated string into floats."""
    return [float(piece.strip()) for piece in value.split(",") if piece.strip()]


__all__ = [
    "slugify",
    "nat_step_from_path",
    "scan_files",
    "coerce_bool",
    "load_pass1_samples_multi",
    "build_problem_step",
    "mark_formal_pairs",
    "bootstrap_problem_ratio",
    "fit_trend_wls",
    "build_positive_delta_flags",
    "parse_float_list",
]
