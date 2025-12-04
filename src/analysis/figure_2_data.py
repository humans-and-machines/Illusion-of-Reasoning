#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data-loading and preprocessing helpers for the uncertainty/correctness figures.
"""

import argparse
import importlib
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.analysis.common.uncertainty import standardize_uncertainty
from src.analysis.core import build_problem_step_from_samples, iter_pass1_records, make_formal_thresholds
from src.analysis.core.h2_uncertainty_helpers import label_formal_samples, mark_formal_pairs
from src.analysis.io import scan_jsonl_files
from src.analysis.labels import aha_words
from src.analysis.utils import (
    build_results_root_argv,
    choose_uncertainty,
    coerce_bool,
    gpt_keys_for_mode,
    problem_key_from_record,
    run_module_main_with_argv,
)


def _any_keys_true(
    pass1_dict: Dict[str, Any],
    record: Dict[str, Any],
    keys: List[str],
) -> int:
    """
    Return 1 if any of the given keys evaluates to a truthy boolean flag.
    """
    for key in keys:
        value = pass1_dict.get(key, record.get(key, None))
        if value is None:
            continue
        coerce_result = coerce_bool(value)
        if coerce_result is not None and coerce_result == 1:
            return 1
    return 0


def _build_pass1_row(
    record: Dict[str, Any],
    step_from_name: Optional[int],
    unc_field: str,
    gpt_keys: List[str],
    gate_gpt_by_words: bool,
) -> Optional[Dict[str, Any]]:
    """
    Build a single PASS-1 row with correctness, uncertainty, and Aha labels.

    Returns ``None`` when any of the required fields is missing.
    """
    pass1_dict = record.get("pass1") or {}
    if not pass1_dict:
        return None

    problem_key = problem_key_from_record(record, missing_default="unknown")
    raw_step = record.get(
        "step",
        step_from_name if step_from_name is not None else None,
    )
    if raw_step is None:
        return None
    step_value = int(raw_step)

    correct_flag = coerce_bool(pass1_dict.get("is_correct_pred"))
    if correct_flag is None:
        return None

    uncertainty_value = choose_uncertainty(pass1_dict, pref=unc_field)
    if uncertainty_value is None:
        return None

    words_flag = aha_words(pass1_dict)
    gpt_flag = _any_keys_true(pass1_dict, record, gpt_keys)
    if gate_gpt_by_words:
        gpt_flag = int(gpt_flag and words_flag)

    return {
        "problem": str(problem_key),
        "step": step_value,
        "correct": int(correct_flag),
        "uncertainty": float(uncertainty_value),
        "aha_words": int(words_flag),
        "aha_gpt": int(gpt_flag),
    }


def load_pass1_rows(
    files: List[str],
    unc_field: str,
    gpt_keys: List[str],
    gate_gpt_by_words: bool,
) -> pd.DataFrame:
    """
    Load PASS-1 rows with correctness, uncertainty, and Aha labels.
    """
    rows: List[Dict[str, Any]] = []

    for _, step_from_name, rec in iter_pass1_records(files):
        row = _build_pass1_row(
            record=rec,
            step_from_name=step_from_name,
            unc_field=unc_field,
            gpt_keys=gpt_keys,
            gate_gpt_by_words=gate_gpt_by_words,
        )
        if row is not None:
            rows.append(row)

    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        raise RuntimeError(
            "No usable PASS-1 rows found (missing labels and/or uncertainty).",
        )
    return dataframe


def wilson_ci(
    num_success: int,
    num_trials: int,
    z_score: float = 1.96,
) -> Tuple[float, float]:
    """
    Wilson confidence interval for a binomial proportion ``num_success/num_trials``.

    Returns a pair ``(lower, upper)``; if ``num_trials`` is not positive, both
    bounds are ``NaN``.
    """
    if num_trials <= 0:
        return (np.nan, np.nan)
    proportion = num_success / num_trials
    z_squared = z_score * z_score
    denominator = 1 + z_squared / num_trials
    center = (proportion + z_squared / (2 * num_trials)) / denominator
    half_width = (
        z_score
        * np.sqrt(
            (proportion * (1 - proportion) / num_trials) + (z_squared / (4 * num_trials * num_trials)),
        )
        / denominator
    )
    return (
        max(0.0, center - half_width),
        min(1.0, center + half_width),
    )


def make_edges_from_std(
    std_vals: np.ndarray,
    bins: int,
    xlim: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Build evenly spaced bin edges over standardized uncertainty values.

    Uses the 1st–99th percentile range by default, or an explicit ``xlim``.
    """
    if xlim is None:
        lower_bound, upper_bound = np.nanpercentile(std_vals, [1, 99])
    else:
        lower_bound, upper_bound = xlim
    return np.linspace(lower_bound, upper_bound, int(max(10, bins)) + 1)


def density_from_hist(
    values_std: np.ndarray,
    edges: np.ndarray,
    smooth_k: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a histogram of standardized values into a (possibly smoothed) density.
    """
    centers = 0.5 * (edges[:-1] + edges[1:])
    finite_vals = values_std[np.isfinite(values_std)]
    if finite_vals.size == 0:
        return centers, np.zeros_like(centers)

    # Filter out values that fall outside the provided edges so histogram density
    # calculations never divide by zero.
    in_bounds = finite_vals[(finite_vals >= edges[0]) & (finite_vals <= edges[-1])]
    if in_bounds.size == 0:
        return centers, np.zeros_like(centers)

    hist, _ = np.histogram(in_bounds, bins=edges, density=True)
    if not np.isfinite(hist).all():
        return centers, np.zeros_like(centers)
    if smooth_k and smooth_k > 1:
        k = max(1, int(smooth_k))
        if k % 2 == 0:
            k += 1
        kernel = np.ones(k, dtype=float) / k
        hist = np.convolve(hist, kernel, mode="same")
    return centers, hist


def compute_correct_hist(
    values_std: np.ndarray,
    correct: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    """Compute per-bin counts of correct responses."""
    hist, _ = np.histogram(
        values_std,
        bins=edges,
        weights=correct.astype(float),
    )
    return hist.astype(int)


def _standardize_uncertainty(d_all: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of d_all with an 'uncertainty_std' column."""
    return standardize_uncertainty(d_all)


def _maybe_run_rq2(args: "argparse.Namespace") -> None:
    """Optionally materialize RQ2 outputs so figures stay in sync."""
    if not args.run_rq2:
        return

    try:  # pragma: no cover - optional dependency
        rq2_module = importlib.import_module("src.analysis.rq2_analysis")
    except ImportError as exc:  # pragma: no cover - optional dependency
        print(
            f"[warn] Could not import src.analysis.rq2_analysis: {exc}",
            file=sys.stderr,
        )
        return

    argv = build_results_root_argv(args.results_root, args.split)
    run_module_main_with_argv(
        rq2_module.main,
        argv,
        prog="rq2_analysis.py",
    )


def _load_all_samples(args: "argparse.Namespace") -> pd.DataFrame:
    """Load or derive the full pass-1 sample table for plotting."""
    rq2_base = args.rq2_dir or os.path.join(args.results_root, "rq2", "h2_analysis")
    rq2_csv = os.path.join(rq2_base, "h2_all3_pass1_samples.csv")

    # Optionally materialize the RQ2 outputs so that GLM/bucket summaries are
    # kept in sync with these descriptive uncertainty plots.
    _maybe_run_rq2(args)

    if os.path.isfile(rq2_csv):
        return pd.read_csv(rq2_csv)

    # Legacy path: derive everything from raw JSONL.
    files = scan_jsonl_files(args.results_root, split_substr=args.split)
    if not files:
        raise SystemExit("No JSONL files found. Check --results_root/--split.")

    # GPT keys
    gpt_keys = gpt_keys_for_mode(args.gpt_mode)

    full_samples = load_pass1_rows(
        files,
        args.unc_field,
        gpt_keys,
        gate_gpt_by_words=args.gpt_gate_by_words,
    )

    # Formal labeling
    problem_step_df = build_problem_step_from_samples(
        full_samples,
        include_native=True,
        native_col="aha_words",
    )
    thresholds = make_formal_thresholds(
        delta1=float(args.delta1),
        delta2=float(args.delta2),
        min_prior_steps=int(args.min_prior_steps),
        delta3=(None if args.delta3 is None else float(args.delta3)),
    )
    problem_step_df = mark_formal_pairs(problem_step_df, thresholds)
    return label_formal_samples(full_samples, problem_step_df)
