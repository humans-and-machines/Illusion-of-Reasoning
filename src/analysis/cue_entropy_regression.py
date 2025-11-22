#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit per-cue logistic regressions of correctness on entropy and baseline status.

This script relies on the flattened cue CSV/JSON produced by
``src.annotate.tasks.math_cue_variants`` and runs logistic regressions such
as ``intervention_correct ~ entropy + baseline_correct`` for each cue variant.
The primary output is the entropy coefficient (estimate, p-value, odds ratio)
so you can compare how the cue-adjusted accuracy shifts with entropy.

Optional CSV output records the summary rows for downstream plotting.
"""

from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.annotate.tasks.math_cue_variants import flatten_math_cue_variants
from src.analysis.cue_delta_accuracy import _cleanup_temp_path
from src.analysis.io import iter_records_from_file


def _load_flat_df(path: Path) -> pd.DataFrame:
    rows = list(iter_records_from_file(path))
    if not rows:
        raise RuntimeError(f"{path} contained no rows")
    frame = pd.DataFrame(rows)
    for col in ("intervention_correct", "baseline_correct"):
        if col in frame:
            frame[col] = frame[col].fillna(False).astype(int)
        else:
            frame[col] = 0
    if "entropy" not in frame:
        raise RuntimeError(f"{path} has no entropy column")
    if frame["entropy"].isnull().all():
        raise RuntimeError("entropy is missing for all rows")
    frame["entropy_rank"] = frame["entropy"].rank(method="dense")
    frame = frame[frame["cue_variant"] != "baseline"]
    return frame


def _build_feature_matrix(subset: pd.DataFrame) -> np.ndarray:
    base_features = subset[["entropy_rank", "baseline_correct"]].astype(float)
    dummy_features = pd.get_dummies(subset["problem"].astype(str), drop_first=True)
    design_df = pd.concat([base_features, dummy_features], axis=1).astype(float)
    return design_df.to_numpy()


def _wald_pvalue(
    feature_matrix: np.ndarray,
    predicted_probabilities: np.ndarray,
    coef_entropy: float,
) -> Optional[float]:
    design_matrix = np.column_stack([np.ones(len(feature_matrix)), feature_matrix])
    weight_matrix = np.diag(predicted_probabilities * (1 - predicted_probabilities))
    try:
        covariance = np.linalg.inv(design_matrix.T @ weight_matrix @ design_matrix)
    except np.linalg.LinAlgError:
        return None
    se_entropy = float(np.sqrt(covariance[1, 1]))
    if se_entropy <= 0:
        return None
    z_score = coef_entropy / se_entropy
    return float(2 * (1 - 0.5 * (1 + math.erf(abs(z_score) / math.sqrt(2)))))


def _fit_logit(
    data_frame: pd.DataFrame,
    cue: str,
) -> Dict[str, Any]:
    subset = data_frame[data_frame["cue_variant"] == cue]
    if subset.empty:
        return {}
    feature_matrix = _build_feature_matrix(subset)
    target = subset["intervention_correct"].astype(int).to_numpy()
    try:
        model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        model.fit(feature_matrix, target)
    except ValueError:
        return {"cue": cue, "error": "logit failed"}

    coef_entropy = float(model.coef_[0][0])
    predict_probs = model.predict_proba(feature_matrix)[:, 1]
    pvalue = _wald_pvalue(feature_matrix, predict_probs, coef_entropy)
    odds_ratio = float(np.exp(coef_entropy))
    return {
        "cue": cue,
        "n": int(subset.shape[0]),
        "entropy_coef": coef_entropy,
        "entropy_pvalue": pvalue,
        "entropy_odds_ratio": odds_ratio,
    }


def parse_args() -> argparse.Namespace:
    """Construct CLI arguments for the cue entropy regression tool."""
    parser = argparse.ArgumentParser(
        description="Fit entropy regressions per cue variant.",
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
    try:
        output = flatten_math_cue_variants(str(args.input_jsonl), str(temp_path))
        return Path(output), Path(output)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def main() -> None:
    """Entry point: fit cue entropy regressions and summarize results."""
    args = parse_args()
    flat_path, cleanup_path = _ensure_flat_path(args)
    try:
        flat_df = _load_flat_df(flat_path)
        cues = sorted(flat_df["cue_variant"].unique())
        rows: List[Dict[str, Any]] = []
        print("Cue regression summaries (entropy coefficient, exp(coef), p-value):")
        for cue in cues:
            summary = _fit_logit(flat_df, cue)
            if not summary:
                continue
            rows.append(summary)
            if summary.get("error"):
                print(f"- {cue}: {summary['error']}")
                continue
            coef = summary["entropy_coef"]
            odds = summary["entropy_odds_ratio"]
            pval = summary["entropy_pvalue"]
            odds_str = f"{odds:.3f}" if odds is not None else "n/a"
            pval_str = f"{pval:.3g}" if pval is not None else "n/a"
            print(
                f"- {cue}: coef={coef:+.4f} (exp={odds_str}) p={pval_str}",
            )
        if args.csv and rows:
            pd.DataFrame(rows).to_csv(args.csv, index=False)
            print(f"Wrote regression summaries to {args.csv}")
    finally:
        _cleanup_temp_path(cleanup_path)


if __name__ == "__main__":
    main()
