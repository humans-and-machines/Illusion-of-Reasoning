#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
H3: Can Aha!/Second-Pass Help When the Model is Uncertain?

We test whether the *second pass* (phase=1) improves accuracy more when the
model is uncertain, using PASS-1 uncertainty as the moderator.

Inputs
------
- Root directory containing step*/.../*.jsonl produced by your two-pass runners.
  Each JSON line has keys:
    problem/clue/row_key, step, sample_idx,
    pass1{...}, pass2{...} with fields:
      is_correct_pred (bool), entropy / entropy_answer / entropy_think (floats)

What this script builds
-----------------------
1) Pair-level (wide) table: one row per (problem, step, sample_idx)
   with PASS-1/2 correctness and PASS-1 uncertainty.

2) Long table: duplicates each pair into two rows:
      phase=0 → PASS-1, phase=1 → PASS-2
   Includes: correct (0/1), uncertainty (from PASS-1), bucket (by quantiles)

3) Pooled GLM (Binomial, robust HC1 SE):
      correct ~ C(problem) + C(step) + phase + uncertainty_std + phase:uncertainty_std
   Effect of "phase" tells us if second-pass helps on average; the interaction
   tells us if that help grows with uncertainty.

4) Bucket GLM (heterogeneous effect across uncertainty buckets):
      correct ~ C(problem) + C(step) + phase + C(bucket) + phase:C(bucket)
   Also computes per-bucket AME of toggling phase 0→1.

5) Plots:
   - Accuracy by uncertainty bucket for phase 0 vs 1
   - Per-bucket AME (phase toggle)

CLI
---
python h3-analysis.py /path/to/results \
  --split test \
  --uncertainty_field entropy_answer \
  --num_buckets 4

Notes
-----
- Uses PASS-1 uncertainty for both phases (keeps moderator fixed within pair).
- Robust to missing fields; pairs without both passes are dropped from modeling.
"""

import os
import argparse
import importlib
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

from src.analysis.core import iter_pass1_records
from src.analysis.io import scan_jsonl_files
from src.analysis.metrics import lazy_import_statsmodels
from src.analysis.utils import coerce_bool


def _get_pyplot():
    """
    Lazily import matplotlib.pyplot and configure the backend.

    This avoids a hard import-time dependency on matplotlib in environments
    where plotting is not required.
    """
    try:
        pyplot_mod = importlib.import_module("matplotlib.pyplot")
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "matplotlib is required for H3 plots; "
            "install it with 'pip install matplotlib'.",
        ) from exc
    pyplot_mod.switch_backend("Agg")
    return pyplot_mod

# -------------------------- data loading -------------------------

def load_pairs(files: List[str], uncertainty_field: str = "entropy") -> pd.DataFrame:
    """
    Build a PAIRS (wide) DataFrame with columns:
      problem, step, sample_idx,
      correct_p1, correct_p2,
      unc1, unc2 (unc2 kept for reference, not used in models),
      source_file
    Only keep pairs that have both pass1 and pass2 correctness.
    """
    rows: List[Dict[str, Any]] = []

    for path, step_from_name, record in iter_pass1_records(files):
        # Identify the "problem" robustly across runners
        problem_key = (
            record.get("problem")
            or record.get("clue")
            or record.get("row_key")
        )
        if problem_key is None:
            problem_key = (
                f"idx:{record.get('dataset_index')}"
                if record.get("dataset_index") is not None
                else "unknown"
            )

        step = record.get(
            "step",
            step_from_name if step_from_name is not None else None,
        )
        if step is None:
            continue

        pass1 = record.get("pass1") or {}
        pass2 = record.get("pass2") or {}
        if not pass1 or not pass2:
            # Need both passes for H3 comparisons
            continue

        correct_pass1 = coerce_bool(pass1.get("is_correct_pred"))
        correct_pass2 = coerce_bool(pass2.get("is_correct_pred"))
        if correct_pass1 is None or correct_pass2 is None:
            continue

        # Uncertainty (we'll use PASS-1 as moderator)
        uncertainty_pass1_raw = pass1.get(uncertainty_field)
        uncertainty_pass2_raw = pass2.get(uncertainty_field)

        rows.append(
            {
                "problem": str(problem_key),
                "step": int(step),
                "sample_idx": record.get("sample_idx", None),
                "correct_p1": int(correct_pass1),
                "correct_p2": int(correct_pass2),
                "unc1": None if uncertainty_pass1_raw is None else float(uncertainty_pass1_raw),
                "unc2": None if uncertainty_pass2_raw is None else float(uncertainty_pass2_raw),
                "source_file": path,
            },
        )

    pairs_df = pd.DataFrame(rows)
    if pairs_df.empty:
        raise RuntimeError("No pairs found with both PASS-1 and PASS-2. "
                           "Check --split, paths, or that pass2 exists in logs.")
    return pairs_df

def pairs_to_long(
    df_pairs: pd.DataFrame,
    num_buckets: int = 4,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert pairs→long:
      phase=0 (pass1), phase=1 (pass2)
    Uncertainty is taken from PASS-1 (unc1) for both rows.

    Adds:
      - uncertainty_std (z-score)
      - bucket (0..num_buckets-1) by quantiles on unc1
    """
    # Drop rows without unc1 (for modeling); keep a copy for descriptive CSV.
    df_pairs_model = df_pairs.dropna(subset=["unc1"]).copy()

    long_rows: List[Dict[str, Any]] = []
    for _, row in df_pairs_model.iterrows():
        pair_id = f"{row['problem']}||{int(row['step'])}||{row.get('sample_idx', 'NA')}"
        for phase in (0, 1):
            long_rows.append(
                {
                    "problem": row["problem"],
                    "step": int(row["step"]),
                    "pair_id": pair_id,
                    "phase": phase,  # 0=P1, 1=P2
                    "correct": int(
                        row["correct_p2"] if phase == 1 else row["correct_p1"]
                    ),
                    "uncertainty": float(row["unc1"]),
                },
            )
    long_df = pd.DataFrame(long_rows)

    # Standardize uncertainty
    mean_uncertainty = long_df["uncertainty"].mean()
    std_uncertainty = long_df["uncertainty"].std(ddof=0)
    long_df["uncertainty_std"] = (
        long_df["uncertainty"] - mean_uncertainty
    ) / (std_uncertainty + 1e-8)

    # Buckets based on PASS-1 uncertainty distribution (shared across phases)
    # Compute on unique pairs to avoid double-counting
    unique_uncertainty = df_pairs_model[["unc1"]].copy()
    unique_uncertainty = unique_uncertainty.rename(columns={"unc1": "unc"})
    try:
        unique_uncertainty["bucket"] = pd.qcut(
            unique_uncertainty["unc"],
            q=num_buckets,
            labels=False,
            duplicates="drop",
        )
    except ValueError:
        # Not enough unique values; fall back to single bucket
        unique_uncertainty["bucket"] = 0
    # Map back
    edges = sorted(unique_uncertainty["bucket"].dropna().unique().tolist())
    if not edges:
        long_df["bucket"] = 0
    else:
        # Build a simple rank-based bucketer against the original unc1 values
        # to keep consistent with qcut above:
        # Create a mapping by merging on unc
        df_pairs_model = df_pairs_model.merge(
            unique_uncertainty[["unc", "bucket"]].drop_duplicates(),
            left_on="unc1",
            right_on="unc",
            how="left",
        )
        if "pair_id" not in df_pairs_model.columns:
            df_pairs_model["pair_id"] = (
                df_pairs_model["problem"].astype(str)
                + "||"
                + df_pairs_model["step"].astype(str)
                + "||"
                + df_pairs_model["sample_idx"].astype(str)
            )
        bucket_map = dict(
            zip(
                df_pairs_model["pair_id"],
                df_pairs_model["bucket"],
            ),
        )
        # annotate long_df
        long_df["bucket"] = long_df["pair_id"].map(bucket_map).fillna(0).astype(int)

    # Ensure categorical types for fixed effects
    long_df["problem"] = long_df["problem"].astype(str)
    long_df["step"] = long_df["step"].astype(int)

    return df_pairs_model, long_df

# --------------------------- modeling -----------------------------

def _compute_phase_ame(result, data_frame: pd.DataFrame) -> float:
    """Compute the AME of toggling phase from 0→1."""
    df_phase_one = data_frame.copy()
    df_phase_zero = data_frame.copy()
    df_phase_one["phase"] = 1
    df_phase_zero["phase"] = 0
    predictions_one = result.predict(df_phase_one)
    predictions_zero = result.predict(df_phase_zero)
    return float(np.mean(predictions_one - predictions_zero))


def fit_pooled_glm(
    data_frame: pd.DataFrame,
    out_txt: str,
) -> Dict[str, float]:
    """
    Fit a pooled GLM with phase and uncertainty, returning key coefficients.

    Model: correct ~ C(problem) + C(step) + phase + uncertainty_std + phase:uncertainty_std
    """
    statsmodels_module, statsmodels_formula = lazy_import_statsmodels()

    model = statsmodels_formula.glm(
        "correct ~ C(problem) + C(step) + phase + uncertainty_std + phase:uncertainty_std",
        data=data_frame,
        family=statsmodels_module.families.Binomial(),
    )
    result = model.fit(cov_type="HC1")

    with open(out_txt, "w", encoding="utf-8") as file_handle:
        file_handle.write(result.summary().as_text())
        file_handle.write("\n")

    params = result.params
    standard_errors = result.bse
    pvalues = result.pvalues

    stats = {
        "b_phase": float(params.get("phase", np.nan)),
        "se_phase": float(standard_errors.get("phase", np.nan)),
        "p_phase": float(pvalues.get("phase", np.nan)),
        "b_unc": float(params.get("uncertainty_std", np.nan)),
        "se_unc": float(standard_errors.get("uncertainty_std", np.nan)),
        "p_unc": float(pvalues.get("uncertainty_std", np.nan)),
        "b_phase_x_unc": float(
            params.get("phase:uncertainty_std", np.nan),
        ),
        "se_phase_x_unc": float(
            standard_errors.get("phase:uncertainty_std", np.nan),
        ),
        "p_phase_x_unc": float(
            pvalues.get("phase:uncertainty_std", np.nan),
        ),
    }

    stats["ame_phase"] = _compute_phase_ame(result, data_frame)

    with open(out_txt, "a", encoding="utf-8") as file_handle:
        file_handle.write(
            f"\nAverage Marginal Effect (phase 0→1): {stats['ame_phase']:.4f}\n",
        )

    return stats


def fit_bucket_glm(
    data_frame: pd.DataFrame,
    out_txt: str,
    _num_buckets: int,
) -> pd.DataFrame:
    """
    Heterogeneous effect by uncertainty bucket:
      correct ~ C(problem) + C(step) + phase + C(bucket) + phase:C(bucket)

    Returns a DataFrame with per-bucket log-odds effect and AME estimates.
    """
    statsmodels_module, statsmodels_formula = lazy_import_statsmodels()

    model = statsmodels_formula.glm(
        "correct ~ C(problem) + C(step) + phase + C(bucket) + phase:C(bucket)",
        data=data_frame,
        family=statsmodels_module.families.Binomial(),
    )
    result = model.fit(cov_type="HC1")

    with open(out_txt, "w", encoding="utf-8") as file_handle:
        file_handle.write(result.summary().as_text())
        file_handle.write("\n")

    # Log-odds base effect for phase
    base_phase = float(result.params.get("phase", np.nan))

    # Build per-bucket summaries
    buckets = sorted(data_frame["bucket"].unique().tolist())
    per_bucket_rows: List[Dict[str, Any]] = []
    for bucket_index in buckets:
        per_bucket_rows.append(
            _bucket_effect_row(
                result=result,
                data_frame=data_frame,
                base_phase=base_phase,
                bucket_index=bucket_index,
            ),
        )

    out = pd.DataFrame(per_bucket_rows).sort_values("bucket").reset_index(drop=True)
    return out


def _bucket_effect_row(
    result,
    data_frame: pd.DataFrame,
    base_phase: float,
    bucket_index: int,
) -> Dict[str, Any]:
    """
    Compute log-odds and AME for a single uncertainty bucket.
    """
    term = f"phase:C(bucket)[T.{bucket_index}]"
    interaction = float(result.params.get(term, 0.0))
    beta = base_phase + interaction  # log-odds change for this bucket

    df_bucket = data_frame[data_frame["bucket"] == bucket_index].copy()
    if df_bucket.empty:
        ame = np.nan
    else:
        df_phase_one = df_bucket.copy()
        df_phase_zero = df_bucket.copy()
        df_phase_one["phase"] = 1
        df_phase_zero["phase"] = 0
        ame = float(
            np.mean(
                result.predict(df_phase_one) - result.predict(df_phase_zero),
            ),
        )

    if term in result.pvalues:
        p_inter = float(result.pvalues[term])
    else:
        p_inter = np.nan

    return {
        "bucket": int(bucket_index),
        "log_odds_phase": beta,
        "p_interaction": p_inter,
        "ame_phase": ame,
    }

# -------------------------- visualization ------------------------

def plot_acc_by_bucket(long_df: pd.DataFrame, out_png: str):
    """
    Accuracy by bucket for phase 0 vs 1.
    """
    agg = (long_df
           .groupby(["bucket", "phase"], as_index=False)
           .agg(acc=("correct","mean"), n=("correct","size")))
    buckets = sorted(agg["bucket"].unique())

    plt_mod = _get_pyplot()
    fig, axes = plt_mod.subplots(figsize=(7.5, 4.5), dpi=140)
    for phase in (0, 1):
        subset = agg[agg["phase"] == phase]
        axes.plot(
            subset["bucket"],
            subset["acc"],
            marker="o",
            label=f"phase={phase}",
        )
    axes.set_xticks(buckets)
    axes.set_xlabel("Uncertainty bucket (by PASS-1)")
    axes.set_ylabel("Accuracy")
    axes.set_title("Accuracy by uncertainty bucket (phase 0 vs 1)")
    axes.grid(True, alpha=0.3)
    axes.legend()
    fig.tight_layout()
    fig.savefig(out_png)
    plt_mod.close(fig)

def plot_ame_by_bucket(bucket_df: pd.DataFrame, out_png: str):
    """
    Per-bucket AME of phase toggle (probability units).
    """
    plt_mod = _get_pyplot()
    fig, axes = plt_mod.subplots(figsize=(7.5, 4.5), dpi=140)
    axes.plot(bucket_df["bucket"], bucket_df["ame_phase"], marker="o")
    axes.axhline(0.0, ls="--", lw=1)
    axes.set_xticks(bucket_df["bucket"].tolist())
    axes.set_xlabel("Uncertainty bucket (by PASS-1)")
    axes.set_ylabel("Δ Accuracy from phase 0→1 (AME)")
    axes.set_title("Phase effect by uncertainty bucket (AME)")
    axes.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png)
    plt_mod.close(fig)

# ----------------------------- main ------------------------------

def main() -> None:
    """CLI entrypoint for H3 pass-1 uncertainty vs. second-pass effect analysis."""
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root", help="Root with step*/.../*.jsonl")
    parser.add_argument(
        "--split",
        default=None,
        help="Substring to filter filenames (e.g. 'test')",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output dir (default: <root>/h3_analysis)",
    )
    parser.add_argument(
        "--uncertainty_field",
        default="entropy",
        choices=["entropy", "entropy_answer", "entropy_think"],
        help="Which field to use as PASS-1 uncertainty moderator",
    )
    parser.add_argument(
        "--num_buckets",
        type=int,
        default=4,
        help="Quantile buckets for uncertainty",
    )
    parser.add_argument("--min_step", type=int, default=None, help="Optional min step")
    parser.add_argument("--max_step", type=int, default=None, help="Optional max step")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.results_root, "h3_analysis")
    os.makedirs(out_dir, exist_ok=True)

    files = scan_jsonl_files(args.results_root, args.split)
    if not files:
        raise SystemExit("No JSONL files found. Check the path or --split.")

    # Build pairs
    pairs = load_pairs(files, uncertainty_field=args.uncertainty_field)

    # Optional step filtering
    if args.min_step is not None:
        pairs = pairs[pairs["step"] >= args.min_step]
    if args.max_step is not None:
        pairs = pairs[pairs["step"] <= args.max_step]

    if pairs.empty:
        raise SystemExit("No pairs after filtering.")

    # Persist the raw/wide pairs
    pairs_csv = os.path.join(out_dir, "h3_pairs.csv")
    pairs.to_csv(pairs_csv, index=False)

    # Long format with standardized uncertainty + buckets
    _, long_df = pairs_to_long(pairs, num_buckets=args.num_buckets)
    long_csv = os.path.join(out_dir, "h3_long.csv")
    long_df.to_csv(long_csv, index=False)

    # Pooled GLM
    pooled_txt = os.path.join(out_dir, "pooled_glm.txt")
    pooled_stats = fit_pooled_glm(long_df, pooled_txt)

    # Bucket GLM
    bucket_txt = os.path.join(out_dir, "bucket_glm.txt")
    bucket_df = fit_bucket_glm(long_df, bucket_txt, _num_buckets=args.num_buckets)
    bucket_csv = os.path.join(out_dir, "bucket_effects.csv")
    bucket_df.to_csv(bucket_csv, index=False)

    # Plots
    acc_png = os.path.join(out_dir, "acc_by_bucket.png")
    plot_acc_by_bucket(long_df, acc_png)

    ame_png = os.path.join(out_dir, "ame_by_bucket.png")
    plot_ame_by_bucket(bucket_df, ame_png)

    # Console recap
    print(f"Wrote PAIRS CSV: {pairs_csv}")
    print(f"Wrote LONG CSV:  {long_csv}")
    print(f"Wrote pooled GLM summary:  {pooled_txt}")
    print(f"Wrote bucket GLM summary:  {bucket_txt}")
    print(f"Wrote bucket effects CSV:  {bucket_csv}")
    print(f"Wrote plots: {acc_png} and {ame_png}\n")

    print("Key pooled effects (log-odds, robust SE):")
    print(f"  β_phase           = {pooled_stats['b_phase']:.4f} "
          f"(se={pooled_stats['se_phase']:.4f}, p={pooled_stats['p_phase']:.3g})")
    print(f"  β_uncertainty     = {pooled_stats['b_unc']:.4f} "
          f"(se={pooled_stats['se_unc']:.4f}, p={pooled_stats['p_unc']:.3g})")
    print(f"  β_phase×uncertainty = {pooled_stats['b_phase_x_unc']:.4f} "
          f"(se={pooled_stats['se_phase_x_unc']:.4f}, p={pooled_stats['p_phase_x_unc']:.3g})")
    print(f"  AME(phase 0→1)    = {pooled_stats['ame_phase']:.4f}")

if __name__ == "__main__":
    main()
