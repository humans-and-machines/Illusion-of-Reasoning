#!/usr/bin/env python3
"""
Compute entropy-stratified shift effects for math runs (e.g., Qwen2.5-7B and Llama3.1-8B).

For each model (one or more roots), this script:
  1) Loads pass-1 records from step*/.../stepXXXX_<split>.jsonl, filtered by step range.
  2) Builds a per-sample table with: problem id, shift flag, correctness, entropy.
  3) Splits samples into:
       - All traces
       - High entropy (top q, default 0.20)
       - Low entropy (bottom 1-q)
  4) For each stratum, reports:
       - N
       - share_shift
       - acc_shift
       - delta_pp = 100 * (acc_shift - acc_no_shift)
       - coef(shift) and p-value from logit(correct ~ shift + C(problem))

Usage example (Math only, steps 0–450):
  python tools/shift_entropy_strata.py \
    --roots artifacts/results/GRPO-7B-math-temp-0 artifacts/results/GRPO-7B-math-temp-0.7 \
    --roots artifacts/results/GRPO-Llama8B-math-temp-0 artifacts/results/GRPO-Llama8B-math-temp-0.7 \
    --split test --min_step 0 --max_step 450 --quantile 0.8
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.api as sm  # type: ignore
from statsmodels.discrete.discrete_model import Logit  # type: ignore

from src.analysis.io import iter_records_from_file, scan_jsonl_files
from src.analysis.metrics import extract_correct
from src.analysis.utils import entropy_from_pass1, extract_pass1_and_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entropy-stratified shift effects (math).")
    parser.add_argument(
        "--roots",
        nargs="+",
        required=True,
        help="List of result roots (step*/.../*.jsonl) for one or more models (temps).",
    )
    parser.add_argument("--split", default="test", help="Split filter (default: test).")
    parser.add_argument("--min_step", type=int, default=0, help="Minimum training step (inclusive).")
    parser.add_argument("--max_step", type=int, default=1000, help="Maximum training step (inclusive).")
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.8,
        help="Entropy quantile threshold for 'high' bucket (default: 0.8 → top 20%%).",
    )
    parser.add_argument(
        "--collapse_temp",
        action="store_true",
        help="Group by model prefix before '-temp-' so all temps are pooled per model.",
    )
    parser.add_argument(
        "--out_csv",
        default=None,
        help="Optional path to save a CSV summary (one row per model × stratum).",
    )
    return parser.parse_args()


def step_from_name(path: str) -> int | None:
    """Extract step number from filenames like step0500_test.jsonl."""
    stem = Path(path).stem
    if "step" not in stem:
        return None
    token = stem.split("step", 1)[1].split("_")[0]
    try:
        return int(token)
    except ValueError:
        return None


def load_rows(root: str, split: str, min_step: int, max_step: int) -> pd.DataFrame:
    files = scan_jsonl_files(root, split)
    rows: List[Dict] = []
    model_name = Path(root).name
    for f in files:
        step_val = step_from_name(f)
        if step_val is None or step_val < min_step or step_val > max_step:
            continue
        for rec in iter_records_from_file(f):
            pass1, _ = extract_pass1_and_step(rec, step_from_name=step_from_name)
            correct = extract_correct(pass1, rec)
            entropy = entropy_from_pass1(pass1, mode="combined")
            shift_flag = pass1.get("shift_in_reasoning_v1")
            if correct is None or entropy is None or shift_flag is None:
                continue
            rows.append(
                {
                    "problem": rec.get("problem_id") or rec.get("problem") or rec.get("row_key"),
                    "shift": int(bool(shift_flag)),
                    "correct": int(bool(correct)),
                    "entropy": float(entropy),
                    "model": model_name,
                }
            )
    return pd.DataFrame(rows)


def logit_with_problem_fe(df_slice: pd.DataFrame) -> tuple[float, float]:
    """Return (coef_shift, p_shift); returns (nan, nan) if fit fails."""
    df_slice = df_slice.dropna(subset=["problem", "shift", "correct"])
    if df_slice["shift"].nunique() < 2:
        return (float("nan"), float("nan"))
    dmat = pd.get_dummies(df_slice["problem"], drop_first=True).astype(float)
    X = sm.add_constant(pd.concat([df_slice["shift"].astype(float), dmat], axis=1)).astype(float)
    y = df_slice["correct"].astype(float)
    try:
        res = Logit(y, X).fit(disp=False, maxiter=200)
        return float(res.params["shift"]), float(res.pvalues["shift"])
    except Exception:
        return (float("nan"), float("nan"))


def summarize(df_slice: pd.DataFrame) -> Dict[str, float]:
    coef_shift, p_shift = logit_with_problem_fe(df_slice)
    # guard empty slices
    has_shift = (df_slice["shift"] == 1).any()
    has_noshift = (df_slice["shift"] == 0).any()
    acc_shift = df_slice.loc[df_slice["shift"] == 1, "correct"].mean() if has_shift else float("nan")
    acc_noshift = df_slice.loc[df_slice["shift"] == 0, "correct"].mean() if has_noshift else float("nan")
    delta_pp = (acc_shift - acc_noshift) * 100.0 if (has_shift and has_noshift) else float("nan")
    return {
        "N": len(df_slice),
        "share_shift": df_slice["shift"].mean(),
        "acc_shift": acc_shift,
        "delta_pp": delta_pp,
        "coef_shift": coef_shift,
        "p": p_shift,
    }


def strata(df_model: pd.DataFrame, quantile: float) -> Dict[str, Dict[str, float]]:
    thr = df_model["entropy"].quantile(quantile)
    hi = df_model[df_model["entropy"] >= thr]
    lo = df_model[df_model["entropy"] < thr]
    return {
        "All traces": summarize(df_model),
        f"High entropy (top {int((1-quantile)*100)}%)": summarize(hi),
        f"Low entropy (bottom {int(quantile*100)}%)": summarize(lo),
    }


def main() -> None:
    args = parse_args()
    all_rows = [load_rows(r, args.split, args.min_step, args.max_step) for r in args.roots]
    df = pd.concat([d for d in all_rows if not d.empty], ignore_index=True)
    if df.empty:
        raise SystemExit("No rows loaded. Check roots/split/step range.")

    if args.collapse_temp:
        df["model"] = df["model"].apply(lambda name: name.split("-temp-")[0] if "-temp-" in name else name)

    model_col = "model"
    summaries: List[Dict[str, str | float]] = []
    for model_name in sorted(df[model_col].unique()):
        df_model = df[df[model_col] == model_name]
        stats = strata(df_model, args.quantile)
        print(f"\n=== {model_name} ===")
        for bucket, vals in stats.items():
            print(
                f"{bucket:28s} N={vals['N']:,}  Δpp={vals['delta_pp']:.2f}  "
                f"coef(shift)={vals['coef_shift']:.2f}  p={vals['p']:.2g}"
            )
            summaries.append(
                {
                    "model": model_name,
                    "bucket": bucket,
                    **vals,
                }
            )

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summaries).to_csv(out_path, index=False)
        print(f"\nSaved CSV -> {out_path}")


if __name__ == "__main__":
    main()
