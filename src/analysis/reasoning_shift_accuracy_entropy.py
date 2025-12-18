#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Estimate whether GPT-labeled reasoning shifts improve accuracy under high entropy.

This script mirrors ``reasoning_shift_entropy.py`` but flips the regression: we
model PASS-1 correctness as the outcome and treat the reasoning shift indicator
as the key predictor while conditioning on entropy. By default we report three
subsets (all rows, high-entropy tail, low-entropy bulk) per domain or domain×temp:

    correct ~ shift + std_entropy + optional controls

where "high entropy" means entropy ≥ q-quantile (default q=0.75) computed within
each domain/temperature subset. Controls include problem fixed effects, step_std,
and temp_std (if available) unless disabled via CLI flags.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm  # type: ignore
import statsmodels.formula.api as smf  # type: ignore

from src.analysis.io import iter_records_from_file, scan_jsonl_files
from src.analysis.labels import aha_gpt
from src.analysis.utils import coerce_bool, entropy_from_pass1, extract_pass1_and_step, problem_key_from_record


def _default_roots(domain: str) -> List[str]:
    if domain == "xword":
        return [
            "artifacts/results/GRPO-1.5B-xword-temp-0",
            "artifacts/results/GRPO-1.5B-xword-temp-0.05",
            "artifacts/results/GRPO-1.5B-xword-temp-0.3",
            "artifacts/results/GRPO-1.5B-xword-temp-0.7",
        ]
    if domain == "math":
        return [
            "artifacts/results/GRPO-1.5B-math-temp-0.0",
            "artifacts/results/GRPO-1.5B-math-temp-0.05",
            "artifacts/results/GRPO-1.5B-math-temp-0.3",
            "artifacts/results/GRPO-1.5B-math-temp-0.7",
        ]
    if domain == "rush":
        return [
            "artifacts/results/GRPO-1.5B-carpark-temp-0",
            "artifacts/results/GRPO-1.5B-carpark-temp-0.05",
            "artifacts/results/GRPO-1.5B-carpark-temp-0.3",
            "artifacts/results/GRPO-1.5B-carpark-temp-0.7",
        ]
    raise ValueError(f"unknown domain for defaults: {domain}")


def _step_from_name(path: str) -> int | None:
    match = re.search(r"step(\d+)", Path(path).stem)
    return int(match.group(1)) if match else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reasoning-shift accuracy vs entropy subsets.")
    parser.add_argument("--xword_roots", nargs="+", default=None, help="Crossword results roots.")
    parser.add_argument("--math_roots", nargs="+", default=None, help="Math results roots.")
    parser.add_argument("--rush_roots", nargs="+", default=None, help="Rush Hour / Carpark results roots.")
    parser.add_argument("--split", default="test", help="Split filter (default: test).")
    parser.add_argument("--min_step", type=int, default=0, help="Minimum training step (inclusive).")
    parser.add_argument("--max_step", type=int, default=950, help="Maximum training step (inclusive).")
    parser.add_argument(
        "--entropy_mode",
        default="combined",
        choices=["combined", "sum", "think", "answer"],
        help="Pass-1 entropy aggregation (default: combined).",
    )
    parser.add_argument(
        "--gpt_mode",
        default="canonical",
        choices=["canonical", "broad"],
        help="GPT shift key set (default: canonical).",
    )
    parser.add_argument(
        "--no_words_gate",
        action="store_true",
        help="Disable gating GPT shifts by native reconsideration cues.",
    )
    parser.add_argument(
        "--control_temp",
        action="store_true",
        help="Include standardized decoding temperature as a covariate when multiple temps are present.",
    )
    parser.add_argument(
        "--by_temp",
        action="store_true",
        help="Fit separate regressions per decoding temperature (disables --control_temp).",
    )
    parser.add_argument(
        "--no_problem_control",
        action="store_true",
        help="Disable problem fixed effects in the accuracy GLMs.",
    )
    parser.add_argument(
        "--no_step_control",
        action="store_true",
        help="Disable the standardized training-step covariate.",
    )
    parser.add_argument(
        "--no_entropy_control",
        action="store_true",
        help="Drop std_entropy from the accuracy regression (not recommended).",
    )
    parser.add_argument(
        "--entropy_quantile",
        type=float,
        default=0.75,
        help="Quantile cutoff for the high-entropy subset (default: 0.75).",
    )
    parser.add_argument(
        "--out_csv",
        default="artifacts/experiments/reasoning_shift_accuracy_entropy_glm.csv",
        help="Where to save the summary CSV.",
    )
    return parser.parse_args()


def iter_rows_for_root(
    root: str,
    *,
    domain: str,
    split: str | None,
    min_step: int,
    max_step: int,
    entropy_mode: str,
    gpt_mode: str,
    gate_by_words: bool,
    temp_value: Optional[float],
) -> Iterable[Dict[str, object]]:
    files = scan_jsonl_files(root, split_substr=None)
    for file_path in files:
        step_hint = _step_from_name(file_path)
        for record in iter_records_from_file(file_path):
            if split and str(record.get("split", "")).lower() != split.lower():
                continue
            pass1, step = extract_pass1_and_step(record, step_hint)
            if not pass1 or step is None:
                continue
            if step < min_step or step > max_step:
                continue
            entropy = entropy_from_pass1(pass1, mode=entropy_mode)
            if entropy is None:
                continue
            shift_flag = aha_gpt(pass1, record, mode=gpt_mode, gate_by_words=gate_by_words, domain=domain)
            correct_flag = coerce_bool(pass1.get("is_correct_pred"))
            if correct_flag is None:
                continue
            yield {
                "problem": problem_key_from_record(record, "unknown"),
                "entropy": float(entropy),
                "shift": int(shift_flag),
                "correct": int(correct_flag),
                "temp_value": float(temp_value) if temp_value is not None else np.nan,
                "step": int(step),
            }


def _infer_temp_from_root(root: str) -> Optional[float]:
    match = re.search(r"temp-?([0-9]+(?:\.[0-9]+)?)", root)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _add_standardized_cols(df: pd.DataFrame, *, control_temp: bool, domain_label: str) -> pd.DataFrame:
    df = df.copy()
    df["std_entropy"] = (df["entropy"] - df["entropy"].mean()) / (df["entropy"].std(ddof=0) + 1e-8)
    df["step_std"] = (df["step"] - df["step"].mean()) / (df["step"].std(ddof=0) + 1e-8)
    if control_temp:
        temps = df["temp_value"].to_numpy(dtype=float)
        finite_mask = np.isfinite(temps)
        if not finite_mask.any() or len(np.unique(temps[finite_mask])) < 2:
            print(f"[warn] {domain_label}: insufficient temperature variation; dropping temp control.")
            df["temp_std"] = np.nan
        else:
            temps = temps[finite_mask]
            mean_temp = temps.mean()
            std_temp = temps.std(ddof=0) + 1e-8
            df.loc[finite_mask, "temp_std"] = (df.loc[finite_mask, "temp_value"] - mean_temp) / std_temp
            df.loc[~finite_mask, "temp_std"] = np.nan
    else:
        df["temp_std"] = np.nan
    return df


def _group_roots_by_temp(roots: Sequence[str]) -> tuple[Dict[float, List[str]], List[str]]:
    grouped: Dict[float, List[str]] = {}
    missing: List[str] = []
    for root in roots:
        temp = _infer_temp_from_root(root)
        if temp is None:
            missing.append(root)
            continue
        grouped.setdefault(temp, []).append(root)
    return grouped, missing


def load_domain(
    domain: str,
    roots: Sequence[str],
    *,
    split: str,
    min_step: int,
    max_step: int,
    entropy_mode: str,
    gpt_mode: str,
    gate_by_words: bool,
    control_temp: bool,
) -> pd.DataFrame:
    actual_roots = [root for root in roots if str(root).strip()]
    if not actual_roots:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for root in actual_roots:
        temp_value = _infer_temp_from_root(root)
        if control_temp and temp_value is None:
            print(f"[warn] {domain}: skipping root without temp pattern -> {root}")
            continue
        rows.extend(
            iter_rows_for_root(
                root,
                domain=domain,
                split=split,
                min_step=min_step,
                max_step=max_step,
                entropy_mode=entropy_mode,
                gpt_mode=gpt_mode,
                gate_by_words=gate_by_words,
                temp_value=temp_value,
            ),
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["problem"] = df["problem"].astype(str)
    df["shift"] = df["shift"].astype(int)
    df["correct"] = df["correct"].astype(int)
    df["step"] = df["step"].astype(float)
    return _add_standardized_cols(df, control_temp=control_temp, domain_label=domain)


def _split_entropy_subsets(df: pd.DataFrame, quantile: float) -> Tuple[Dict[str, pd.DataFrame], float]:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("--entropy_quantile must be within [0, 1].")
    if df.empty or not np.isfinite(df["entropy"]).any():
        return {"all": df.copy(), "high": df.iloc[0:0], "low": df.iloc[0:0]}, float("nan")
    quantile_value = float(np.quantile(df["entropy"].to_numpy(dtype=float), quantile))
    subsets = {
        "all": df,
        "high": df[df["entropy"] >= quantile_value],
        "low": df[df["entropy"] < quantile_value],
    }
    return subsets, quantile_value


@dataclass
class AccuracySummary:
    domain: str
    temp_value: float
    subset: str
    entropy_quantile: float
    entropy_threshold: float
    N: int
    share_shift: float
    acc_shift1: float
    acc_shift0: float
    delta_pp: float
    entropy_sd: float
    coef_std_entropy: float
    se_std_entropy: float
    p_std_entropy: float
    ci_entropy_low: float
    ci_entropy_high: float
    coef_shift: float
    se_shift: float
    p_shift: float
    ci_shift_low: float
    ci_shift_high: float
    odds_ratio_shift: float
    odds_shift_ci_low: float
    odds_shift_ci_high: float
    step_sd: float
    coef_step_std: float
    se_step_std: float
    p_step_std: float
    ci_step_low: float
    ci_step_high: float
    temp_sd: float
    coef_temp_std: float
    se_temp_std: float
    p_temp_std: float
    ci_temp_low: float
    ci_temp_high: float


def fit_accuracy_glm(
    domain: str,
    df: pd.DataFrame,
    include_temp: bool,
    *,
    temp_value: float = float("nan"),
    control_problem: bool,
    control_step: bool,
    control_entropy: bool,
    subset_label: str,
    entropy_quantile: float,
    entropy_threshold: float,
) -> AccuracySummary:
    formula_terms: List[str] = []
    if control_problem:
        formula_terms.append("C(problem)")
    if control_entropy:
        formula_terms.append("std_entropy")
    if control_step:
        formula_terms.append("step_std")
    formula_terms.append("shift")
    if include_temp and df["temp_std"].notna().any():
        formula_terms.append("temp_std")
        df = df.dropna(subset=["temp_std"])
    include_temp = include_temp and ("temp_std" in formula_terms)
    formula = "correct ~ " + " + ".join(formula_terms)
    model = smf.glm(formula, data=df, family=sm.families.Binomial())
    result = model.fit(
        cov_type="cluster",
        cov_kwds={
            "groups": pd.Categorical(df["problem"]).codes,
            "use_correction": True,
            "df_correction": True,
        },
    )
    coef_entropy = float(result.params.get("std_entropy", float("nan")))
    se_entropy = float(result.bse.get("std_entropy", float("nan")))
    p_entropy = float(result.pvalues.get("std_entropy", float("nan")))
    ci_entropy_low = coef_entropy - 1.959964 * se_entropy if np.isfinite(se_entropy) else float("nan")
    ci_entropy_high = coef_entropy + 1.959964 * se_entropy if np.isfinite(se_entropy) else float("nan")

    coef_shift = float(result.params.get("shift", float("nan")))
    se_shift = float(result.bse.get("shift", float("nan")))
    p_shift = float(result.pvalues.get("shift", float("nan")))
    ci_shift_low = coef_shift - 1.959964 * se_shift if np.isfinite(se_shift) else float("nan")
    ci_shift_high = coef_shift + 1.959964 * se_shift if np.isfinite(se_shift) else float("nan")
    odds_shift = float(np.exp(coef_shift)) if np.isfinite(coef_shift) else float("nan")
    odds_shift_low = float(np.exp(ci_shift_low)) if np.isfinite(ci_shift_low) else float("nan")
    odds_shift_high = float(np.exp(ci_shift_high)) if np.isfinite(ci_shift_high) else float("nan")

    coef_step = float(result.params.get("step_std", float("nan")))
    se_step = float(result.bse.get("step_std", float("nan")))
    p_step = float(result.pvalues.get("step_std", float("nan")))
    ci_step_low = coef_step - 1.959964 * se_step if np.isfinite(se_step) else float("nan")
    ci_step_high = coef_step + 1.959964 * se_step if np.isfinite(se_step) else float("nan")

    if include_temp:
        coef_temp = float(result.params.get("temp_std", float("nan")))
        se_temp = float(result.bse.get("temp_std", float("nan")))
        p_temp = float(result.pvalues.get("temp_std", float("nan")))
        ci_temp_low = coef_temp - 1.959964 * se_temp if np.isfinite(se_temp) else float("nan")
        ci_temp_high = coef_temp + 1.959964 * se_temp if np.isfinite(se_temp) else float("nan")
    else:
        coef_temp = float("nan")
        se_temp = float("nan")
        p_temp = float("nan")
        ci_temp_low = float("nan")
        ci_temp_high = float("nan")
    acc_by_shift = df.groupby("shift")["correct"].mean()
    acc_shift1 = float(acc_by_shift.get(1, np.nan))
    acc_shift0 = float(acc_by_shift.get(0, np.nan))
    delta_pp = acc_shift1 - acc_shift0

    return AccuracySummary(
        domain=domain,
        temp_value=float(temp_value),
        subset=subset_label,
        entropy_quantile=float(entropy_quantile),
        entropy_threshold=float(entropy_threshold),
        N=int(len(df)),
        share_shift=float(df["shift"].mean()),
        acc_shift1=acc_shift1,
        acc_shift0=acc_shift0,
        delta_pp=delta_pp,
        entropy_sd=float(df["entropy"].std(ddof=0)),
        coef_std_entropy=coef_entropy,
        se_std_entropy=se_entropy,
        p_std_entropy=p_entropy,
        ci_entropy_low=ci_entropy_low,
        ci_entropy_high=ci_entropy_high,
        coef_shift=coef_shift,
        se_shift=se_shift,
        p_shift=p_shift,
        ci_shift_low=ci_shift_low,
        ci_shift_high=ci_shift_high,
        odds_ratio_shift=odds_shift,
        odds_shift_ci_low=odds_shift_low,
        odds_shift_ci_high=odds_shift_high,
        step_sd=float(df["step"].std(ddof=0)),
        coef_step_std=coef_step,
        se_step_std=se_step,
        p_step_std=p_step,
        ci_step_low=ci_step_low,
        ci_step_high=ci_step_high,
        temp_sd=float(df["temp_value"].std(ddof=0)) if include_temp else float("nan"),
        coef_temp_std=coef_temp,
        se_temp_std=se_temp,
        p_temp_std=p_temp,
        ci_temp_low=ci_temp_low,
        ci_temp_high=ci_temp_high,
    )


def analyze_dataframe(
    *,
    domain: str,
    df: pd.DataFrame,
    temp_value: float,
    include_temp: bool,
    control_problem: bool,
    control_step: bool,
    control_entropy: bool,
    entropy_quantile: float,
) -> List[AccuracySummary]:
    subsets, threshold = _split_entropy_subsets(df, entropy_quantile)
    summaries: List[AccuracySummary] = []
    for label, subset_df in subsets.items():
        if subset_df.empty:
            print(f"[warn] skipping {domain} temp={temp_value} subset={label}: no rows after filtering")
            continue
        if subset_df["shift"].nunique() < 2:
            print(
                f"[warn] skipping {domain} temp={temp_value} subset={label}: "
                "shift column has <2 unique values",
            )
            continue
        if subset_df["correct"].nunique() < 2:
            print(
                f"[warn] skipping {domain} temp={temp_value} subset={label}: "
                "correct column has <2 unique values",
            )
            continue
        summary = fit_accuracy_glm(
            domain,
            subset_df,
            include_temp=include_temp,
            temp_value=temp_value,
            control_problem=control_problem,
            control_step=control_step,
            control_entropy=control_entropy,
            subset_label=label,
            entropy_quantile=entropy_quantile,
            entropy_threshold=threshold,
        )
        summaries.append(summary)
        print(
            f"[info] {domain} temp={temp_value:g} subset={label}: "
            f"N={summary.N:,} delta_pp={summary.delta_pp:+.4f} "
            f"coef(shift)={summary.coef_shift:+.4f} p={summary.p_shift:.3g} "
            f"(entropy_threshold={threshold:.3f})",
        )
    return summaries


def main() -> None:
    args = parse_args()
    gate_by_words = not args.no_words_gate
    if args.by_temp and args.control_temp:
        print("[warn] --by_temp specified; ignoring --control_temp.")
    control_temp = args.control_temp and not args.by_temp
    control_problem = not args.no_problem_control
    control_step = not args.no_step_control
    control_entropy = not args.no_entropy_control
    domain_roots = {
        "Crossword": args.xword_roots or _default_roots("xword"),
        "Math": args.math_roots or _default_roots("math"),
        "Rush Hour": args.rush_roots or _default_roots("rush"),
    }
    summaries: List[AccuracySummary] = []
    for domain, roots in domain_roots.items():
        print(f"[info] loading {domain} ({len(roots)} roots)")
        if args.by_temp:
            grouped, missing = _group_roots_by_temp(roots)
            if missing:
                print(f"[warn] {domain}: skipping roots without temp tag -> {missing}")
            if not grouped:
                print(f"[warn] skipping {domain}: no temperature-tagged roots")
                continue
            for temp_value in sorted(grouped):
                df = load_domain(
                    domain,
                    grouped[temp_value],
                    split=args.split,
                    min_step=args.min_step,
                    max_step=args.max_step,
                    entropy_mode=args.entropy_mode,
                    gpt_mode=args.gpt_mode,
                    gate_by_words=gate_by_words,
                    control_temp=False,
                )
                if df.empty:
                    print(f"[warn] skipping {domain} temp={temp_value}: no rows in requested window")
                    continue
                summs = analyze_dataframe(
                    domain=domain,
                    df=df,
                    temp_value=temp_value,
                    include_temp=False,
                    control_problem=control_problem,
                    control_step=control_step,
                    control_entropy=control_entropy,
                    entropy_quantile=args.entropy_quantile,
                )
                summaries.extend(summs)
        else:
            df = load_domain(
                domain,
                roots,
                split=args.split,
                min_step=args.min_step,
                max_step=args.max_step,
                entropy_mode=args.entropy_mode,
                gpt_mode=args.gpt_mode,
                gate_by_words=gate_by_words,
                control_temp=control_temp,
            )
            if df.empty:
                print(f"[warn] skipping {domain}: no rows in requested window")
                continue
            include_temp = control_temp and df["temp_std"].notna().any()
            summs = analyze_dataframe(
                domain=domain,
                df=df,
                temp_value=float("nan"),
                include_temp=include_temp,
                control_problem=control_problem,
                control_step=control_step,
                control_entropy=control_entropy,
                entropy_quantile=args.entropy_quantile,
            )
            summaries.extend(summs)

    if not summaries:
        raise SystemExit("[error] No domains produced results. Check roots/filters.")

    summary_df = pd.DataFrame([s.__dict__ for s in summaries])[
        [
            "domain",
            "temp_value",
            "subset",
            "entropy_quantile",
            "entropy_threshold",
            "N",
            "share_shift",
            "acc_shift1",
            "acc_shift0",
            "delta_pp",
            "coef_shift",
            "se_shift",
            "p_shift",
            "ci_shift_low",
            "ci_shift_high",
            "odds_ratio_shift",
            "odds_shift_ci_low",
            "odds_shift_ci_high",
            "coef_std_entropy",
            "se_std_entropy",
            "p_std_entropy",
            "ci_entropy_low",
            "ci_entropy_high",
            "step_sd",
            "coef_step_std",
            "se_step_std",
            "p_step_std",
            "ci_step_low",
            "ci_step_high",
            "temp_sd",
            "coef_temp_std",
            "se_temp_std",
            "p_temp_std",
            "ci_temp_low",
            "ci_temp_high",
        ]
    ]
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_path, index=False)
    print(f"[info] wrote summary -> {out_path}")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
