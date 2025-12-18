#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fit per-domain GLMs for the prevalence of GPT-labeled reasoning shifts. By default
the specification is ``shift ~ C(problem) + std_entropy + step_std``, but the fixed
effects and step control can be disabled via CLI flags.

All temperature folders for a domain are pooled by default, steps are filtered via
``[min_step, max_step]``, and pass-1 entropy is standardized (z-score). Optionally
fit separate regressions per decoding temperature with ``--by_temp`` or pool all
domains into a single regression via ``--combine_domains``.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm  # type: ignore
import statsmodels.formula.api as smf  # type: ignore

from src.analysis.io import iter_records_from_file, scan_jsonl_files
from src.analysis.labels import aha_gpt
from src.analysis.utils import entropy_from_pass1, extract_pass1_and_step, problem_key_from_record


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
    parser = argparse.ArgumentParser(description="Reasoning-shift prevalence vs entropy regressions.")
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
        "--combine_domains",
        action="store_true",
        help="Fit a single regression pooling all domains (incompatible with --by_temp).",
    )
    parser.add_argument(
        "--no_problem_control",
        action="store_true",
        help="Disable problem fixed effects; run shift ~ entropy (+ optional covariates).",
    )
    parser.add_argument(
        "--no_step_control",
        action="store_true",
        help="Disable the standardized training-step covariate.",
    )
    parser.add_argument(
        "--out_csv",
        default="artifacts/experiments/reasoning_shift_entropy_glm.csv",
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
            yield {
                "problem": problem_key_from_record(record, "unknown"),
                "entropy": float(entropy),
                "shift": int(shift_flag),
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
    standardize: bool = True,
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
    df["problem"] = df["problem"].astype(str).map(lambda value: f"{domain}:{value}")
    df["shift"] = df["shift"].astype(int)
    df["step"] = df["step"].astype(float)
    df["domain_label"] = domain
    if standardize:
        return _add_standardized_cols(df, control_temp=control_temp, domain_label=domain)
    return df


@dataclass
class DomainSummary:
    domain: str
    temp_value: float
    N: int
    share_shift: float
    entropy_sd: float
    coef_std_entropy: float
    se_std_entropy: float
    p_std_entropy: float
    ci_low: float
    ci_high: float
    odds_ratio_1sd: float
    odds_ci_low: float
    odds_ci_high: float
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


def fit_shift_glm(
    domain: str,
    df: pd.DataFrame,
    include_temp: bool,
    *,
    temp_value: float = float("nan"),
    control_problem: bool = True,
    control_step: bool = True,
    control_domain: bool = False,
) -> DomainSummary:
    formula_terms: List[str] = []
    if control_problem:
        formula_terms.append("C(problem)")
    if control_domain:
        if "domain_label" not in df.columns:
            raise ValueError("domain_label column missing; cannot control for domain effects.")
        formula_terms.append("C(domain_label)")
    formula_terms.append("std_entropy")
    if control_step:
        formula_terms.append("step_std")
    if include_temp and df["temp_std"].notna().any():
        formula_terms.append("temp_std")
        df = df.dropna(subset=["temp_std"])
    include_temp = include_temp and ("temp_std" in formula_terms)
    formula = "shift ~ " + " + ".join(formula_terms)
    model = smf.glm(formula, data=df, family=sm.families.Binomial())
    result = model.fit(
        cov_type="cluster",
        cov_kwds={
            "groups": pd.Categorical(df["problem"]).codes,
            "use_correction": True,
            "df_correction": True,
        },
    )
    coef = float(result.params.get("std_entropy", float("nan")))
    se = float(result.bse.get("std_entropy", float("nan")))
    p_val = float(result.pvalues.get("std_entropy", float("nan")))
    ci_low = coef - 1.959964 * se if np.isfinite(se) else float("nan")
    ci_high = coef + 1.959964 * se if np.isfinite(se) else float("nan")
    odds = float(np.exp(coef)) if np.isfinite(coef) else float("nan")
    odds_low = float(np.exp(ci_low)) if np.isfinite(ci_low) else float("nan")
    odds_high = float(np.exp(ci_high)) if np.isfinite(ci_high) else float("nan")
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
    return DomainSummary(
        domain=domain,
        N=int(len(df)),
        share_shift=float(df["shift"].mean()),
        entropy_sd=float(df["entropy"].std(ddof=0)),
        coef_std_entropy=coef,
        se_std_entropy=se,
        p_std_entropy=p_val,
        ci_low=ci_low,
        ci_high=ci_high,
        odds_ratio_1sd=odds,
        odds_ci_low=odds_low,
        odds_ci_high=odds_high,
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
        temp_value=float(temp_value),
    )


def main() -> None:
    args = parse_args()
    gate_by_words = not args.no_words_gate
    if args.by_temp and args.control_temp:
        print("[warn] --by_temp specified; ignoring --control_temp.")
    control_temp = args.control_temp and not args.by_temp
    if args.combine_domains and args.by_temp:
        raise SystemExit("--combine_domains cannot be used together with --by_temp.")
    control_problem = not args.no_problem_control
    control_step = not args.no_step_control
    domain_roots = {
        "Crossword": args.xword_roots or _default_roots("xword"),
        "Math": args.math_roots or _default_roots("math"),
        "Rush Hour": args.rush_roots or _default_roots("rush"),
    }
    summaries: List[DomainSummary] = []
    if args.combine_domains:
        combined_frames: List[pd.DataFrame] = []
        for domain, roots in domain_roots.items():
            print(f"[info] loading {domain} ({len(roots)} roots) for combined regression")
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
                standardize=False,
            )
            if df.empty:
                print(f"[warn] skipping {domain}: no rows in requested window")
                continue
            combined_frames.append(df)
        if not combined_frames:
            raise SystemExit("[error] No data available for combined regression.")
        combined_df = pd.concat(combined_frames, ignore_index=True)
        combined_df = _add_standardized_cols(
            combined_df,
            control_temp=control_temp,
            domain_label="All Domains",
        )
        if combined_df["shift"].nunique() < 2:
            raise SystemExit("[error] shift column has <2 unique values after combining domains.")
        include_temp = control_temp and combined_df["temp_std"].notna().any()
        control_domain = combined_df["domain_label"].nunique() > 1
        summary = fit_shift_glm(
            "All Domains",
            combined_df,
            include_temp=include_temp,
            control_problem=control_problem,
            control_step=control_step,
            control_domain=control_domain,
        )
        summaries.append(summary)
        print(
            "[info] All Domains: "
            f"N={summary.N:,} share_shift={summary.share_shift:.4f} "
            f"coef(std_entropy)={summary.coef_std_entropy:+.4f} p={summary.p_std_entropy:.3g}",
        )
    else:
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
                    if df["shift"].nunique() < 2:
                        print(f"[warn] skipping {domain} temp={temp_value}: shift column has <2 unique values")
                        continue
                    summary = fit_shift_glm(
                        domain,
                        df,
                        include_temp=False,
                        temp_value=temp_value,
                        control_problem=control_problem,
                        control_step=control_step,
                    )
                    summaries.append(summary)
                    print(
                        f"[info] {domain} temp={temp_value:g}: N={summary.N:,} share_shift={summary.share_shift:.4f} "
                        f"coef(std_entropy)={summary.coef_std_entropy:+.4f} p={summary.p_std_entropy:.3g}",
                    )
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
                if df["shift"].nunique() < 2:
                    print(f"[warn] skipping {domain}: shift column has <2 unique values")
                    continue
                include_temp = control_temp and df["temp_std"].notna().any()
                summary = fit_shift_glm(
                    domain,
                    df,
                    include_temp=include_temp,
                    control_problem=control_problem,
                    control_step=control_step,
                )
                summaries.append(summary)
                print(
                    f"[info] {domain}: N={summary.N:,} share_shift={summary.share_shift:.4f} "
                    f"coef(std_entropy)={summary.coef_std_entropy:+.4f} p={summary.p_std_entropy:.3g}",
                )

    if not summaries:
        raise SystemExit("[error] No domains produced results. Check roots/filters.")

    summary_df = pd.DataFrame([s.__dict__ for s in summaries])[
        [
            "domain",
            "temp_value",
            "N",
            "share_shift",
            "entropy_sd",
            "coef_std_entropy",
            "se_std_entropy",
            "p_std_entropy",
            "ci_low",
            "ci_high",
            "odds_ratio_1sd",
            "odds_ci_low",
            "odds_ci_high",
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
