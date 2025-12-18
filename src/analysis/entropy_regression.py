#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Entropy-only pooled regressions:

    correct ~ C(problem) + entropy_std

We aggregate all available temperatures per domain (Crossword/Math/Rush Hour by default),
restrict to the requested step window, and fit a single GLM with cluster-robust SEs.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm  # type: ignore
import statsmodels.formula.api as smf  # type: ignore

from src.analysis.io import iter_records_from_file, scan_jsonl_files
from src.analysis.utils import extract_pass1_and_step, problem_key_from_record


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
    raise ValueError(f"Unknown domain: {domain}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entropy-only GLMs per domain.")
    parser.add_argument("--xword_roots", nargs="+", default=None, help="Crossword roots.")
    parser.add_argument("--math_roots", nargs="+", default=None, help="Math roots.")
    parser.add_argument("--rush_roots", nargs="+", default=None, help="Rush-Hour roots.")
    parser.add_argument("--split", default="test", help="Split filter (default: test).")
    parser.add_argument("--min_step", type=int, default=0)
    parser.add_argument("--max_step", type=int, default=950)
    parser.add_argument(
        "--out_csv",
        default="artifacts/experiments/entropy_regression_summary.csv",
        help="Summary CSV output path.",
    )
    return parser.parse_args()


def _step_from_name(path: str) -> int | None:
    match = re.search(r"step(\d+)", Path(path).stem)
    return int(match.group(1)) if match else None


def iter_rows(root: str, split: str | None, min_step: int, max_step: int) -> Iterable[Dict[str, object]]:
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
            entropy = pass1.get("entropy")
            correct = pass1.get("is_correct_pred")
            if entropy is None or correct is None:
                continue
            yield {
                "problem": problem_key_from_record(record, "unknown"),
                "correct": int(correct),
                "entropy": float(entropy),
            }


def load_domain(roots: Sequence[str], split: str, min_step: int, max_step: int) -> pd.DataFrame:
    actual_roots = [root for root in roots if str(root).strip()]
    if not actual_roots:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for root in actual_roots:
        rows.extend(iter_rows(root, split, min_step, max_step))
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No rows loaded for {roots}")
    df["entropy_std"] = (df["entropy"] - df["entropy"].mean()) / (df["entropy"].std(ddof=0) + 1e-8)
    df["problem"] = df["problem"].astype(str)
    return df


def fit_entropy_glm(df: pd.DataFrame) -> Dict[str, float]:
    model = smf.glm("correct ~ C(problem) + entropy_std", data=df, family=sm.families.Binomial())
    result = model.fit(
        cov_type="cluster",
        cov_kwds={
            "groups": pd.Categorical(df["problem"]).codes,
            "use_correction": True,
            "df_correction": True,
        },
    )
    coef = float(result.params.get("entropy_std", float("nan")))
    pval = float(result.pvalues.get("entropy_std", float("nan")))
    se = float(result.bse.get("entropy_std", float("nan")))
    ci_low = coef - 1.959964 * se if np.isfinite(se) else float("nan")
    ci_high = coef + 1.959964 * se if np.isfinite(se) else float("nan")
    odds_ratio = float(np.exp(coef))
    odds_ci_low = float(np.exp(ci_low)) if np.isfinite(ci_low) else float("nan")
    odds_ci_high = float(np.exp(ci_high)) if np.isfinite(ci_high) else float("nan")
    return {
        "N": len(df),
        "entropy_sd": float(df["entropy"].std(ddof=0)),
        "coef_entropy_std": coef,
        "p_entropy_std": pval,
        "entropy_std_ci_low": ci_low,
        "entropy_std_ci_high": ci_high,
        "odds_ratio_1sd": odds_ratio,
        "odds_ratio_ci_low": odds_ci_low,
        "odds_ratio_ci_high": odds_ci_high,
    }


def main() -> None:
    args = parse_args()
    domain_roots = {
        "Crossword": args.xword_roots or _default_roots("xword"),
        "Math": args.math_roots or _default_roots("math"),
        "Rush Hour": args.rush_roots or _default_roots("rush"),
    }
    rows: List[Dict[str, float | str]] = []
    for domain, roots in domain_roots.items():
        print(f"[info] loading {domain}")
        df = load_domain(roots, args.split, args.min_step, args.max_step)
        if df.empty:
            print(f"[warn] skipping {domain}: no rows")
            continue
        stats = fit_entropy_glm(df)
        stats["domain"] = domain
        rows.append(stats)
        print(
            f"[info] {domain}: N={stats['N']:,} coef_entropy_std={stats['coef_entropy_std']:+.4f} "
            f"p={stats['p_entropy_std']:.3g}",
        )

    summary = pd.DataFrame(rows)[
        [
            "domain",
            "N",
            "entropy_sd",
            "coef_entropy_std",
            "p_entropy_std",
            "entropy_std_ci_low",
            "entropy_std_ci_high",
            "odds_ratio_1sd",
            "odds_ratio_ci_low",
            "odds_ratio_ci_high",
        ]
    ]
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"[info] wrote summary -> {out_path}")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
