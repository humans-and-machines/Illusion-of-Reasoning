#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run pooled temperature regressions for Crossword, Math, and Rush Hour.

For each domain we load the four decoding-temperature runs (T in {0, 0.05, 0.3, 0.7}),
stack all rows (step 0–950 by default), compute GPT shifts, and fit:

    correct ~ C(problem) + temp_std + shift

We report N, shift rate, conditional accuracies, Δpp, AME, and the GLM
coefficients / p-values for the shift and temperature terms.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm  # type: ignore
import statsmodels.formula.api as smf  # type: ignore

from src.analysis.io import iter_records_from_file, scan_jsonl_files
from src.analysis.labels import aha_gpt_canonical, aha_words
from src.analysis.utils import extract_pass1_and_step, problem_key_from_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-temperature shift regressions.")
    parser.add_argument(
        "--xword_roots",
        nargs="+",
        default=[
            "artifacts/results/GRPO-1.5B-xword-temp-0",
            "artifacts/results/GRPO-1.5B-xword-temp-0.05",
            "artifacts/results/GRPO-1.5B-xword-temp-0.3",
            "artifacts/results/GRPO-1.5B-xword-temp-0.7",
        ],
        help="Crossword temperature runs (default: GRPO-1.5B-xword-temp-*)",
    )
    parser.add_argument(
        "--math_roots",
        nargs="+",
        default=[
            "artifacts/results/GRPO-1.5B-math-temp-0.0",
            "artifacts/results/GRPO-1.5B-math-temp-0.05",
            "artifacts/results/GRPO-1.5B-math-temp-0.3",
            "artifacts/results/GRPO-1.5B-math-temp-0.7",
        ],
        help="Math temperature runs (default: GRPO-1.5B-math-temp-*)",
    )
    parser.add_argument(
        "--rush_roots",
        nargs="+",
        default=[
            "artifacts/results/GRPO-1.5B-carpark-temp-0",
            "artifacts/results/GRPO-1.5B-carpark-temp-0.05",
            "artifacts/results/GRPO-1.5B-carpark-temp-0.3",
            "artifacts/results/GRPO-1.5B-carpark-temp-0.7",
        ],
        help="Rush Hour temperature runs (default: GRPO-1.5B-carpark-temp-*)",
    )
    parser.add_argument("--split", default="test", help="Split filter (default: test).")
    parser.add_argument("--min_step", type=int, default=0, help="Minimum step (inclusive).")
    parser.add_argument("--max_step", type=int, default=950, help="Maximum step (inclusive).")
    parser.add_argument(
        "--out_csv",
        default="artifacts/experiments/temp_shift_multi_glm_summary.csv",
        help="Where to write the summary CSV.",
    )
    return parser.parse_args()


def parse_temp_from_path(path_str: str) -> float:
    match = re.search(r"temp[-_]?([0-9.]+)", Path(path_str).name.lower())
    if not match:
        raise ValueError(f"Cannot parse temperature from path: {path_str}")
    return float(match.group(1))


def iter_rows_for_root(
    root: str,
    *,
    split: str | None,
    min_step: int,
    max_step: int,
) -> Iterable[Dict[str, object]]:
    files = scan_jsonl_files(root, split_substr=None)
    temp_value = parse_temp_from_path(root)
    for file_path in files:
        step_hint = int(re.search(r"step(\d+)", Path(file_path).stem).group(1)) if "step" in file_path else None
        for record in iter_records_from_file(file_path):
            if split and str(record.get("split", "")).lower() != split.lower():
                continue
            pass1, step = extract_pass1_and_step(record, step_hint)
            if not pass1 or step is None:
                continue
            if step < min_step or step > max_step:
                continue
            correct = pass1.get("is_correct_pred")
            if correct is None:
                continue
            words_flag = aha_words(pass1)
            shift_flag = aha_gpt_canonical(pass1, record) and words_flag
            entropy = pass1.get("entropy")
            yield {
                "problem": problem_key_from_record(record, "unknown"),
                "correct": int(correct),
                "shift": int(shift_flag),
                "temp_value": temp_value,
            }


def load_domain(roots: Sequence[str], split: str, min_step: int, max_step: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for root in roots:
        rows.extend(iter_rows_for_root(root, split=split, min_step=min_step, max_step=max_step))
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No rows loaded for roots: {roots}")
    df["problem"] = df["problem"].astype(str)
    df["temp_std"] = (df["temp_value"] - df["temp_value"].mean()) / (df["temp_value"].std(ddof=0) + 1e-8)
    return df


@dataclass
class DomainResult:
    domain: str
    N: int
    share_shift: float
    acc_shift: float
    acc_no_shift: float
    delta_pp: float
    AME: float
    coef_shift: float
    p_shift: float
    coef_temp_std: float
    p_temp_std: float


def fit_domain(domain: str, df: pd.DataFrame) -> DomainResult:
    model = smf.glm("correct ~ C(problem) + temp_std + shift", data=df, family=sm.families.Binomial())
    result = model.fit(
        cov_type="cluster",
        cov_kwds={
            "groups": pd.Categorical(df["problem"]).codes,
            "use_correction": True,
            "df_correction": True,
        },
    )

    share_shift = float(df["shift"].mean())
    mask_shift = df["shift"] == 1
    mask_no = df["shift"] == 0
    acc_shift = float(df.loc[mask_shift, "correct"].mean()) if mask_shift.any() else float("nan")
    acc_no = float(df.loc[mask_no, "correct"].mean()) if mask_no.any() else float("nan")
    delta_pp = (acc_shift - acc_no) * 100.0 if math.isfinite(acc_shift) and math.isfinite(acc_no) else float("nan")

    df_shift1 = df.copy()
    df_shift1["shift"] = 1
    df_shift0 = df.copy()
    df_shift0["shift"] = 0
    ame = float(np.mean(result.predict(df_shift1) - result.predict(df_shift0)))

    return DomainResult(
        domain=domain,
        N=int(len(df)),
        share_shift=share_shift,
        acc_shift=acc_shift,
        acc_no_shift=acc_no,
        delta_pp=delta_pp,
        AME=ame,
        coef_shift=float(result.params.get("shift", float("nan"))),
        p_shift=float(result.pvalues.get("shift", float("nan"))),
        coef_temp_std=float(result.params.get("temp_std", float("nan"))),
        p_temp_std=float(result.pvalues.get("temp_std", float("nan"))),
    )


def main() -> None:
    args = parse_args()
    domain_roots = {
        "Crossword": args.xword_roots,
        "Math": args.math_roots,
        "Rush Hour": args.rush_roots,
    }
    rows: List[Dict[str, object]] = []
    for domain, roots in domain_roots.items():
        print(f"[info] Loading {domain} ({len(roots)} temps)")
        df = load_domain(roots, args.split, args.min_step, args.max_step)
        res = fit_domain(domain, df)
        rows.append(res.__dict__)
        print(
            f"[info] {domain}: N={res.N:,} share_shift={res.share_shift:.4f} "
            f"Δpp={res.delta_pp:+.2f} AME={res.AME:+.4f} coef_shift={res.coef_shift:+.3f} p_shift={res.p_shift:.3g}",
        )

    summary = pd.DataFrame(rows)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"[info] Wrote summary -> {out_path}")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
