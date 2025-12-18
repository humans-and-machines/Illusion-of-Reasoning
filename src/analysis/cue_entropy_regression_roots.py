#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate forced-intervention cue runs across temperature roots and fit entropy regressions.

We reuse the ``cue_entropy_regression`` helpers but add a CLI that mirrors
``entropy_regression.py`` so the user can point at multiple Math/Xword/Carpark
roots (e.g., ``--math_roots artifacts/results/GRPO-1.5B-math-temp-0.0 ...``).

For each domain we:
  • scan every JSONL under the requested roots (respecting split + step bounds),
  • build cue rows that include ``entropy``, ``baseline_correct``,
    ``intervention_correct``, and a ``problem`` identifier, and
  • fit logistic regressions of the form
        intervention_correct ~ entropy + baseline_correct + C(problem).

In the GRPO runs used here there is a single injected reconsideration cue per
trace (stored under ``pass2``); ``--cues`` simply controls the label attached
to that single cue (e.g., ``C1``) rather than selecting different passes.
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.analysis import cue_entropy_regression as cue_reg
from src.analysis.io import iter_records_from_file, scan_jsonl_files
from src.analysis.utils import coerce_bool, coerce_float, extract_pass1_and_step, nat_step_from_path, problem_key_from_record

CUE_TO_PASS = {
    # Single injected cue used in GRPO runs (pass2).
    "C1": "pass2",
}

DEFAULT_MATH_ROOTS = [
    "artifacts/results/GRPO-1.5B-math-temp-0.0",
    "artifacts/results/GRPO-1.5B-math-temp-0.05",
    "artifacts/results/GRPO-1.5B-math-temp-0.3",
    "artifacts/results/GRPO-1.5B-math-temp-0.7",
]
DEFAULT_XWORD_ROOTS = [
    "artifacts/results/GRPO-1.5B-xword-temp-0",
    "artifacts/results/GRPO-1.5B-xword-temp-0.05",
    "artifacts/results/GRPO-1.5B-xword-temp-0.3",
    "artifacts/results/GRPO-1.5B-xword-temp-0.7",
]
DEFAULT_RUSH_ROOTS = [
    "artifacts/results/GRPO-1.5B-carpark-temp-0",
    "artifacts/results/GRPO-1.5B-carpark-temp-0.05",
    "artifacts/results/GRPO-1.5B-carpark-temp-0.3",
    "artifacts/results/GRPO-1.5B-carpark-temp-0.7",
]


def parse_args() -> argparse.Namespace:
    """Build the CLI."""
    parser = argparse.ArgumentParser(
        description="Fit cue-specific entropy regressions by scanning multi-temperature roots.",
    )
    parser.add_argument("--math_roots", nargs="+", default=None, help="Math roots (default: canonical GRPO runs).")
    parser.add_argument("--xword_roots", nargs="+", default=None, help="Crossword roots.")
    parser.add_argument("--rush_roots", nargs="+", default=None, help="Rush-Hour roots (carpark).")
    parser.add_argument("--split", default="test", help="Split filter (default: test).")
    parser.add_argument("--min_step", type=int, default=0, help="Minimum step to include (default: 0).")
    parser.add_argument("--max_step", type=int, default=950, help="Maximum step to include (default: 950).")
    parser.add_argument(
        "--cues",
        nargs="+",
        default=["C1"],
        help="Cue labels to analyse (subset of C1). All map to the single injected cue stored in pass2.",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse regularization strength for the logistic regression (passed to cue_entropy_regression).",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=2500,
        help="Maximum solver iterations for the logistic regression.",
    )
    parser.add_argument(
        "--out_csv",
        default="artifacts/plots/intervention_cue_entropy_roots.csv",
        help="Optional CSV path for regression summaries.",
    )
    return parser.parse_args()


def _normalize_roots(values: Optional[Sequence[str]], defaults: Sequence[str]) -> List[str]:
    actual = [os.fspath(path) for path in values or [] if str(path).strip()]
    return actual or list(defaults)


def _split_matches(value: object, target: Optional[str]) -> bool:
    if target is None:
        return True
    if value is None:
        return False
    return str(value).lower() == str(target).lower()


def _parse_temp_from_root(root: str) -> Optional[float]:
    match = re.search(r"temp-?([0-9]+(?:\.[0-9]+)?)", root)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _iter_cue_rows_for_file(
    path: str,
    domain_label: str,
    cues_map: Dict[str, str],
    split_filter: Optional[str],
    min_step: Optional[int],
    max_step: Optional[int],
    temp_value: Optional[float],
) -> Iterable[Dict[str, object]]:
    step_from_name = nat_step_from_path(path)
    for record in iter_records_from_file(path):
        if not isinstance(record, dict):
            continue
        if not _split_matches(record.get("split"), split_filter):
            continue

        pass1_data, step_value = extract_pass1_and_step(record, step_from_name)
        if not pass1_data or step_value is None:
            continue
        if min_step is not None and step_value < min_step:
            continue
        if max_step is not None and step_value > max_step:
            continue

        baseline = coerce_bool(
            pass1_data.get("is_correct_pred")
            if "is_correct_pred" in pass1_data
            else pass1_data.get("is_correct"),
        )
        if baseline is None:
            continue

        problem = problem_key_from_record(record, missing_default="unknown")
        problem_token = f"{domain_label}::{problem}"

        for cue_label, pass_key in cues_map.items():
            pass_section = record.get(pass_key)
            if not isinstance(pass_section, dict):
                continue
            intervention = coerce_bool(pass_section.get("is_correct_pred"))
            if intervention is None:
                continue
            entropy_value = coerce_float(pass_section.get("entropy"))
            if entropy_value is None or not np.isfinite(entropy_value):
                continue

            yield {
                "domain": domain_label,
                "problem": problem_token,
                "problem_raw": problem,
                "cue_variant": cue_label,
                "entropy": float(entropy_value),
                "baseline_correct": int(baseline),
                "intervention_correct": int(intervention),
                "step": int(step_value),
                "sample_idx": record.get("sample_idx"),
                "temp_value": temp_value,
            }


def _load_domain_rows(
    domain_label: str,
    roots: Sequence[str],
    cues: Sequence[str],
    split_filter: Optional[str],
    min_step: Optional[int],
    max_step: Optional[int],
) -> pd.DataFrame:
    cues_upper = {cue.upper(): CUE_TO_PASS[cue.upper()] for cue in cues if cue.upper() in CUE_TO_PASS}
    if not cues_upper:
        raise SystemExit(f"No valid cues selected for {domain_label}; choose from {sorted(CUE_TO_PASS)}.")

    rows: List[Dict[str, object]] = []
    for root in roots:
        temp_value = _parse_temp_from_root(root)
        files = scan_jsonl_files(root, split_substr=None)
        if not files:
            continue
        for file_path in files:
            rows.extend(
                _iter_cue_rows_for_file(
                    path=file_path,
                    domain_label=domain_label,
                    cues_map=cues_upper,
                    split_filter=split_filter,
                    min_step=min_step,
                    max_step=max_step,
                    temp_value=temp_value,
                ),
            )

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["problem"] = df["problem"].astype(str)
    df["cue_variant"] = df["cue_variant"].astype(str)
    return df


def _run_domain_regressions(
    domain_label: str,
    df: pd.DataFrame,
    cues: Sequence[str],
    regularization_strength: float,
    max_iter: int,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    if df.empty:
        print(f"[warn] {domain_label}: no cue rows loaded; skipping.")
        return results

    print(f"\n[domain] {domain_label}: {len(df):,} cue rows available.")
    for cue_label in cues:
        cue_upper = cue_label.upper()
        if cue_upper not in df["cue_variant"].unique():
            print(f"  - {domain_label} / {cue_upper}: no rows; skipping.")
            continue

        summary = cue_reg._fit_logit(  # pylint: disable=protected-access
            data_frame=df,
            cue=cue_upper,
            regularization_strength=regularization_strength,
            max_iter=max_iter,
        )
        if not summary:
            print(f"  - {domain_label} / {cue_upper}: insufficient data.")
            continue
        summary["domain"] = domain_label
        results.append(summary)
        print(f"  {domain_label} / ", end="")
        cue_reg._print_summary_line(cue_upper, summary)  # pylint: disable=protected-access
    return results


def main() -> None:
    args = parse_args()
    cues = [cue.upper() for cue in args.cues]

    domains = {
        "Math": _normalize_roots(args.math_roots, DEFAULT_MATH_ROOTS),
        "Crossword": _normalize_roots(args.xword_roots, DEFAULT_XWORD_ROOTS),
        "Carpark": _normalize_roots(args.rush_roots, DEFAULT_RUSH_ROOTS),
    }

    all_summaries: List[Dict[str, object]] = []
    for domain_label, roots in domains.items():
        df = _load_domain_rows(
            domain_label=domain_label,
            roots=roots,
            cues=cues,
            split_filter=args.split,
            min_step=args.min_step,
            max_step=args.max_step,
        )
        summaries = _run_domain_regressions(
            domain_label=domain_label,
            df=df,
            cues=cues,
            regularization_strength=args.C,
            max_iter=args.max_iter,
        )
        all_summaries.extend(summaries)

    if not all_summaries:
        print("[warn] No regression summaries produced.")
        return

    out_df = pd.DataFrame(all_summaries)
    out_path = os.fspath(args.out_csv)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out_df.insert(0, "domain", out_df.pop("domain"))
    out_df.to_csv(out_path, index=False)
    print(f"\n[info] Wrote cue regression summaries -> {out_path}")


if __name__ == "__main__":
    main()
