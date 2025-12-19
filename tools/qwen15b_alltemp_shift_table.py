#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate shift metrics across all temperatures for Qwen2.5-1.5B.

For each domain (Crossword / Math / Rush Hour), this script pools
all four GRPO runs at T ∈ {0, 0.05, 0.3, 0.7} and computes:

  - N        : total number of samples (after split/step filters)
  - %S       : 100 × share of samples with a GPT-labeled shift
  - p_Y|S=1  : P(correct | shift = 1)
  - Δ%       : (P(correct | shift=1) − P(correct | shift=0)) in percentage points
  - AME      : average marginal effect of shift from a Binomial GLM
  - p        : GLM p-value for the shift coefficient

Usage (canonical GPT shifts, no words gate, steps [0, 950]):

  python tools/qwen15b_alltemp_shift_table.py
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

# Ensure the repo root (containing the ``src`` package) is on sys.path when
# this script is invoked as ``python tools/qwen15b_alltemp_shift_table.py``.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.analysis.io import iter_records_from_file, scan_jsonl_files  # noqa: E402
from src.analysis.labels import aha_gpt  # noqa: E402
from src.analysis.metrics import extract_correct, lazy_import_statsmodels  # noqa: E402
from src.analysis.utils import (  # noqa: E402
    extract_pass1_and_step,
    nat_step_from_path,
    problem_key_from_record,
)


DOMAIN_ROOTS: Dict[str, List[str]] = {
    "Xword": [
        "artifacts/results/GRPO-1.5B-xword-temp-0",
        "artifacts/results/GRPO-1.5B-xword-temp-0.05",
        "artifacts/results/GRPO-1.5B-xword-temp-0.3",
        "artifacts/results/GRPO-1.5B-xword-temp-0.7",
    ],
    "Math": [
        "artifacts/results/GRPO-1.5B-math-temp-0.0",
        "artifacts/results/GRPO-1.5B-math-temp-0.05",
        "artifacts/results/GRPO-1.5B-math-temp-0.3",
        "artifacts/results/GRPO-1.5B-math-temp-0.7",
    ],
    "RHour": [
        "artifacts/results/GRPO-1.5B-carpark-temp-0",
        "artifacts/results/GRPO-1.5B-carpark-temp-0.05",
        "artifacts/results/GRPO-1.5B-carpark-temp-0.3",
        "artifacts/results/GRPO-1.5B-carpark-temp-0.7",
    ],
}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute all-temperature aggregated shift metrics for Qwen2.5-1.5B "
            "(Crossword / Math / Rush Hour)."
        ),
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Record-level split filter (default: 'test').",
    )
    parser.add_argument(
        "--min_step",
        type=int,
        default=0,
        help="Minimum training step (inclusive, default: 0).",
    )
    parser.add_argument(
        "--max_step",
        type=int,
        default=950,
        help="Maximum training step (inclusive, default: 950).",
    )
    parser.add_argument(
        "--gpt_mode",
        choices=["canonical", "broad"],
        default="canonical",
        help="GPT shift key set (default: canonical).",
    )
    parser.add_argument(
        "--gate_by_words",
        action="store_true",
        help="Gate GPT shifts by words-based cues (default: off).",
    )
    return parser


def _load_domain_rows(
    domain_label: str,
    roots: List[str],
    split_filter: str,
    min_step: int,
    max_step: int,
    gpt_mode: str,
    gate_by_words: bool,
) -> pd.DataFrame:
    rows: List[tuple[int, int, str]] = []

    for root in roots:
        files = scan_jsonl_files(root, split_substr=None)
        for path in files:
            step_hint = nat_step_from_path(path)
            for record in iter_records_from_file(path):
                if split_filter and str(record.get("split", "")).lower() != split_filter:
                    continue
                pass1, step = extract_pass1_and_step(record, step_hint)
                if not pass1 or step is None:
                    continue
                if step < min_step or step > max_step:
                    continue

                correct_flag = extract_correct(pass1, record)
                if correct_flag is None:
                    continue

                shift_flag = aha_gpt(
                    pass1,
                    record,
                    mode=gpt_mode,
                    gate_by_words=gate_by_words,
                    domain=domain_label,
                )
                problem = problem_key_from_record(record, "unknown")
                rows.append((int(correct_flag), int(shift_flag), str(problem)))

    if not rows:
        return pd.DataFrame(columns=["correct", "shift", "problem"])
    return pd.DataFrame(rows, columns=["correct", "shift", "problem"])


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    try:
        sm, smf = lazy_import_statsmodels()
    except RuntimeError:
        sm = smf = None
        print(
            "[warn] statsmodels/scipy not available; AME and GLM p-values "
            "will be reported as NaN.",
        )

    print("domain,N,%S,p_Y|S=1,Delta_pp,AME,p")

    for domain_label, roots in DOMAIN_ROOTS.items():
        df = _load_domain_rows(
            domain_label=domain_label,
            roots=roots,
            split_filter=str(args.split or ""),
            min_step=args.min_step,
            max_step=args.max_step,
            gpt_mode=args.gpt_mode,
            gate_by_words=bool(args.gate_by_words),
        )
        if df.empty:
            continue

        N = int(len(df))
        n_shift = int((df["shift"] == 1).sum())
        n_noshift = N - n_shift

        share_shift = (n_shift / N) * 100.0 if N else float("nan")
        p_shift = float(df.loc[df["shift"] == 1, "correct"].mean()) if n_shift else float("nan")
        p_noshift = (
            float(df.loc[df["shift"] == 0, "correct"].mean()) if n_noshift else float("nan")
        )
        if np.isfinite(p_shift) and np.isfinite(p_noshift):
            delta_pp = (p_shift - p_noshift) * 100.0
        else:
            delta_pp = float("nan")

        ame = float("nan")
        p_glm = float("nan")

        if sm is not None and smf is not None and n_shift > 0 and n_noshift > 0:
            model = smf.glm("correct ~ shift", data=df, family=sm.families.Binomial())
            try:
                res = model.fit()
                exog = res.model.exog.copy()
                colnames = list(res.model.exog_names)
                try:
                    idx_shift = colnames.index("shift")
                except ValueError:
                    idx_shift = max(i for i, name in enumerate(colnames) if "shift" in name)
                params = res.params.to_numpy()
                logits1 = exog @ params
                probs1 = 1.0 / (1.0 + np.exp(-logits1))
                exog[:, idx_shift] = 0.0
                logits0 = exog @ params
                probs0 = 1.0 / (1.0 + np.exp(-logits0))
                ame = float(np.mean(probs1 - probs0))
                p_glm = float(res.pvalues.get("shift", float("nan")))
            except Exception:
                ame = float("nan")
                p_glm = float("nan")

        print(
            f"{domain_label},{N},{share_shift:.2f},{p_shift:.4f},"
            f"{delta_pp:.2f},{ame:.4f},{p_glm:.3g}",
        )


if __name__ == "__main__":
    main()

