#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-temperature Formal Aha! metrics for Qwen2.5-1.5B.

For each decoding temperature T ∈ {0, 0.05, 0.3, 0.7} and domain
(Crossword / Math / Rush Hour), this script:

  1) Loads per-sample rows with canonical GPT Aha flags.
  2) Computes problem–step level Formal Aha flags using the shared
     thresholds (δ1, δ2, min_prior_steps).
  3) Attaches a sample-level ``aha_formal`` indicator
     (problem–step Formal AND sample-level GPT Aha).
  4) Treats ``aha_formal`` as the shift indicator S and reports:

       - N          : number of samples
       - %S         : 100 × share of samples with Formal Aha
       - p_Y|S=1    : P(correct | Formal Aha)
       - Δ%         : (P(correct | S=1) − P(correct | S=0)) in percentage points
       - AME        : average marginal effect of S from a Binomial GLM
       - p          : GLM p-value for the S coefficient

Usage (defaults: canonical GPT, gated by words, δ1=δ2=0.13, min_prior_steps=2):

  python tools/qwen15b_temp_formal_table.py
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd

# Ensure the repo root (containing the ``src`` package) is on sys.path when
# this script is invoked as ``python tools/qwen15b_temp_formal_table.py``.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.analysis.h2_temp_aha_eval import (  # noqa: E402
    attach_formal_sample_level,
    load_samples,
)
from src.analysis.metrics import lazy_import_statsmodels  # noqa: E402


ROOTS_BY_TEMP: Dict[float, Dict[str, str]] = {
    0.0: {
        "Crossword": "artifacts/results/GRPO-1.5B-xword-temp-0",
        "Math": "artifacts/results/GRPO-1.5B-math-temp-0.0",
        "RHour": "artifacts/results/GRPO-1.5B-carpark-temp-0",
    },
    0.05: {
        "Crossword": "artifacts/results/GRPO-1.5B-xword-temp-0.05",
        "Math": "artifacts/results/GRPO-1.5B-math-temp-0.05",
        "RHour": "artifacts/results/GRPO-1.5B-carpark-temp-0.05",
    },
    0.3: {
        "Crossword": "artifacts/results/GRPO-1.5B-xword-temp-0.3",
        "Math": "artifacts/results/GRPO-1.5B-math-temp-0.3",
        "RHour": "artifacts/results/GRPO-1.5B-carpark-temp-0.3",
    },
    0.7: {
        "Crossword": "artifacts/results/GRPO-1.5B-xword-temp-0.7",
        "Math": "artifacts/results/GRPO-1.5B-math-temp-0.7",
        "RHour": "artifacts/results/GRPO-1.5B-carpark-temp-0.7",
    },
}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-temperature Formal Aha! metrics (N, %S, p_Y|S=1, Δ%, AME, p) "
            "for Qwen2.5-1.5B across Crossword / Math / Rush Hour."
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
        "--delta1",
        type=float,
        default=0.13,
        help="Formal δ1 prior-failure threshold (default: 0.13).",
    )
    parser.add_argument(
        "--delta2",
        type=float,
        default=0.13,
        help="Formal δ2 prior-stability threshold (default: 0.13).",
    )
    parser.add_argument(
        "--min_prior_steps",
        type=int,
        default=2,
        help="Formal minimum prior steps (default: 2).",
    )
    parser.add_argument(
        "--gpt_mode",
        choices=["gpt", "gpt_broad"],
        default="gpt",
        help="Sample-level GPT Aha mode used inside Formal (default: gpt).",
    )
    parser.add_argument(
        "--gate_gpt_by_words",
        action="store_true",
        help="Gate GPT Aha by words cue when constructing aha_gpt (default: off).",
    )
    return parser


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

    print("T,domain,N,%S,p_Y|S=1,Delta_pp,AME,p")

    for temp_value, dom_roots in sorted(ROOTS_BY_TEMP.items()):
        for domain_key, root in dom_roots.items():
            # 1) Load per-sample rows with GPT Aha (words + gpt flags).
            samples = load_samples(
                root,
                split_filter=str(args.split or None),
                aha_mode="gpt" if args.gpt_mode == "gpt" else "gpt_broad",
                gate_gpt_by_words=bool(args.gate_gpt_by_words),
            )
            if samples.empty:
                continue

            # Step filter
            samples = samples[
                (samples["step"] >= args.min_step)
                & (samples["step"] <= args.max_step)
            ]
            if samples.empty:
                continue

            # 2) Attach Formal flags at problem–step level and sample level.
            samples = samples.copy()
            samples["run_label"] = f"T={temp_value:g}"
            samples = attach_formal_sample_level(
                samples,
                delta1=args.delta1,
                delta2=args.delta2,
                min_prior_steps=args.min_prior_steps,
            )

            # Use aha_formal as the shift indicator S
            df = samples[["correct", "aha_formal"]].rename(columns={"aha_formal": "S"})
            df["correct"] = df["correct"].astype(int)
            df["S"] = df["S"].astype(int)

            N = int(len(df))
            n_shift = int((df["S"] == 1).sum())
            n_noshift = N - n_shift

            share_S = (n_shift / N) * 100.0 if N else float("nan")
            p_shift = float(df.loc[df["S"] == 1, "correct"].mean()) if n_shift else float("nan")
            p_noshift = (
                float(df.loc[df["S"] == 0, "correct"].mean()) if n_noshift else float("nan")
            )
            if np.isfinite(p_shift) and np.isfinite(p_noshift):
                delta_pp = (p_shift - p_noshift) * 100.0
            else:
                delta_pp = float("nan")

            ame = float("nan")
            p_glm = float("nan")
            if sm is not None and smf is not None and n_shift > 0 and n_noshift > 0:
                model = smf.glm("correct ~ S", data=df, family=sm.families.Binomial())
                try:
                    res = model.fit()
                    exog = res.model.exog.copy()
                    colnames = list(res.model.exog_names)
                    try:
                        idx_S = colnames.index("S")
                    except ValueError:
                        idx_S = max(i for i, name in enumerate(colnames) if "S" in name)
                    params = res.params.to_numpy()
                    logits1 = exog @ params
                    probs1 = 1.0 / (1.0 + np.exp(-logits1))
                    exog[:, idx_S] = 0.0
                    logits0 = exog @ params
                    probs0 = 1.0 / (1.0 + np.exp(-logits0))
                    ame = float(np.mean(probs1 - probs0))
                    p_glm = float(res.pvalues.get("S", float("nan")))
                except Exception:
                    ame = float("nan")
                    p_glm = float("nan")

            print(
                f"{temp_value:.2g},{domain_key},{N},"
                f"{share_S:.3f},{p_shift:.4f},{delta_pp:.2f},{ame:.4f},{p_glm:.3g}",
            )


if __name__ == "__main__":
    main()

