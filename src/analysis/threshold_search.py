#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Grid-search Formal Aha thresholds with bootstrap diagnostics.

This script searches over (delta1, delta2, delta3) grids, computes Formal Aha
flags on a development slab, and reports bootstrap confidence intervals for the
gain-at-shift term. The intended use is to pick thresholds that (a) have
positive lower CIs and (b) are reasonably prevalent, matching the paper's
"dev-slab + bootstrap" description.

Typical usage (from repo root):

    python -m src.analysis.threshold_search \
        --results_root logs/math_dev \
        --delta1_grid 0,0.125,0.25 \
        --delta2_grid 0,0.125,0.25 \
        --delta3_grid none,0.0,0.05,0.125 \
        --bootstrap_draws 2000 \
        --out_csv /tmp/formal_threshold_search.csv

The search objective is controllable via flags (see CLI help). By default we
pick the configuration with the largest bootstrap lower CI; ties are broken by
prevalence and mean gain.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.analysis.core import (
    build_problem_step_from_samples,
    make_formal_thresholds,
    mark_formal_pairs_with_gain,
)
from src.analysis.figure_2_data import load_pass1_rows
from src.analysis.io import scan_jsonl_files
from src.analysis.utils import gpt_keys_for_mode


# ---------------------------------------------------------------------------
# Dataclasses and small helpers
# ---------------------------------------------------------------------------


@dataclass
class ThresholdResult:
    """Container for metrics computed at a single threshold triple."""

    delta1: float
    delta2: float
    delta3: Optional[float]
    min_prior_steps: int
    events: int
    pairs: int
    prevalence: float
    mean_gain: float
    ci_lo: float
    ci_hi: float
    finite_events: int

    @property
    def score(self) -> float:
        """Alias used for sorting; populated externally."""
        return getattr(self, "_score", float("nan"))

    def set_score(self, value: float) -> None:
        self._score = float(value)


def _parse_float_grid(text: str) -> List[Optional[float]]:
    """Parse a comma-separated grid, accepting "none" for delta3."""

    grid: List[Optional[float]] = []
    for token in text.split(','):
        token = token.strip().lower()
        if not token:
            continue
        if token in {"none", "null"}:
            grid.append(None)
        else:
            try:
                grid.append(float(token))
            except ValueError as exc:  # pragma: no cover - user error path
                raise argparse.ArgumentTypeError(f"Cannot parse grid value '{token}'") from exc
    if not grid:
        raise argparse.ArgumentTypeError("Grid cannot be empty")
    return grid


def _bootstrap_mean(values: np.ndarray, draws: int, seed: int) -> Tuple[float, float]:
    """Return (lo, hi) bootstrap percentiles for the mean of ``values``."""

    sample_count = values.size
    if draws <= 0 or sample_count <= 1:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = np.empty(int(draws), dtype=float)
    for draw_index in range(int(draws)):
        take = rng.integers(0, sample_count, sample_count)
        means[draw_index] = float(values[take].mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _evaluate_thresholds(
    problem_step_df: pd.DataFrame,
    delta1: float,
    delta2: float,
    delta3: Optional[float],
    min_prior_steps: int,
    bootstrap_draws: int,
    seed: int,
) -> ThresholdResult:
    """Compute Formal Aha flags and bootstrap gain stats for one config."""

    thresholds = make_formal_thresholds(
        delta1=delta1,
        delta2=delta2,
        min_prior_steps=min_prior_steps,
        delta3=delta3,
    )
    flagged = mark_formal_pairs_with_gain(problem_step_df, thresholds)
    mask = flagged["aha_formal_pair"] == 1
    events = int(mask.sum())
    pairs = int(len(flagged))
    prevalence = float(events / pairs) if pairs else math.nan

    deltas = (
        flagged.loc[mask, "p_correct_given_shift"].to_numpy(float)
        - flagged.loc[mask, "freq_correct"].to_numpy(float)
    )
    finite = np.isfinite(deltas)
    finite_events = int(finite.sum())
    deltas = deltas[finite]
    mean_gain = float(deltas.mean()) if deltas.size else math.nan
    ci_lo, ci_hi = _bootstrap_mean(deltas, draws=bootstrap_draws, seed=seed)

    return ThresholdResult(
        delta1=delta1,
        delta2=delta2,
        delta3=delta3,
        min_prior_steps=min_prior_steps,
        events=events,
        pairs=pairs,
        prevalence=prevalence,
        mean_gain=mean_gain,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        finite_events=finite_events,
    )


def _score_result(
    result: ThresholdResult,
    *,
    require_positive_ci: bool,
    min_events: int,
) -> float:
    """Return a scalar score used for model selection."""

    if result.events < min_events:
        return float("-inf")
    if require_positive_ci and (not math.isfinite(result.ci_lo) or result.ci_lo <= 0):
        return float("-inf")
    # Primary criterion: lower CI; tie-breakers happen via sort order later.
    return float(result.ci_lo if math.isfinite(result.ci_lo) else float("-inf"))


def _sorted_results(results: Sequence[ThresholdResult]) -> List[ThresholdResult]:
    """Sort by score desc, then prevalence, then mean_gain."""

    return sorted(
        results,
        key=lambda r: (
            r.score,
            r.prevalence if math.isfinite(r.prevalence) else -1.0,
            r.mean_gain if math.isfinite(r.mean_gain) else -1.0,
        ),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", required=True, help="Root directory containing PASS-1 JSONL logs")
    parser.add_argument("--split", default=None, help="Substring to filter files (e.g., 'step0950_test')")
    parser.add_argument("--unc_field", choices=["answer", "overall", "think"], default="answer")
    parser.add_argument("--gpt_mode", choices=["canonical", "broad"], default="canonical")
    parser.add_argument(
        "--no_gate_gpt_by_words",
        action="store_true",
        help="If set, GPT shift labels are NOT gated on lexical cues",
    )

    parser.add_argument("--delta1_grid", type=_parse_float_grid, required=True, help="Comma-separated grid for delta1")
    parser.add_argument("--delta2_grid", type=_parse_float_grid, required=True, help="Comma-separated grid for delta2")
    parser.add_argument(
        "--delta3_grid",
        type=_parse_float_grid,
        default="none",
        help="Comma-separated grid for delta3; use 'none' to disable gain-thresholding",
    )
    parser.add_argument("--min_prior_steps", type=int, default=2, help="Minimum prior checkpoints required")
    parser.add_argument("--bootstrap_draws", type=int, default=2000, help="Bootstrap draws for gain CI")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for bootstrapping")
    parser.add_argument("--min_events", type=int, default=5, help="Require at least this many events to score")
    parser.add_argument(
        "--require_positive_ci",
        action="store_true",
        help="Discard configs whose bootstrap lower CI is non-positive",
    )
    parser.add_argument("--top_k", type=int, default=5, help="How many configurations to display")
    parser.add_argument("--out_csv", type=str, default=None, help="Optional path to save full grid as CSV")
    return parser


def _load_problem_step_df(args: argparse.Namespace) -> pd.DataFrame:
    files = scan_jsonl_files(args.results_root, split_substr=args.split)
    if not files:
        raise SystemExit("No JSONL files found. Check --results_root/--split.")

    gpt_keys = gpt_keys_for_mode(args.gpt_mode)
    samples = load_pass1_rows(
        files,
        unc_field=args.unc_field,
        gpt_keys=gpt_keys,
        gate_gpt_by_words=not args.no_gate_gpt_by_words,
    )
    problem_step_df = build_problem_step_from_samples(
        samples,
        include_native=True,
        native_col="aha_words",
    )
    return problem_step_df


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    problem_step_df = _load_problem_step_df(args)

    results: List[ThresholdResult] = []
    for delta1 in args.delta1_grid:
        for delta2 in args.delta2_grid:
            for delta3 in args.delta3_grid:
                res = _evaluate_thresholds(
                    problem_step_df,
                    delta1=delta1 if delta1 is not None else 0.0,
                    delta2=delta2 if delta2 is not None else 0.0,
                    delta3=delta3,
                    min_prior_steps=args.min_prior_steps,
                    bootstrap_draws=int(args.bootstrap_draws),
                    seed=int(args.seed),
                )
                res.set_score(
                    _score_result(
                        res,
                        require_positive_ci=bool(args.require_positive_ci),
                        min_events=int(args.min_events),
                    ),
                )
                results.append(res)

    ranked = _sorted_results(results)
    best = ranked[0]

    def _fmt(x: Optional[float]) -> str:
        return "none" if x is None else f"{x:.4f}"

    print("# Best thresholds (by score → CI_lo):")
    print(
        f"  delta1={_fmt(best.delta1)}  delta2={_fmt(best.delta2)}  delta3={_fmt(best.delta3)}  "
        f"min_prior_steps={best.min_prior_steps}\n"
        f"  events={best.events} / {best.pairs} (prev={best.prevalence:.4f})  "
        f"mean_gain={best.mean_gain:.4f}  CI=[{best.ci_lo:.4f}, {best.ci_hi:.4f}]",
    )

    print("\n# Top configs:")
    for row in ranked[: max(1, int(args.top_k))]:
        print(
            f"  d1={_fmt(row.delta1)} d2={_fmt(row.delta2)} d3={_fmt(row.delta3)} | "
            f"events={row.events}/{row.pairs} prev={row.prevalence:.4f} | "
            f"gain={row.mean_gain:.4f} CI=[{row.ci_lo:.4f}, {row.ci_hi:.4f}]",
        )

    if args.out_csv:
        records = []
        for row in ranked:
            records.append(
                {
                    "delta1": row.delta1,
                    "delta2": row.delta2,
                    "delta3": row.delta3,
                    "min_prior_steps": row.min_prior_steps,
                    "events": row.events,
                    "pairs": row.pairs,
                    "prevalence": row.prevalence,
                    "mean_gain": row.mean_gain,
                    "ci_lo": row.ci_lo,
                    "ci_hi": row.ci_hi,
                    "finite_events": row.finite_events,
                    "score": row.score,
                },
            )
        df = pd.DataFrame.from_records(records)
        df.to_csv(args.out_csv, index=False)
        print(f"\n[info] wrote grid results to {args.out_csv}")


if __name__ == "__main__":
    main()
