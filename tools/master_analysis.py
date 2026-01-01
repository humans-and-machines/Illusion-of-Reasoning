#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master launcher for analysis utilities.

Step 1: shift-prevalence
------------------------
Compute per-checkpoint (%) prevalence of reasoning shifts using shift labels
written by ``src.annotate.cli.shift_cli`` (field: ``pass1.shift_in_reasoning_v1``).

This produces a row per (experiment, temperature, checkpoint step) including:
  - N_total:        total records counted
  - N_labeled:      records where the shift label was present
  - shift_count:    labeled records with shift == 1
  - shift_pct:      100 * shift_count / N_labeled
  - label_coverage: 100 * N_labeled / N_total

Default grid (as requested):
  - Qwen-1.5B: Math, Crossword, Carpark   (steps <= 950)
  - Qwen-7B:   Math                       (steps <= 450)
  - Llama-8B:  Math                       (steps <= 450)
  - Temps: {0, 0.05, 0.3, 0.7}
  - Split: test

Example:
  python tools/master_analysis.py shift-prevalence
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# Cap CPU thread usage to reduce crashes on shared nodes.
def _cap_threads(max_threads: int = 20) -> None:
    env_override = os.environ.get("MASTER_ANALYSIS_MAX_THREADS")
    if env_override:
        try:
            max_threads = int(env_override)
        except ValueError:
            max_threads = max_threads
    env_keys = [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ]
    cap = str(int(max_threads))
    for key in env_keys:
        raw = os.environ.get(key)
        if raw is None:
            os.environ[key] = cap
            continue
        try:
            if int(raw) > max_threads:
                os.environ[key] = cap
        except ValueError:
            os.environ[key] = cap


_cap_threads(20)


# Ensure the repo root (containing the ``src`` package) is on sys.path when
# invoked as ``python tools/master_analysis.py ...``.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.analysis.io import iter_records_from_file, scan_files_step_only  # noqa: E402
from src.analysis.labels import aha_gpt  # noqa: E402
from src.analysis.metrics import extract_correct, glm_fit_with_covariance, lazy_import_statsmodels, predict_formula  # noqa: E402
from src.analysis.utils import (  # noqa: E402
    coerce_bool,
    entropy_from_pass1,
    extract_pass1_and_step,
    nat_step_from_path,
    problem_key_from_record,
    step_from_rec_or_path,
)


DEFAULT_TEMPS: Tuple[float, ...] = (0.0, 0.05, 0.3, 0.7)


@dataclass(frozen=True)
class ExperimentSpec:
    key: str
    model: str
    domain: str
    max_step: int
    # Template for the directory name under results_base, where "{temp}" is substituted.
    # For temp == 0, multiple candidates may exist (e.g., "0" vs "0.0").
    dir_template: str
    zero_temp_candidates: Tuple[str, ...] = ("0", "0.0")


def build_default_grid() -> List[ExperimentSpec]:
    return [
        ExperimentSpec(
            key="qwen15b_math",
            model="Qwen-1.5B",
            domain="Math",
            max_step=950,
            dir_template="GRPO-1.5B-math-temp-{temp}",
            zero_temp_candidates=("0.0", "0"),
        ),
        ExperimentSpec(
            key="qwen15b_crossword",
            model="Qwen-1.5B",
            domain="Crossword",
            max_step=950,
            dir_template="GRPO-1.5B-xword-temp-{temp}",
        ),
        ExperimentSpec(
            key="qwen15b_carpark",
            model="Qwen-1.5B",
            domain="Carpark",
            max_step=950,
            dir_template="GRPO-1.5B-carpark-temp-{temp}",
        ),
        ExperimentSpec(
            key="qwen7b_math",
            model="Qwen-7B",
            domain="Math",
            max_step=450,
            dir_template="GRPO-7B-math-temp-{temp}",
        ),
        ExperimentSpec(
            key="llama8b_math",
            model="Llama-8B",
            domain="Math",
            max_step=450,
            dir_template="GRPO-Llama8B-math-temp-{temp}",
        ),
    ]


def _format_temp_for_dir(temp: float) -> str:
    if float(temp) == 0.0:
        return "0"
    # Keep user-facing set stable for the requested values.
    if float(temp) == 0.05:
        return "0.05"
    if float(temp) == 0.3:
        return "0.3"
    if float(temp) == 0.7:
        return "0.7"
    # Fallback: compact string.
    return str(temp)


def resolve_results_root(
    results_base: Path,
    spec: ExperimentSpec,
    temp: float,
) -> Optional[Path]:
    """
    Resolve an existing results root on disk for (spec, temp), accounting for
    minor naming differences at temp=0.
    """
    if float(temp) == 0.0:
        for tok in spec.zero_temp_candidates:
            candidate = results_base / spec.dir_template.format(temp=tok)
            if candidate.is_dir():
                return candidate
        return None

    candidate = results_base / spec.dir_template.format(temp=_format_temp_for_dir(temp))
    return candidate if candidate.is_dir() else None


@dataclass
class StepCounts:
    n_total: int = 0
    n_labeled: int = 0
    n_shift: int = 0


@dataclass
class AccuracyCounts:
    n_total: int = 0
    n_labeled: int = 0
    n_shift: int = 0
    k_shift: int = 0
    n_noshift: int = 0
    k_noshift: int = 0
    n_shift_p2: int = 0
    k_shift_p2: int = 0
    n_noshift_p2: int = 0
    k_noshift_p2: int = 0


def iter_step_counts(
    root: Path,
    *,
    split: str,
    min_step: int,
    max_step: int,
    pass_key: str,
    label_key: str,
) -> Dict[int, StepCounts]:
    """
    Aggregate (N_total, N_labeled, N_shift) per step from a results root.
    """
    counts_by_step: Dict[int, StepCounts] = {}
    files = scan_files_step_only(str(root), split_substr=split)
    for file_path in files:
        step_hint = nat_step_from_path(file_path)
        if step_hint is not None and (step_hint < min_step or step_hint > max_step):
            continue
        for record in iter_records_from_file(file_path):
            if split and str(record.get("split", "")).lower() != split.lower():
                continue

            pass_obj = record.get(pass_key) or {}
            if not isinstance(pass_obj, dict):
                continue

            step = int(step_from_rec_or_path(record, file_path))
            if step < min_step or step > max_step:
                continue

            bucket = counts_by_step.setdefault(step, StepCounts())
            bucket.n_total += 1

            label_val: Any = pass_obj.get(label_key, record.get(label_key))
            coerced = coerce_bool(label_val)
            if coerced is None:
                continue
            bucket.n_labeled += 1
            bucket.n_shift += int(coerced == 1)

    return counts_by_step


def iter_accuracy_counts(
    root: Path,
    *,
    split: str,
    min_step: int,
    max_step: int,
    pass_key: str,
    label_key: str,
) -> Tuple[AccuracyCounts, set[int]]:
    """
    Aggregate correctness conditional on shift labels for a results root.

    - n_total: records with a defined correctness flag
    - n_labeled: subset with a defined shift label
    """
    counts = AccuracyCounts()
    steps: set[int] = set()

    files = scan_files_step_only(str(root), split_substr=split)
    for file_path in files:
        step_hint = nat_step_from_path(file_path)
        if step_hint is not None and (step_hint < min_step or step_hint > max_step):
            continue
        for record in iter_records_from_file(file_path):
            if split and str(record.get("split", "")).lower() != split.lower():
                continue

            pass_obj = record.get(pass_key) or {}
            if not isinstance(pass_obj, dict):
                continue

            step = int(step_from_rec_or_path(record, file_path))
            if step < min_step or step > max_step:
                continue

            correct_flag = extract_correct(pass_obj, record)
            if correct_flag is None:
                continue

            pass2_obj = record.get("pass2") or {}
            pass2_correct = None
            if isinstance(pass2_obj, dict):
                pass2_correct = extract_correct(pass2_obj, record)

            steps.add(int(step))
            counts.n_total += 1

            label_val: Any = pass_obj.get(label_key, record.get(label_key))
            shift_flag = coerce_bool(label_val)
            if shift_flag is None:
                continue

            counts.n_labeled += 1
            if int(shift_flag) == 1:
                counts.n_shift += 1
                counts.k_shift += int(correct_flag == 1)
                if pass2_correct is not None:
                    counts.n_shift_p2 += 1
                    counts.k_shift_p2 += int(pass2_correct == 1)
            else:
                counts.n_noshift += 1
                counts.k_noshift += int(correct_flag == 1)
                if pass2_correct is not None:
                    counts.n_noshift_p2 += 1
                    counts.k_noshift_p2 += int(pass2_correct == 1)

    return counts, steps


def _pct(num: int, den: int) -> float:
    return (100.0 * float(num) / float(den)) if den else 0.0


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _print_aligned_table(
    headers: List[str],
    rows: List[List[str]],
    *,
    right_align: Optional[Iterable[int]] = None,
) -> None:
    """
    Print a simple fixed-width table with aligned columns (no external deps).
    """
    if not rows:
        print("[warn] no rows to print")
        return

    right = set(right_align or [])
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt_row(row: List[str]) -> str:
        out: List[str] = []
        for i, cell in enumerate(row):
            txt = str(cell)
            out.append(txt.rjust(widths[i]) if i in right else txt.ljust(widths[i]))
        return " | ".join(out)

    print(fmt_row(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))


def cmd_shift_prevalence(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="master_analysis.py shift-prevalence",
        description="Compute per-checkpoint shift prevalence across the standard experiment grid.",
    )
    parser.add_argument(
        "--results_base",
        default="artifacts/results",
        help="Base directory containing results roots (default: artifacts/results).",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Record-level split filter (default: test).",
    )
    parser.add_argument(
        "--min_step",
        type=int,
        default=0,
        help="Minimum checkpoint step to include (inclusive, default: 0).",
    )
    parser.add_argument(
        "--max_step_cap",
        type=int,
        default=None,
        help=(
            "Optional global cap on checkpoint steps (inclusive), applied on top "
            "of each experiment's default max_step. Useful for quick smoke tests."
        ),
    )
    parser.add_argument(
        "--temps",
        nargs="+",
        type=float,
        default=list(DEFAULT_TEMPS),
        help="Temperatures to include (default: 0 0.05 0.3 0.7).",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=[],
        help=(
            "Optional subset of experiment keys to run "
            "(e.g., --only qwen15b_math qwen7b_math). Default: all."
        ),
    )
    parser.add_argument(
        "--pass_key",
        default="pass1",
        help="Which pass dict to read the label from (default: pass1).",
    )
    parser.add_argument(
        "--label_key",
        default="shift_in_reasoning_v1",
        help="Label field name inside the pass dict (default: shift_in_reasoning_v1).",
    )
    parser.add_argument(
        "--out_json",
        default="artifacts/analysis/shift_prevalence_by_step.json",
        help="Write rows to this JSON file (default: artifacts/analysis/shift_prevalence_by_step.json).",
    )
    parser.add_argument(
        "--out_csv",
        default="artifacts/analysis/shift_prevalence_by_step.csv",
        help="Write rows to this CSV file (default: artifacts/analysis/shift_prevalence_by_step.csv).",
    )
    parser.add_argument(
        "--no_summary",
        action="store_true",
        help="Disable printing per-(experiment,temp) totals to stdout.",
    )
    parser.add_argument(
        "--print_summary_only",
        action="store_true",
        help="Only print the summary (skip writing JSON/CSV).",
    )
    args = parser.parse_args(argv)

    results_base = Path(args.results_base)
    grid = build_default_grid()
    if args.only:
        allowed = set(args.only)
        grid = [spec for spec in grid if spec.key in allowed]
    temps = list(args.temps)

    rows: List[Dict[str, Any]] = []
    missing: List[Tuple[str, float]] = []
    summary: Dict[Tuple[str, float], StepCounts] = {}
    steps_seen: Dict[Tuple[str, float], set[int]] = {}
    exp_meta: Dict[str, Tuple[str, str]] = {}

    for spec in grid:
        for temp in temps:
            root = resolve_results_root(results_base, spec, temp)
            if root is None:
                missing.append((spec.key, float(temp)))
                continue

            max_step = int(spec.max_step)
            if args.max_step_cap is not None:
                max_step = min(max_step, int(args.max_step_cap))

            counts_by_step = iter_step_counts(
                root,
                split=str(args.split or ""),
                min_step=int(args.min_step),
                max_step=max_step,
                pass_key=str(args.pass_key),
                label_key=str(args.label_key),
            )
            sum_key = (spec.key, float(temp))
            sum_bucket = summary.setdefault(sum_key, StepCounts())
            seen = steps_seen.setdefault(sum_key, set())
            exp_meta[spec.key] = (spec.model, spec.domain)
            for step in sorted(counts_by_step):
                c = counts_by_step[step]
                sum_bucket.n_total += int(c.n_total)
                sum_bucket.n_labeled += int(c.n_labeled)
                sum_bucket.n_shift += int(c.n_shift)
                seen.add(int(step))
                row = {
                    "experiment": spec.key,
                    "model": spec.model,
                    "domain": spec.domain,
                    "temp": float(temp),
                    "min_step": int(args.min_step),
                    "max_step": int(max_step),
                    "step": int(step),
                    "n_total": int(c.n_total),
                    "n_labeled": int(c.n_labeled),
                    "shift_count": int(c.n_shift),
                    "shift_pct": float(_pct(c.n_shift, c.n_labeled)),
                    "label_coverage_pct": float(_pct(c.n_labeled, c.n_total)),
                    "root": str(root),
                    "split": str(args.split),
                    "pass_key": str(args.pass_key),
                    "label_key": str(args.label_key),
                }
                rows.append(row)

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    if not args.print_summary_only:
        _ensure_parent(out_json)
        _ensure_parent(out_csv)

    # Build rollups (currently: Qwen-1.5B across all three domains).
    qwen15b_keys = {k for k, (m, _) in exp_meta.items() if m == "Qwen-1.5B"}
    qwen_llama_keys = {k for k, (m, _) in exp_meta.items() if m in {"Qwen-7B", "Llama-8B"}}
    qwen15b_by_temp: Dict[float, StepCounts] = {}
    qwen15b_steps_by_temp: Dict[float, set[Tuple[str, int]]] = {}
    qwen15b_all_temps = StepCounts()
    qwen15b_all_temps_steps: set[Tuple[float, str, int]] = set()
    qwen_llama_by_temp: Dict[float, StepCounts] = {}
    qwen_llama_steps_by_temp: Dict[float, set[Tuple[str, int]]] = {}
    qwen_llama_all_temps = StepCounts()
    qwen_llama_all_temps_steps: set[Tuple[float, str, int]] = set()
    for (exp_key, temp_value), c in summary.items():
        if exp_key not in qwen15b_keys:
            if exp_key in qwen_llama_keys:
                bucket = qwen_llama_by_temp.setdefault(float(temp_value), StepCounts())
                bucket.n_total += c.n_total
                bucket.n_labeled += c.n_labeled
                bucket.n_shift += c.n_shift
                qwen_llama_all_temps.n_total += c.n_total
                qwen_llama_all_temps.n_labeled += c.n_labeled
                qwen_llama_all_temps.n_shift += c.n_shift
                stset = qwen_llama_steps_by_temp.setdefault(float(temp_value), set())
                for st in steps_seen.get((exp_key, temp_value), set()):
                    stset.add((exp_key, int(st)))
                    qwen_llama_all_temps_steps.add((float(temp_value), exp_key, int(st)))
            continue
        bucket = qwen15b_by_temp.setdefault(float(temp_value), StepCounts())
        bucket.n_total += c.n_total
        bucket.n_labeled += c.n_labeled
        bucket.n_shift += c.n_shift
        qwen15b_all_temps.n_total += c.n_total
        qwen15b_all_temps.n_labeled += c.n_labeled
        qwen15b_all_temps.n_shift += c.n_shift
        stset = qwen15b_steps_by_temp.setdefault(float(temp_value), set())
        for st in steps_seen.get((exp_key, temp_value), set()):
            stset.add((exp_key, int(st)))
            qwen15b_all_temps_steps.add((float(temp_value), exp_key, int(st)))

    summary_rows: List[Dict[str, Any]] = []
    for (exp_key, temp_value), c in summary.items():
        model, domain = exp_meta.get(exp_key, ("", ""))
        n_steps = len(steps_seen.get((exp_key, temp_value), set()))
        summary_rows.append(
            {
                "experiment": exp_key,
                "model": model,
                "domain": domain,
                "temp": float(temp_value),
                "steps": int(n_steps),
                "n_total": int(c.n_total),
                "n_labeled": int(c.n_labeled),
                "shift_count": int(c.n_shift),
                "shift_pct": float(_pct(c.n_shift, c.n_labeled)),
                "label_coverage_pct": float(_pct(c.n_labeled, c.n_total)),
            },
        )
    for temp_value, c in sorted(qwen15b_by_temp.items()):
        summary_rows.append(
            {
                "experiment": "qwen15b_all_domains",
                "model": "Qwen-1.5B",
                "domain": "ALL",
                "temp": float(temp_value),
                "steps": int(len(qwen15b_steps_by_temp.get(float(temp_value), set()))),
                "n_total": int(c.n_total),
                "n_labeled": int(c.n_labeled),
                "shift_count": int(c.n_shift),
                "shift_pct": float(_pct(c.n_shift, c.n_labeled)),
                "label_coverage_pct": float(_pct(c.n_labeled, c.n_total)),
            },
        )
    if qwen15b_all_temps.n_total > 0:
        summary_rows.append(
            {
                "experiment": "qwen15b_all_temps",
                "model": "Qwen-1.5B",
                "domain": "ALL",
                "temp": None,
                "steps": int(len(qwen15b_all_temps_steps)),
                "n_total": int(qwen15b_all_temps.n_total),
                "n_labeled": int(qwen15b_all_temps.n_labeled),
                "shift_count": int(qwen15b_all_temps.n_shift),
                "shift_pct": float(_pct(qwen15b_all_temps.n_shift, qwen15b_all_temps.n_labeled)),
                "label_coverage_pct": float(_pct(qwen15b_all_temps.n_labeled, qwen15b_all_temps.n_total)),
            },
        )
    for temp_value, c in sorted(qwen_llama_by_temp.items()):
        summary_rows.append(
            {
                "experiment": "qwen7b_llama8b_math",
                "model": "Qwen-7B + Llama-8B",
                "domain": "Math",
                "temp": float(temp_value),
                "steps": int(len(qwen_llama_steps_by_temp.get(float(temp_value), set()))),
                "n_total": int(c.n_total),
                "n_labeled": int(c.n_labeled),
                "shift_count": int(c.n_shift),
                "shift_pct": float(_pct(c.n_shift, c.n_labeled)),
                "label_coverage_pct": float(_pct(c.n_labeled, c.n_total)),
            },
        )
    if qwen_llama_all_temps.n_total > 0:
        summary_rows.append(
            {
                "experiment": "qwen7b_llama8b_all_temps",
                "model": "Qwen-7B + Llama-8B",
                "domain": "Math",
                "temp": None,
                "steps": int(len(qwen_llama_all_temps_steps)),
                "n_total": int(qwen_llama_all_temps.n_total),
                "n_labeled": int(qwen_llama_all_temps.n_labeled),
                "shift_count": int(qwen_llama_all_temps.n_shift),
                "shift_pct": float(_pct(qwen_llama_all_temps.n_shift, qwen_llama_all_temps.n_labeled)),
                "label_coverage_pct": float(_pct(qwen_llama_all_temps.n_labeled, qwen_llama_all_temps.n_total)),
            },
        )

    if not args.no_summary or args.print_summary_only:
        overall = StepCounts()
        overall_steps: set[Tuple[str, float, int]] = set()
        headers = [
            "experiment",
            "temp",
            "steps",
            "N_total",
            "N_labeled",
            "shifts",
            "shift_%",
            "label_cov_%",
        ]
        table_rows: List[List[str]] = []
        for (exp_key, temp_value) in sorted(summary, key=lambda x: (x[0][0], x[0][1])):
            c = summary[(exp_key, temp_value)]
            n_steps = len(steps_seen.get((exp_key, temp_value), set()))
            overall.n_total += c.n_total
            overall.n_labeled += c.n_labeled
            overall.n_shift += c.n_shift
            for st in steps_seen.get((exp_key, temp_value), set()):
                overall_steps.add((exp_key, float(temp_value), int(st)))
            table_rows.append(
                [
                    str(exp_key),
                    f"{temp_value:.3g}",
                    str(n_steps),
                    str(c.n_total),
                    str(c.n_labeled),
                    str(c.n_shift),
                    f"{_pct(c.n_shift, c.n_labeled):.6f}",
                    f"{_pct(c.n_labeled, c.n_total):.6f}",
                ],
            )
        # Insert Qwen-1.5B all-domains rollups (per temp).
        for temp_value, c in sorted(qwen15b_by_temp.items()):
            n_steps = len(qwen15b_steps_by_temp.get(float(temp_value), set()))
            table_rows.append(
                [
                    "qwen15b_all_domains",
                    f"{temp_value:.3g}",
                    str(n_steps),
                    str(c.n_total),
                    str(c.n_labeled),
                    str(c.n_shift),
                    f"{_pct(c.n_shift, c.n_labeled):.6f}",
                    f"{_pct(c.n_labeled, c.n_total):.6f}",
                ],
            )
        for temp_value, c in sorted(qwen_llama_by_temp.items()):
            n_steps = len(qwen_llama_steps_by_temp.get(float(temp_value), set()))
            table_rows.append(
                [
                    "qwen7b_llama8b_math",
                    f"{temp_value:.3g}",
                    str(n_steps),
                    str(c.n_total),
                    str(c.n_labeled),
                    str(c.n_shift),
                    f"{_pct(c.n_shift, c.n_labeled):.6f}",
                    f"{_pct(c.n_labeled, c.n_total):.6f}",
                ],
            )
        if qwen15b_all_temps.n_total > 0:
            table_rows.append(
                [
                    "qwen15b_all_temps",
                    "NA",
                    str(len(qwen15b_all_temps_steps)),
                    str(qwen15b_all_temps.n_total),
                    str(qwen15b_all_temps.n_labeled),
                    str(qwen15b_all_temps.n_shift),
                    f"{_pct(qwen15b_all_temps.n_shift, qwen15b_all_temps.n_labeled):.6f}",
                    f"{_pct(qwen15b_all_temps.n_labeled, qwen15b_all_temps.n_total):.6f}",
                ],
            )
        if qwen_llama_all_temps.n_total > 0:
            table_rows.append(
                [
                    "qwen7b_llama8b_all_temps",
                    "NA",
                    str(len(qwen_llama_all_temps_steps)),
                    str(qwen_llama_all_temps.n_total),
                    str(qwen_llama_all_temps.n_labeled),
                    str(qwen_llama_all_temps.n_shift),
                    f"{_pct(qwen_llama_all_temps.n_shift, qwen_llama_all_temps.n_labeled):.6f}",
                    f"{_pct(qwen_llama_all_temps.n_labeled, qwen_llama_all_temps.n_total):.6f}",
                ],
            )
        table_rows.append(
            [
                "OVERALL",
                "NA",
                str(len(overall_steps)),
                str(overall.n_total),
                str(overall.n_labeled),
                str(overall.n_shift),
                f"{_pct(overall.n_shift, overall.n_labeled):.6f}",
                f"{_pct(overall.n_labeled, overall.n_total):.6f}",
            ],
        )
        _print_aligned_table(headers, table_rows, right_align=[1, 2, 3, 4, 5, 6, 7])

    if args.print_summary_only:
        return 0

    # JSON
    import json  # local import to keep tool startup light

    payload = {
        "meta": {
            "results_base": str(results_base),
            "temps": [float(t) for t in temps],
            "split": str(args.split),
            "min_step": int(args.min_step),
            "max_step_cap": (int(args.max_step_cap) if args.max_step_cap is not None else None),
            "pass_key": str(args.pass_key),
            "label_key": str(args.label_key),
            "grid": [spec.__dict__ for spec in grid],
            "missing_roots": [{"experiment": k, "temp": t} for k, t in missing],
        },
        "summary": summary_rows,
        "rows": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # CSV
    fieldnames = [
        "experiment",
        "model",
        "domain",
        "temp",
        "min_step",
        "max_step",
        "step",
        "n_total",
        "n_labeled",
        "shift_count",
        "shift_pct",
        "label_coverage_pct",
        "root",
        "split",
        "pass_key",
        "label_key",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    # Console summary (compact, for quick checks).
    print(f"[ok] wrote {len(rows)} rows")
    print(f"[ok] json: {out_json}")
    print(f"[ok] csv : {out_csv}")
    if missing:
        print(f"[warn] missing roots ({len(missing)}):")
        for key, temp in missing:
            print(f"  - {key} @ temp={temp}")

    return 0


def cmd_shift_accuracy(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="master_analysis.py shift-accuracy",
        description="Compute accuracy for shifted vs non-shifted traces across the standard experiment grid.",
    )
    parser.add_argument(
        "--results_base",
        default="artifacts/results",
        help="Base directory containing results roots (default: artifacts/results).",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Record-level split filter (default: test).",
    )
    parser.add_argument(
        "--min_step",
        type=int,
        default=0,
        help="Minimum checkpoint step to include (inclusive, default: 0).",
    )
    parser.add_argument(
        "--max_step_cap",
        type=int,
        default=None,
        help=(
            "Optional global cap on checkpoint steps (inclusive), applied on top "
            "of each experiment's default max_step."
        ),
    )
    parser.add_argument(
        "--temps",
        nargs="+",
        type=float,
        default=list(DEFAULT_TEMPS),
        help="Temperatures to include (default: 0 0.05 0.3 0.7).",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=[],
        help="Optional subset of experiment keys to run (default: all).",
    )
    parser.add_argument(
        "--pass_key",
        default="pass1",
        help="Which pass dict to read labels/correctness from (default: pass1).",
    )
    parser.add_argument(
        "--label_key",
        default="shift_in_reasoning_v1",
        help="Shift label field name inside the pass dict (default: shift_in_reasoning_v1).",
    )
    parser.add_argument(
        "--out_json",
        default="artifacts/analysis/shift_accuracy_summary.json",
        help="Write rows to this JSON file (default: artifacts/analysis/shift_accuracy_summary.json).",
    )
    parser.add_argument(
        "--out_csv",
        default="artifacts/analysis/shift_accuracy_summary.csv",
        help="Write rows to this CSV file (default: artifacts/analysis/shift_accuracy_summary.csv).",
    )
    parser.add_argument(
        "--no_summary",
        action="store_true",
        help="Disable printing the summary table to stdout.",
    )
    parser.add_argument(
        "--print_summary_only",
        action="store_true",
        help="Only print the summary (skip writing JSON/CSV).",
    )
    args = parser.parse_args(argv)

    results_base = Path(args.results_base)
    grid = build_default_grid()
    if args.only:
        allowed = set(args.only)
        grid = [spec for spec in grid if spec.key in allowed]

    temps = list(args.temps)
    missing: List[Tuple[str, float]] = []
    per_exp_temp: Dict[Tuple[str, float], Tuple[AccuracyCounts, set[int]]] = {}
    exp_meta: Dict[str, Tuple[str, str]] = {}

    for spec in grid:
        for temp in temps:
            root = resolve_results_root(results_base, spec, temp)
            if root is None:
                missing.append((spec.key, float(temp)))
                continue

            max_step = int(spec.max_step)
            if args.max_step_cap is not None:
                max_step = min(max_step, int(args.max_step_cap))

            counts, steps = iter_accuracy_counts(
                root,
                split=str(args.split or ""),
                min_step=int(args.min_step),
                max_step=max_step,
                pass_key=str(args.pass_key),
                label_key=str(args.label_key),
            )
            per_exp_temp[(spec.key, float(temp))] = (counts, steps)
            exp_meta[spec.key] = (spec.model, spec.domain)

    # Rollups for Qwen-1.5B across all three domains.
    qwen15b_keys = {k for k, (m, _) in exp_meta.items() if m == "Qwen-1.5B"}
    qwen15b_by_temp: Dict[float, AccuracyCounts] = {}
    qwen15b_steps_by_temp: Dict[float, set[Tuple[str, int]]] = {}
    qwen15b_all_temps = AccuracyCounts()
    qwen15b_all_temps_steps: set[Tuple[float, str, int]] = set()
    qwen_llama_keys = {k for k, (m, _) in exp_meta.items() if m in {"Qwen-7B", "Llama-8B"}}
    qwen_llama_by_temp: Dict[float, AccuracyCounts] = {}
    qwen_llama_steps_by_temp: Dict[float, set[Tuple[str, int]]] = {}
    qwen_llama_all_temps = AccuracyCounts()
    qwen_llama_all_temps_steps: set[Tuple[float, str, int]] = set()

    for (exp_key, temp_value), (c, steps) in per_exp_temp.items():
        if exp_key not in qwen15b_keys:
            continue
        bucket = qwen15b_by_temp.setdefault(float(temp_value), AccuracyCounts())
        bucket.n_total += c.n_total
        bucket.n_labeled += c.n_labeled
        bucket.n_shift += c.n_shift
        bucket.k_shift += c.k_shift
        bucket.n_noshift += c.n_noshift
        bucket.k_noshift += c.k_noshift
        bucket.n_shift_p2 += c.n_shift_p2
        bucket.k_shift_p2 += c.k_shift_p2
        bucket.n_noshift_p2 += c.n_noshift_p2
        bucket.k_noshift_p2 += c.k_noshift_p2

        qwen15b_all_temps.n_total += c.n_total
        qwen15b_all_temps.n_labeled += c.n_labeled
        qwen15b_all_temps.n_shift += c.n_shift
        qwen15b_all_temps.k_shift += c.k_shift
        qwen15b_all_temps.n_noshift += c.n_noshift
        qwen15b_all_temps.k_noshift += c.k_noshift
        qwen15b_all_temps.n_shift_p2 += c.n_shift_p2
        qwen15b_all_temps.k_shift_p2 += c.k_shift_p2
        qwen15b_all_temps.n_noshift_p2 += c.n_noshift_p2
        qwen15b_all_temps.k_noshift_p2 += c.k_noshift_p2

        stset = qwen15b_steps_by_temp.setdefault(float(temp_value), set())
        for st in steps:
            stset.add((exp_key, int(st)))
            qwen15b_all_temps_steps.add((float(temp_value), exp_key, int(st)))

        if exp_key in qwen_llama_keys:
            bucket = qwen_llama_by_temp.setdefault(float(temp_value), AccuracyCounts())
            bucket.n_total += c.n_total
            bucket.n_labeled += c.n_labeled
            bucket.n_shift += c.n_shift
            bucket.k_shift += c.k_shift
            bucket.n_noshift += c.n_noshift
            bucket.k_noshift += c.k_noshift
            bucket.n_shift_p2 += c.n_shift_p2
            bucket.k_shift_p2 += c.k_shift_p2
            bucket.n_noshift_p2 += c.n_noshift_p2
            bucket.k_noshift_p2 += c.k_noshift_p2

            qwen_llama_all_temps.n_total += c.n_total
            qwen_llama_all_temps.n_labeled += c.n_labeled
            qwen_llama_all_temps.n_shift += c.n_shift
            qwen_llama_all_temps.k_shift += c.k_shift
            qwen_llama_all_temps.n_noshift += c.n_noshift
            qwen_llama_all_temps.k_noshift += c.k_noshift
            qwen_llama_all_temps.n_shift_p2 += c.n_shift_p2
            qwen_llama_all_temps.k_shift_p2 += c.k_shift_p2
            qwen_llama_all_temps.n_noshift_p2 += c.n_noshift_p2
            qwen_llama_all_temps.k_noshift_p2 += c.k_noshift_p2

            stset = qwen_llama_steps_by_temp.setdefault(float(temp_value), set())
            for st in steps:
                stset.add((exp_key, int(st)))
                qwen_llama_all_temps_steps.add((float(temp_value), exp_key, int(st)))

    def acc_pct(k: int, n: int) -> float:
        return (100.0 * float(k) / float(n)) if n else 0.0

    def delta_pp(acc_a: float, acc_b: float) -> float:
        return float(acc_a - acc_b)

    summary_rows: List[Dict[str, Any]] = []
    overall = AccuracyCounts()
    overall_steps: set[Tuple[str, float, int]] = set()
    for (exp_key, temp_value), (c, steps) in sorted(per_exp_temp.items(), key=lambda x: (x[0][0], x[0][1])):
        model, domain = exp_meta.get(exp_key, ("", ""))
        p_shift = acc_pct(c.k_shift, c.n_shift)
        p_noshift = acc_pct(c.k_noshift, c.n_noshift)
        p_shift_p2 = acc_pct(c.k_shift_p2, c.n_shift_p2)
        p_noshift_p2 = acc_pct(c.k_noshift_p2, c.n_noshift_p2)
        overall.n_total += c.n_total
        overall.n_labeled += c.n_labeled
        overall.n_shift += c.n_shift
        overall.k_shift += c.k_shift
        overall.n_noshift += c.n_noshift
        overall.k_noshift += c.k_noshift
        overall.n_shift_p2 += c.n_shift_p2
        overall.k_shift_p2 += c.k_shift_p2
        overall.n_noshift_p2 += c.n_noshift_p2
        overall.k_noshift_p2 += c.k_noshift_p2
        for st in steps:
            overall_steps.add((exp_key, float(temp_value), int(st)))
        summary_rows.append(
            {
                "experiment": exp_key,
                "model": model,
                "domain": domain,
                "temp": float(temp_value),
                "steps": int(len(steps)),
                "n_total_with_correct": int(c.n_total),
                "n_labeled": int(c.n_labeled),
                "n_shift": int(c.n_shift),
                "k_shift": int(c.k_shift),
                "n_noshift": int(c.n_noshift),
                "k_noshift": int(c.k_noshift),
                "acc_shift_pct": float(p_shift),
                "acc_noshift_pct": float(p_noshift),
                "delta_pp": float(delta_pp(p_shift, p_noshift)),
                "n_labeled_p2": int(c.n_shift_p2 + c.n_noshift_p2),
                "n_shift_p2": int(c.n_shift_p2),
                "k_shift_p2": int(c.k_shift_p2),
                "n_noshift_p2": int(c.n_noshift_p2),
                "k_noshift_p2": int(c.k_noshift_p2),
                "acc_shift_p2_pct": float(p_shift_p2),
                "acc_noshift_p2_pct": float(p_noshift_p2),
                "delta_pp_p2": float(delta_pp(p_shift_p2, p_noshift_p2)),
                "shift_pct": float(_pct(c.n_shift, c.n_labeled)),
                "label_coverage_pct": float(_pct(c.n_labeled, c.n_total)),
            },
        )

    for temp_value, c in sorted(qwen15b_by_temp.items()):
        p_shift = acc_pct(c.k_shift, c.n_shift)
        p_noshift = acc_pct(c.k_noshift, c.n_noshift)
        p_shift_p2 = acc_pct(c.k_shift_p2, c.n_shift_p2)
        p_noshift_p2 = acc_pct(c.k_noshift_p2, c.n_noshift_p2)
        summary_rows.append(
            {
                "experiment": "qwen15b_all_domains",
                "model": "Qwen-1.5B",
                "domain": "ALL",
                "temp": float(temp_value),
                "steps": int(len(qwen15b_steps_by_temp.get(float(temp_value), set()))),
                "n_total_with_correct": int(c.n_total),
                "n_labeled": int(c.n_labeled),
                "n_shift": int(c.n_shift),
                "k_shift": int(c.k_shift),
                "n_noshift": int(c.n_noshift),
                "k_noshift": int(c.k_noshift),
                "acc_shift_pct": float(p_shift),
                "acc_noshift_pct": float(p_noshift),
                "delta_pp": float(delta_pp(p_shift, p_noshift)),
                "n_labeled_p2": int(c.n_shift_p2 + c.n_noshift_p2),
                "n_shift_p2": int(c.n_shift_p2),
                "k_shift_p2": int(c.k_shift_p2),
                "n_noshift_p2": int(c.n_noshift_p2),
                "k_noshift_p2": int(c.k_noshift_p2),
                "acc_shift_p2_pct": float(p_shift_p2),
                "acc_noshift_p2_pct": float(p_noshift_p2),
                "delta_pp_p2": float(delta_pp(p_shift_p2, p_noshift_p2)),
                "shift_pct": float(_pct(c.n_shift, c.n_labeled)),
                "label_coverage_pct": float(_pct(c.n_labeled, c.n_total)),
            },
        )

    if qwen15b_all_temps.n_total > 0:
        p_shift = acc_pct(qwen15b_all_temps.k_shift, qwen15b_all_temps.n_shift)
        p_noshift = acc_pct(qwen15b_all_temps.k_noshift, qwen15b_all_temps.n_noshift)
        p_shift_p2 = acc_pct(qwen15b_all_temps.k_shift_p2, qwen15b_all_temps.n_shift_p2)
        p_noshift_p2 = acc_pct(qwen15b_all_temps.k_noshift_p2, qwen15b_all_temps.n_noshift_p2)
        summary_rows.append(
            {
                "experiment": "qwen15b_all_temps",
                "model": "Qwen-1.5B",
                "domain": "ALL",
                "temp": None,
                "steps": int(len(qwen15b_all_temps_steps)),
                "n_total_with_correct": int(qwen15b_all_temps.n_total),
                "n_labeled": int(qwen15b_all_temps.n_labeled),
                "n_shift": int(qwen15b_all_temps.n_shift),
                "k_shift": int(qwen15b_all_temps.k_shift),
                "n_noshift": int(qwen15b_all_temps.n_noshift),
                "k_noshift": int(qwen15b_all_temps.k_noshift),
                "acc_shift_pct": float(p_shift),
                "acc_noshift_pct": float(p_noshift),
                "delta_pp": float(delta_pp(p_shift, p_noshift)),
                "n_labeled_p2": int(qwen15b_all_temps.n_shift_p2 + qwen15b_all_temps.n_noshift_p2),
                "n_shift_p2": int(qwen15b_all_temps.n_shift_p2),
                "k_shift_p2": int(qwen15b_all_temps.k_shift_p2),
                "n_noshift_p2": int(qwen15b_all_temps.n_noshift_p2),
                "k_noshift_p2": int(qwen15b_all_temps.k_noshift_p2),
                "acc_shift_p2_pct": float(p_shift_p2),
                "acc_noshift_p2_pct": float(p_noshift_p2),
                "delta_pp_p2": float(delta_pp(p_shift_p2, p_noshift_p2)),
                "shift_pct": float(_pct(qwen15b_all_temps.n_shift, qwen15b_all_temps.n_labeled)),
                "label_coverage_pct": float(_pct(qwen15b_all_temps.n_labeled, qwen15b_all_temps.n_total)),
            },
        )

    for temp_value, c in sorted(qwen_llama_by_temp.items()):
        p_shift = acc_pct(c.k_shift, c.n_shift)
        p_noshift = acc_pct(c.k_noshift, c.n_noshift)
        p_shift_p2 = acc_pct(c.k_shift_p2, c.n_shift_p2)
        p_noshift_p2 = acc_pct(c.k_noshift_p2, c.n_noshift_p2)
        summary_rows.append(
            {
                "experiment": "qwen7b_llama8b_math",
                "model": "Qwen-7B + Llama-8B",
                "domain": "Math",
                "temp": float(temp_value),
                "steps": int(len(qwen_llama_steps_by_temp.get(float(temp_value), set()))),
                "n_total_with_correct": int(c.n_total),
                "n_labeled": int(c.n_labeled),
                "n_shift": int(c.n_shift),
                "k_shift": int(c.k_shift),
                "n_noshift": int(c.n_noshift),
                "k_noshift": int(c.k_noshift),
                "acc_shift_pct": float(p_shift),
                "acc_noshift_pct": float(p_noshift),
                "delta_pp": float(delta_pp(p_shift, p_noshift)),
                "n_labeled_p2": int(c.n_shift_p2 + c.n_noshift_p2),
                "n_shift_p2": int(c.n_shift_p2),
                "k_shift_p2": int(c.k_shift_p2),
                "n_noshift_p2": int(c.n_noshift_p2),
                "k_noshift_p2": int(c.k_noshift_p2),
                "acc_shift_p2_pct": float(p_shift_p2),
                "acc_noshift_p2_pct": float(p_noshift_p2),
                "delta_pp_p2": float(delta_pp(p_shift_p2, p_noshift_p2)),
                "shift_pct": float(_pct(c.n_shift, c.n_labeled)),
                "label_coverage_pct": float(_pct(c.n_labeled, c.n_total)),
            },
        )

    if qwen_llama_all_temps.n_total > 0:
        p_shift = acc_pct(qwen_llama_all_temps.k_shift, qwen_llama_all_temps.n_shift)
        p_noshift = acc_pct(qwen_llama_all_temps.k_noshift, qwen_llama_all_temps.n_noshift)
        p_shift_p2 = acc_pct(qwen_llama_all_temps.k_shift_p2, qwen_llama_all_temps.n_shift_p2)
        p_noshift_p2 = acc_pct(qwen_llama_all_temps.k_noshift_p2, qwen_llama_all_temps.n_noshift_p2)
        summary_rows.append(
            {
                "experiment": "qwen7b_llama8b_all_temps",
                "model": "Qwen-7B + Llama-8B",
                "domain": "Math",
                "temp": None,
                "steps": int(len(qwen_llama_all_temps_steps)),
                "n_total_with_correct": int(qwen_llama_all_temps.n_total),
                "n_labeled": int(qwen_llama_all_temps.n_labeled),
                "n_shift": int(qwen_llama_all_temps.n_shift),
                "k_shift": int(qwen_llama_all_temps.k_shift),
                "n_noshift": int(qwen_llama_all_temps.n_noshift),
                "k_noshift": int(qwen_llama_all_temps.k_noshift),
                "acc_shift_pct": float(p_shift),
                "acc_noshift_pct": float(p_noshift),
                "delta_pp": float(delta_pp(p_shift, p_noshift)),
                "n_labeled_p2": int(qwen_llama_all_temps.n_shift_p2 + qwen_llama_all_temps.n_noshift_p2),
                "n_shift_p2": int(qwen_llama_all_temps.n_shift_p2),
                "k_shift_p2": int(qwen_llama_all_temps.k_shift_p2),
                "n_noshift_p2": int(qwen_llama_all_temps.n_noshift_p2),
                "k_noshift_p2": int(qwen_llama_all_temps.k_noshift_p2),
                "acc_shift_p2_pct": float(p_shift_p2),
                "acc_noshift_p2_pct": float(p_noshift_p2),
                "delta_pp_p2": float(delta_pp(p_shift_p2, p_noshift_p2)),
                "shift_pct": float(_pct(qwen_llama_all_temps.n_shift, qwen_llama_all_temps.n_labeled)),
                "label_coverage_pct": float(_pct(qwen_llama_all_temps.n_labeled, qwen_llama_all_temps.n_total)),
            },
        )

    if overall.n_total > 0:
        p_shift = acc_pct(overall.k_shift, overall.n_shift)
        p_noshift = acc_pct(overall.k_noshift, overall.n_noshift)
        p_shift_p2 = acc_pct(overall.k_shift_p2, overall.n_shift_p2)
        p_noshift_p2 = acc_pct(overall.k_noshift_p2, overall.n_noshift_p2)
        summary_rows.append(
            {
                "experiment": "OVERALL",
                "model": "ALL",
                "domain": "ALL",
                "temp": None,
                "steps": int(len(overall_steps)),
                "n_total_with_correct": int(overall.n_total),
                "n_labeled": int(overall.n_labeled),
                "n_shift": int(overall.n_shift),
                "k_shift": int(overall.k_shift),
                "n_noshift": int(overall.n_noshift),
                "k_noshift": int(overall.k_noshift),
                "acc_shift_pct": float(p_shift),
                "acc_noshift_pct": float(p_noshift),
                "delta_pp": float(delta_pp(p_shift, p_noshift)),
                "n_labeled_p2": int(overall.n_shift_p2 + overall.n_noshift_p2),
                "n_shift_p2": int(overall.n_shift_p2),
                "k_shift_p2": int(overall.k_shift_p2),
                "n_noshift_p2": int(overall.n_noshift_p2),
                "k_noshift_p2": int(overall.k_noshift_p2),
                "acc_shift_p2_pct": float(p_shift_p2),
                "acc_noshift_p2_pct": float(p_noshift_p2),
                "delta_pp_p2": float(delta_pp(p_shift_p2, p_noshift_p2)),
                "shift_pct": float(_pct(overall.n_shift, overall.n_labeled)),
                "label_coverage_pct": float(_pct(overall.n_labeled, overall.n_total)),
            },
        )

    if not args.no_summary:
        headers = [
            "experiment",
            "temp",
            "steps",
            "N",
            "acc_shift_%",
            "acc_noshift_%",
            "delta_pp",
            "acc_shift_p2_%",
            "acc_noshift_p2_%",
            "delta_pp_p2",
            "shift_%",
        ]
        table_rows: List[List[str]] = []
        for row in summary_rows:
            # Print the "main grid" rows first, then rollups (which we appended last).
            temp_val = row["temp"]
            temp_str = "NA" if temp_val is None else f"{float(temp_val):.3g}"
            table_rows.append(
                [
                    str(row["experiment"]),
                    temp_str,
                    str(row["steps"]),
                    str(row["n_labeled"]),
                    f"{float(row['acc_shift_pct']):.2f}",
                    f"{float(row['acc_noshift_pct']):.2f}",
                    f"{float(row['delta_pp']):.2f}",
                    f"{float(row['acc_shift_p2_pct']):.2f}",
                    f"{float(row['acc_noshift_p2_pct']):.2f}",
                    f"{float(row['delta_pp_p2']):.2f}",
                    f"{float(row['shift_pct']):.2f}",
                ],
            )
        _print_aligned_table(headers, table_rows, right_align=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        if missing:
            print(f"\n[warn] missing roots ({len(missing)}):")
            for key, temp in missing:
                print(f"  - {key} @ temp={temp}")

    if args.print_summary_only:
        return 0

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    _ensure_parent(out_json)
    _ensure_parent(out_csv)

    import json  # local import

    payload = {
        "meta": {
            "results_base": str(results_base),
            "temps": [float(t) for t in temps],
            "split": str(args.split),
            "min_step": int(args.min_step),
            "max_step_cap": (int(args.max_step_cap) if args.max_step_cap is not None else None),
            "pass_key": str(args.pass_key),
            "label_key": str(args.label_key),
            "grid": [spec.__dict__ for spec in grid],
            "missing_roots": [{"experiment": k, "temp": t} for k, t in missing],
        },
        "summary": summary_rows,
    }
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fieldnames = [
        "experiment",
        "model",
        "domain",
        "temp",
        "steps",
        "n_total_with_correct",
        "n_labeled",
        "n_shift",
        "k_shift",
        "n_noshift",
        "k_noshift",
        "acc_shift_pct",
        "acc_noshift_pct",
        "delta_pp",
        "n_labeled_p2",
        "n_shift_p2",
        "k_shift_p2",
        "n_noshift_p2",
        "k_noshift_p2",
        "acc_shift_p2_pct",
        "acc_noshift_p2_pct",
        "delta_pp_p2",
        "shift_pct",
        "label_coverage_pct",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"[ok] json: {out_json}")
    print(f"[ok] csv : {out_csv}")
    return 0


def cmd_table2(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="master_analysis.py table2",
        description="Emit Table 2 (LaTeX): shift prevalence and conditional accuracy (RQ1).",
    )
    parser.add_argument(
        "--accuracy_json",
        default="artifacts/analysis/shift_accuracy_summary.json",
        help="Input JSON produced by shift-accuracy (default: artifacts/analysis/shift_accuracy_summary.json).",
    )
    parser.add_argument(
        "--out_tex",
        default="latex/table2_shift_accuracy.tex",
        help="Write LaTeX table to this path (default: latex/table2_shift_accuracy.tex).",
    )
    parser.add_argument(
        "--label",
        default="tab:shift-accuracy",
        help="LaTeX label (default: tab:shift-accuracy).",
    )
    parser.add_argument(
        "--caption",
        default=(
            "\\textbf{Shift prevalence and conditional accuracy (RQ1).} "
            "(\\%S_{i,j}) gives the fraction of traces labeled as containing a reasoning shift. "
            "(P(\\checkmark \\mid S_{i,j}{=}0)) and "
            "(P(\\checkmark \\mid S_{i,j}{=}1)) report accuracy without vs.\\ with a detected shift, "
            "pooled across all problems, temperatures $\\{0, 0.05, 0.3, 0.7\\}$, checkpoints, and samples using "
            "count-weighted (not simple) averages. Across models and domains, shifted traces are consistently less accurate. "
            "\\includegraphics[width=.7em]{icons/qwen.pdf} = Qwen\\,2.5; "
            "\\includegraphics[width=.7em]{latex/icons/llamma.pdf} = Llama\\,3.1."
        ),
        help="LaTeX caption (default: paper caption).",
    )
    args = parser.parse_args(argv)

    import json  # local import

    in_path = Path(args.accuracy_json)
    payload = json.loads(in_path.read_text(encoding="utf-8"))
    rows = payload.get("summary") or []
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"No rows found in {in_path}")

    # Aggregate per experiment key across all temps.
    want_keys = [
        "qwen15b_crossword",
        "qwen15b_math",
        "qwen15b_carpark",
        "qwen7b_math",
        "llama8b_math",
    ]
    agg: Dict[str, AccuracyCounts] = {k: AccuracyCounts() for k in want_keys}
    missing_k = False
    for row in rows:
        exp = row.get("experiment")
        temp = row.get("temp")
        if exp not in agg:
            continue
        # Skip rollup rows where temp is None.
        if temp is None:
            continue
        if row.get("k_shift") is None or row.get("k_noshift") is None:
            missing_k = True
        c = agg[exp]
        c.n_labeled += int(row.get("n_labeled") or 0)
        c.n_shift += int(row.get("n_shift") or 0)
        c.k_shift += int(row.get("k_shift") or 0)
        c.n_noshift += int(row.get("n_noshift") or 0)
        c.k_noshift += int(row.get("k_noshift") or 0)

    if missing_k:
        raise SystemExit(
            f"{in_path} is missing exact success counts (k_shift/k_noshift). "
            "Re-run `python tools/master_analysis.py shift-accuracy` (or `... all`) "
            "to regenerate it, then rerun `... table2`."
        )

    def pct(num: int, den: int) -> float:
        return (100.0 * float(num) / float(den)) if den else 0.0

    def prob(k: int, n: int) -> float:
        return (float(k) / float(n)) if n else 0.0

    domain_label = {"Crossword": "Xwords", "Math": "Math", "Carpark": "RHour"}
    model_cell = {
        "qwen15b": "\\\\includegraphics[width=1em]{icons/qwen.pdf}-1.5B",
        "qwen7b": "\\\\includegraphics[width=1em]{icons/qwen.pdf}-7B",
        "llama8b": "\\\\includegraphics[width=1em]{latex/icons/llamma.pdf}-8B",
        "qwen7b_llama8b": (
            "\\\\includegraphics[width=1em]{icons/qwen.pdf}-7B + "
            "\\\\includegraphics[width=1em]{latex/icons/llamma.pdf}-8B"
        ),
    }

    def row_for(exp_key: str) -> Tuple[float, float, float]:
        c = agg[exp_key]
        share_shift = pct(c.n_shift, c.n_labeled)
        p0 = prob(c.k_noshift, c.n_noshift)
        p1 = prob(c.k_shift, c.n_shift)
        return share_shift, p0, p1

    # Overall across all rows (count-weighted).
    overall = AccuracyCounts()
    for k in want_keys:
        c = agg[k]
        overall.n_labeled += c.n_labeled
        overall.n_shift += c.n_shift
        overall.k_shift += c.k_shift
        overall.n_noshift += c.n_noshift
        overall.k_noshift += c.k_noshift

    def fmt_share(x: float) -> str:
        return f"{x:.2f}"

    def fmt_prob(x: float) -> str:
        return f"{x:.3f}"

    s_all = pct(overall.n_shift, overall.n_labeled)
    p0_all = prob(overall.k_noshift, overall.n_noshift)
    p1_all = prob(overall.k_shift, overall.n_shift)

    s_xw, p0_xw, p1_xw = row_for("qwen15b_crossword")
    s_m, p0_m, p1_m = row_for("qwen15b_math")
    s_rh, p0_rh, p1_rh = row_for("qwen15b_carpark")
    qwen_combo = AccuracyCounts()
    for key in ("qwen15b_crossword", "qwen15b_math", "qwen15b_carpark"):
        c = agg[key]
        qwen_combo.n_labeled += c.n_labeled
        qwen_combo.n_shift += c.n_shift
        qwen_combo.k_shift += c.k_shift
        qwen_combo.n_noshift += c.n_noshift
        qwen_combo.k_noshift += c.k_noshift
    s_q_all = pct(qwen_combo.n_shift, qwen_combo.n_labeled)
    p0_q_all = prob(qwen_combo.k_noshift, qwen_combo.n_noshift)
    p1_q_all = prob(qwen_combo.k_shift, qwen_combo.n_shift)
    s_7, p0_7, p1_7 = row_for("qwen7b_math")
    s_l, p0_l, p1_l = row_for("llama8b_math")
    combo = AccuracyCounts()
    for key in ("qwen7b_math", "llama8b_math"):
        c = agg[key]
        combo.n_labeled += c.n_labeled
        combo.n_shift += c.n_shift
        combo.k_shift += c.k_shift
        combo.n_noshift += c.n_noshift
        combo.k_noshift += c.k_noshift
    s_c = pct(combo.n_shift, combo.n_labeled)
    p0_c = prob(combo.k_noshift, combo.n_noshift)
    p1_c = prob(combo.k_shift, combo.n_shift)

    tex = (
        "\\begin{table}[t]\n"
        "\\footnotesize\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}} l l r r r @{}}\n"
        "\\toprule\n"
        "{\\textbf{Model}} &\n"
        "{\\textbf{Domain}} &\n"
        "{\\(\\%{S_{i,j}}\\)} &\n"
        "{\\scriptsize \\(P(\\checkmark\\mid S_{i,j}\\!=\\!0)\\)} &\n"
        "{\\scriptsize \\(P(\\checkmark\\mid S_{i,j}\\!=\\!1)\\)} \\\\\n"
        "\\midrule\n"
        f"{model_cell['qwen15b']} \n"
        f"  & {domain_label['Crossword']}      & {fmt_share(s_xw)} & {fmt_prob(p0_xw)} & {fmt_prob(p1_xw)} \\\\\n"
        f"  & {domain_label['Math']}       & {fmt_share(s_m)} & {fmt_prob(p0_m)} & {fmt_prob(p1_m)} \\\\\n"
        f"  & {domain_label['Carpark']}      & {fmt_share(s_rh)} & {fmt_prob(p0_rh)} & {fmt_prob(p1_rh)} \\\\\n"
        f"  & All       & {fmt_share(s_q_all)} & {fmt_prob(p0_q_all)} & {fmt_prob(p1_q_all)} \\\\\n"
        "\\midrule\n"
        f"{model_cell['qwen7b']}   \n"
        f"  & {domain_label['Math']}       & {fmt_share(s_7)} & {fmt_prob(p0_7)} & {fmt_prob(p1_7)} \\\\\n"
        "\\midrule\n"
        f"{model_cell['llama8b']} \n"
        f"  & {domain_label['Math']}       & {fmt_share(s_l)} & {fmt_prob(p0_l)} & {fmt_prob(p1_l)} \\\\\n"
        "\\midrule\n"
        f"{model_cell['qwen7b_llama8b']} \n"
        f"  & {domain_label['Math']}       & {fmt_share(s_c)} & {fmt_prob(p0_c)} & {fmt_prob(p1_c)} \\\\\n"
        "\\midrule\n"
        "\\multicolumn{2}{l}{\\textbf{Overall (all rows)}} \n"
        f"  & {fmt_share(s_all)} & {fmt_prob(p0_all)} & {fmt_prob(p1_all)} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular*}\n"
        f"\\caption{{{args.caption}}}\n"
        f"\\label{{{args.label}}}\n"
        "\\end{table}\n"
    )
    out_path = Path(args.out_tex)
    _ensure_parent(out_path)
    out_path.write_text(tex, encoding="utf-8")
    print(tex)
    print(f"[ok] wrote: {out_path}")
    return 0


def _two_sided_log10_p_from_z(z_value: float) -> float:
    """
    Approximate log10(two-sided p-value) for a Normal(0,1) z-stat.

    Uses math.erfc for moderate values and an asymptotic tail approximation
    for very large |z| to avoid underflow.
    """
    z = abs(float(z_value))
    if z == 0.0:
        return 0.0

    # For moderate z, compute p directly from erfc.
    if z < 8.0:
        p = math.erfc(z / math.sqrt(2.0))
        # erfc gives two-sided already? For Normal, two-sided p = erfc(z/sqrt(2)).
        # (Because 2*(1-Phi(z)) == erfc(z/sqrt(2))).
        return math.log10(p) if p > 0.0 else float("-inf")

    # Asymptotic: log(2*(1-Phi(z))) ~= log(2) - 0.5 z^2 - log(z) - 0.5 log(2*pi)
    ln_p = math.log(2.0) - 0.5 * z * z - math.log(z) - 0.5 * math.log(2.0 * math.pi)
    return ln_p / math.log(10.0)


def cmd_pooled_logit(argv: Optional[List[str]] = None) -> int:
    """
    Emit the pooled logistic regression numbers for correctness ~ shift.

    This matches the 2x2 table implied by the binary predictor and uses the
    analytic Wald test on the log-odds ratio:
      beta = log(OR),  SE(beta) = sqrt(sum 1/cell)
    """
    parser = argparse.ArgumentParser(
        prog="master_analysis.py pooled-logit",
        description="Compute pooled logistic regression stats for correct ~ shift from shift-accuracy outputs.",
    )
    parser.add_argument(
        "--accuracy_json",
        default="artifacts/analysis/shift_accuracy_summary.json",
        help="Input JSON produced by shift-accuracy (default: artifacts/analysis/shift_accuracy_summary.json).",
    )
    parser.add_argument(
        "--pool_experiment",
        default="qwen15b_all_temps",
        help=(
            "Which summary row to use for pooling (default: qwen15b_all_temps). "
            "This should be pooled across Crossword/Math/RHour and all temps."
        ),
    )
    parser.add_argument(
        "--out_tex",
        default="latex/table2_pooled_logit_sentence.tex",
        help="Write a LaTeX-ready sentence snippet to this path (default: latex/table2_pooled_logit_sentence.tex).",
    )
    parser.add_argument(
        "--out_tex_summary",
        default="latex/table2_qwen15b_summary_sentence.tex",
        help=(
            "Write the Qwen2.5--1.5B pooled prevalence/accuracy sentence to this path "
            "(default: latex/table2_qwen15b_summary_sentence.tex)."
        ),
    )
    args = parser.parse_args(argv)

    import json  # local import

    in_path = Path(args.accuracy_json)
    payload = json.loads(in_path.read_text(encoding="utf-8"))
    rows = payload.get("summary") or []
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"No rows found in {in_path}")

    target = None
    for row in rows:
        if row.get("experiment") == args.pool_experiment:
            target = row
            break
    if target is None:
        raise SystemExit(f"Could not find experiment={args.pool_experiment!r} in {in_path}")

    n_shift = int(target.get("n_shift") or 0)
    k_shift = int(target.get("k_shift") or 0)
    n_noshift = int(target.get("n_noshift") or 0)
    k_noshift = int(target.get("k_noshift") or 0)
    n = int(target.get("n_labeled") or 0)

    if n <= 0 or n_shift <= 0 or n_noshift <= 0:
        raise SystemExit(
            f"Invalid counts for {args.pool_experiment}: "
            f"n={n}, n_shift={n_shift}, n_noshift={n_noshift}",
        )

    # 2x2 cells
    a = k_shift
    b = n_shift - k_shift
    c = k_noshift
    d = n_noshift - k_noshift

    if min(a, b, c, d) <= 0:
        # Haldane-Anscombe correction to avoid infinite log-odds.
        a += 0.5
        b += 0.5
        c += 0.5
        d += 0.5

    beta = math.log((a / b) / (c / d))
    se = math.sqrt((1.0 / a) + (1.0 / b) + (1.0 / c) + (1.0 / d))
    z = beta / se if se > 0.0 else float("inf")
    log10_p = _two_sided_log10_p_from_z(z)

    acc_shift = float(k_shift) / float(n_shift) if n_shift else 0.0
    acc_noshift = float(k_noshift) / float(n_noshift) if n_noshift else 0.0
    shift_share_pct = _pct(n_shift, n)

    # Pretty p-value bound for LaTeX (avoid printing 0.0).
    if math.isfinite(log10_p):
        exp_floor = int(math.floor(log10_p))
        # exp_floor is negative for small p (e.g. -305). Bound: p < 10^{exp_floor}.
        p_bound_tex = f"10^{{{exp_floor}}}"
    else:
        p_bound_tex = "10^{-1000}"

    sentence = (
        "A pooled logistic regression of correctness on a shift indicator confirms that this difference is highly "
        f"significant (p $< {p_bound_tex}$)."
        "\\footnote{In R-style notation,\n"
        "\\(\n"
        "\\texttt{correct} \\sim \\texttt{shift}.\n"
        "\\)\n"
        "\\texttt{correct} is a binary outcome, and \\texttt{shift} is a binary indicator for an annotator-labeled "
        "reasoning shift. The pooled regression aggregates all test-set traces across Crossword, Math, and RHour.}"
    )

    def latex_int(value: int) -> str:
        return f"{int(value):,}".replace(",", "{,}")

    summary_sentence = (
        "Across domains, temperatures, and checkpoints for Qwen2.5--1.5B, reasoning shifts remain uncommon "
        f"(approximately ${shift_share_pct:.1f}\\%$ of samples) and are associated with substantially "
        f"\\emph{{lower}} accuracy: ${acc_shift * 100.0:.2f}\\%$ for shifted traces versus "
        f"${acc_noshift * 100.0:.2f}\\%$ for non-shifted traces, $N{{=}}{latex_int(n)}$."
    )

    # Console diagnostics for sanity checking.
    print("=== pooled-logit ===")
    print(f"pool_experiment={args.pool_experiment}")
    print(f"N_labeled={n}  n_shift={n_shift}  n_noshift={n_noshift}")
    print(f"acc_shift={acc_shift:.6f}  acc_noshift={acc_noshift:.6f}")
    print(f"shift_share_pct={shift_share_pct:.6f}")
    print(f"beta(log-odds)={beta:.6f}  se={se:.6f}  z={z:.3f}  log10_p={log10_p:.2f}")
    print("\n" + sentence + "\n")
    print(summary_sentence + "\n")

    out_path = Path(args.out_tex)
    _ensure_parent(out_path)
    out_path.write_text(sentence + "\n", encoding="utf-8")
    print(f"[ok] wrote: {out_path}")

    out_path2 = Path(args.out_tex_summary)
    _ensure_parent(out_path2)
    out_path2.write_text(summary_sentence + "\n", encoding="utf-8")
    print(f"[ok] wrote: {out_path2}")
    return 0


def _infer_temp_from_root_name(root: str) -> Optional[float]:
    """
    Best-effort temperature inference for external-model roots.

    Supports conventions like:
      - *temp005  -> 0.05
      - *temp03   -> 0.3
      - *temp07   -> 0.7

    Falls back to None when no token is found.
    """
    low = str(root).lower()
    if "temp005" in low or "temp05" in low:
        return 0.05
    if "temp03" in low or "temp3" in low:
        return 0.3
    if "temp07" in low or "temp7" in low:
        return 0.7
    return None


def cmd_external_models(argv: Optional[List[str]] = None) -> int:
    """
    Summarize shift rates and conditional accuracies for external models on MATH-500.
    """
    parser = argparse.ArgumentParser(
        prog="master_analysis.py external-models",
        description="Compute external-model shift rates/conditional accuracy and emit a LaTeX table + sentence snippet.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Record-level split filter (default: test).",
    )
    parser.add_argument(
        "--pass_key",
        default="pass1",
        help="Which pass dict to read labels/correctness from (default: pass1).",
    )
    parser.add_argument(
        "--label_key",
        default="shift_in_reasoning_v1",
        help="Shift label field name inside the pass dict (default: shift_in_reasoning_v1).",
    )
    parser.add_argument(
        "--out_tex",
        default="latex/table_external_models.tex",
        help="Write LaTeX table to this path (default: latex/table_external_models.tex).",
    )
    parser.add_argument(
        "--out_sentence_tex",
        default="latex/external_models_sentence.tex",
        help="Write the updated paper sentence snippet to this path (default: latex/external_models_sentence.tex).",
    )
    parser.add_argument(
        "--label",
        default="tab:external-models",
        help="LaTeX label for the table (default: tab:external-models).",
    )
    args = parser.parse_args(argv)

    # Hardcoded roots discovered in this repo (MATH-500, external API evals).
    # We treat each root as one decoding-temperature condition.
    models = [
        ("DeepSeek\\textendash R1", "artifacts/results/deepseek-r1-openrouter"),
        ("DeepSeek\\textendash R1", "artifacts/results/deepseek-r1-openrouter-temp005"),
        ("GPT\\textendash 4o", "artifacts/results/gpt4o-math-portkey"),
        ("GPT\\textendash 4o", "artifacts/results/gpt4o-math-portkey-temp005"),
    ]

    per_model_temp: Dict[str, Dict[float, AccuracyCounts]] = {}
    for model_name, root_str in models:
        root = Path(root_str)
        if not root.exists():
            continue

        temp = _infer_temp_from_root_name(root_str)
        # If the root doesn't encode temp, we assume it's the "high temp" condition
        # (mirroring the convention used by gpt4o-math-portkey vs *-temp005).
        if temp is None:
            temp = 0.0

        # Optionally warn if record-level temperature disagrees.
        files = scan_files_step_only(str(root), split_substr=str(args.split or ""))
        if not files:
            continue
        try:
            import json  # local import

            with open(files[0], "r", encoding="utf-8") as f:
                obj = json.loads(next(f))
            temp_val = obj.get("temperature")
            if temp_val is not None and abs(float(temp_val) - float(temp)) > 1e-9:
                print(
                    f"[warn] root temp inferred as {temp} but file reports temperature={temp_val} "
                    f"({root_str}); using {temp}",
                )
        except Exception:
            pass

        counts, _steps = iter_accuracy_counts(
            root,
            split=str(args.split or ""),
            min_step=0,
            max_step=10**9,
            pass_key=str(args.pass_key),
            label_key=str(args.label_key),
        )
        per_model_temp.setdefault(model_name, {}).setdefault(float(temp), AccuracyCounts())
        bucket = per_model_temp[model_name][float(temp)]
        bucket.n_total += counts.n_total
        bucket.n_labeled += counts.n_labeled
        bucket.n_shift += counts.n_shift
        bucket.k_shift += counts.k_shift
        bucket.n_noshift += counts.n_noshift
        bucket.k_noshift += counts.k_noshift

    def shift_pct(c: AccuracyCounts) -> float:
        return _pct(c.n_shift, c.n_labeled)

    def acc(c: AccuracyCounts, which: str) -> float:
        if which == "shift":
            return (100.0 * c.k_shift / c.n_shift) if c.n_shift else 0.0
        return (100.0 * c.k_noshift / c.n_noshift) if c.n_noshift else 0.0

    # Compute shift-rate ranges for the sentence.
    def fmt_range(values: List[float]) -> str:
        values_sorted = sorted(values)
        if not values_sorted:
            return "NA"
        if len(values_sorted) == 1:
            return f"{values_sorted[0]:.2f}\\%"
        return f"{values_sorted[0]:.2f}--{values_sorted[-1]:.2f}\\%"

    deepseek_vals = [shift_pct(c) for _, c in sorted(per_model_temp.get("DeepSeek\\textendash R1", {}).items())]
    gpt4o_vals = [shift_pct(c) for _, c in sorted(per_model_temp.get("GPT\\textendash 4o", {}).items())]

    sentence = (
        "To test whether this pattern is specific to small GRPO-tuned models, we evaluate "
        "DeepSeek\\textendash R1 and GPT\\textendash 4o under matched decoding conditions on "
        "\\textsc{MATH\\textendash 500}. As shown in Table~\\ref{tab:external-models}, both models "
        "exhibit \\emph{very low} canonical shift rates across temperatures "
        f"({fmt_range(deepseek_vals)} for DeepSeek\\textendash R1 and {fmt_range(gpt4o_vals)} for "
        "GPT\\textendash 4o), and accuracy conditioned on a shift shows no systematic benefit, suggesting "
        "that the phenomenon generalizes across model families and training paradigms."
    )

    # Build LaTeX table (one row per model x temp).
    lines: List[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\footnotesize")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}} l r r r @{}}")
    lines.append("\\toprule")
    lines.append("{\\textbf{Model}} & {\\textbf{Temp}} & {\\(\\%S\\)} & "
                 "{\\scriptsize \\(P(\\checkmark\\mid S{=}0)\\)} & "
                 "{\\scriptsize \\(P(\\checkmark\\mid S{=}1)\\)} \\\\")
    lines.append("\\midrule")
    for model_name in sorted(per_model_temp):
        temps = sorted(per_model_temp[model_name])
        first = True
        for t in temps:
            c = per_model_temp[model_name][t]
            row_model = model_name if first else ""
            first = False
            lines.append(
                f"{row_model} & {t:.2g} & {shift_pct(c):.2f} & {acc(c,'noshift')/100.0:.3f} & {acc(c,'shift')/100.0:.3f} \\\\"
            )
        lines.append("\\midrule")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular*}")
    lines.append("\\caption{\\textbf{External models on \\textsc{MATH\\textendash 500}.} "
                 "Canonical shift rates and conditional accuracy by decoding temperature.}")
    lines.append(f"\\label{{{args.label}}}")
    lines.append("\\end{table}")

    tex = "\n".join(lines) + "\n"
    out_path = Path(args.out_tex)
    _ensure_parent(out_path)
    out_path.write_text(tex, encoding="utf-8")
    print(tex)
    print(sentence + "\n")

    out_sent = Path(args.out_sentence_tex)
    _ensure_parent(out_sent)
    out_sent.write_text(sentence + "\n", encoding="utf-8")
    print(f"[ok] wrote: {out_path}")
    print(f"[ok] wrote: {out_sent}")
    return 0


def _read_text_or_die(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Missing required file: {p} (run `python tools/master_analysis.py all` first)")
    return p.read_text(encoding="utf-8").strip()


def _compute_formal_aha_pooled_pct_qwen15b(
    *,
    delta1: float = 0.125,
    delta2: float = 0.125,
) -> float:
    """
    Compute the pooled formal-Aha prevalence across Qwen2.5--1.5B temps for
    the "overall_all" scope in the saved heatmap CSVs.
    """
    paths = [
        "paper_figs/aha_heatmaps/Q1p5B_T0/aha_heatmap__Crossword+Math+RushHour__Qwen2.5-1.5B_T0.csv",
        "paper_figs/aha_heatmaps/Q1p5B_T0.05/aha_heatmap__Crossword+Math+RushHour__Qwen2.5-1.5B_T0.05.csv",
        "paper_figs/aha_heatmaps/Q1p5B_T0.3/aha_heatmap__Crossword+Math+RushHour__Qwen2.5-1.5B_T0.3.csv",
        "paper_figs/aha_heatmaps/Q1p5B_T0.7/aha_heatmap__Crossword+Math+RushHour__Qwen2.5-1.5B_T0.7.csv",
    ]
    total_events = 0
    total_pairs = 0
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            matched = False
            for row in reader:
                if row.get("scope") != "overall_all":
                    continue
                if abs(float(row["delta1"]) - float(delta1)) > 1e-9:
                    continue
                if abs(float(row["delta2"]) - float(delta2)) > 1e-9:
                    continue
                total_events += int(row["n_events"])
                total_pairs += int(row["n_pairs"])
                matched = True
                break
        if not matched:
            raise SystemExit(f"Could not find overall_all row for delta1={delta1}, delta2={delta2} in {path}")
    return (100.0 * total_events / total_pairs) if total_pairs else 0.0


def cmd_rq1_section(argv: Optional[List[str]] = None) -> int:
    """
    Emit the full RQ1 section LaTeX snippet with up-to-date numbers.
    """
    parser = argparse.ArgumentParser(
        prog="master_analysis.py rq1-section",
        description="Write the RQ1 section LaTeX (with computed numbers and \\input tables).",
    )
    parser.add_argument(
        "--out_tex",
        default="latex/section_rq1.tex",
        help="Write the LaTeX section to this path (default: latex/section_rq1.tex).",
    )
    parser.add_argument(
        "--table2_path",
        default="latex/table2_shift_accuracy.tex",
        help="Path to Table 2 LaTeX (default: latex/table2_shift_accuracy.tex).",
    )
    parser.add_argument(
        "--external_table_path",
        default="latex/table_external_models.tex",
        help="Path to external-models table LaTeX (default: latex/table_external_models.tex).",
    )
    parser.add_argument(
        "--qwen15b_sentence_path",
        default="latex/table2_qwen15b_summary_sentence.tex",
        help="Path to the Qwen2.5--1.5B pooled summary sentence snippet.",
    )
    parser.add_argument(
        "--pooled_logit_sentence_path",
        default="latex/table2_pooled_logit_sentence.tex",
        help="Path to the pooled-logit sentence snippet.",
    )
    parser.add_argument(
        "--external_sentence_path",
        default="latex/external_models_sentence.tex",
        help="Path to the external-models sentence snippet.",
    )
    parser.add_argument(
        "--delta1",
        type=float,
        default=0.125,
        help="Formal Aha delta1 for the pooled rarity line (default: 1/8).",
    )
    parser.add_argument(
        "--delta2",
        type=float,
        default=0.125,
        help="Formal Aha delta2 for the pooled rarity line (default: 1/8).",
    )
    args = parser.parse_args(argv)

    qwen_sentence = _read_text_or_die(str(args.qwen15b_sentence_path))
    logit_sentence = _read_text_or_die(str(args.pooled_logit_sentence_path))
    external_sentence = _read_text_or_die(str(args.external_sentence_path))

    table2_path = Path(args.table2_path)
    external_table_path = Path(args.external_table_path)
    if not table2_path.exists():
        raise SystemExit(f"Missing required file: {table2_path} (run `python tools/master_analysis.py table2`)")
    if not external_table_path.exists():
        raise SystemExit(f"Missing required file: {external_table_path} (run `python tools/master_analysis.py external-models`)")

    formal_pct = _compute_formal_aha_pooled_pct_qwen15b(delta1=float(args.delta1), delta2=float(args.delta2))

    section = (
        "\\subsection{RQ1: Reasoning Shifts \\& Model Accuracy}\n"
        "\\label{sec:results-rq1}\n"
        "\\noindent \\textbf{Do reasoning shifts improve accuracy?}\n"
        "Before analyzing formal ``Aha!'' moments, we first consider the broader class of "
        "\\emph{reasoning shifts}---any mid-trace pivot detected by our annotator, irrespective of whether it "
        "satisfies the stricter criteria in Def.~\\ref{def:aha-moment-lrms}. If such shifts reflected genuine "
        "insight, traces containing them should be \\emph{more} accurate than those without them.\n\n"
        f"{qwen_sentence} {logit_sentence}\n\n"
        f"{external_sentence} These results characterize the ``raw'' behavioral signature of mid-trace shifts, "
        "independent of any stricter ``Aha!'' interpretation.\n\n"
        f"\\input{{{table2_path.as_posix()}}}\n\n"
        "\\vspace{1mm}\n"
        "\\noindent \\textbf{How frequent are formal ``Aha!'' moments?}\n"
        "We now restrict attention to the much smaller subset of events that satisfy \\emph{all three} criteria "
        "in Def.~\\ref{subsec:aha-moment-def}. In Fig.~\\ref{fig:aha-heatmap-overall}, by varying "
        "$\\delta_1,\\delta_2\\in\\{0,\\,1/8,\\,2/8\\}$ and fixing $\\delta_3=\\epsilon>0$, we find that formal "
        "``Aha!'' moments are extremely rare, even with relatively lax constraints. Similar patterns hold for "
        "Qwen2.5--7B and Llama3.1--8B (App.~\\ref{sec:app-additional-results}). "
        f"Pooling every Crossword/Math/RHour checkpoint and temperature, the formal detector fires on only "
        f"${formal_pct:.2f}\\%$ of samples.\n"
    )

    out_path = Path(args.out_tex)
    _ensure_parent(out_path)
    out_path.write_text(section + "\n", encoding="utf-8")
    print(section)
    print(f"[ok] wrote: {out_path}")
    return 0


def _std_col(values: List[float]) -> List[float]:
    if not values:
        return []
    mean = sum(values) / float(len(values))
    var = sum((x - mean) ** 2 for x in values) / float(len(values))
    std = (var**0.5) if var > 0 else 1.0
    return [(x - mean) / std for x in values]


def _fmt_int_tex(value: int) -> str:
    return f"{int(value):,}".replace(",", "{,}")


def _fmt_shift_pct(value: float) -> str:
    return f"{float(value):.3f}"


def _fmt_prob(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _fmt_delta_pp(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{float(value):.2f}"


def _fmt_ame(value: float) -> str:
    return f"{float(value):.4f}"


def _fmt_p_tex(p_value: float, z_value: Optional[float] = None) -> str:
    """
    Format a p-value for LaTeX, using scientific notation for small values.
    """
    try:
        p = float(p_value)
    except (TypeError, ValueError):
        p = float("nan")

    if p != p:  # NaN
        return "NaN"

    if p == 0.0 and z_value is not None:
        log10_p = _two_sided_log10_p_from_z(float(z_value))
        if math.isfinite(log10_p):
            exp = int(math.floor(log10_p))
            mant = 10 ** (log10_p - exp)
            return f"{mant:.3g}\\times10^{{{exp}}}"
        return "0"

    if p >= 1e-3:
        # Keep as decimal (matches examples like 0.69, 0.0125).
        return f"{p:.4g}"

    exp = int(math.floor(math.log10(p)))
    mant = p / (10 ** exp)
    return f"{mant:.3g}\\times10^{{{exp}}}"


def _ame_for_shift(
    res: Any,
    model: Any,
    df: Any,
    *,
    shift_col: str = "shift",
) -> float:
    """
    Average marginal effect for a binary shift indicator via counterfactual prediction.
    """
    df0 = df.copy()
    df1 = df.copy()
    df0[shift_col] = 0
    df1[shift_col] = 1
    p0 = predict_formula(res, model, df0)
    p1 = predict_formula(res, model, df1)
    return float((p1 - p0).mean())


def _try_import_statsmodels_silently() -> Tuple[Optional[Any], Optional[Any]]:
    """
    Try to import statsmodels, suppressing noisy binary-incompatibility tracebacks.
    """
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
        try:
            sm, smf = lazy_import_statsmodels()
            return sm, smf
        except Exception:
            return None, None


def _sigmoid(x: "Any") -> "Any":
    import numpy as np

    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def _fit_fe_logit_irls_multi(
    y: "Any",
    X: "Any",
    pid: "Any",
    *,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> Tuple["Any", "Any", "Any", "Any"]:
    """
    Fixed-effects logistic regression with per-problem intercepts via block IRLS.

    Returns:
      beta (k,), se (k,), z (k,), alpha (n_groups,)
    """
    import numpy as np

    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    pid = np.asarray(pid, dtype=int)

    n = int(y.size)
    k = int(X.shape[1]) if n else 0
    p = int(pid.max() + 1) if n else 0
    if n == 0 or p == 0 or k == 0:
        return np.zeros(k), np.full(k, float("nan")), np.full(k, float("nan")), np.zeros(p)

    alpha = np.zeros(p, dtype=float)
    beta = np.zeros(k, dtype=float)

    eps = 1e-12
    for _ in range(max_iter):
        eta = alpha[pid] + X @ beta
        mu = _sigmoid(eta)
        w = mu * (1.0 - mu) + eps
        resid = y - mu

        sum_w = np.bincount(pid, weights=w, minlength=p)
        denom = sum_w + eps
        g_alpha = np.bincount(pid, weights=resid, minlength=p)

        sum_wx = np.zeros((k, p), dtype=float)
        for j in range(k):
            sum_wx[j] = np.bincount(pid, weights=w * X[:, j], minlength=p)

        g_beta = X.T @ resid
        X_w = X * w[:, None]
        C = X_w.T @ X

        B = np.zeros((k, k), dtype=float)
        rhs_adj = np.zeros(k, dtype=float)
        for g in range(p):
            if denom[g] <= 0:
                continue
            v = sum_wx[:, g]
            B += np.outer(v, v) / denom[g]
            rhs_adj += v * (g_alpha[g] / denom[g])

        M = C - B
        rhs = g_beta - rhs_adj
        try:
            delta_beta = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            break

        delta_alpha = (g_alpha - sum_wx.T @ delta_beta) / denom

        beta += delta_beta
        alpha += delta_alpha

        if float(np.max(np.abs(delta_beta))) < tol and float(np.max(np.abs(delta_alpha))) < tol:
            break

    # Final scores / covariance for betas (cluster-robust by problem).
    eta = alpha[pid] + X @ beta
    mu = _sigmoid(eta)
    w = mu * (1.0 - mu) + eps
    resid = y - mu

    sum_w = np.bincount(pid, weights=w, minlength=p)
    denom = sum_w + eps
    sum_wx = np.zeros((k, p), dtype=float)
    for j in range(k):
        sum_wx[j] = np.bincount(pid, weights=w * X[:, j], minlength=p)

    X_w = X * w[:, None]
    C = X_w.T @ X
    B = np.zeros((k, k), dtype=float)
    for g in range(p):
        if denom[g] <= 0:
            continue
        v = sum_wx[:, g]
        B += np.outer(v, v) / denom[g]
    M = C - B
    try:
        invM = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return beta, np.full(k, float("nan")), np.full(k, float("nan")), alpha

    s = np.zeros((k, p), dtype=float)
    for j in range(k):
        s[j] = np.bincount(pid, weights=X[:, j] * resid, minlength=p)
    S = s @ s.T
    V = invM @ S @ invM

    g = int(np.unique(pid).size)
    if g > 1:
        V *= float(g) / float(g - 1)

    se = np.sqrt(np.clip(np.diag(V), 0.0, None))
    z = np.divide(beta, se, out=np.full_like(beta, float("nan")), where=se > 0)
    return beta, se, z, alpha

def _load_rows_for_root(
    root: str,
    *,
    split: str,
    min_step: int,
    max_step: int,
    temp_value: Optional[float] = None,
    pass_key: str = "pass1",
    label_key: str = "shift_in_reasoning_v1",
) -> "Any":
    """
    Load a sample-level DataFrame with (correct, shift, problem, step, temp).
    """
    import pandas as pd

    rows: List[Tuple[int, int, str, int, float]] = []
    files = scan_files_step_only(str(root), split_substr=str(split or ""))
    for file_path in files:
        step_hint = nat_step_from_path(file_path)
        if step_hint is not None and (step_hint < min_step or step_hint > max_step):
            continue
        for record in iter_records_from_file(file_path):
            if split and str(record.get("split", "")).lower() != split.lower():
                continue
            step = int(step_from_rec_or_path(record, file_path))
            if step < min_step or step > max_step:
                continue
            pass_obj = record.get(pass_key) or {}
            if not isinstance(pass_obj, dict):
                continue
            correct_flag = extract_correct(pass_obj, record)
            if correct_flag is None:
                continue
            shift_flag = coerce_bool(pass_obj.get(label_key, record.get(label_key)))
            if shift_flag is None:
                continue
            problem = problem_key_from_record(record, "unknown")
            temp = float(temp_value) if temp_value is not None else float("nan")
            rows.append((int(correct_flag), int(shift_flag), str(problem), int(step), temp))

    df = pd.DataFrame(rows, columns=["correct", "shift", "problem", "step", "temp"])
    return df


def _compute_rq2_domain_metrics(
    *,
    df: "Any",
    formula: str,
    std_src_col: str,
    std_dest_col: str,
    cluster_by: str = "problem",
    use_statsmodels: bool = False,
) -> Dict[str, Any]:
    """
    Compute raw metrics + GLM AME/p for a domain.
    """
    import numpy as np

    if df.empty:
        return {
            "N": 0,
            "shift_pct": float("nan"),
            "p_shift": float("nan"),
            "delta_pp": float("nan"),
            "ame": float("nan"),
            "p": float("nan"),
            "z": float("nan"),
        }

    # Standardize the requested covariate.
    values = df[std_src_col].astype(float).tolist()
    df[std_dest_col] = _std_col(values)

    N = int(len(df))
    n_shift = int((df["shift"] == 1).sum())
    n_noshift = N - n_shift
    p_shift = float(df.loc[df["shift"] == 1, "correct"].mean()) if n_shift else 0.0
    p_noshift = float(df.loc[df["shift"] == 0, "correct"].mean()) if n_noshift else 0.0
    delta_pp = (p_shift - p_noshift) * 100.0
    shift_pct = (100.0 * n_shift / N) if N else float("nan")

    sm = smf = None
    z_val = float("nan")
    if use_statsmodels:
        sm, smf = _try_import_statsmodels_silently()
    try:
        if not use_statsmodels or sm is None or smf is None:
            raise RuntimeError("statsmodels unavailable or disabled")
        model = smf.glm(formula, data=df, family=sm.families.Binomial())
        res, _cov_type, _cov_kwds = glm_fit_with_covariance(model, df, cluster_by=cluster_by)
        ame = _ame_for_shift(res, model, df, shift_col="shift")
        coef = float(res.params.get("shift", np.nan))
        se = float(res.bse.get("shift", np.nan))
        z_val = float(coef / se) if se and np.isfinite(se) else float("nan")
        p_val = float(res.pvalues.get("shift", float("nan")))
    except Exception:
        # Fallback: fixed-effects logistic regression with block-IRLS (no SciPy).
        import numpy as np

        pid = np.asarray(df["problem"].astype("category").cat.codes, dtype=int)
        x0 = np.asarray(df[std_dest_col], dtype=float)
        x1 = np.asarray(df["shift"], dtype=float)
        y = np.asarray(df["correct"], dtype=float)
        X = np.column_stack([x0, x1])
        beta, se, z, alpha = _fit_fe_logit_irls_multi(y, X, pid)
        b1 = float(beta[1]) if beta.size > 1 else float("nan")
        z_val = float(z[1]) if z.size > 1 else float("nan")

        eta = alpha[pid] + X @ beta
        eta0 = eta - b1 * x1
        eta1 = eta0 + b1
        ame = float((_sigmoid(eta1) - _sigmoid(eta0)).mean())

        log10_p = _two_sided_log10_p_from_z(z_val) if z_val == z_val else float("nan")
        p_val = float(10 ** log10_p) if log10_p > -300 else 0.0

    return {
        "N": N,
        "shift_pct": float(shift_pct),
        "p_shift": float(p_shift),
        "delta_pp": float(delta_pp),
        "ame": float(ame),
        "p": float(p_val),
        "z": float(z_val),
    }


def cmd_rq2_section(argv: Optional[List[str]] = None) -> int:
    """
    Emit the full RQ2 section LaTeX snippet + the Table~\\ref{tab:rs}.
    """
    parser = argparse.ArgumentParser(
        prog="master_analysis.py rq2-section",
        description="Write the RQ2 section LaTeX (with computed Table~\\ref{tab:rs}).",
    )
    parser.add_argument(
        "--out_section_tex",
        default="latex/section_rq2.tex",
        help="Write the RQ2 section to this path (default: latex/section_rq2.tex).",
    )
    parser.add_argument(
        "--out_table_tex",
        default="latex/table_rq2_rs.tex",
        help="Write the RQ2 table (tab:rs) to this path (default: latex/table_rq2_rs.tex).",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Split filter (default: test).",
    )
    parser.add_argument(
        "--min_step",
        type=int,
        default=0,
        help="Minimum checkpoint step (default: 0).",
    )
    parser.add_argument(
        "--max_step",
        type=int,
        default=950,
        help="Maximum checkpoint step (default: 950).",
    )
    parser.add_argument(
        "--stage_temp",
        type=float,
        default=0.7,
        help="Training-stage analysis temperature (default: 0.7).",
    )
    parser.add_argument(
        "--use_statsmodels",
        action="store_true",
        help="Use statsmodels GLM (higher memory). Default: use lightweight IRLS.",
    )
    args = parser.parse_args(argv)

    # Ensure the figure PDFs exist at the paths used by the LaTeX snippet.
    import shutil

    fig_a_src = Path("paper_figs/raw_effects/raw_effect_per_step_crossword_math_linear.pdf")
    fig_a_dst = Path("latex/raw_effect_per_step_crossword_math_linear.pdf")
    if fig_a_src.exists() and not fig_a_dst.exists():
        _ensure_parent(fig_a_dst)
        shutil.copyfile(fig_a_src, fig_a_dst)

    fig_b_src = Path(
        "artifacts/plots/temperature_raw_effects/raw_effects_plot__Crossword+Math+RushHour__Qwen2.5-1.5B.pdf"
    )
    fig_b_dst = Path("latex/raw_effects_plot__Crossword+Math+RushHour__Qwen2.5-1.5B.pdf")
    if fig_b_src.exists() and not fig_b_dst.exists():
        _ensure_parent(fig_b_dst)
        shutil.copyfile(fig_b_src, fig_b_dst)

    split = str(args.split or "")
    min_step = int(args.min_step)
    max_step = int(args.max_step)

    # (a) Training stage: fixed T=0.7, include step_std.
    roots_stage = {
        "Xword": "artifacts/results/GRPO-1.5B-xword-temp-0.7",
        "Math": "artifacts/results/GRPO-1.5B-math-temp-0.7",
        "RHour": "artifacts/results/GRPO-1.5B-carpark-temp-0.7",
    }
    stage_metrics: Dict[str, Dict[str, Any]] = {}
    stage_dfs: List["Any"] = []
    for dom, root in roots_stage.items():
        df = _load_rows_for_root(root, split=split, min_step=min_step, max_step=max_step, temp_value=float(args.stage_temp))
        stage_dfs.append(df)
        stage_metrics[dom] = _compute_rq2_domain_metrics(
            df=df,
            formula="correct ~ C(problem) + step_std + shift",
            std_src_col="step",
            std_dest_col="step_std",
            cluster_by="problem",
            use_statsmodels=bool(args.use_statsmodels),
        )
    import pandas as pd
    combined_stage = pd.concat(stage_dfs, ignore_index=True) if stage_dfs else pd.DataFrame()
    stage_metrics["Combined"] = _compute_rq2_domain_metrics(
        df=combined_stage,
        formula="correct ~ C(problem) + step_std + shift",
        std_src_col="step",
        std_dest_col="step_std",
        cluster_by="problem",
        use_statsmodels=bool(args.use_statsmodels),
    )

    # (b) Temperature: pool over steps, include temp_std.
    roots_by_temp = {
        0.0: {
            "Xword": "artifacts/results/GRPO-1.5B-xword-temp-0",
            "Math": "artifacts/results/GRPO-1.5B-math-temp-0.0",
            "RHour": "artifacts/results/GRPO-1.5B-carpark-temp-0",
        },
        0.05: {
            "Xword": "artifacts/results/GRPO-1.5B-xword-temp-0.05",
            "Math": "artifacts/results/GRPO-1.5B-math-temp-0.05",
            "RHour": "artifacts/results/GRPO-1.5B-carpark-temp-0.05",
        },
        0.3: {
            "Xword": "artifacts/results/GRPO-1.5B-xword-temp-0.3",
            "Math": "artifacts/results/GRPO-1.5B-math-temp-0.3",
            "RHour": "artifacts/results/GRPO-1.5B-carpark-temp-0.3",
        },
        0.7: {
            "Xword": "artifacts/results/GRPO-1.5B-xword-temp-0.7",
            "Math": "artifacts/results/GRPO-1.5B-math-temp-0.7",
            "RHour": "artifacts/results/GRPO-1.5B-carpark-temp-0.7",
        },
    }
    temp_metrics: Dict[str, Dict[str, Any]] = {}
    temp_dfs: List["Any"] = []
    for dom in ("Xword", "Math", "RHour"):
        dfs = []
        for t, dom_roots in roots_by_temp.items():
            root = dom_roots[dom]
            dfs.append(_load_rows_for_root(root, split=split, min_step=min_step, max_step=max_step, temp_value=float(t)))
        df_all = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        temp_dfs.append(df_all)
        temp_metrics[dom] = _compute_rq2_domain_metrics(
            df=df_all,
            formula="correct ~ C(problem) + temp_std + shift",
            std_src_col="temp",
            std_dest_col="temp_std",
            cluster_by="problem",
            use_statsmodels=bool(args.use_statsmodels),
        )
    combined_temp = pd.concat(temp_dfs, ignore_index=True) if temp_dfs else pd.DataFrame()
    temp_metrics["Combined"] = _compute_rq2_domain_metrics(
        df=combined_temp,
        formula="correct ~ C(problem) + temp_std + shift",
        std_src_col="temp",
        std_dest_col="temp_std",
        cluster_by="problem",
        use_statsmodels=bool(args.use_statsmodels),
    )

    # Write Table~\\ref{tab:rs}.
    table_lines: List[str] = []
    table_lines.append("\\begin{table}[t]")
    table_lines.append("\\centering")
    table_lines.append("\\small")
    table_lines.append("\\begin{tabular}{lrrrr}")
    table_lines.append("\\toprule")
    table_lines.append("\\multicolumn{5}{c}{\\textbf{(a) Training stage}} \\\\")
    table_lines.append("\\midrule")
    table_lines.append("\\textbf{Metric} & \\textbf{Xword} & \\textbf{Math} & \\textbf{RHour} & \\textbf{Combined} \\\\")
    table_lines.append("\\midrule")
    table_lines.append(
        f"$N$                   & {_fmt_int_tex(stage_metrics['Xword']['N'])} & "
        f"{_fmt_int_tex(stage_metrics['Math']['N'])} & {_fmt_int_tex(stage_metrics['RHour']['N'])} & "
        f"{_fmt_int_tex(stage_metrics['Combined']['N'])} \\\\"
    )
    table_lines.append(
        f"$\\%S$                 & {_fmt_shift_pct(stage_metrics['Xword']['shift_pct'])}    & "
        f"{_fmt_shift_pct(stage_metrics['Math']['shift_pct'])}    & {_fmt_shift_pct(stage_metrics['RHour']['shift_pct'])}  & "
        f"{_fmt_shift_pct(stage_metrics['Combined']['shift_pct'])}  \\\\"
    )
    table_lines.append(
        f"$\\hat{{p}}_{{Y\\mid S=1}}$ & {_fmt_prob(stage_metrics['Xword']['p_shift'], 4)}   & "
        f"{_fmt_prob(stage_metrics['Math']['p_shift'], 4)}   & {_fmt_prob(stage_metrics['RHour']['p_shift'], 4)}  & "
        f"{_fmt_prob(stage_metrics['Combined']['p_shift'], 4)}  \\\\"
    )
    table_lines.append(
        f"$\\Delta\\mathrm{{pp}}$   & ${_fmt_delta_pp(stage_metrics['Xword']['delta_pp'])}$  & "
        f"${_fmt_delta_pp(stage_metrics['Math']['delta_pp'])}$ & ${_fmt_delta_pp(stage_metrics['RHour']['delta_pp'])}$  & "
        f"${_fmt_delta_pp(stage_metrics['Combined']['delta_pp'])}$  \\\\"
    )
    table_lines.append(
        f"$\\mathrm{{AME}}$        & ${_fmt_ame(stage_metrics['Xword']['ame'])}$& "
        f"${_fmt_ame(stage_metrics['Math']['ame'])}$& ${_fmt_ame(stage_metrics['RHour']['ame'])}$ & "
        f"${_fmt_ame(stage_metrics['Combined']['ame'])}$\\\\"
    )
    table_lines.append(
        f"$p$                   & ${_fmt_p_tex(stage_metrics['Xword']['p'], stage_metrics['Xword']['z'])}$ & "
        f"${_fmt_p_tex(stage_metrics['Math']['p'], stage_metrics['Math']['z'])}$ & "
        f"${_fmt_p_tex(stage_metrics['RHour']['p'], stage_metrics['RHour']['z'])}$ & "
        f"${_fmt_p_tex(stage_metrics['Combined']['p'], stage_metrics['Combined']['z'])}$ \\\\"
    )
    table_lines.append("\\midrule")
    table_lines.append("\\multicolumn{5}{c}{ \\textbf{(b) Temperature}} \\\\")
    table_lines.append("\\textbf{Metric} & \\textbf{Xword} & \\textbf{Math} & \\textbf{RHour} & \\textbf{Combined} \\\\")
    table_lines.append("\\midrule")
    table_lines.append(
        f"$N$                       & {_fmt_int_tex(temp_metrics['Xword']['N'])}  & "
        f"{_fmt_int_tex(temp_metrics['Math']['N'])} & {_fmt_int_tex(temp_metrics['RHour']['N'])} & "
        f"{_fmt_int_tex(temp_metrics['Combined']['N'])} \\\\"
    )
    table_lines.append(
        f"\\(\\%S\\)                   & {_fmt_shift_pct(temp_metrics['Xword']['shift_pct'])}      & "
        f"{_fmt_shift_pct(temp_metrics['Math']['shift_pct'])}      & {_fmt_shift_pct(temp_metrics['RHour']['shift_pct'])}     & "
        f"{_fmt_shift_pct(temp_metrics['Combined']['shift_pct'])}     \\\\"
    )
    table_lines.append(
        f"$\\hat{{p}}_{{Y\\mid S=1}}$     & {_fmt_prob(temp_metrics['Xword']['p_shift'], 4)}    & "
        f"{_fmt_prob(temp_metrics['Math']['p_shift'], 4)}    & {_fmt_prob(temp_metrics['RHour']['p_shift'], 4)}    & "
        f"{_fmt_prob(temp_metrics['Combined']['p_shift'], 4)}    \\\\"
    )
    table_lines.append(
        f"$\\Delta\\mathrm{{pp}}$       & ${_fmt_delta_pp(temp_metrics['Xword']['delta_pp'])}$  & "
        f"${_fmt_delta_pp(temp_metrics['Math']['delta_pp'])}$  & ${_fmt_delta_pp(temp_metrics['RHour']['delta_pp'])}$   & "
        f"${_fmt_delta_pp(temp_metrics['Combined']['delta_pp'])}$   \\\\"
    )
    table_lines.append(
        f"$\\mathrm{{AME}}$            & ${_fmt_ame(temp_metrics['Xword']['ame'])}$    & "
        f"${_fmt_ame(temp_metrics['Math']['ame'])}$ & ${_fmt_ame(temp_metrics['RHour']['ame'])}$ & "
        f"${_fmt_ame(temp_metrics['Combined']['ame'])}$ \\\\"
    )
    table_lines.append(
        f"$p$                       & ${_fmt_p_tex(temp_metrics['Xword']['p'], temp_metrics['Xword']['z'])}$ & "
        f"${_fmt_p_tex(temp_metrics['Math']['p'], temp_metrics['Math']['z'])}$ & "
        f"${_fmt_p_tex(temp_metrics['RHour']['p'], temp_metrics['RHour']['z'])}$ & "
        f"${_fmt_p_tex(temp_metrics['Combined']['p'], temp_metrics['Combined']['z'])}$ \\\\"
    )
    table_lines.append("\\bottomrule")
    table_lines.append("\\end{tabular}")
    table_lines.append(
        "\\caption{\\textbf{Effect of detected reasoning shifts on accuracy (Qwen2.5-1.5B).}\n"
        "For each domain, \\(\\%S\\) is the share of samples where the LLM-as-judge detects a shift "
        "(\\(S_{i,j}=1\\)); \\(\\hat{p}_{Y\\mid S=1}\\) is the empirical accuracy among shifted traces; and "
        "\\(\\Delta\\mathrm{pp}\\) is the raw accuracy difference (in percentage points) between shifted and non-shifted traces.\n"
        "\\emph{(a)} controls for training step (standardized) at fixed training decoding temperature "
        "$T=0.7$; \\emph{(b)} controls for decoding temperature $T\\in\\{0.0,0.05,0.3,0.7\\}$ while aggregating over steps.\n"
        "\\(\\mathrm{AME}\\) is the average marginal effect of a shift from a logistic regression with problem fixed effects "
        "and cluster-robust SEs; negative values mean that, holding problem and step/temperature fixed, traces with shifts are "
        "less likely to be correct.\n"
        "See \\S\\ref{sec:results-rq2} for the full regression specification.}"
    )
    table_lines.append("\\label{tab:rs}")
    table_lines.append("\\vspace{-5mm}")
    table_lines.append("\\end{table}")
    table_tex = "\n".join(table_lines) + "\n"

    table_path = Path(args.out_table_tex)
    _ensure_parent(table_path)
    table_path.write_text(table_tex, encoding="utf-8")
    print(table_tex)

    # Narrative numbers for the section.
    s_x = stage_metrics["Xword"]
    s_m = stage_metrics["Math"]
    s_r = stage_metrics["RHour"]
    t_x = temp_metrics["Xword"]
    t_m = temp_metrics["Math"]
    t_r = temp_metrics["RHour"]

    section = (
        "\\subsection{RQ2: Training Stage \\& Temperature}\n"
        "\\label{sec:results-rq2}\n\n"
        "RQ1 establishes two constraints on ``insight-like'' behavior: broad reasoning shifts are uncommon and tend to "
        "coincide with worse outcomes, while \\emph{formal} ``Aha!'' events are so rare that they contribute little to "
        "overall model performance.\n"
        "This raises a natural question: are we simply averaging over regimes where shifts sometimes help and sometimes hurt?\n"
        "We test two plausible sources of heterogeneity:\n"
        "(i) shifts might become more (or less) effective at different \\emph{stages} of training; and\n"
        "(ii) their impact might depend on the \\emph{decoding temperature} (and thus sampling entropy).\n\n"
        "\\begin{figure}[t]\n"
        "\\centering\n\n"
        "\\begin{subfigure}[t]{\\linewidth}\n"
        "  \\centering\n"
        "  \\includegraphics[width=\\linewidth]{latex/raw_effect_per_step_crossword_math_linear.pdf}\n"
        "  \\caption{Raw effect of reasoning shifts over training for Qwen2.5-1.5B finetuning across domains (same evaluation at every step).}\n"
        "  \\label{fig:raw-effect-overlay:a}\n"
        "\\end{subfigure}\n\n"
        "\\vspace{4pt}\n\n"
        "\\begin{subfigure}[t]{\\linewidth}\n"
        "  \\centering\n"
        "  \\includegraphics[width=\\linewidth]{latex/raw_effects_plot__Crossword+Math+RushHour__Qwen2.5-1.5B.pdf}\n"
        "  \\caption{Raw effect of reasoning shifts over Qwen2.5-1.5B finetuning across domains (same evaluation at every temperature).}\n"
        "  \\label{fig:temp-raw-effect}\n"
        "\\end{subfigure}\n\n"
        "\\caption{\\textbf{Reasoning shifts across training and temperature (Qwen2.5-1.5B).}\n"
        "We plot the raw accuracy gap $\\widehat{\\Delta}=\\widehat{p}_{Y\\mid S=1}-\\widehat{p}_{Y\\mid S=0}$ (pp).\n"
        f"(a)~At fixed $T={float(args.stage_temp):.1f}$, $\\widehat{{\\Delta}}$ stays near zero or negative across training.\n"
        "(b)~Across $T$, shifts align with correction on \\emph{Xword} at lower $T$, remain harmful on \\emph{Math}, and are near-zero on \\emph{RHour}.}\n"
        "\\label{fig:raw-effect-overlay}\n"
        "\\end{figure}\n\n"
        "\\vspace{2mm}\n"
        "\\noindent\n"
        "\\textbf{How does the effect of reasoning shifts vary across training?}\n"
        "To test whether the shift--accuracy relationship changes as training progresses,\n"
        "we regress correctness on problem fixed effects, standardized training step, and the shift indicator.\n"
        "We report average marginal effects (AME) with cluster--robust SEs at the problem level.%\n"
        "\\footnote{In R-style notation:\n"
        "\\(\n"
        "\\texttt{correct} \\sim \\texttt{C(problem)} + \\texttt{step\\_std} + \\texttt{shift}.\n"
        "\\)\n"
        "\\texttt{correct} is a binary outcome;\n"
        "\\texttt{C(problem)} are problem fixed effects;\n"
        "\\texttt{step\\_std} is the standardized checkpoint index.}\n\n"
        "At $T{=}0.7$, we find no evidence that shifts become beneficial later in training.\n"
        f"In \\emph{{Xwords}} and \\emph{{Math}} shifts are uncommon ($\\%S={s_x['shift_pct']:.3f}$; "
        f"$\\%S={s_m['shift_pct']:.3f}$) and are mildly harmful "
        f"($\\mathrm{{AME}}={s_x['ame']:.4f}$, $p={_fmt_p_tex(s_x['p'], s_x['z'])}$; "
        f"$\\mathrm{{AME}}={s_m['ame']:.4f}$, $p={_fmt_p_tex(s_m['p'], s_m['z'])}$).\n\n"
        f"In \\emph{{RHour}}, shifts are comparatively frequent ($\\%S={s_r['shift_pct']:.3f}$) but have no measurable effect on accuracy "
        f"($\\mathrm{{AME}}\\approx{s_r['ame']:.4f}$, $p={_fmt_p_tex(s_r['p'], s_r['z'])}$).\n"
        "Analogous results for $T\\in\\{0.0,0.05,0.3\\}$ are reported in Appendix~\\ref{sec:app-rs-temp}.\n"
        "Figure~\\ref{fig:raw-effect-overlay}a echoes this pattern: across checkpoints, shifted traces are not systematically more accurate than non-shifted ones.\n"
        "We repeat robustness checks using alternative detector variants across $T\\in\\{0,0.05,0.3,0.7\\}$ in App.~\\ref{spp:ablations}.\n"
        "We observe the same qualitative pattern with the stricter \\emph{formal} ``Aha!'' detector (Appendix~\\ref{sec:app-formal-aha-temp}), "
        "but because it fires on only $\\approx 10^{-3}$ of traces at $T{=}0.7$, estimates are underpowered for fine-grained stage-by-stage heterogeneity; "
        "critically, we do not see a consistent late-training transition to positive effects.\n\n"
        "\\vspace{1mm}\n"
        "\\noindent\n"
        "\\textbf{How does the effect of reasoning shifts vary with decoding temperature?}\n"
        "We next ask whether temperature modulates the relationship between shifts and correctness.\n"
        "We regress correctness on problem fixed effects, standardized temperature, and the shift indicator, aggregating across training steps.%\n"
        "\\footnote{R-style notation:\n"
        "\\(\n"
        "\\texttt{correct} \\sim \\texttt{C(problem)} + \\texttt{temp\\_std} + \\texttt{shift}.\n"
        "\\)\n"
        "\\texttt{temp\\_std} is the standardized decoding temperature.}\n\n"
        "Table~\\ref{tab:rs} summarizes the average association between shifts and correctness while controlling for standardized decoding temperature "
        "(via \\texttt{temp\\_std}).\n"
        "Figure~\\ref{fig:temp-raw-effect} shows the corresponding per-$T$ raw pattern.\n"
        f"On \\emph{{Xwords}}, shifts are associated with a small but statistically significant positive marginal effect "
        f"($\\mathrm{{AME}}={t_x['ame']:.4f}$, $p={_fmt_p_tex(t_x['p'], t_x['z'])}$), consistent with the raw contrast "
        f"$\\Delta={_fmt_delta_pp(t_x['delta_pp'])}$pp.\n"
        f"On \\emph{{Math}}, shifts are strongly harmful ($\\mathrm{{AME}}={t_m['ame']:.4f}$, $p={_fmt_p_tex(t_m['p'], t_m['z'])}$; "
        f"$\\Delta={_fmt_delta_pp(t_m['delta_pp'])}$pp).\n"
        f"On \\emph{{RHour}}, shifts are frequent ($\\%S={t_r['shift_pct']:.2f}$) but correctness is extremely low overall; accordingly, the estimated effect is "
        f"statistically detectable yet numerically negligible ($\\mathrm{{AME}}\\approx{t_r['ame']:.4f}$, $p={_fmt_p_tex(t_r['p'], t_r['z'])}$; "
        f"$\\Delta\\approx{_fmt_delta_pp(t_r['delta_pp'])}$pp).\n\n"
        "Raw per-temperature contrasts (Fig.~\\ref{fig:temp-raw-effect}) sharpen the interpretation:\n"
        "on \\emph{Xwords}, shifts can coincide with productive correction at low $T$, but the benefit weakens and may reverse by $T{=}0.7$.\n"
        "In \\emph{Math}, shifts remain harmful across temperatures, though the raw penalty attenuates as $T$ increases.\n"
        "In \\emph{RHour}, the curve stays close to zero in magnitude, reflecting the near-zero accuracy regime.\n\n"
        "\\vspace{1mm}\n"
        "\\noindent\n"
        "\\textbf{Takeaway.}\n"
        "We find that reasoning shifts do not reliably yield higher accuracy across specific training phases or at particular temperatures.\n\n"
        f"\\input{{{table_path.as_posix()}}}\n"
    )

    section_path = Path(args.out_section_tex)
    _ensure_parent(section_path)
    section_path.write_text(section + "\n", encoding="utf-8")
    print(section)
    print(f"[ok] wrote: {table_path}")
    print(f"[ok] wrote: {section_path}")
    return 0


def cmd_rq2_stage_by_temp_tables(argv: Optional[List[str]] = None) -> int:
    """
    Emit fixed-temperature training-stage tables for Qwen2.5-1.5B and Qwen2.5-7B/Llama3.1-8B.
    """
    parser = argparse.ArgumentParser(
        prog="master_analysis.py rq2-stage-by-temp-tables",
        description="Write fixed-temperature training-stage tables (RQ2) for Qwen1.5B and Qwen7B/Llama8B.",
    )
    parser.add_argument(
        "--out_table_tex",
        default="latex/table_rq2_stage_by_temp.tex",
        help="Write the Qwen1.5B table to this path (default: latex/table_rq2_stage_by_temp.tex).",
    )
    parser.add_argument(
        "--out_table_7b8b_tex",
        default="latex/table_rq2_stage_by_temp_7b8b.tex",
        help="Write the Qwen7B/Llama8B table to this path (default: latex/table_rq2_stage_by_temp_7b8b.tex).",
    )
    parser.add_argument(
        "--temps",
        nargs="+",
        type=float,
        default=[0.0, 0.05, 0.3],
        help="Temperatures to include (default: 0 0.05 0.3).",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Split filter (default: test).",
    )
    parser.add_argument(
        "--min_step",
        type=int,
        default=0,
        help="Minimum checkpoint step (default: 0).",
    )
    parser.add_argument(
        "--max_step",
        type=int,
        default=950,
        help="Maximum checkpoint step for Qwen1.5B (default: 950).",
    )
    parser.add_argument(
        "--max_step_7b8b",
        type=int,
        default=450,
        help="Maximum checkpoint step for Qwen7B/Llama8B (default: 450).",
    )
    parser.add_argument(
        "--use_statsmodels",
        action="store_true",
        help="Use statsmodels GLM (higher memory). Default: use lightweight IRLS.",
    )
    args = parser.parse_args(argv)

    temps = [float(t) for t in args.temps]
    split = str(args.split or "")
    min_step = int(args.min_step)
    max_step = int(args.max_step)
    max_step_7b8b = int(args.max_step_7b8b)

    roots_by_temp = {
        0.0: {
            "Xword": "artifacts/results/GRPO-1.5B-xword-temp-0",
            "Math": "artifacts/results/GRPO-1.5B-math-temp-0.0",
            "RHour": "artifacts/results/GRPO-1.5B-carpark-temp-0",
        },
        0.05: {
            "Xword": "artifacts/results/GRPO-1.5B-xword-temp-0.05",
            "Math": "artifacts/results/GRPO-1.5B-math-temp-0.05",
            "RHour": "artifacts/results/GRPO-1.5B-carpark-temp-0.05",
        },
        0.3: {
            "Xword": "artifacts/results/GRPO-1.5B-xword-temp-0.3",
            "Math": "artifacts/results/GRPO-1.5B-math-temp-0.3",
            "RHour": "artifacts/results/GRPO-1.5B-carpark-temp-0.3",
        },
        0.7: {
            "Xword": "artifacts/results/GRPO-1.5B-xword-temp-0.7",
            "Math": "artifacts/results/GRPO-1.5B-math-temp-0.7",
            "RHour": "artifacts/results/GRPO-1.5B-carpark-temp-0.7",
        },
    }

    import pandas as pd

    table_lines: List[str] = []
    table_lines.append("\\begin{table}[t]")
    table_lines.append("\\centering")
    table_lines.append("\\small")
    table_lines.append("\\setlength{\\tabcolsep}{4pt}")
    table_lines.append("\\renewcommand{\\arraystretch}{1.05}")
    table_lines.append("\\begin{tabular}{lrrrr}")
    table_lines.append("\\toprule")
    for temp_value in temps:
        temp_value = float(temp_value)
        dom_roots = roots_by_temp.get(temp_value)
        if not dom_roots:
            raise SystemExit(f"No roots configured for temp={temp_value}")
        stage_metrics: Dict[str, Dict[str, Any]] = {}
        stage_dfs: List["pd.DataFrame"] = []
        for dom in ("Xword", "Math", "RHour"):
            root = dom_roots[dom]
            df = _load_rows_for_root(
                root,
                split=split,
                min_step=min_step,
                max_step=max_step,
                temp_value=temp_value,
            )
            stage_dfs.append(df)
            stage_metrics[dom] = _compute_rq2_domain_metrics(
                df=df,
                formula="correct ~ C(problem) + step_std + shift",
                std_src_col="step",
                std_dest_col="step_std",
                cluster_by="problem",
                use_statsmodels=bool(args.use_statsmodels),
            )
        combined_df = pd.concat(stage_dfs, ignore_index=True) if stage_dfs else pd.DataFrame()
        stage_metrics["Combined"] = _compute_rq2_domain_metrics(
            df=combined_df,
            formula="correct ~ C(problem) + step_std + shift",
            std_src_col="step",
            std_dest_col="step_std",
            cluster_by="problem",
            use_statsmodels=bool(args.use_statsmodels),
        )
        table_lines.append(
            f"\\multicolumn{{5}}{{c}}{{\\textbf{{Training stage at fixed decoding temperature $T={temp_value:.2g}$}}}} \\\\"
        )
        table_lines.append("\\midrule")
        table_lines.append("\\textbf{Metric} & \\textbf{Xword} & \\textbf{Math} & \\textbf{RHour} & \\textbf{Combined} \\\\")
        table_lines.append("\\midrule")
        table_lines.append(
            f"$N$                   & {_fmt_int_tex(stage_metrics['Xword']['N'])} & "
            f"{_fmt_int_tex(stage_metrics['Math']['N'])} & {_fmt_int_tex(stage_metrics['RHour']['N'])} & "
            f"{_fmt_int_tex(stage_metrics['Combined']['N'])} \\\\"
        )
        table_lines.append(
            f"$\\%S$                 & {_fmt_shift_pct(stage_metrics['Xword']['shift_pct'])}    & "
            f"{_fmt_shift_pct(stage_metrics['Math']['shift_pct'])}    & {_fmt_shift_pct(stage_metrics['RHour']['shift_pct'])}  & "
            f"{_fmt_shift_pct(stage_metrics['Combined']['shift_pct'])}  \\\\"
        )
        table_lines.append(
            f"$\\hat{{p}}_{{Y\\mid S=1}}$ & {_fmt_prob(stage_metrics['Xword']['p_shift'], 4)}   & "
            f"{_fmt_prob(stage_metrics['Math']['p_shift'], 4)}   & {_fmt_prob(stage_metrics['RHour']['p_shift'], 4)}  & "
            f"{_fmt_prob(stage_metrics['Combined']['p_shift'], 4)}  \\\\"
        )
        table_lines.append(
            f"$\\Delta\\mathrm{{pp}}$   & ${_fmt_delta_pp(stage_metrics['Xword']['delta_pp'])}$  & "
            f"${_fmt_delta_pp(stage_metrics['Math']['delta_pp'])}$ & ${_fmt_delta_pp(stage_metrics['RHour']['delta_pp'])}$  & "
            f"${_fmt_delta_pp(stage_metrics['Combined']['delta_pp'])}$  \\\\"
        )
        table_lines.append(
            f"$\\mathrm{{AME}}$        & ${_fmt_ame(stage_metrics['Xword']['ame'])}$& "
            f"${_fmt_ame(stage_metrics['Math']['ame'])}$& ${_fmt_ame(stage_metrics['RHour']['ame'])}$ & "
            f"${_fmt_ame(stage_metrics['Combined']['ame'])}$\\\\"
        )
        table_lines.append(
            f"$p$                   & ${_fmt_p_tex(stage_metrics['Xword']['p'], stage_metrics['Xword']['z'])}$ & "
            f"${_fmt_p_tex(stage_metrics['Math']['p'], stage_metrics['Math']['z'])}$ & "
            f"${_fmt_p_tex(stage_metrics['RHour']['p'], stage_metrics['RHour']['z'])}$ & "
            f"${_fmt_p_tex(stage_metrics['Combined']['p'], stage_metrics['Combined']['z'])}$ \\\\"
        )
        if temp_value != temps[-1]:
            table_lines.append("\\midrule")
    table_lines.append("\\bottomrule")
    table_lines.append("\\end{tabular}")
    table_lines.append(
        "\\caption{\\textbf{Effect of detected reasoning shifts on accuracy (Qwen2.5-1.5B): training-stage analysis at fixed temperature.} "
        "For each fixed decoding temperature $T\\in\\{0.0,0.05,0.3\\}$, we report the share of traces with a detected shift "
        "(\\%S), accuracy among shifted traces (\\(\\hat{p}_{Y\\mid S=1}\\)), the raw accuracy difference in percentage points "
        "(\\(\\Delta\\mathrm{pp}\\)) between shifted and non-shifted traces, and the average marginal effect (\\(\\mathrm{AME}\\)) "
        "from a logistic regression with problem fixed effects, a standardized training-step control, and cluster-robust SEs "
        "(clustered by problem). Negative AME values indicate that shifted traces are less likely to be correct holding problem "
        "and training stage fixed.}"
    )
    table_lines.append("\\label{tab:rs_stage_T0005T03}")
    table_lines.append("\\vspace{-3mm}")
    table_lines.append("\\end{table}")
    table_tex = "\n".join(table_lines) + "\n"

    out_path = Path(args.out_table_tex)
    _ensure_parent(out_path)
    out_path.write_text(table_tex, encoding="utf-8")
    print(table_tex)
    print(f"[ok] wrote: {out_path}")

    specs = {spec.key: spec for spec in build_default_grid()}

    def root_for(key: str, temp_value: float) -> Path:
        spec = specs.get(key)
        if spec is None:
            raise SystemExit(f"Unknown experiment key: {key}")
        root = resolve_results_root(Path("artifacts/results"), spec, float(temp_value))
        if root is None:
            raise SystemExit(f"Missing results root for {key} at temp={temp_value}")
        return root

    model_keys = {
        "Qwen2.5-7B": "qwen7b_math",
        "Llama3.1-8B": "llama8b_math",
    }

    table7_lines: List[str] = []
    table7_lines.append("\\begin{table}[t]")
    table7_lines.append("\\centering")
    table7_lines.append("\\small")
    table7_lines.append("\\setlength{\\tabcolsep}{4pt}")
    table7_lines.append("\\renewcommand{\\arraystretch}{1.05}")
    table7_lines.append("\\begin{tabular}{lrrr}")
    table7_lines.append("\\toprule")
    for temp_value in temps:
        temp_value = float(temp_value)
        stage_metrics: Dict[str, Dict[str, Any]] = {}
        stage_dfs: List["pd.DataFrame"] = []
        for label, key in model_keys.items():
            root = root_for(key, temp_value)
            df = _load_rows_for_root(
                str(root),
                split=split,
                min_step=min_step,
                max_step=max_step_7b8b,
                temp_value=temp_value,
            )
            stage_dfs.append(df)
            stage_metrics[label] = _compute_rq2_domain_metrics(
                df=df,
                formula="correct ~ C(problem) + step_std + shift",
                std_src_col="step",
                std_dest_col="step_std",
                cluster_by="problem",
                use_statsmodels=bool(args.use_statsmodels),
            )
        combined_df = pd.concat(stage_dfs, ignore_index=True) if stage_dfs else pd.DataFrame()
        stage_metrics["Combined"] = _compute_rq2_domain_metrics(
            df=combined_df,
            formula="correct ~ C(problem) + step_std + shift",
            std_src_col="step",
            std_dest_col="step_std",
            cluster_by="problem",
            use_statsmodels=bool(args.use_statsmodels),
        )
        table7_lines.append(
            f"\\multicolumn{{4}}{{c}}{{\\textbf{{Training stage at fixed decoding temperature $T={temp_value:.2g}$}}}} \\\\"
        )
        table7_lines.append("\\midrule")
        table7_lines.append("\\textbf{Metric} & \\textbf{Qwen2.5-7B} & \\textbf{Llama3.1-8B} & \\textbf{Combined} \\\\")
        table7_lines.append("\\midrule")
        table7_lines.append(
            f"$N$                   & {_fmt_int_tex(stage_metrics['Qwen2.5-7B']['N'])} & "
            f"{_fmt_int_tex(stage_metrics['Llama3.1-8B']['N'])} & {_fmt_int_tex(stage_metrics['Combined']['N'])} \\\\"
        )
        table7_lines.append(
            f"$\\%S$                 & {_fmt_shift_pct(stage_metrics['Qwen2.5-7B']['shift_pct'])}    & "
            f"{_fmt_shift_pct(stage_metrics['Llama3.1-8B']['shift_pct'])}    & "
            f"{_fmt_shift_pct(stage_metrics['Combined']['shift_pct'])}  \\\\"
        )
        table7_lines.append(
            f"$\\hat{{p}}_{{Y\\mid S=1}}$ & {_fmt_prob(stage_metrics['Qwen2.5-7B']['p_shift'], 4)}   & "
            f"{_fmt_prob(stage_metrics['Llama3.1-8B']['p_shift'], 4)}   & "
            f"{_fmt_prob(stage_metrics['Combined']['p_shift'], 4)}  \\\\"
        )
        table7_lines.append(
            f"$\\Delta\\mathrm{{pp}}$   & ${_fmt_delta_pp(stage_metrics['Qwen2.5-7B']['delta_pp'])}$  & "
            f"${_fmt_delta_pp(stage_metrics['Llama3.1-8B']['delta_pp'])}$ & "
            f"${_fmt_delta_pp(stage_metrics['Combined']['delta_pp'])}$  \\\\"
        )
        table7_lines.append(
            f"$\\mathrm{{AME}}$        & ${_fmt_ame(stage_metrics['Qwen2.5-7B']['ame'])}$& "
            f"${_fmt_ame(stage_metrics['Llama3.1-8B']['ame'])}$& "
            f"${_fmt_ame(stage_metrics['Combined']['ame'])}$\\\\"
        )
        table7_lines.append(
            f"$p$                   & ${_fmt_p_tex(stage_metrics['Qwen2.5-7B']['p'], stage_metrics['Qwen2.5-7B']['z'])}$ & "
            f"${_fmt_p_tex(stage_metrics['Llama3.1-8B']['p'], stage_metrics['Llama3.1-8B']['z'])}$ & "
            f"${_fmt_p_tex(stage_metrics['Combined']['p'], stage_metrics['Combined']['z'])}$ \\\\"
        )
        if temp_value != temps[-1]:
            table7_lines.append("\\midrule")
    table7_lines.append("\\bottomrule")
    table7_lines.append("\\end{tabular}")
    table7_lines.append(
        "\\caption{\\textbf{Effect of detected reasoning shifts on accuracy (Qwen2.5-7B/Llama3.1-8B): training-stage analysis at fixed temperature.} "
        "For each fixed decoding temperature $T\\in\\{0.0,0.05,0.3\\}$, we report shift prevalence (\\%S), accuracy among shifted "
        "traces (\\(\\hat{p}_{Y\\mid S=1}\\)), the raw accuracy difference in percentage points (\\(\\Delta\\mathrm{pp}\\)), and the "
        "average marginal effect (\\(\\mathrm{AME}\\)) from a logistic regression with problem fixed effects, a standardized "
        "training-step control, and cluster-robust SEs. Negative AME values indicate that shifted traces are less likely to be correct "
        "holding problem and training stage fixed.}"
    )
    table7_lines.append("\\label{tab:rs_stage_T0005T03_7b8b}")
    table7_lines.append("\\vspace{-3mm}")
    table7_lines.append("\\end{table}")
    table7_tex = "\n".join(table7_lines) + "\n"

    out_path_7b8b = Path(args.out_table_7b8b_tex)
    _ensure_parent(out_path_7b8b)
    out_path_7b8b.write_text(table7_tex, encoding="utf-8")
    print(table7_tex)
    print(f"[ok] wrote: {out_path_7b8b}")
    return 0


def cmd_formal_aha_temp_tables(argv: Optional[List[str]] = None) -> int:
    """
    Emit formal-Aha temperature-sweep tables (sample-level, Qwen2.5-1.5B and Qwen2.5-7B/Llama3.1-8B).
    """
    parser = argparse.ArgumentParser(
        prog="master_analysis.py formal-aha-temp-tables",
        description="Write formal Aha temperature-sweep tables.",
    )
    parser.add_argument(
        "--out_table_tex",
        default="latex/table_formal_aha_temp.tex",
        help="Write the Qwen1.5B table to this path (default: latex/table_formal_aha_temp.tex).",
    )
    parser.add_argument(
        "--out_table_7b8b_tex",
        default="latex/table_formal_aha_temp_7b8b.tex",
        help="Write the Qwen7B/Llama8B table to this path (default: latex/table_formal_aha_temp_7b8b.tex).",
    )
    parser.add_argument(
        "--temps",
        nargs="+",
        type=float,
        default=[0.0, 0.05, 0.3, 0.7],
        help="Temperatures to include (default: 0 0.05 0.3 0.7).",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Split filter (default: test).",
    )
    parser.add_argument(
        "--min_step",
        type=int,
        default=0,
        help="Minimum checkpoint step (default: 0).",
    )
    parser.add_argument(
        "--max_step",
        type=int,
        default=950,
        help="Maximum checkpoint step for Qwen1.5B (default: 950).",
    )
    parser.add_argument(
        "--max_step_7b8b",
        type=int,
        default=450,
        help="Maximum checkpoint step for Qwen7B/Llama8B (default: 450).",
    )
    parser.add_argument(
        "--delta1",
        type=float,
        default=0.125,
        help="Formal Aha delta1 threshold (default: 0.125).",
    )
    parser.add_argument(
        "--delta2",
        type=float,
        default=0.125,
        help="Formal Aha delta2 threshold (default: 0.125).",
    )
    parser.add_argument(
        "--min_prior_steps",
        type=int,
        default=2,
        help="Formal Aha min prior steps (default: 2).",
    )
    parser.add_argument(
        "--use_statsmodels",
        action="store_true",
        help="Use statsmodels GLM (higher memory). Default: use lightweight IRLS.",
    )
    args = parser.parse_args(argv)

    import numpy as np
    import pandas as pd

    max_step = int(args.max_step)
    max_step_7b8b = int(args.max_step_7b8b)

    def load_samples(root: str, *, max_step: int, temp_value: float) -> "pd.DataFrame":
        df = _load_rows_for_root(
            root,
            split=str(args.split or ""),
            min_step=int(args.min_step),
            max_step=int(max_step),
            temp_value=float(temp_value),
            pass_key="pass1",
            label_key="shift_in_reasoning_v1",
        ).copy()
        if df.empty:
            return df
        df = df.rename(columns={"shift": "aha_gpt"})
        df["aha_gpt"] = df["aha_gpt"].astype(int)
        return df

    def mark_formal(problem_step_df: "pd.DataFrame") -> "pd.DataFrame":
        required_cols = {"step", "problem", "freq_correct", "aha_rate_gpt", "aha_any_gpt"}
        missing = required_cols.difference(problem_step_df.columns)
        if missing:
            raise ValueError("mark_formal: missing required columns: " + ", ".join(sorted(missing)))

        problem_step_df = problem_step_df.sort_values(["problem", "step"]).copy()
        flags = np.zeros(len(problem_step_df), dtype=int)
        idx = 0
        for _, sub in problem_step_df.groupby("problem", sort=False):
            sub_sorted = sub.sort_values("step")
            freq = sub_sorted["freq_correct"].to_numpy(float)
            rate = sub_sorted["aha_rate_gpt"].to_numpy(float)
            shift = sub_sorted["aha_any_gpt"].to_numpy(int)
            for j in range(len(sub_sorted)):
                if j < int(args.min_prior_steps):
                    flags[idx] = 0
                else:
                    prior_ok = float(np.max(freq[:j])) < float(args.delta1) and float(np.max(rate[:j])) < float(
                        args.delta2
                    )
                    flags[idx] = int(prior_ok and (shift[j] == 1))
                idx += 1
        problem_step_df["aha_formal_ps"] = flags
        return problem_step_df

    def attach_formal(samples_df: "pd.DataFrame") -> "pd.DataFrame":
        if samples_df.empty:
            return samples_df
        grouped = samples_df.groupby(["step", "problem"], as_index=False).agg(
            n_samples=("correct", "size"),
            freq_correct=("correct", "mean"),
            aha_any_gpt=("aha_gpt", "max"),
            aha_rate_gpt=("aha_gpt", "mean"),
        )
        for col in ("n_samples", "aha_any_gpt"):
            grouped[col] = grouped[col].astype(int)
        grouped["freq_correct"] = grouped["freq_correct"].astype(float)
        grouped["aha_rate_gpt"] = grouped["aha_rate_gpt"].astype(float)
        grouped = mark_formal(grouped)
        merged = samples_df.merge(
            grouped[["step", "problem", "aha_formal_ps"]],
            on=["step", "problem"],
            how="left",
        ).fillna({"aha_formal_ps": 0})
        merged["aha_formal_ps"] = merged["aha_formal_ps"].astype(int)
        merged["aha_formal"] = (merged["aha_formal_ps"] & merged["aha_gpt"]).astype(int)
        return merged

    def formal_metrics(df: "pd.DataFrame") -> Dict[str, Any]:
        if df.empty:
            return {
                "N": 0,
                "shift_pct": float("nan"),
                "p_shift": float("nan"),
                "delta_pp": float("nan"),
                "ame": float("nan"),
                "p": float("nan"),
                "z": float("nan"),
                "has_shift": False,
            }
        N = int(len(df))
        n_shift = int((df["aha_formal"] == 1).sum())
        n_noshift = N - n_shift
        shift_pct = (100.0 * n_shift / N) if N else float("nan")
        has_shift = n_shift > 0 and n_noshift > 0
        p_shift = float(df.loc[df["aha_formal"] == 1, "correct"].mean()) if n_shift else float("nan")
        p_noshift = float(df.loc[df["aha_formal"] == 0, "correct"].mean()) if n_noshift else float("nan")
        delta_pp = (p_shift - p_noshift) * 100.0 if np.isfinite(p_shift) and np.isfinite(p_noshift) else float("nan")
        if not has_shift:
            return {
                "N": N,
                "shift_pct": shift_pct,
                "p_shift": p_shift,
                "delta_pp": delta_pp,
                "ame": float("nan"),
                "p": float("nan"),
                "z": float("nan"),
                "has_shift": False,
            }

        df = df.copy()
        df["step_std"] = _std_col(df["step"].astype(float).tolist())
        df["shift"] = df["aha_formal"].astype(int)
        pid = df["problem"].astype("category").cat.codes.to_numpy()
        y = df["correct"].astype(float).to_numpy()
        X = np.column_stack([df["step_std"].astype(float).to_numpy(), df["shift"].astype(float).to_numpy()])
        try:
            beta, se, z, alpha = _fit_fe_logit_irls_multi(y, X, pid)
            b1 = float(beta[1]) if beta.size > 1 else float("nan")
            z_val = float(z[1]) if z.size > 1 else float("nan")
            eta = alpha[pid] + X @ beta
            eta0 = eta - b1 * X[:, 1]
            eta1 = eta0 + b1
            ame = float((_sigmoid(eta1) - _sigmoid(eta0)).mean())
            log10_p = _two_sided_log10_p_from_z(z_val) if z_val == z_val else float("nan")
            p_val = float(10 ** log10_p) if log10_p > -300 else 0.0
        except Exception:
            ame = float("nan")
            p_val = float("nan")
            z_val = float("nan")

        return {
            "N": N,
            "shift_pct": shift_pct,
            "p_shift": p_shift,
            "delta_pp": delta_pp,
            "ame": ame,
            "p": p_val,
            "z": z_val,
            "has_shift": has_shift,
        }

    def fmt_or_dash(value: float, fmt: str) -> str:
        if value != value:
            return "--"
        return fmt.format(value)

    def fmt_p(value: float, z: float) -> str:
        if value != value or z != z:
            return "--"
        return _fmt_p_tex(value, z)

    temps = [float(t) for t in args.temps]

    roots_by_temp = {
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

    table_lines: List[str] = []
    table_lines.append("\\begin{table}[t]")
    table_lines.append("\\centering")
    table_lines.append("\\small")
    table_lines.append("\\setlength{\\tabcolsep}{4pt}")
    table_lines.append("\\renewcommand{\\arraystretch}{1.05}")
    table_lines.append("\\begin{tabular}{lrrrr}")
    table_lines.append("\\toprule")
    table_lines.append("\\textbf{Metric} & \\textbf{Crossword} & \\textbf{Math} & \\textbf{RHour} & \\textbf{Combined} \\\\")
    table_lines.append("\\midrule")
    for temp_value in temps:
        dom_roots = roots_by_temp.get(float(temp_value))
        if not dom_roots:
            raise SystemExit(f"No roots configured for temp={temp_value}")
        metrics: Dict[str, Dict[str, Any]] = {}
        frames: List["pd.DataFrame"] = []
        for dom in ("Crossword", "Math", "RHour"):
            df = load_samples(dom_roots[dom], max_step=max_step, temp_value=float(temp_value))
            df = attach_formal(df)
            metrics[dom] = formal_metrics(df)
            frames.append(df)
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        metrics["Combined"] = formal_metrics(combined)
        table_lines.append(f"\\multicolumn{{5}}{{c}}{{\\textbf{{$T={float(temp_value):.2g}$}}}} \\\\")
        table_lines.append("\\midrule")
        table_lines.append(
            f"$N$                   & {_fmt_int_tex(metrics['Crossword']['N'])} & {_fmt_int_tex(metrics['Math']['N'])} & "
            f"{_fmt_int_tex(metrics['RHour']['N'])} & {_fmt_int_tex(metrics['Combined']['N'])} \\\\"
        )
        table_lines.append(
            f"$\\%S$                 & {_fmt_shift_pct(metrics['Crossword']['shift_pct'])}    & "
            f"{_fmt_shift_pct(metrics['Math']['shift_pct'])}    & {_fmt_shift_pct(metrics['RHour']['shift_pct'])}  & "
            f"{_fmt_shift_pct(metrics['Combined']['shift_pct'])} \\\\"
        )
        table_lines.append(
            f"$\\hat{{p}}_{{Y\\mid S=1}}$ & {fmt_or_dash(metrics['Crossword']['p_shift'], '{:.4f}')}   & "
            f"{fmt_or_dash(metrics['Math']['p_shift'], '{:.4f}')}   & {fmt_or_dash(metrics['RHour']['p_shift'], '{:.4f}')} & "
            f"{fmt_or_dash(metrics['Combined']['p_shift'], '{:.4f}')} \\\\"
        )
        table_lines.append(
            f"$\\Delta\\mathrm{{pp}}$   & {fmt_or_dash(metrics['Crossword']['delta_pp'], '${:+.2f}$')}  & "
            f"{fmt_or_dash(metrics['Math']['delta_pp'], '${:+.2f}$')}  & {fmt_or_dash(metrics['RHour']['delta_pp'], '${:+.2f}$')} & "
            f"{fmt_or_dash(metrics['Combined']['delta_pp'], '${:+.2f}$')} \\\\"
        )
        table_lines.append(
            f"$\\mathrm{{AME}}$        & {fmt_or_dash(metrics['Crossword']['ame'], '${:+.4f}$')} & "
            f"{fmt_or_dash(metrics['Math']['ame'], '${:+.4f}$')} & {fmt_or_dash(metrics['RHour']['ame'], '${:+.4f}$')} & "
            f"{fmt_or_dash(metrics['Combined']['ame'], '${:+.4f}$')} \\\\"
        )
        table_lines.append(
            f"$p$                   & {fmt_p(metrics['Crossword']['p'], metrics['Crossword']['z'])} & "
            f"{fmt_p(metrics['Math']['p'], metrics['Math']['z'])} & {fmt_p(metrics['RHour']['p'], metrics['RHour']['z'])} & "
            f"{fmt_p(metrics['Combined']['p'], metrics['Combined']['z'])} \\\\"
        )
        if temp_value != temps[-1]:
            table_lines.append("\\midrule")
    table_lines.append("\\bottomrule")
    table_lines.append("\\end{tabular}")
    table_lines.append(
        "\\caption{\\textbf{Formal ``Aha!'' detector (Def.~3.1): temperature sweep.} "
        "For each domain and decoding temperature, \\(\\%S\\) is the share of traces flagged by the formal detector; "
        "\\(\\hat{p}_{Y\\mid S=1}\\) is empirical accuracy among flagged traces; and "
        "\\(\\Delta\\mathrm{pp}\\) is the raw accuracy difference (percentage points) between flagged and non-flagged traces. "
        "\\(\\mathrm{AME}\\) is the average marginal effect of a formal-Aha flag from a logistic regression with problem fixed effects, "
        "a standardized training-step control, and cluster-robust SEs. "
        "Cells marked ``--'' indicate the detector never fired in that regime, making conditional quantities undefined.}"
    )
    table_lines.append("\\label{tab:formal-aha-temp}")
    table_lines.append("\\vspace{-3mm}")
    table_lines.append("\\end{table}")
    table_tex = "\n".join(table_lines) + "\n"

    out_path = Path(args.out_table_tex)
    _ensure_parent(out_path)
    out_path.write_text(table_tex, encoding="utf-8")
    print(table_tex)
    print(f"[ok] wrote: {out_path}")

    specs = {spec.key: spec for spec in build_default_grid()}

    def root_for(key: str, temp_value: float) -> Path:
        spec = specs.get(key)
        if spec is None:
            raise SystemExit(f"Unknown experiment key: {key}")
        root = resolve_results_root(Path("artifacts/results"), spec, float(temp_value))
        if root is None:
            raise SystemExit(f"Missing results root for {key} at temp={temp_value}")
        return root

    model_keys = {
        "Qwen2.5-7B": "qwen7b_math",
        "Llama3.1-8B": "llama8b_math",
    }

    table7_lines: List[str] = []
    table7_lines.append("\\begin{table}[t]")
    table7_lines.append("\\centering")
    table7_lines.append("\\small")
    table7_lines.append("\\setlength{\\tabcolsep}{4pt}")
    table7_lines.append("\\renewcommand{\\arraystretch}{1.05}")
    table7_lines.append("\\begin{tabular}{lrrr}")
    table7_lines.append("\\toprule")
    table7_lines.append("\\textbf{Metric} & \\textbf{Qwen2.5-7B} & \\textbf{Llama3.1-8B} & \\textbf{Combined} \\\\")
    table7_lines.append("\\midrule")
    for temp_value in temps:
        metrics: Dict[str, Dict[str, Any]] = {}
        frames = []
        for label, key in model_keys.items():
            root = root_for(key, float(temp_value))
            df = load_samples(str(root), max_step=max_step_7b8b, temp_value=float(temp_value))
            df = attach_formal(df)
            metrics[label] = formal_metrics(df)
            frames.append(df)
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        metrics["Combined"] = formal_metrics(combined)
        table7_lines.append(f"\\multicolumn{{4}}{{c}}{{\\textbf{{$T={float(temp_value):.2g}$}}}} \\\\")
        table7_lines.append("\\midrule")
        table7_lines.append(
            f"$N$                   & {_fmt_int_tex(metrics['Qwen2.5-7B']['N'])} & "
            f"{_fmt_int_tex(metrics['Llama3.1-8B']['N'])} & {_fmt_int_tex(metrics['Combined']['N'])} \\\\"
        )
        table7_lines.append(
            f"$\\%S$                 & {_fmt_shift_pct(metrics['Qwen2.5-7B']['shift_pct'])}    & "
            f"{_fmt_shift_pct(metrics['Llama3.1-8B']['shift_pct'])}    & {_fmt_shift_pct(metrics['Combined']['shift_pct'])} \\\\"
        )
        table7_lines.append(
            f"$\\hat{{p}}_{{Y\\mid S=1}}$ & {fmt_or_dash(metrics['Qwen2.5-7B']['p_shift'], '{:.4f}')}   & "
            f"{fmt_or_dash(metrics['Llama3.1-8B']['p_shift'], '{:.4f}')}   & {fmt_or_dash(metrics['Combined']['p_shift'], '{:.4f}')} \\\\"
        )
        table7_lines.append(
            f"$\\Delta\\mathrm{{pp}}$   & {fmt_or_dash(metrics['Qwen2.5-7B']['delta_pp'], '${:+.2f}$')}  & "
            f"{fmt_or_dash(metrics['Llama3.1-8B']['delta_pp'], '${:+.2f}$')}  & "
            f"{fmt_or_dash(metrics['Combined']['delta_pp'], '${:+.2f}$')} \\\\"
        )
        table7_lines.append(
            f"$\\mathrm{{AME}}$        & {fmt_or_dash(metrics['Qwen2.5-7B']['ame'], '${:+.4f}$')} & "
            f"{fmt_or_dash(metrics['Llama3.1-8B']['ame'], '${:+.4f}$')} & "
            f"{fmt_or_dash(metrics['Combined']['ame'], '${:+.4f}$')} \\\\"
        )
        table7_lines.append(
            f"$p$                   & {fmt_p(metrics['Qwen2.5-7B']['p'], metrics['Qwen2.5-7B']['z'])} & "
            f"{fmt_p(metrics['Llama3.1-8B']['p'], metrics['Llama3.1-8B']['z'])} & "
            f"{fmt_p(metrics['Combined']['p'], metrics['Combined']['z'])} \\\\"
        )
        if temp_value != temps[-1]:
            table7_lines.append("\\midrule")
    table7_lines.append("\\bottomrule")
    table7_lines.append("\\end{tabular}")
    table7_lines.append(
        "\\caption{\\textbf{Formal ``Aha!'' detector (Def.~3.1): temperature sweep for Qwen2.5-7B/Llama3.1-8B.} "
        "For each decoding temperature, \\(\\%S\\) is the share of traces flagged by the formal detector; "
        "\\(\\hat{p}_{Y\\mid S=1}\\) is empirical accuracy among flagged traces; and "
        "\\(\\Delta\\mathrm{pp}\\) is the raw accuracy difference (percentage points) between flagged and non-flagged traces. "
        "\\(\\mathrm{AME}\\) is the average marginal effect from a logistic regression with problem fixed effects and "
        "a standardized training-step control. Cells marked ``--'' indicate the detector never fired in that regime.}"
    )
    table7_lines.append("\\label{tab:formal-aha-temp-7b8b}")
    table7_lines.append("\\vspace{-3mm}")
    table7_lines.append("\\end{table}")
    table7_tex = "\n".join(table7_lines) + "\n"

    out_path_7b8b = Path(args.out_table_7b8b_tex)
    _ensure_parent(out_path_7b8b)
    out_path_7b8b.write_text(table7_tex, encoding="utf-8")
    print(table7_tex)
    print(f"[ok] wrote: {out_path_7b8b}")
    return 0


@dataclass
class RQ3DomainData:
    domain: str
    correct: "Any"
    shift: "Any"
    entropy: "Any"
    entropy_std: "Any"
    step_std: "Any"
    problem_ids: "Any"
    entropy_q80: float
    share_shift: float


def _make_problem_ids(keys: List[str]) -> "Any":
    import numpy as np

    mapping: Dict[str, int] = {}
    ids = np.empty(len(keys), dtype=int)
    next_id = 0
    for i, key in enumerate(keys):
        if key not in mapping:
            mapping[key] = next_id
            next_id += 1
        ids[i] = mapping[key]
    return ids


def _load_rq3_domain_data(
    domain_label: str,
    roots: List[str],
    *,
    split: str,
    min_step: int,
    max_step: int,
    entropy_mode: str,
    gpt_mode: str,
    gate_by_words: bool,
) -> RQ3DomainData:
    import numpy as np

    correct: List[int] = []
    shift: List[int] = []
    entropy: List[float] = []
    step: List[int] = []
    problem_keys: List[str] = []

    for root in roots:
        files = scan_files_step_only(str(root), split_substr=str(split or ""))
        for file_path in files:
            step_hint = nat_step_from_path(file_path)
            if step_hint is not None and (step_hint < min_step or step_hint > max_step):
                continue
            for record in iter_records_from_file(file_path):
                if split and str(record.get("split", "")).lower() != split.lower():
                    continue
                pass1, step_val = extract_pass1_and_step(record, step_hint)
                if not pass1 or step_val is None:
                    continue
                if step_val < min_step or step_val > max_step:
                    continue
                ent = entropy_from_pass1(pass1, mode=entropy_mode)
                if ent is None:
                    continue
                shift_flag = aha_gpt(
                    pass1,
                    record,
                    mode=gpt_mode,
                    gate_by_words=gate_by_words,
                    domain=domain_label,
                )
                correct_flag = extract_correct(pass1, record)
                if correct_flag is None:
                    continue
                problem = problem_key_from_record(record, "unknown")
                problem_key = f"{domain_label}:{problem}"

                correct.append(int(correct_flag))
                shift.append(int(shift_flag))
                entropy.append(float(ent))
                step.append(int(step_val))
                problem_keys.append(problem_key)

    if not correct:
        raise SystemExit(f"No RQ3 rows found for domain={domain_label}")

    entropy_arr = np.asarray(entropy, dtype=float)
    step_arr = np.asarray(step, dtype=float)
    mean_ent = float(entropy_arr.mean())
    std_ent = float(entropy_arr.std(ddof=0) + 1e-8)
    mean_step = float(step_arr.mean())
    std_step = float(step_arr.std(ddof=0) + 1e-8)

    entropy_std = (entropy_arr - mean_ent) / std_ent
    step_std = (step_arr - mean_step) / std_step
    entropy_q80 = float(np.quantile(entropy_arr, 0.8))
    shift_arr = np.asarray(shift, dtype=int)

    return RQ3DomainData(
        domain=domain_label,
        correct=np.asarray(correct, dtype=int),
        shift=shift_arr,
        entropy=entropy_arr,
        entropy_std=entropy_std,
        step_std=step_std,
        problem_ids=_make_problem_ids(problem_keys),
        entropy_q80=entropy_q80,
        share_shift=float(shift_arr.mean()) if shift_arr.size else 0.0,
    )


def _fit_shift_entropy_fe(
    data: RQ3DomainData,
) -> Dict[str, Any]:
    import numpy as np

    y = data.shift.astype(float)
    X = data.entropy_std.astype(float)
    pid = data.problem_ids
    beta, se, z, _alpha = _fit_fe_logit_irls_multi(y, X, pid)
    coef = float(beta[0]) if beta.size else float("nan")
    se_val = float(se[0]) if se.size else float("nan")
    z_val = float(z[0]) if z.size else float("nan")
    or_val = float(np.exp(coef)) if np.isfinite(coef) else float("nan")
    ci_low = float(np.exp(coef - 1.959964 * se_val)) if np.isfinite(se_val) else float("nan")
    ci_high = float(np.exp(coef + 1.959964 * se_val)) if np.isfinite(se_val) else float("nan")
    log10_p = _two_sided_log10_p_from_z(z_val) if z_val == z_val else float("nan")
    p_val = float(10 ** log10_p) if log10_p > -300 else 0.0
    return {
        "N": int(y.size),
        "coef": coef,
        "se": se_val,
        "z": z_val,
        "p": p_val,
        "or": or_val,
        "or_ci_low": ci_low,
        "or_ci_high": ci_high,
    }


def _rq3_stratum_metrics(
    data: RQ3DomainData,
    mask: "Any",
) -> Dict[str, Any]:
    import numpy as np

    y = data.correct[mask].astype(float)
    shift = data.shift[mask].astype(float)
    entropy_std = data.entropy_std[mask].astype(float)
    step_std = data.step_std[mask].astype(float)
    pid = data.problem_ids[mask]

    N = int(y.size)
    n_shift = int((shift == 1).sum())
    n_noshift = N - n_shift
    p_shift = float(y[shift == 1].mean()) if n_shift else 0.0
    p_noshift = float(y[shift == 0].mean()) if n_noshift else 0.0
    delta_pp = (p_shift - p_noshift) * 100.0

    if N == 0 or n_shift == 0 or n_noshift == 0:
        return {
            "N": N,
            "delta_pp": delta_pp,
            "coef": float("nan"),
            "p": float("nan"),
            "z": float("nan"),
        }

    X = np.column_stack([entropy_std, step_std, shift])
    beta, se, z, _alpha = _fit_fe_logit_irls_multi(y, X, pid)
    coef = float(beta[2]) if beta.size > 2 else float("nan")
    z_val = float(z[2]) if z.size > 2 else float("nan")
    log10_p = _two_sided_log10_p_from_z(z_val) if z_val == z_val else float("nan")
    p_val = float(10 ** log10_p) if log10_p > -300 else 0.0
    return {
        "N": N,
        "delta_pp": delta_pp,
        "coef": coef,
        "p": p_val,
        "z": z_val,
    }


def _load_forced_aha_summary(root: str) -> Optional[Dict[str, Any]]:
    path = Path(root) / "forced_aha_effect" / "forced_aha_summary.csv"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("metric") == "sample":
                return row
    return None


def _compute_forced_aha_summary(roots: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
    forced_summary: Dict[str, Dict[str, Any]] = {}
    for dom, dom_roots in roots.items():
        total_n = 0.0
        sum_acc1 = 0.0
        sum_acc2 = 0.0
        wins2 = 0.0
        wins1 = 0.0
        for root in dom_roots:
            row = _load_forced_aha_summary(root)
            if not row:
                continue
            n_units = float(row.get("n_units", 0) or 0)
            acc1 = float(row.get("acc_pass1", 0) or 0)
            acc2 = float(row.get("acc_pass2", 0) or 0)
            w2 = float(row.get("wins_pass2", 0) or 0)
            w1 = float(row.get("wins_pass1", 0) or 0)
            total_n += n_units
            sum_acc1 += acc1 * n_units
            sum_acc2 += acc2 * n_units
            wins2 += w2
            wins1 += w1
        if total_n <= 0:
            raise SystemExit(f"No forced_aha_summary rows found for domain={dom}")
        acc1 = sum_acc1 / total_n
        acc2 = sum_acc2 / total_n
        delta_pp = (acc2 - acc1) * 100.0
        forced_summary[dom] = {
            "N": int(round(total_n)),
            "p1": acc1,
            "p2": acc2,
            "delta_pp": delta_pp,
            "wins2": int(round(wins2)),
            "wins1": int(round(wins1)),
        }
    return forced_summary


def cmd_rq3_section(argv: Optional[List[str]] = None) -> int:
    """
    Emit the RQ3 section LaTeX snippet with updated numbers + tables.
    """
    parser = argparse.ArgumentParser(
        prog="master_analysis.py rq3-section",
        description="Write the RQ3 section LaTeX and supporting tables.",
    )
    parser.add_argument(
        "--out_section_tex",
        default="latex/section_rq3.tex",
        help="Write the RQ3 section to this path (default: latex/section_rq3.tex).",
    )
    parser.add_argument(
        "--out_table_entropy_tex",
        default="latex/table_rq3_shift_entropy_strata.tex",
        help="Write the entropy-strata table to this path (default: latex/table_rq3_shift_entropy_strata.tex).",
    )
    parser.add_argument(
        "--out_table_forced_tex",
        default="latex/table_rq3_forced_aha.tex",
        help="Write the forced-aha table to this path (default: latex/table_rq3_forced_aha.tex).",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Split filter (default: test).",
    )
    parser.add_argument(
        "--min_step",
        type=int,
        default=0,
        help="Minimum checkpoint step (default: 0).",
    )
    parser.add_argument(
        "--max_step",
        type=int,
        default=950,
        help="Maximum checkpoint step (default: 950).",
    )
    parser.add_argument(
        "--entropy_mode",
        default="combined",
        choices=["combined", "sum", "think", "answer"],
        help="Pass-1 entropy mode (default: combined).",
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
    args = parser.parse_args(argv)

    split = str(args.split or "")
    min_step = int(args.min_step)
    max_step = int(args.max_step)
    gate_by_words = not args.no_words_gate

    roots = {
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

    domain_data: Dict[str, RQ3DomainData] = {}
    for domain_key, domain_roots in roots.items():
        domain_label = "Crossword" if domain_key == "Xword" else ("Carpark" if domain_key == "RHour" else "Math")
        domain_data[domain_key] = _load_rq3_domain_data(
            domain_label,
            domain_roots,
            split=split,
            min_step=min_step,
            max_step=max_step,
            entropy_mode=args.entropy_mode,
            gpt_mode=args.gpt_mode,
            gate_by_words=gate_by_words,
        )

    # Shift ~ std_entropy regressions.
    shift_entropy_stats = {k: _fit_shift_entropy_fe(v) for k, v in domain_data.items()}

    # Pooled across domains (std_entropy standardized within domain).
    import numpy as np

    pooled_correct = np.concatenate([d.correct for d in domain_data.values()])
    pooled_shift = np.concatenate([d.shift for d in domain_data.values()])
    pooled_entropy_std = np.concatenate([d.entropy_std for d in domain_data.values()])
    pooled_problem_keys: List[str] = []
    for k, d in domain_data.items():
        pooled_problem_keys.extend([f"{k}:{pid}" for pid in d.problem_ids.tolist()])
    pooled_pid = _make_problem_ids(pooled_problem_keys)
    pooled_data = RQ3DomainData(
        domain="All",
        correct=pooled_correct,
        shift=pooled_shift,
        entropy=np.concatenate([d.entropy for d in domain_data.values()]),
        entropy_std=pooled_entropy_std,
        step_std=np.concatenate([d.step_std for d in domain_data.values()]),
        problem_ids=pooled_pid,
        entropy_q80=float("nan"),
        share_shift=float(pooled_shift.mean()) if pooled_shift.size else 0.0,
    )
    pooled_stats = _fit_shift_entropy_fe(pooled_data)

    # Shift rarity range across domains.
    shift_pcts = [domain_data[k].share_shift * 100.0 for k in ("Xword", "Math", "RHour")]
    shift_pct_min = min(shift_pcts)
    shift_pct_max = max(shift_pcts)

    # Entropy strata table metrics.
    strata_labels = [
        ("all", "All traces"),
        ("high", "High entropy (top 20\\%)"),
        ("low", "Low entropy (bottom 80\\%)"),
    ]
    strata_metrics: Dict[str, Dict[str, Dict[str, Any]]] = {k: {} for k in ("Xword", "Math", "RHour", "Combined")}
    combined_masks_all: List["Any"] = []
    combined_masks_high: List["Any"] = []
    combined_masks_low: List["Any"] = []
    for dom, data in domain_data.items():
        mask_all = np.ones_like(data.entropy, dtype=bool)
        mask_high = data.entropy >= data.entropy_q80
        mask_low = data.entropy < data.entropy_q80
        strata_metrics[dom]["all"] = _rq3_stratum_metrics(data, mask_all)
        strata_metrics[dom]["high"] = _rq3_stratum_metrics(data, mask_high)
        strata_metrics[dom]["low"] = _rq3_stratum_metrics(data, mask_low)
        combined_masks_all.append(mask_all)
        combined_masks_high.append(mask_high)
        combined_masks_low.append(mask_low)

    if combined_masks_all:
        mask_all = np.concatenate(combined_masks_all)
        mask_high = np.concatenate(combined_masks_high)
        mask_low = np.concatenate(combined_masks_low)
        strata_metrics["Combined"]["all"] = _rq3_stratum_metrics(pooled_data, mask_all)
        strata_metrics["Combined"]["high"] = _rq3_stratum_metrics(pooled_data, mask_high)
        strata_metrics["Combined"]["low"] = _rq3_stratum_metrics(pooled_data, mask_low)

    # Build entropy-strata table LaTeX.
    def fmt_delta(value: float) -> str:
        sign = "+" if value > 0 else ""
        return f"{sign}{value:.2f}"

    def fmt_coef(value: float) -> str:
        sign = "+" if value > 0 else ""
        return f"{sign}{value:.2f}"

    def fmt_p(value: float, z: float) -> str:
        if value == 0.0 and z == z:
            log10_p = _two_sided_log10_p_from_z(z)
            if log10_p < -1e6:
                return "\\approx 0"
            return _fmt_p_tex(0.0, z)
        if value != value:
            return "NaN"
        if value < 1e-6:
            return _fmt_p_tex(value, z)
        return f"{value:.3g}"

    def fmt_delta_prec(value: float, digits: int) -> str:
        sign = "+" if value > 0 else ""
        return f"{sign}{value:.{digits}f}"

    table_lines: List[str] = []
    table_lines.append("\\begin{table}[t]")
    table_lines.append("  \\centering")
    table_lines.append("  \\small")
    table_lines.append("  \\setlength{\\tabcolsep}{4pt}")
    table_lines.append("  \\renewcommand{\\arraystretch}{1.05}")
    table_lines.append("  \\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}} l r r r r @{}}")
    table_lines.append("    \\toprule")
    table_lines.append("    \\textbf{Metric} & \\textbf{Xword} & \\textbf{Math} & \\textbf{RHour} & \\textbf{Combined} \\\\")
    table_lines.append("    \\midrule")
    for key, label in strata_labels:
        table_lines.append(f"    \\multicolumn{{5}}{{c}}{{\\textbf{{{label}}}}} \\\\")
        table_lines.append("    \\midrule")
        table_lines.append(
            f"    $N$                       & {_fmt_int_tex(strata_metrics['Xword'][key]['N'])}  & "
            f"{_fmt_int_tex(strata_metrics['Math'][key]['N'])} & {_fmt_int_tex(strata_metrics['RHour'][key]['N'])} & "
            f"{_fmt_int_tex(strata_metrics['Combined'][key]['N'])} \\\\"
        )
        table_lines.append(
            f"    $\\Delta$ (pp)             & ${fmt_delta(strata_metrics['Xword'][key]['delta_pp'])}$   & "
            f"${fmt_delta(strata_metrics['Math'][key]['delta_pp'])}$  & ${fmt_delta(strata_metrics['RHour'][key]['delta_pp'])}$   & "
            f"${fmt_delta(strata_metrics['Combined'][key]['delta_pp'])}$   \\\\"
        )
        table_lines.append(
            f"    $\\mathrm{{coef(shift)}}$    & ${fmt_coef(strata_metrics['Xword'][key]['coef'])}$   & "
            f"${fmt_coef(strata_metrics['Math'][key]['coef'])}$   & ${fmt_coef(strata_metrics['RHour'][key]['coef'])}$  & "
            f"${fmt_coef(strata_metrics['Combined'][key]['coef'])}$  \\\\"
        )
        table_lines.append(
            f"    $p$                       & ${fmt_p(strata_metrics['Xword'][key]['p'], strata_metrics['Xword'][key]['z'])}$     & "
            f"${fmt_p(strata_metrics['Math'][key]['p'], strata_metrics['Math'][key]['z'])}$ & "
            f"${fmt_p(strata_metrics['RHour'][key]['p'], strata_metrics['RHour'][key]['z'])}$ & "
            f"${fmt_p(strata_metrics['Combined'][key]['p'], strata_metrics['Combined'][key]['z'])}$ \\\\"
        )
        if key != "low":
            table_lines.append("    \\midrule")
    table_lines.append("    \\bottomrule")
    table_lines.append("  \\end{tabular*}")
    table_lines.append("  \\caption{\\textbf{Do spontaneous reasoning shifts help under high uncertainty?}\n"
                       "  We stratify traces within each domain by sequence entropy (high = top 20\\% at the within-domain 80th percentile; "
                       "low = bottom 80\\%), and compare shifted vs.\\ non-shifted traces. $\\Delta$ (pp) is the raw accuracy difference\n"
                       "  $\\hat p(\\checkmark\\!\\mid\\!S{=}1) - \\hat p(\\checkmark\\!\\mid\\!S{=}0)$. "
                       "$\\mathrm{coef(shift)}$ and $p$ report the shift coefficient and $p$-value from a logistic regression with problem fixed effects "
                       "and covariates. Across domains, shifts do not become reliably beneficial in the high-entropy regime.}")
    table_lines.append("  \\label{tab:rq3-shift-entropy-strata}")
    table_lines.append("\\end{table}")
    table_entropy_tex = "\n".join(table_lines) + "\n"

    table_entropy_path = Path(args.out_table_entropy_tex)
    _ensure_parent(table_entropy_path)
    table_entropy_path.write_text(table_entropy_tex, encoding="utf-8")

    # Forced-aha table.
    forced_roots = {
        "Xword": roots["Xword"],
        "Math": roots["Math"],
        "RHour": roots["RHour"],
    }
    forced_summary = _compute_forced_aha_summary(forced_roots)
    combo_n = sum(forced_summary[dom]["N"] for dom in forced_summary)
    if combo_n > 0:
        combo_p1 = sum(forced_summary[dom]["p1"] * forced_summary[dom]["N"] for dom in forced_summary) / combo_n
        combo_p2 = sum(forced_summary[dom]["p2"] * forced_summary[dom]["N"] for dom in forced_summary) / combo_n
        combo_wins2 = sum(forced_summary[dom]["wins2"] for dom in forced_summary)
        combo_wins1 = sum(forced_summary[dom]["wins1"] for dom in forced_summary)
        forced_summary["Combined"] = {
            "N": int(combo_n),
            "p1": combo_p1,
            "p2": combo_p2,
            "delta_pp": (combo_p2 - combo_p1) * 100.0,
            "wins2": int(combo_wins2),
            "wins1": int(combo_wins1),
        }

    forced_lines: List[str] = []
    forced_lines.append("\\begin{table}[t]")
    forced_lines.append("  \\centering")
    forced_lines.append("  \\small")
    forced_lines.append("  \\setlength{\\tabcolsep}{4pt}")
    forced_lines.append("  \\renewcommand{\\arraystretch}{1.05}")
    forced_lines.append("  \\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}} l r r r r @{}}")
    forced_lines.append("    \\toprule")
    forced_lines.append("    \\textbf{Metric} & \\textbf{Xword} & \\textbf{Math} & \\textbf{RHour} & \\textbf{Combined} \\\\")
    forced_lines.append("    \\midrule")
    forced_lines.append(
        f"    $N$ (paired samples)      & {_fmt_int_tex(forced_summary['Xword']['N'])}  & "
        f"{_fmt_int_tex(forced_summary['Math']['N'])}  & {_fmt_int_tex(forced_summary['RHour']['N'])} & "
        f"{_fmt_int_tex(forced_summary['Combined']['N'])} \\\\"
    )
    forced_lines.append(
        f"    $\\hat p_{{\\text{{P1}}}}$      & {forced_summary['Xword']['p1']:.4f}    & "
        f"{forced_summary['Math']['p1']:.4f}    & {forced_summary['RHour']['p1']:.6f} & "
        f"{forced_summary['Combined']['p1']:.4f} \\\\"
    )
    forced_lines.append(
        f"    $\\hat p_{{\\text{{P2}}}}$      & {forced_summary['Xword']['p2']:.4f}    & "
        f"{forced_summary['Math']['p2']:.4f}    & {forced_summary['RHour']['p2']:.6f} & "
        f"{forced_summary['Combined']['p2']:.4f} \\\\"
    )
    forced_lines.append(
        f"    $\\Delta$ (pp)             & ${fmt_delta(forced_summary['Xword']['delta_pp'])}$   & "
        f"${fmt_delta(forced_summary['Math']['delta_pp'])}$   & ${fmt_delta(forced_summary['RHour']['delta_pp'])}$ & "
        f"${fmt_delta(forced_summary['Combined']['delta_pp'])}$ \\\\"
    )
    forced_lines.append(
        f"    wins (P2 $\\uparrow$)      & {_fmt_int_tex(forced_summary['Xword']['wins2'])}   & "
        f"{_fmt_int_tex(forced_summary['Math']['wins2'])}  & {_fmt_int_tex(forced_summary['RHour']['wins2'])} & "
        f"{_fmt_int_tex(forced_summary['Combined']['wins2'])} \\\\"
    )
    forced_lines.append(
        f"    wins (P1 $\\uparrow$)      & {_fmt_int_tex(forced_summary['Xword']['wins1'])}   & "
        f"{_fmt_int_tex(forced_summary['Math']['wins1'])}  & {_fmt_int_tex(forced_summary['RHour']['wins1'])} & "
        f"{_fmt_int_tex(forced_summary['Combined']['wins1'])} \\\\"
    )
    forced_lines.append("    \\bottomrule")
    forced_lines.append("  \\end{tabular*}")
    forced_lines.append(
        "  \\caption{\\textbf{Forced ``Aha'' (triggered reconsideration), sample-level results.}\n"
        "  We compare paired outcomes between a baseline generation (Pass~1) and a second generation with an appended "
        "reconsideration cue (Pass~2).\n"
        "  $\\hat p_{\\text{P1}}$ and $\\hat p_{\\text{P2}}$ denote accuracies in each pass; $\\Delta$ (pp) is the percentage-point gain.}"
    )
    forced_lines.append("  \\label{tab:rq3-forced-aha}")
    forced_lines.append("  \\vspace{-4mm}")
    forced_lines.append("\\end{table}")
    forced_tex = "\n".join(forced_lines) + "\n"

    forced_path = Path(args.out_table_forced_tex)
    _ensure_parent(forced_path)
    forced_path.write_text(forced_tex, encoding="utf-8")

    # Narrative numbers for section.
    pooled_or = pooled_stats["or"]
    pooled_beta = pooled_stats["coef"]
    pooled_se = pooled_stats["se"]
    pooled_p = pooled_stats["p"]
    pooled_ci_low = pooled_stats["or_ci_low"]
    pooled_ci_high = pooled_stats["or_ci_high"]
    pooled_N = pooled_stats["N"]

    or_x = shift_entropy_stats["Xword"]["or"]
    or_m = shift_entropy_stats["Math"]["or"]
    or_r = shift_entropy_stats["RHour"]["or"]

    def fmt_or(x: float) -> str:
        return f"{x:.2f}"

    def fmt_beta(x: float) -> str:
        return f"{x:.3f}"

    def fmt_se(x: float) -> str:
        return f"{x:.4f}"

    p_tex = _fmt_p_tex(pooled_p, pooled_stats["z"])
    section = (
        "\\subsection{RQ3: Reasoning Shifts \\& Uncertainty}\n"
        "\\label{sec:rq3-uncertainty}\n\n"
        "The results above (particularly \\textit{Xwords}, see Fig.~\\ref{fig:temp-raw-effect}) suggest that decoding temperature "
        "may modulate the effect of reasoning shifts: at low $T$ they sometimes align with productive corrections, while at higher "
        "$T$ they resemble noise. Because temperature primarily alters sampling entropy rather than the model's underlying reasoning "
        "process~\\citep{hinton2015distilling,holtzman2019degeneration}, this points to a link between shifts and internal uncertainty. "
        "We thus ask whether, under high-uncertainty regimens, reasoning shifts are more frequent or become more helpful.\n\n"
        "\\vspace{1mm}\n"
        "\\noindent\n"
        "\\textbf{Are reasoning shifts more likely under high uncertainty?}\n"
        "To test whether shifts preferentially occur when the model is uncertain, we relate each trace's reasoning shift indicator to "
        "its sequence-level entropy. We pool traces across all decoding temperatures and training checkpoints, and fit a logistic regression "
        "of shift prevalence on standardized entropy with problem fixed effects and cluster-robust SEs (clustered by problem).%\n"
        "\\footnote{In R-style notation:\n"
        "\\(\n"
        "\\texttt{shift} \\sim \\texttt{C(problem)} + \\texttt{std\\_entropy}.\n"
        "\\)\n"
        "Here \\texttt{shift} is a binary indicator for reasoning shift, \\texttt{C(problem)} denotes problem fixed effects, and "
        "\\texttt{std\\_entropy} is the within-domain $z$-scored pass-1 sequence entropy. We estimate a Binomial(logit) GLM with "
        "cluster-robust SEs at the problem level.}\n\n"
        "Pooling \\emph{all} traces across domains (\\textit{Xwords}, \\textit{Math}, \\textit{RHour}), we find that higher entropy is "
        "associated with a higher probability of a shift on average: a $+1$ SD increase in entropy corresponds to an odds ratio of "
        f"$\\approx${fmt_or(pooled_or)}$\\times$ (log-odds $\\beta={fmt_beta(pooled_beta)}$, $\\mathrm{{SE}}={fmt_se(pooled_se)}$, "
        f"$p={p_tex}$; 95\\% CI OR [{fmt_or(pooled_ci_low)}, {fmt_or(pooled_ci_high)}]; $N={_fmt_int_tex(pooled_N)}$). "
        "However, this aggregate effect masks domain heterogeneity: the entropy--shift association is positive in \\textit{Xwords} "
        f"(OR$\\approx${fmt_or(or_x)}$\\times$) and \\textit{{RHour}} (OR$\\approx${fmt_or(or_r)}$\\times$), but negative in \\textit{{Math}} "
        f"(OR$\\approx${fmt_or(or_m)}$\\times$).\n"
        "One possible intuition is that in \\emph{Math}, higher-entropy generations more often reflect diffuse exploration or verbose "
        "``flailing'' rather than a discrete mid-trace pivot, so shifts concentrate in comparatively lower-entropy traces.\n"
        "Across domains, shifts remain rare in absolute terms ($\\approx$"
        f"{shift_pct_min:.2f}\\%--{shift_pct_max:.2f}\\%), so these effects primarily indicate "
        "\\emph{where the rare shifts concentrate} in the entropy distribution, rather than implying that shifts become common.\n\n"
        "\\vspace{1.5mm}\n"
        "\\noindent\n"
        "\\textbf{Do reasoning shifts improve performance under high uncertainty?}\n"
        "A natural hypothesis is that when the model is uncertain, a mid-trace pivot might reflect productive self-correction. We test "
        "this by stratifying traces into \\emph{high-entropy} instances (top 20\\% within domain) and \\emph{low-entropy} instances "
        "(bottom 80\\%), using a fixed entropy threshold per domain. Within each stratum, we estimate the effect of a shift on correctness "
        "using a logistic regression with problem fixed effects and controls for continuous entropy and training step, and report the "
        "shift coefficient alongside the raw accuracy difference between shifted and non-shifted traces.%\n"
        "\\footnote{Within each domain, we split at the 80th percentile of sequence entropy and fit a Binomial(logit) GLM predicting "
        "\\texttt{correct} from \\texttt{shift} with problem fixed effects and covariates. We report both regression and raw contrasts "
        "for interpretability.}\n\n"
        "Table~\\ref{tab:rq3-shift-entropy-strata} shows that shifts do \\emph{not} become reliably beneficial in the high-entropy regime. "
        f"In \\emph{{Math}}, shifts remain harmful even among high-entropy traces (raw $\\Delta={fmt_delta(strata_metrics['Math']['high']['delta_pp'])}$pp) "
        f"and are substantially more harmful in the low-entropy majority (raw $\\Delta={fmt_delta(strata_metrics['Math']['low']['delta_pp'])}$pp). "
        f"In \\emph{{Xwords}}, the point estimate in the high-entropy stratum is near zero (raw $\\Delta={fmt_delta(strata_metrics['Xword']['high']['delta_pp'])}$pp), "
        "but shifts are so rare that estimates are noisy and not statistically distinguishable from zero. "
        "In \\emph{RHour}, accuracy is near-zero throughout, so estimated effects are statistically detectable due to sample size but "
        "negligible in magnitude.\n\n"
        f"\\input{{{table_entropy_path.as_posix()}}}\n\n"
        "\\vspace{1mm}\n"
        "\\noindent\n"
        "\\textbf{Can artificially triggered reasoning shifts improve performance?}\n"
        "The negative results above suggest that \\emph{spontaneous} shifts are not a dependable self-correction mechanism, even when the "
        "model is uncertain.\n"
        "We therefore test an \\emph{extrinsically triggered} ``forced Aha'' intervention: for each prompt we generate a baseline completion "
        "(Pass~1), then re-query the model under identical decoding settings while appending a reconsideration cue (Pass~2), and compare "
        "paired correctness outcomes.\n"
        "Here Pass~1 is the model's initial answer, while Pass~2 appends a short instruction that explicitly asks the model to re-check and "
        "revise if needed (the same cue across all domains; see App.~\\ref{app:uncertainty-interventions} for the exact wording).\n\n"
        "Table~\\ref{tab:rq3-forced-aha} reports sample-level paired results aggregated across checkpoints and decoding temperatures, reported "
        "separately by domain. Forced reconsideration yields a large gain on \\emph{Math} "
        f"($${forced_summary['Math']['p1']:.3f}\\!\\rightarrow\\!{forced_summary['Math']['p2']:.3f}$$; "
        f"${fmt_delta(forced_summary['Math']['delta_pp'])}$pp) and a small gain on Xwords "
        f"(${fmt_delta_prec(forced_summary['Xword']['delta_pp'], 2)}$pp), while remaining negligible in absolute terms on RHour "
        f"(${fmt_delta_prec(forced_summary['RHour']['delta_pp'], 3)}$pp) due to its near-zero base rate.\n"
        "Importantly, the paired ``win'' counts show that improvements dominate backslides in \\emph{Math} "
        f"({_fmt_int_tex(forced_summary['Math']['wins2'])} wrong$\\rightarrow$right vs. "
        f"{_fmt_int_tex(forced_summary['Math']['wins1'])} right$\\rightarrow$wrong), indicating that the effect is not merely random flipping.\n"
        "In contrast, \\emph{Xwords} shows near-balanced wins and losses "
        f"({_fmt_int_tex(forced_summary['Xword']['wins2'])} vs. "
        f"{_fmt_int_tex(forced_summary['Xword']['wins1'])}), consistent with a much smaller net gain. "
        "Across domains, triggered reconsideration improves accuracy (Table~\\ref{tab:rq3-forced-aha}). In \\emph{Math}, these gains are "
        "amplified on high-entropy instances, consistent with uncertainty serving as a useful gate for reflection "
        "(App.~\\ref{app:uncertainty-interventions}, Table~\\ref{tab:cue-regressions}).\n\n"
        "\\vspace{1mm}\n"
        "\\noindent\n"
        "\\textbf{Takeaway.} Reasoning shifts are a low-base-rate event that concentrates in higher-entropy (more uncertain) generations on "
        "average, though the direction and strength of this association varies by domain.\n"
        "Conditioning on uncertainty does not reveal a ``hidden regime'' where spontaneous shifts reliably help. If anything, the primary pattern "
        "is that shifts are least harmful in the high-entropy tail, but they do not turn into a consistent mechanism for improving correctness.\n"
        "Artificially triggering reasoning shifts yields performance gains, particularly in \\emph{Math} under high-entropy.\n\n"
        f"\\input{{{forced_path.as_posix()}}}\n"
    )

    out_section_path = Path(args.out_section_tex)
    _ensure_parent(out_section_path)
    out_section_path.write_text(section + "\n", encoding="utf-8")
    print(section)
    print(f"[ok] wrote: {table_entropy_path}")
    print(f"[ok] wrote: {forced_path}")
    print(f"[ok] wrote: {out_section_path}")
    return 0


def _select_pass2_obj(record: Dict[str, Any], pass2_key: Optional[str]) -> Optional[Dict[str, Any]]:
    if pass2_key:
        candidate = record.get(pass2_key)
        if isinstance(candidate, dict):
            return candidate
        if candidate is not None:
            return None
    for key in ("pass2", "pass2c", "pass2b", "pass2a"):
        candidate = record.get(key)
        if isinstance(candidate, dict):
            return candidate
    return None


def _entropy_gate_stats(
    entropy_arr: "Any",
    pass1_corr: "Any",
    pass2_corr: "Any",
    *,
    mask: "Any",
) -> Dict[str, Any]:
    import numpy as np

    n = int(np.sum(mask))
    if n <= 0:
        return {"n": 0, "acc_p1": 0.0, "acc_p2": 0.0, "delta_pp": 0.0}
    p1 = float(np.mean(pass1_corr[mask]))
    p2 = float(np.mean(pass2_corr[mask]))
    return {"n": n, "acc_p1": p1, "acc_p2": p2, "delta_pp": (p2 - p1) * 100.0}


def _load_entropy_gate_data(
    domain_label: str,
    roots: List[str],
    *,
    split: str,
    min_step: int,
    max_step: int,
    entropy_mode: str,
    pass2_key: Optional[str],
) -> Dict[str, Any]:
    import numpy as np

    entropy: List[float] = []
    p1: List[int] = []
    p2: List[int] = []

    for root in roots:
        files = scan_files_step_only(str(root), split_substr=str(split or ""))
        for file_path in files:
            step_hint = nat_step_from_path(file_path)
            if step_hint is not None and (step_hint < min_step or step_hint > max_step):
                continue
            for record in iter_records_from_file(file_path):
                if split and str(record.get("split", "")).lower() != split.lower():
                    continue
                pass1, step_val = extract_pass1_and_step(record, step_hint)
                if not pass1 or step_val is None:
                    continue
                if step_val < min_step or step_val > max_step:
                    continue
                ent = entropy_from_pass1(pass1, mode=entropy_mode)
                if ent is None:
                    continue
                correct1 = extract_correct(pass1, record)
                if correct1 is None:
                    continue
                pass2_obj = _select_pass2_obj(record, pass2_key)
                if not pass2_obj:
                    continue
                correct2 = extract_correct(pass2_obj, record)
                if correct2 is None:
                    continue

                entropy.append(float(ent))
                p1.append(int(correct1))
                p2.append(int(correct2))

    if not entropy:
        raise SystemExit(f"No pass2 entropy-gate rows found for domain={domain_label}")

    entropy_arr = np.asarray(entropy, dtype=float)
    p1_arr = np.asarray(p1, dtype=int)
    p2_arr = np.asarray(p2, dtype=int)
    q80 = float(np.quantile(entropy_arr, 0.8))

    mask_all = np.ones_like(entropy_arr, dtype=bool)
    mask_high = entropy_arr >= q80
    mask_low = entropy_arr < q80

    return {
        "entropy": entropy_arr,
        "pass1": p1_arr,
        "pass2": p2_arr,
        "q80": q80,
        "stats": {
            "all": _entropy_gate_stats(entropy_arr, p1_arr, p2_arr, mask=mask_all),
            "high": _entropy_gate_stats(entropy_arr, p1_arr, p2_arr, mask=mask_high),
            "low": _entropy_gate_stats(entropy_arr, p1_arr, p2_arr, mask=mask_low),
        },
    }


def _load_pass2_entropy_regression_data(
    domain_label: str,
    roots: List[str],
    *,
    split: str,
    min_step: int,
    max_step: int,
    entropy_mode: str,
    pass2_key: Optional[str],
) -> Dict[str, Any]:
    import numpy as np

    entropy: List[float] = []
    p1: List[int] = []
    p2: List[int] = []
    problem_keys: List[str] = []

    for root in roots:
        files = scan_files_step_only(str(root), split_substr=str(split or ""))
        for file_path in files:
            step_hint = nat_step_from_path(file_path)
            if step_hint is not None and (step_hint < min_step or step_hint > max_step):
                continue
            for record in iter_records_from_file(file_path):
                if split and str(record.get("split", "")).lower() != split.lower():
                    continue
                pass1, step_val = extract_pass1_and_step(record, step_hint)
                if not pass1 or step_val is None:
                    continue
                if step_val < min_step or step_val > max_step:
                    continue
                ent = entropy_from_pass1(pass1, mode=entropy_mode)
                if ent is None:
                    continue
                correct1 = extract_correct(pass1, record)
                if correct1 is None:
                    continue
                pass2_obj = _select_pass2_obj(record, pass2_key)
                if not pass2_obj:
                    continue
                correct2 = extract_correct(pass2_obj, record)
                if correct2 is None:
                    continue

                entropy.append(float(ent))
                p1.append(int(correct1))
                p2.append(int(correct2))
                problem = problem_key_from_record(record, "unknown")
                problem_keys.append(f"{domain_label}:{problem}")

    if not entropy:
        raise SystemExit(f"No pass2 entropy-regression rows found for domain={domain_label}")

    entropy_arr = np.asarray(entropy, dtype=float)
    p1_arr = np.asarray(p1, dtype=int)
    p2_arr = np.asarray(p2, dtype=int)
    mean_ent = float(entropy_arr.mean())
    std_ent = float(entropy_arr.std(ddof=0) + 1e-8)
    entropy_std = (entropy_arr - mean_ent) / std_ent
    pid = _make_problem_ids(problem_keys)

    return {
        "entropy": entropy_arr,
        "entropy_std": entropy_std,
        "entropy_mean": mean_ent,
        "entropy_stddev": std_ent,
        "pass1": p1_arr,
        "pass2": p2_arr,
        "problem_keys": problem_keys,
        "problem_ids": pid,
    }


def _fit_pass2_entropy_fe(data: Dict[str, Any]) -> Dict[str, Any]:
    import numpy as np

    entropy_std = np.asarray(data["entropy_std"], dtype=float)
    pass1 = np.asarray(data["pass1"], dtype=int)
    pass2 = np.asarray(data["pass2"], dtype=int)
    pid = np.asarray(data["problem_ids"], dtype=int)

    n = int(entropy_std.size)
    if n == 0:
        return {
            "N": n,
            "beta": float("nan"),
            "or": float("nan"),
            "ame": float("nan"),
            "p": float("nan"),
            "z": float("nan"),
        }

    X = np.column_stack([entropy_std.astype(float), pass1.astype(float)])
    beta, se, z, alpha = _fit_fe_logit_irls_multi(pass2.astype(float), X, pid)
    b_ent = float(beta[0]) if beta.size > 0 else float("nan")
    z_ent = float(z[0]) if z.size > 0 else float("nan")

    eta = alpha[pid] + X @ beta
    eta0 = eta - b_ent * X[:, 0]
    eta1 = eta0 + b_ent
    ame = float((_sigmoid(eta1) - _sigmoid(eta0)).mean())

    log10_p = _two_sided_log10_p_from_z(z_ent) if z_ent == z_ent else float("nan")
    p_val = float(10 ** log10_p) if log10_p > -300 else 0.0

    return {
        "N": n,
        "beta": b_ent,
        "or": float(math.exp(b_ent)),
        "ame": ame,
        "p": p_val,
        "z": z_ent,
    }


def cmd_pass2_entropy_gate(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="master_analysis.py pass2-entropy-gate",
        description="Summarize pass-2 accuracy gains stratified by pass-1 entropy.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Record-level split filter (default: test).",
    )
    parser.add_argument(
        "--min_step",
        type=int,
        default=0,
        help="Minimum checkpoint step to include (inclusive, default: 0).",
    )
    parser.add_argument(
        "--max_step",
        type=int,
        default=10**9,
        help="Maximum checkpoint step to include (inclusive, default: 1e9).",
    )
    parser.add_argument(
        "--entropy_mode",
        default="combined",
        choices=["combined", "answer", "think", "full"],
        help="Pass-1 entropy mode (default: combined).",
    )
    parser.add_argument(
        "--pass2_key",
        default="pass2",
        help="Pass-2 key to use when present (default: pass2).",
    )
    parser.add_argument(
        "--out_json",
        default="artifacts/analysis/pass2_entropy_gate.json",
        help="Write summary rows to this JSON file.",
    )
    parser.add_argument(
        "--out_csv",
        default="artifacts/analysis/pass2_entropy_gate.csv",
        help="Write summary rows to this CSV file.",
    )
    parser.add_argument(
        "--no_summary",
        action="store_true",
        help="Disable printing the summary table to stdout.",
    )
    parser.add_argument(
        "--print_summary_only",
        action="store_true",
        help="Only print the summary (skip writing JSON/CSV).",
    )
    args = parser.parse_args(argv)

    roots = {
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

    summary_rows: List[Dict[str, Any]] = []
    domain_stats: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for domain_label, domain_roots in roots.items():
        data = _load_entropy_gate_data(
            domain_label,
            domain_roots,
            split=str(args.split or ""),
            min_step=int(args.min_step),
            max_step=int(args.max_step),
            entropy_mode=str(args.entropy_mode),
            pass2_key=str(args.pass2_key or ""),
        )
        stats = data["stats"]
        domain_stats[domain_label] = stats
        for bucket_key, label in (
            ("all", "all"),
            ("high", "high"),
            ("low", "low"),
        ):
            row = stats[bucket_key]
            summary_rows.append(
                {
                    "domain": domain_label,
                    "bucket": label,
                    "n": int(row["n"]),
                    "acc_pass1_pct": float(row["acc_p1"] * 100.0),
                    "acc_pass2_pct": float(row["acc_p2"] * 100.0),
                    "delta_pp": float(row["delta_pp"]),
                },
            )

    if domain_stats:
        for bucket_key, label in (
            ("all", "all"),
            ("high", "high"),
            ("low", "low"),
        ):
            total_n = 0
            sum_p1 = 0.0
            sum_p2 = 0.0
            for stats in domain_stats.values():
                row = stats.get(bucket_key, {})
                n = int(row.get("n", 0) or 0)
                total_n += n
                sum_p1 += float(row.get("acc_p1", 0.0)) * n
                sum_p2 += float(row.get("acc_p2", 0.0)) * n
            if total_n <= 0:
                continue
            acc_p1 = sum_p1 / total_n
            acc_p2 = sum_p2 / total_n
            summary_rows.append(
                {
                    "domain": "ALL",
                    "bucket": label,
                    "n": int(total_n),
                    "acc_pass1_pct": float(acc_p1 * 100.0),
                    "acc_pass2_pct": float(acc_p2 * 100.0),
                    "delta_pp": float((acc_p2 - acc_p1) * 100.0),
                },
            )

    if not args.no_summary:
        headers = ["domain", "bucket", "N", "acc_p1_%", "acc_p2_%", "delta_pp"]
        table_rows: List[List[str]] = []
        for row in summary_rows:
            table_rows.append(
                [
                    str(row["domain"]),
                    str(row["bucket"]),
                    str(row["n"]),
                    f"{float(row['acc_pass1_pct']):.2f}",
                    f"{float(row['acc_pass2_pct']):.2f}",
                    f"{float(row['delta_pp']):.2f}",
                ],
            )
        _print_aligned_table(headers, table_rows, right_align=[2, 3, 4, 5])

    if args.print_summary_only:
        return 0

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    _ensure_parent(out_json)
    _ensure_parent(out_csv)

    import json  # local import

    payload = {
        "meta": {
            "split": str(args.split),
            "min_step": int(args.min_step),
            "max_step": int(args.max_step),
            "entropy_mode": str(args.entropy_mode),
            "pass2_key": str(args.pass2_key),
            "roots": roots,
        },
        "summary": summary_rows,
    }
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fieldnames = ["domain", "bucket", "n", "acc_pass1_pct", "acc_pass2_pct", "delta_pp"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"[ok] json: {out_json}")
    print(f"[ok] csv : {out_csv}")
    return 0


def cmd_pass2_entropy_regression(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="master_analysis.py pass2-entropy-regression",
        description="Run pass-2 correctness regressions against high-entropy indicators.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Record-level split filter (default: test).",
    )
    parser.add_argument(
        "--min_step",
        type=int,
        default=0,
        help="Minimum checkpoint step to include (inclusive, default: 0).",
    )
    parser.add_argument(
        "--max_step",
        type=int,
        default=10**9,
        help="Maximum checkpoint step to include (inclusive, default: 1e9).",
    )
    parser.add_argument(
        "--max_step_7b8b",
        type=int,
        default=450,
        help="Max checkpoint step for Qwen7B/Llama8B (default: 450).",
    )
    parser.add_argument(
        "--entropy_mode",
        default="combined",
        choices=["combined", "answer", "think", "full"],
        help="Pass-1 entropy mode (default: combined).",
    )
    parser.add_argument(
        "--pass2_key",
        default="pass2",
        help="Pass-2 key to use when present (default: pass2).",
    )
    parser.add_argument(
        "--out_table_tex",
        default="latex/table_pass2_entropy_regression.tex",
        help="Write the Qwen1.5B table to this path.",
    )
    parser.add_argument(
        "--out_table_7b8b_tex",
        default="latex/table_pass2_entropy_regression_7b8b.tex",
        help="Write the Qwen7B/Llama8B table to this path.",
    )
    parser.add_argument(
        "--out_json",
        default="artifacts/analysis/pass2_entropy_regression.json",
        help="Write regression rows to this JSON file.",
    )
    parser.add_argument(
        "--out_csv",
        default="artifacts/analysis/pass2_entropy_regression.csv",
        help="Write regression rows to this CSV file.",
    )
    parser.add_argument(
        "--no_summary",
        action="store_true",
        help="Disable printing the summary table to stdout.",
    )
    parser.add_argument(
        "--print_summary_only",
        action="store_true",
        help="Only print the summary (skip writing JSON/CSV).",
    )
    args = parser.parse_args(argv)

    import numpy as np

    roots = {
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

    stats_by_domain: Dict[str, Dict[str, Any]] = {}
    data_by_domain: Dict[str, Dict[str, Any]] = {}
    for domain_label, domain_roots in roots.items():
        data = _load_pass2_entropy_regression_data(
            domain_label,
            domain_roots,
            split=str(args.split or ""),
            min_step=int(args.min_step),
            max_step=int(args.max_step),
            entropy_mode=str(args.entropy_mode),
            pass2_key=str(args.pass2_key or ""),
        )
        data_by_domain[domain_label] = data
        stats_by_domain[domain_label] = _fit_pass2_entropy_fe(data)

    if data_by_domain:
        combined_entropy: List["Any"] = []
        combined_pass1: List["Any"] = []
        combined_pass2: List["Any"] = []
        combined_keys: List[str] = []
        for label, data in data_by_domain.items():
            combined_entropy.append(data["entropy"])
            combined_pass1.append(data["pass1"])
            combined_pass2.append(data["pass2"])
            combined_keys.extend(data["problem_keys"])
        pooled_entropy = np.concatenate(combined_entropy)
        pooled_mean = float(pooled_entropy.mean())
        pooled_std = float(pooled_entropy.std(ddof=0) + 1e-8)
        combined_data = {
            "entropy": pooled_entropy,
            "entropy_std": (pooled_entropy - pooled_mean) / pooled_std,
            "pass1": np.concatenate(combined_pass1),
            "pass2": np.concatenate(combined_pass2),
            "problem_ids": _make_problem_ids(combined_keys),
        }
        stats_by_domain["Combined"] = _fit_pass2_entropy_fe(combined_data)

    specs = {spec.key: spec for spec in build_default_grid()}

    def roots_for(key: str) -> List[str]:
        roots_list: List[str] = []
        for t in DEFAULT_TEMPS:
            root = resolve_results_root(Path("artifacts/results"), specs[key], float(t))
            if root is None:
                raise SystemExit(f"Missing results root for {key} at temp={t}")
            roots_list.append(str(root))
        return roots_list

    model_roots = {
        "Qwen2.5-7B": roots_for("qwen7b_math"),
        "Llama3.1-8B": roots_for("llama8b_math"),
    }

    stats_by_model: Dict[str, Dict[str, Any]] = {}
    data_by_model: Dict[str, Dict[str, Any]] = {}
    for model_label, model_roots_list in model_roots.items():
        data = _load_pass2_entropy_regression_data(
            model_label,
            model_roots_list,
            split=str(args.split or ""),
            min_step=int(args.min_step),
            max_step=int(args.max_step_7b8b),
            entropy_mode=str(args.entropy_mode),
            pass2_key=str(args.pass2_key or ""),
        )
        data_by_model[model_label] = data
        stats_by_model[model_label] = _fit_pass2_entropy_fe(data)

    if data_by_model:
        combined_entropy = []
        combined_pass1 = []
        combined_pass2 = []
        combined_keys = []
        for label, data in data_by_model.items():
            combined_entropy.append(data["entropy"])
            combined_pass1.append(data["pass1"])
            combined_pass2.append(data["pass2"])
            combined_keys.extend(data["problem_keys"])
        pooled_entropy = np.concatenate(combined_entropy)
        pooled_mean = float(pooled_entropy.mean())
        pooled_std = float(pooled_entropy.std(ddof=0) + 1e-8)
        combined_data = {
            "entropy": pooled_entropy,
            "entropy_std": (pooled_entropy - pooled_mean) / pooled_std,
            "pass1": np.concatenate(combined_pass1),
            "pass2": np.concatenate(combined_pass2),
            "problem_ids": _make_problem_ids(combined_keys),
        }
        stats_by_model["Combined"] = _fit_pass2_entropy_fe(combined_data)

    def fmt_or_dash(value: float, fmt: str) -> str:
        if value != value:
            return "--"
        return fmt.format(value)

    def fmt_p(value: float, z: float) -> str:
        if value != value or z != z:
            return "--"
        return _fmt_p_tex(value, z)

    if not args.no_summary:
        headers = ["group", "N", "beta", "OR_1sd", "AME_pp", "p"]
        rows: List[List[str]] = []
        for label in ("Xword", "Math", "RHour", "Combined"):
            stats = stats_by_domain.get(label, {})
            rows.append(
                [
                    label,
                    str(int(stats.get("N", 0) or 0)),
                    fmt_or_dash(float(stats.get("beta", float("nan"))), "{:+.3f}"),
                    fmt_or_dash(float(stats.get("or", float("nan"))), "{:.2f}"),
                    fmt_or_dash(float(stats.get("ame", float("nan")) * 100.0), "{:+.2f}"),
                    fmt_p(float(stats.get("p", float("nan"))), float(stats.get("z", float("nan")))),
                ],
            )
        _print_aligned_table(headers, rows, right_align=[1, 2, 3, 4, 5])

        rows = []
        for label in ("Qwen2.5-7B", "Llama3.1-8B", "Combined"):
            stats = stats_by_model.get(label, {})
            rows.append(
                [
                    label,
                    str(int(stats.get("N", 0) or 0)),
                    fmt_or_dash(float(stats.get("beta", float("nan"))), "{:+.3f}"),
                    fmt_or_dash(float(stats.get("or", float("nan"))), "{:.2f}"),
                    fmt_or_dash(float(stats.get("ame", float("nan")) * 100.0), "{:+.2f}"),
                    fmt_p(float(stats.get("p", float("nan"))), float(stats.get("z", float("nan")))),
                ],
            )
        _print_aligned_table(headers, rows, right_align=[1, 2, 3, 4, 5])

    if args.print_summary_only:
        return 0

    rows = []
    for label, stats in stats_by_domain.items():
        rows.append(
            {
                "group": label,
                "scope": "qwen15b",
                "n": int(stats.get("N", 0) or 0),
                "beta": float(stats.get("beta", float("nan"))),
                "or": float(stats.get("or", float("nan"))),
                "ame_pp": float(stats.get("ame", float("nan")) * 100.0),
                "p": float(stats.get("p", float("nan"))),
                "z": float(stats.get("z", float("nan"))),
            },
        )
    for label, stats in stats_by_model.items():
        rows.append(
            {
                "group": label,
                "scope": "qwen7b_llama8b",
                "n": int(stats.get("N", 0) or 0),
                "beta": float(stats.get("beta", float("nan"))),
                "or": float(stats.get("or", float("nan"))),
                "ame_pp": float(stats.get("ame", float("nan")) * 100.0),
                "p": float(stats.get("p", float("nan"))),
                "z": float(stats.get("z", float("nan"))),
            },
        )

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    _ensure_parent(out_json)
    _ensure_parent(out_csv)

    import json  # local import

    payload = {
        "meta": {
            "split": str(args.split),
            "min_step": int(args.min_step),
            "max_step": int(args.max_step),
            "max_step_7b8b": int(args.max_step_7b8b),
            "entropy_mode": str(args.entropy_mode),
            "pass2_key": str(args.pass2_key),
        },
        "rows": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    fieldnames = ["group", "scope", "n", "beta", "or", "ame_pp", "p", "z"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    def fmt_cell(value: float, fmt: str) -> str:
        if value != value:
            return "--"
        return fmt.format(value)

    table_lines: List[str] = []
    table_lines.append("\\begin{table}[t]")
    table_lines.append("\\centering")
    table_lines.append("\\small")
    table_lines.append("\\setlength{\\tabcolsep}{4pt}")
    table_lines.append("\\renewcommand{\\arraystretch}{1.05}")
    table_lines.append("\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}} l r r r @{}}")
    table_lines.append("\\toprule")
    table_lines.append("\\textbf{Metric} & \\textbf{Xword} & \\textbf{Math} & \\textbf{RHour} & \\textbf{Combined} \\\\")
    table_lines.append("\\midrule")
    table_lines.append(
        f"$N$ & {_fmt_int_tex(stats_by_domain['Xword']['N'])} & {_fmt_int_tex(stats_by_domain['Math']['N'])} & "
        f"{_fmt_int_tex(stats_by_domain['RHour']['N'])} & {_fmt_int_tex(stats_by_domain['Combined']['N'])} \\\\"
    )
    table_lines.append(
        f"$\\beta_{{\\mathrm{{ent}}}}$ & {fmt_cell(stats_by_domain['Xword']['beta'], '{:+.3f}')} & "
        f"{fmt_cell(stats_by_domain['Math']['beta'], '{:+.3f}')} & {fmt_cell(stats_by_domain['RHour']['beta'], '{:+.3f}')} & "
        f"{fmt_cell(stats_by_domain['Combined']['beta'], '{:+.3f}')} \\\\"
    )
    table_lines.append(
        f"$\\mathrm{{OR}}_{{1\\sigma}}$ & {fmt_cell(stats_by_domain['Xword']['or'], '{:.2f}')} & "
        f"{fmt_cell(stats_by_domain['Math']['or'], '{:.2f}')} & {fmt_cell(stats_by_domain['RHour']['or'], '{:.2f}')} & "
        f"{fmt_cell(stats_by_domain['Combined']['or'], '{:.2f}')} \\\\"
    )
    table_lines.append(
        f"$\\mathrm{{AME}}$ (pp) & {fmt_cell(stats_by_domain['Xword']['ame'] * 100.0, '{:+.2f}')} & "
        f"{fmt_cell(stats_by_domain['Math']['ame'] * 100.0, '{:+.2f}')} & {fmt_cell(stats_by_domain['RHour']['ame'] * 100.0, '{:+.2f}')} & "
        f"{fmt_cell(stats_by_domain['Combined']['ame'] * 100.0, '{:+.2f}')} \\\\"
    )
    table_lines.append(
        f"$p$ & {fmt_p(stats_by_domain['Xword']['p'], stats_by_domain['Xword']['z'])} & "
        f"{fmt_p(stats_by_domain['Math']['p'], stats_by_domain['Math']['z'])} & "
        f"{fmt_p(stats_by_domain['RHour']['p'], stats_by_domain['RHour']['z'])} & "
        f"{fmt_p(stats_by_domain['Combined']['p'], stats_by_domain['Combined']['z'])} \\\\"
    )
    table_lines.append("\\bottomrule")
    table_lines.append("\\end{tabular*}")
    table_lines.append(
        "\\caption{\\textbf{Pass-2 accuracy rises with pass-1 entropy (Qwen2.5-1.5B).} "
        "We regress pass-2 correctness on standardized pass-1 entropy, controlling for pass-1 correctness and problem fixed effects. "
        "$\\beta_{\\mathrm{ent}}$ is the log-odds coefficient for a 1 SD entropy increase; "
        "$\\mathrm{OR}_{1\\sigma}=\\exp(\\beta_{\\mathrm{ent}})$; and $\\mathrm{AME}$ is the average marginal effect in percentage points. "
        "Combined rows re-standardize entropy using a pooled mean/SD across domains.}"
    )
    table_lines.append("\\label{tab:pass2-entropy-regression}")
    table_lines.append("\\end{table}")
    table_tex = "\n".join(table_lines) + "\n"

    table7_lines: List[str] = []
    table7_lines.append("\\begin{table}[t]")
    table7_lines.append("\\centering")
    table7_lines.append("\\small")
    table7_lines.append("\\setlength{\\tabcolsep}{4pt}")
    table7_lines.append("\\renewcommand{\\arraystretch}{1.05}")
    table7_lines.append("\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}} l r r r @{}}")
    table7_lines.append("\\toprule")
    table7_lines.append("\\textbf{Metric} & \\textbf{Qwen2.5-7B} & \\textbf{Llama3.1-8B} & \\textbf{Combined} \\\\")
    table7_lines.append("\\midrule")
    table7_lines.append(
        f"$N$ & {_fmt_int_tex(stats_by_model['Qwen2.5-7B']['N'])} & "
        f"{_fmt_int_tex(stats_by_model['Llama3.1-8B']['N'])} & {_fmt_int_tex(stats_by_model['Combined']['N'])} \\\\"
    )
    table7_lines.append(
        f"$\\beta_{{\\mathrm{{ent}}}}$ & {fmt_cell(stats_by_model['Qwen2.5-7B']['beta'], '{:+.3f}')} & "
        f"{fmt_cell(stats_by_model['Llama3.1-8B']['beta'], '{:+.3f}')} & {fmt_cell(stats_by_model['Combined']['beta'], '{:+.3f}')} \\\\"
    )
    table7_lines.append(
        f"$\\mathrm{{OR}}_{{1\\sigma}}$ & {fmt_cell(stats_by_model['Qwen2.5-7B']['or'], '{:.2f}')} & "
        f"{fmt_cell(stats_by_model['Llama3.1-8B']['or'], '{:.2f}')} & {fmt_cell(stats_by_model['Combined']['or'], '{:.2f}')} \\\\"
    )
    table7_lines.append(
        f"$\\mathrm{{AME}}$ (pp) & {fmt_cell(stats_by_model['Qwen2.5-7B']['ame'] * 100.0, '{:+.2f}')} & "
        f"{fmt_cell(stats_by_model['Llama3.1-8B']['ame'] * 100.0, '{:+.2f}')} & {fmt_cell(stats_by_model['Combined']['ame'] * 100.0, '{:+.2f}')} \\\\"
    )
    table7_lines.append(
        f"$p$ & {fmt_p(stats_by_model['Qwen2.5-7B']['p'], stats_by_model['Qwen2.5-7B']['z'])} & "
        f"{fmt_p(stats_by_model['Llama3.1-8B']['p'], stats_by_model['Llama3.1-8B']['z'])} & "
        f"{fmt_p(stats_by_model['Combined']['p'], stats_by_model['Combined']['z'])} \\\\"
    )
    table7_lines.append("\\bottomrule")
    table7_lines.append("\\end{tabular*}")
    table7_lines.append(
        "\\caption{\\textbf{Pass-2 accuracy rises with pass-1 entropy (Qwen2.5-7B/Llama3.1-8B).} "
        "We regress pass-2 correctness on standardized pass-1 entropy, controlling for pass-1 correctness "
        "and problem fixed effects. Combined rows re-standardize entropy using a pooled mean/SD across models.}"
    )
    table7_lines.append("\\label{tab:pass2-entropy-regression-7b8b}")
    table7_lines.append("\\end{table}")
    table7_tex = "\n".join(table7_lines) + "\n"

    out_table = Path(args.out_table_tex)
    out_table_7b8b = Path(args.out_table_7b8b_tex)
    _ensure_parent(out_table)
    _ensure_parent(out_table_7b8b)
    out_table.write_text(table_tex, encoding="utf-8")
    out_table_7b8b.write_text(table7_tex, encoding="utf-8")

    print(table_tex)
    print(table7_tex)
    print(f"[ok] wrote: {out_table}")
    print(f"[ok] wrote: {out_table_7b8b}")
    print(f"[ok] wrote: {out_json}")
    print(f"[ok] wrote: {out_csv}")
    return 0


def cmd_appendix_qwen_llama(argv: Optional[List[str]] = None) -> int:
    """
    Emit appendix subsection for Qwen2.5-7B and Llama3.1-8B analyses.
    """
    parser = argparse.ArgumentParser(
        prog="master_analysis.py appendix-qwen-llama",
        description="Write appendix LaTeX for Qwen2.5-7B/Llama3.1-8B regressions.",
    )
    parser.add_argument(
        "--template",
        default="latex/appendix_qwen_llama_template.tex",
        help="Template LaTeX file with placeholders.",
    )
    parser.add_argument(
        "--out_section_tex",
        default="latex/section_app_qwen_llama.tex",
        help="Write the appendix section to this path (default: latex/section_app_qwen_llama.tex).",
    )
    parser.add_argument(
        "--out_table_rs_tex",
        default="latex/table_rs_7b8b.tex",
        help="Write the rs-7b8b table to this path (default: latex/table_rs_7b8b.tex).",
    )
    parser.add_argument(
        "--out_table_entropy_tex",
        default="latex/table_shift_entropy_strata_7b8b.tex",
        help="Write the entropy-strata table to this path (default: latex/table_shift_entropy_strata_7b8b.tex).",
    )
    parser.add_argument(
        "--out_table_forced_tex",
        default="latex/table_forced_aha_math_7b8b.tex",
        help="Write the forced-aha table to this path (default: latex/table_forced_aha_math_7b8b.tex).",
    )
    parser.add_argument(
        "--out_table_external_tex",
        default="latex/table_external_models_appendix.tex",
        help="Write the external-model appendix table to this path (default: latex/table_external_models_appendix.tex).",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Split filter (default: test).",
    )
    parser.add_argument(
        "--min_step",
        type=int,
        default=0,
        help="Minimum checkpoint step (default: 0).",
    )
    parser.add_argument(
        "--max_step",
        type=int,
        default=450,
        help="Maximum checkpoint step (default: 450).",
    )
    parser.add_argument(
        "--stage_temp",
        type=float,
        default=0.7,
        help="Training-stage analysis temperature (default: 0.7).",
    )
    parser.add_argument(
        "--entropy_mode",
        default="combined",
        choices=["combined", "sum", "think", "answer"],
        help="Pass-1 entropy mode (default: combined).",
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
    args = parser.parse_args(argv)

    template = _read_text_or_die(str(args.template))
    split = str(args.split or "")
    min_step = int(args.min_step)
    max_step = int(args.max_step)
    stage_temp = float(args.stage_temp)
    gate_by_words = not args.no_words_gate

    # Ensure figure PDFs exist at the paths referenced by the appendix.
    import shutil

    def copy_if_exists(src: Path, dst: Path) -> None:
        if src.exists() and not dst.exists():
            _ensure_parent(dst)
            shutil.copyfile(src, dst)

    copy_if_exists(
        Path("paper_figs/aha_heatmaps/Q7B_L8B_Math_T0.7/aha_heatmap_overall.pdf"),
        Path("latex/aha_heatmaps/Q7B_L8B_Math_T0.7/aha_heatmap_overall.pdf"),
    )
    copy_if_exists(
        Path("paper_figs/raw_effects/raw_effect_per_step_overlay_linear.pdf"),
        Path("latex/raw_effect_per_step_overlay_linear-llama.pdf"),
    )
    copy_if_exists(
        Path("paper_figs/temperature_effects/raw_effects_plot__MIXED__7B_vs_8B.pdf"),
        Path("latex/raw_effects_plot__MIXED__7B_vs_8B.pdf"),
    )

    specs = {spec.key: spec for spec in build_default_grid()}

    def root_for(key: str, temp_value: float) -> Path:
        spec = specs.get(key)
        if spec is None:
            raise SystemExit(f"Unknown experiment key: {key}")
        root = resolve_results_root(Path("artifacts/results"), spec, float(temp_value))
        if root is None:
            raise SystemExit(f"Missing results root for {key} at temp={temp_value}")
        return root

    model_keys = {
        "Qwen2.5-7B": "qwen7b_math",
        "Llama3.1-8B": "llama8b_math",
    }

    import pandas as pd

    stage_metrics: Dict[str, Dict[str, Any]] = {}
    stage_dfs: Dict[str, "pd.DataFrame"] = {}
    for label, key in model_keys.items():
        root = root_for(key, stage_temp)
        df = _load_rows_for_root(
            str(root),
            split=split,
            min_step=min_step,
            max_step=max_step,
            temp_value=stage_temp,
        )
        stage_dfs[label] = df
        stage_metrics[label] = _compute_rq2_domain_metrics(
            df=df,
            formula="correct ~ C(problem) + step_std + shift",
            std_src_col="step",
            std_dest_col="step_std",
            cluster_by="problem",
            use_statsmodels=False,
        )
    if stage_dfs:
        combined_df = pd.concat(stage_dfs.values(), ignore_index=True)
        stage_metrics["Combined"] = _compute_rq2_domain_metrics(
            df=combined_df,
            formula="correct ~ C(problem) + step_std + shift",
            std_src_col="step",
            std_dest_col="step_std",
            cluster_by="problem",
            use_statsmodels=False,
        )

    temp_metrics: Dict[str, Dict[str, Any]] = {}
    temp_dfs: Dict[str, "pd.DataFrame"] = {}
    for label, key in model_keys.items():
        dfs = []
        for t in DEFAULT_TEMPS:
            root = root_for(key, float(t))
            dfs.append(
                _load_rows_for_root(
                    str(root),
                    split=split,
                    min_step=min_step,
                    max_step=max_step,
                    temp_value=float(t),
                )
            )
        df_all = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        temp_dfs[label] = df_all
        temp_metrics[label] = _compute_rq2_domain_metrics(
            df=df_all,
            formula="correct ~ C(problem) + temp_std + shift",
            std_src_col="temp",
            std_dest_col="temp_std",
            cluster_by="problem",
            use_statsmodels=False,
        )
    if temp_dfs:
        combined_df = pd.concat(temp_dfs.values(), ignore_index=True)
        temp_metrics["Combined"] = _compute_rq2_domain_metrics(
            df=combined_df,
            formula="correct ~ C(problem) + temp_std + shift",
            std_src_col="temp",
            std_dest_col="temp_std",
            cluster_by="problem",
            use_statsmodels=False,
        )

    def fmt_shift_pct_short(value: float) -> str:
        return f"{float(value):.2f}"

    def fmt_prob4(value: float) -> str:
        return f"{float(value):.4f}"

    rs_lines: List[str] = []
    rs_lines.append("\\begin{table}[h]")
    rs_lines.append("\\centering")
    rs_lines.append("\\small")
    rs_lines.append("\\setlength{\\tabcolsep}{4pt}")
    rs_lines.append("\\renewcommand{\\arraystretch}{1.05}")
    rs_lines.append("\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}} l r r r @{}}")
    rs_lines.append("\\toprule")
    rs_lines.append("\\multicolumn{4}{c}{\\textbf{(a) Training stage}} \\\\")
    rs_lines.append("\\midrule")
    rs_lines.append("\\textbf{Metric} & \\textbf{Qwen2.5-7B} & \\textbf{Llama3.1-8B} & \\textbf{Combined} \\\\")
    rs_lines.append("\\midrule")
    rs_lines.append(
        f"$N$                       & {_fmt_int_tex(stage_metrics['Qwen2.5-7B']['N'])} & "
        f"{_fmt_int_tex(stage_metrics['Llama3.1-8B']['N'])} & {_fmt_int_tex(stage_metrics['Combined']['N'])} \\\\"
    )
    rs_lines.append(
        f"$\\%S$                     & {fmt_shift_pct_short(stage_metrics['Qwen2.5-7B']['shift_pct'])}     & "
        f"{fmt_shift_pct_short(stage_metrics['Llama3.1-8B']['shift_pct'])}    & "
        f"{fmt_shift_pct_short(stage_metrics['Combined']['shift_pct'])}    \\\\"
    )
    rs_lines.append(
        f"$\\hat{{p}}_{{Y\\mid S=1}}$     & {fmt_prob4(stage_metrics['Qwen2.5-7B']['p_shift'])}   & "
        f"{fmt_prob4(stage_metrics['Llama3.1-8B']['p_shift'])}  & "
        f"{fmt_prob4(stage_metrics['Combined']['p_shift'])}  \\\\"
    )
    rs_lines.append(
        f"$\\Delta\\%$                & ${_fmt_delta_pp(stage_metrics['Qwen2.5-7B']['delta_pp'])}$ & "
        f"${_fmt_delta_pp(stage_metrics['Llama3.1-8B']['delta_pp'])}$ & "
        f"${_fmt_delta_pp(stage_metrics['Combined']['delta_pp'])}$ \\\\"
    )
    rs_lines.append(
        f"$\\mathrm{{AME}}$            & ${_fmt_ame(stage_metrics['Qwen2.5-7B']['ame'])}$ & "
        f"${_fmt_ame(stage_metrics['Llama3.1-8B']['ame'])}$ & "
        f"${_fmt_ame(stage_metrics['Combined']['ame'])}$ \\\\"
    )
    rs_lines.append(
        f"$p$                       & ${_fmt_p_tex(stage_metrics['Qwen2.5-7B']['p'], stage_metrics['Qwen2.5-7B']['z'])}$       & "
        f"${_fmt_p_tex(stage_metrics['Llama3.1-8B']['p'], stage_metrics['Llama3.1-8B']['z'])}$ & "
        f"${_fmt_p_tex(stage_metrics['Combined']['p'], stage_metrics['Combined']['z'])}$ \\\\"
    )
    rs_lines.append("\\midrule")
    rs_lines.append("\\multicolumn{4}{c}{\\textbf{(b) Temperature}} \\\\")
    rs_lines.append("\\midrule")
    rs_lines.append("\\textbf{Metric} & \\textbf{Qwen2.5-7B} & \\textbf{Llama3.1-8B} & \\textbf{Combined} \\\\")
    rs_lines.append("\\midrule")
    rs_lines.append(
        f"$N$                       & {_fmt_int_tex(temp_metrics['Qwen2.5-7B']['N'])} & "
        f"{_fmt_int_tex(temp_metrics['Llama3.1-8B']['N'])} & {_fmt_int_tex(temp_metrics['Combined']['N'])} \\\\"
    )
    rs_lines.append(
        f"$\\%S$                     & {fmt_shift_pct_short(temp_metrics['Qwen2.5-7B']['shift_pct'])}      & "
        f"{fmt_shift_pct_short(temp_metrics['Llama3.1-8B']['shift_pct'])}     & "
        f"{fmt_shift_pct_short(temp_metrics['Combined']['shift_pct'])}     \\\\"
    )
    rs_lines.append(
        f"$\\hat{{p}}_{{Y\\mid S=1}}$     & {fmt_prob4(temp_metrics['Qwen2.5-7B']['p_shift'])}    & "
        f"{fmt_prob4(temp_metrics['Llama3.1-8B']['p_shift'])}   & "
        f"{fmt_prob4(temp_metrics['Combined']['p_shift'])}   \\\\"
    )
    rs_lines.append(
        f"$\\Delta\\%$                & ${_fmt_delta_pp(temp_metrics['Qwen2.5-7B']['delta_pp'])}$  & "
        f"${_fmt_delta_pp(temp_metrics['Llama3.1-8B']['delta_pp'])}$ & "
        f"${_fmt_delta_pp(temp_metrics['Combined']['delta_pp'])}$ \\\\"
    )
    rs_lines.append(
        f"$\\mathrm{{AME}}$            & ${_fmt_ame(temp_metrics['Qwen2.5-7B']['ame'])}$ & "
        f"${_fmt_ame(temp_metrics['Llama3.1-8B']['ame'])}$ & "
        f"${_fmt_ame(temp_metrics['Combined']['ame'])}$ \\\\"
    )
    rs_lines.append(
        f"$p$                       & ${_fmt_p_tex(temp_metrics['Qwen2.5-7B']['p'], temp_metrics['Qwen2.5-7B']['z'])}$ & "
        f"${_fmt_p_tex(temp_metrics['Llama3.1-8B']['p'], temp_metrics['Llama3.1-8B']['z'])}$ & "
        f"${_fmt_p_tex(temp_metrics['Combined']['p'], temp_metrics['Combined']['z'])}$\\\\"
    )
    rs_lines.append("\\bottomrule")
    rs_lines.append("\\end{tabular*}")
    rs_lines.append(
        "\\caption{\\textbf{Effect of detected reasoning shifts on accuracy, controlling for training stage or temperature (separately).}\n"
        "Cells report shift prevalence (\\(\\%S\\)), accuracy among shifted traces (\\(\\hat{p}_{Y\\mid S=1}\\)),\n"
        "raw accuracy difference (\\(\\Delta\\%=\\hat{p}_{Y\\mid S=1}-\\hat{p}_{Y\\mid S=0}\\), percentage points), and the average marginal effect (AME)\n"
        "from GLM Binomial(logit) regressions with problem fixed effects and cluster-robust SEs.\n"
        "Negative AMEs indicate that shifts reduce accuracy. Patterns match the main-text \\textsc{Math} results for Qwen2.5-1.5B.}"
    )
    rs_lines.append("\\label{tab:rs-7b8b}")
    rs_lines.append("\\vspace{-5mm}")
    rs_lines.append("\\end{table}")
    rs_table = "\n".join(rs_lines) + "\n"

    rs_path = Path(args.out_table_rs_tex)
    _ensure_parent(rs_path)
    rs_path.write_text(rs_table, encoding="utf-8")

    import numpy as np

    strata_stats: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for label, key in model_keys.items():
        roots = [str(root_for(key, float(t))) for t in DEFAULT_TEMPS]
        data = _load_rq3_domain_data(
            "Math",
            roots,
            split=split,
            min_step=min_step,
            max_step=max_step,
            entropy_mode=args.entropy_mode,
            gpt_mode=args.gpt_mode,
            gate_by_words=gate_by_words,
        )
        mask_all = np.ones_like(data.correct, dtype=bool)
        mask_high = data.entropy >= data.entropy_q80
        mask_low = data.entropy < data.entropy_q80
        strata_stats[label] = {
            "all": _rq3_stratum_metrics(data, mask_all),
            "high": _rq3_stratum_metrics(data, mask_high),
            "low": _rq3_stratum_metrics(data, mask_low),
        }
    combined_roots: List[str] = []
    for t in DEFAULT_TEMPS:
        combined_roots.append(str(root_for("qwen7b_math", float(t))))
        combined_roots.append(str(root_for("llama8b_math", float(t))))
    combined_data = _load_rq3_domain_data(
        "Math",
        combined_roots,
        split=split,
        min_step=min_step,
        max_step=max_step,
        entropy_mode=args.entropy_mode,
        gpt_mode=args.gpt_mode,
        gate_by_words=gate_by_words,
    )
    mask_all = np.ones_like(combined_data.correct, dtype=bool)
    mask_high = combined_data.entropy >= combined_data.entropy_q80
    mask_low = combined_data.entropy < combined_data.entropy_q80
    strata_stats["Combined"] = {
        "all": _rq3_stratum_metrics(combined_data, mask_all),
        "high": _rq3_stratum_metrics(combined_data, mask_high),
        "low": _rq3_stratum_metrics(combined_data, mask_low),
    }

    def fmt_delta(value: float) -> str:
        return _fmt_delta_pp(value)

    entropy_lines: List[str] = []
    entropy_lines.append("\\begin{table}[t]")
    entropy_lines.append("  \\centering")
    entropy_lines.append("  \\small")
    entropy_lines.append("  \\setlength{\\tabcolsep}{4pt}")
    entropy_lines.append("  \\renewcommand{\\arraystretch}{1.05}")
    entropy_lines.append("  \\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}} l r r r @{}}")
    entropy_lines.append("    \\toprule")
    entropy_lines.append("    \\textbf{Metric} & \\textbf{Qwen2.5-7B} & \\textbf{Llama3.1-8B} & \\textbf{Combined} \\\\")
    entropy_lines.append("    \\midrule")
    entropy_lines.append("    \\multicolumn{4}{c}{\\textbf{All traces}} \\\\")
    entropy_lines.append("    \\midrule")
    entropy_lines.append(
        f"    $N$                       & {_fmt_int_tex(strata_stats['Qwen2.5-7B']['all']['N'])} & "
        f"{_fmt_int_tex(strata_stats['Llama3.1-8B']['all']['N'])} & "
        f"{_fmt_int_tex(strata_stats['Combined']['all']['N'])} \\\\"
    )
    entropy_lines.append(
        f"    $\\Delta$ (pp)             & ${fmt_delta(strata_stats['Qwen2.5-7B']['all']['delta_pp'])}$  & "
        f"${fmt_delta(strata_stats['Llama3.1-8B']['all']['delta_pp'])}$  & "
        f"${fmt_delta(strata_stats['Combined']['all']['delta_pp'])}$  \\\\"
    )
    entropy_lines.append(
        f"    $p$                       & ${_fmt_p_tex(strata_stats['Qwen2.5-7B']['all']['p'], strata_stats['Qwen2.5-7B']['all']['z'])}$ & "
        f"${_fmt_p_tex(strata_stats['Llama3.1-8B']['all']['p'], strata_stats['Llama3.1-8B']['all']['z'])}$ & "
        f"${_fmt_p_tex(strata_stats['Combined']['all']['p'], strata_stats['Combined']['all']['z'])}$ \\\\"
    )
    entropy_lines.append("    \\midrule")
    entropy_lines.append("    \\multicolumn{4}{c}{\\textbf{High entropy (top 20\\%)}} \\\\")
    entropy_lines.append("    \\midrule")
    entropy_lines.append(
        f"    $N$                       & {_fmt_int_tex(strata_stats['Qwen2.5-7B']['high']['N'])}  & "
        f"{_fmt_int_tex(strata_stats['Llama3.1-8B']['high']['N'])} & "
        f"{_fmt_int_tex(strata_stats['Combined']['high']['N'])} \\\\"
    )
    entropy_lines.append(
        f"    $\\Delta$ (pp)             & ${fmt_delta(strata_stats['Qwen2.5-7B']['high']['delta_pp'])}$  & "
        f"${fmt_delta(strata_stats['Llama3.1-8B']['high']['delta_pp'])}$  & "
        f"${fmt_delta(strata_stats['Combined']['high']['delta_pp'])}$  \\\\"
    )
    entropy_lines.append(
        f"    $p$                       & ${_fmt_p_tex(strata_stats['Qwen2.5-7B']['high']['p'], strata_stats['Qwen2.5-7B']['high']['z'])}$ & "
        f"${_fmt_p_tex(strata_stats['Llama3.1-8B']['high']['p'], strata_stats['Llama3.1-8B']['high']['z'])}$ & "
        f"${_fmt_p_tex(strata_stats['Combined']['high']['p'], strata_stats['Combined']['high']['z'])}$ \\\\"
    )
    entropy_lines.append("    \\midrule")
    entropy_lines.append("    \\multicolumn{4}{c}{\\textbf{Low entropy (bottom 80\\%)}} \\\\")
    entropy_lines.append("    \\midrule")
    entropy_lines.append(
        f"    $N$                       & {_fmt_int_tex(strata_stats['Qwen2.5-7B']['low']['N'])} & "
        f"{_fmt_int_tex(strata_stats['Llama3.1-8B']['low']['N'])} & "
        f"{_fmt_int_tex(strata_stats['Combined']['low']['N'])} \\\\"
    )
    entropy_lines.append(
        f"    $\\Delta$ (pp)             & ${fmt_delta(strata_stats['Qwen2.5-7B']['low']['delta_pp'])}$  & "
        f"${fmt_delta(strata_stats['Llama3.1-8B']['low']['delta_pp'])}$  & "
        f"${fmt_delta(strata_stats['Combined']['low']['delta_pp'])}$ \\\\"
    )
    entropy_lines.append(
        f"    $p$                       & ${_fmt_p_tex(strata_stats['Qwen2.5-7B']['low']['p'], strata_stats['Qwen2.5-7B']['low']['z'])}$ & "
        f"${_fmt_p_tex(strata_stats['Llama3.1-8B']['low']['p'], strata_stats['Llama3.1-8B']['low']['z'])}$ & "
        f"${_fmt_p_tex(strata_stats['Combined']['low']['p'], strata_stats['Combined']['low']['z'])}$ \\\\"
    )
    entropy_lines.append("    \\bottomrule")
    entropy_lines.append("  \\end{tabular*}")
    entropy_lines.append(
        "  \\caption{Entropy-stratified shift effects (Math, steps $\\le\\!450$, temps pooled). "
        "$\\Delta$ is the raw accuracy gap $\\hat p(\\checkmark\\mid S{=}1) - \\hat p(\\checkmark\\mid S{=}0)$ (pp). "
        "$\\mathrm{coef(shift)}$ and $p$ come from logit(correct $\\sim$ shift + problem FEs); dashes indicate the model didn't converge or lacked variation.}"
    )
    entropy_lines.append("  \\label{tab:shift-entropy-strata-7b8b}")
    entropy_lines.append("\\end{table}")
    entropy_table = "\n".join(entropy_lines) + "\n"

    entropy_path = Path(args.out_table_entropy_tex)
    _ensure_parent(entropy_path)
    entropy_path.write_text(entropy_table, encoding="utf-8")

    def load_forced_summary(root_dir: Path) -> Dict[str, Any]:
        path = root_dir / "forced_aha_summary.csv"
        if not path.exists():
            raise SystemExit(f"Missing forced-aha summary: {path}")
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("metric") == "sample":
                    return row
        raise SystemExit(f"No sample row in forced-aha summary: {path}")

    def forced_from_root(root_dir: Path) -> Dict[str, Any]:
        row = load_forced_summary(root_dir)
        n_units = float(row.get("n_units", 0) or 0)
        acc1 = float(row.get("acc_pass1", 0) or 0)
        acc2 = float(row.get("acc_pass2", 0) or 0)
        wins2 = float(row.get("wins_pass2", 0) or 0)
        wins1 = float(row.get("wins_pass1", 0) or 0)
        return {
            "N": int(round(n_units)),
            "p1": acc1,
            "p2": acc2,
            "delta_pp": (acc2 - acc1) * 100.0,
            "wins2": int(round(wins2)),
            "wins1": int(round(wins1)),
        }

    forced_roots = {
        "Qwen2.5-7B": Path("artifacts/forced_aha/GRPO-7B-math_alltemps"),
        "Llama3.1-8B": Path("artifacts/forced_aha/GRPO-Llama8B-math_alltemps"),
    }
    forced_stats: Dict[str, Dict[str, Any]] = {}
    for label, root in forced_roots.items():
        if not root.exists():
            alt = Path(str(root).replace("_alltemps", ""))
            root = alt if alt.exists() else root
        forced_stats[label] = forced_from_root(root)
    combo_n = forced_stats["Qwen2.5-7B"]["N"] + forced_stats["Llama3.1-8B"]["N"]
    if combo_n > 0:
        combo_p1 = (
            forced_stats["Qwen2.5-7B"]["p1"] * forced_stats["Qwen2.5-7B"]["N"]
            + forced_stats["Llama3.1-8B"]["p1"] * forced_stats["Llama3.1-8B"]["N"]
        ) / combo_n
        combo_p2 = (
            forced_stats["Qwen2.5-7B"]["p2"] * forced_stats["Qwen2.5-7B"]["N"]
            + forced_stats["Llama3.1-8B"]["p2"] * forced_stats["Llama3.1-8B"]["N"]
        ) / combo_n
        forced_stats["Combined"] = {
            "N": combo_n,
            "p1": combo_p1,
            "p2": combo_p2,
            "delta_pp": (combo_p2 - combo_p1) * 100.0,
            "wins2": forced_stats["Qwen2.5-7B"]["wins2"] + forced_stats["Llama3.1-8B"]["wins2"],
            "wins1": forced_stats["Qwen2.5-7B"]["wins1"] + forced_stats["Llama3.1-8B"]["wins1"],
        }

    forced_lines: List[str] = []
    forced_lines.append("\\begin{table}[t]")
    forced_lines.append("  \\centering")
    forced_lines.append("  \\small")
    forced_lines.append("  \\setlength{\\tabcolsep}{4pt}")
    forced_lines.append("  \\renewcommand{\\arraystretch}{1.05}")
    forced_lines.append("  \\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}} l r r r @{}}")
    forced_lines.append("    \\toprule")
    forced_lines.append("    \\textbf{Metric} & \\textbf{Qwen2.5-7B} & \\textbf{Llama3.1-8B} & \\textbf{Combined} \\\\")
    forced_lines.append("    \\midrule")
    forced_lines.append(
        f"    $N$ (paired samples) & {_fmt_int_tex(forced_stats['Qwen2.5-7B']['N'])} & "
        f"{_fmt_int_tex(forced_stats['Llama3.1-8B']['N'])} & {_fmt_int_tex(forced_stats['Combined']['N'])} \\\\"
    )
    forced_lines.append(
        f"    $\\hat p_{{\\text{{P1}}}}$ & {fmt_prob4(forced_stats['Qwen2.5-7B']['p1'])} & "
        f"{fmt_prob4(forced_stats['Llama3.1-8B']['p1'])} & {fmt_prob4(forced_stats['Combined']['p1'])} \\\\"
    )
    forced_lines.append(
        f"    $\\hat p_{{\\text{{P2}}}}$ & {fmt_prob4(forced_stats['Qwen2.5-7B']['p2'])} & "
        f"{fmt_prob4(forced_stats['Llama3.1-8B']['p2'])} & {fmt_prob4(forced_stats['Combined']['p2'])} \\\\"
    )
    forced_lines.append(
        f"    $\\Delta$ (pp)        & ${_fmt_delta_pp(forced_stats['Qwen2.5-7B']['delta_pp'])}$ & "
        f"${_fmt_delta_pp(forced_stats['Llama3.1-8B']['delta_pp'])}$ & "
        f"${_fmt_delta_pp(forced_stats['Combined']['delta_pp'])}$ \\\\"
    )
    forced_lines.append(
        f"    wins (P2 $\\uparrow$) & {_fmt_int_tex(forced_stats['Qwen2.5-7B']['wins2'])}     & "
        f"{_fmt_int_tex(forced_stats['Llama3.1-8B']['wins2'])} & {_fmt_int_tex(forced_stats['Combined']['wins2'])} \\\\"
    )
    forced_lines.append(
        f"    wins (P1 $\\uparrow$) & {_fmt_int_tex(forced_stats['Qwen2.5-7B']['wins1'])}     & "
        f"{_fmt_int_tex(forced_stats['Llama3.1-8B']['wins1'])} & {_fmt_int_tex(forced_stats['Combined']['wins1'])} \\\\"
    )
    forced_lines.append("    \\bottomrule")
    forced_lines.append("  \\end{tabular*}")
    forced_lines.append(
        "  \\caption{\\textbf{Forced ``Aha'' (triggered reconsideration), sample-level results on Math.}\n"
        "  $\\hat p_{\\text{P1}}$ and $\\hat p_{\\text{P2}}$ are accuracies in baseline vs forced pass; $\\Delta$ is the percentage-point gain; "
        "``wins'' count paired samples where one pass is correct and the other is not.}"
    )
    forced_lines.append("  \\label{tab:forced-aha-math-7b8b}")
    forced_lines.append("\\end{table}")
    forced_table = "\n".join(forced_lines) + "\n"

    forced_path = Path(args.out_table_forced_tex)
    _ensure_parent(forced_path)
    forced_path.write_text(forced_table, encoding="utf-8")

    # Shift~entropy regressions per temperature (pooled across models).
    temp_or: Dict[float, Dict[str, Any]] = {}
    for t in (0.05, 0.7):
        roots = [
            str(root_for("qwen7b_math", float(t))),
            str(root_for("llama8b_math", float(t))),
        ]
        data = _load_rq3_domain_data(
            "Math",
            roots,
            split=split,
            min_step=min_step,
            max_step=max_step,
            entropy_mode=args.entropy_mode,
            gpt_mode=args.gpt_mode,
            gate_by_words=gate_by_words,
        )
        temp_or[float(t)] = _fit_shift_entropy_fe(data)

    or_005 = temp_or[0.05]["or"]
    p_005 = _fmt_p_tex(temp_or[0.05]["p"], temp_or[0.05]["z"])
    or_07 = temp_or[0.7]["or"]
    p_07 = _fmt_p_tex(temp_or[0.7]["p"], temp_or[0.7]["z"])

    # External models table for appendix format.
    ext_models = [
        ("DeepSeek\\textendash R1", "artifacts/results/deepseek-r1-openrouter"),
        ("DeepSeek\\textendash R1", "artifacts/results/deepseek-r1-openrouter-temp005"),
        ("GPT\\textendash 4o", "artifacts/results/gpt4o-math-portkey"),
        ("GPT\\textendash 4o", "artifacts/results/gpt4o-math-portkey-temp005"),
    ]
    ext_counts: Dict[str, Dict[float, AccuracyCounts]] = {}
    for model_name, root_str in ext_models:
        root = Path(root_str)
        if not root.exists():
            continue
        temp = _infer_temp_from_root_name(root_str)
        if temp is None:
            temp = 0.0
        counts, _steps = iter_accuracy_counts(
            root,
            split=split,
            min_step=0,
            max_step=10**9,
            pass_key="pass1",
            label_key="shift_in_reasoning_v1",
        )
        ext_counts.setdefault(model_name, {})[float(temp)] = counts

    ext_lines: List[str] = []
    ext_lines.append("\\begin{table*}[t]")
    ext_lines.append("\\centering")
    ext_lines.append("\\footnotesize")
    ext_lines.append("\\setlength{\\tabcolsep}{5pt}")
    ext_lines.append("\\begin{tabular}{l c r r r}")
    ext_lines.append("\\toprule")
    ext_lines.append("\\textbf{Model} & \\textbf{$T$} & \\textbf{\\# Problems} & \\textbf{\\% Shifts (count)} & \\textbf{$P(\\checkmark \\mid S{=}1)$} \\\\")
    ext_lines.append("\\midrule")
    for model_name in sorted(ext_counts):
        for t in sorted(ext_counts[model_name]):
            c = ext_counts[model_name][t]
            shift_pct = _pct(c.n_shift, c.n_labeled)
            acc_shift = (c.k_shift / c.n_shift) if c.n_shift else 0.0
            ext_lines.append(
                f"{model_name} & {t:.2g} & {_fmt_int_tex(c.n_labeled)} & "
                f"{shift_pct:.2f}\\% ({int(c.n_shift)}) & {acc_shift:.2f} \\\\"
            )
    ext_lines.append("\\bottomrule")
    ext_lines.append("\\end{tabular}")
    ext_lines.append(
        "\\caption{\\textbf{Canonical reasoning shifts for external models on \\textsc{MATH\\textendash 500} by decoding temperature.}\n"
        "Shift rates remain extremely low across $T{\\in}\\{0,0.05\\}$, and accuracy conditioned on a shift shows no systematic benefit.}"
    )
    ext_lines.append("\\label{tab:external-models}")
    ext_lines.append("\\end{table*}")
    ext_table = "\n".join(ext_lines) + "\n"

    ext_path = Path(args.out_table_external_tex)
    _ensure_parent(ext_path)
    ext_path.write_text(ext_table, encoding="utf-8")

    replacements = {
        "{{OR_005}}": f"{or_005:.2f}",
        "{{P_005}}": p_005,
        "{{OR_07}}": f"{or_07:.2f}",
        "{{P_07}}": p_07,
        "{{TABLE_RS_7B8B}}": rs_table.strip(),
        "{{TABLE_SHIFT_ENTROPY_7B8B}}": entropy_table.strip(),
        "{{TABLE_FORCED_AHA_7B8B}}": forced_table.strip(),
        "{{TABLE_EXTERNAL_MODELS}}": ext_table.strip(),
    }

    for key, value in replacements.items():
        if key not in template:
            raise SystemExit(f"Missing placeholder {key} in {args.template}")
        template = template.replace(key, value)

    if "{{" in template:
        raise SystemExit("Unexpanded placeholders remain in appendix output.")

    out_path = Path(args.out_section_tex)
    _ensure_parent(out_path)
    out_path.write_text(template + "\n", encoding="utf-8")
    print(template)
    print(f"[ok] wrote: {rs_path}")
    print(f"[ok] wrote: {entropy_path}")
    print(f"[ok] wrote: {forced_path}")
    print(f"[ok] wrote: {ext_path}")
    print(f"[ok] wrote: {out_path}")
    return 0


def cmd_models_tasks_table(argv: Optional[List[str]] = None) -> int:
    """
    Emit the methods table for model coverage and learning progress.
    """
    parser = argparse.ArgumentParser(
        prog="master_analysis.py models-tasks-table",
        description="Write the models/tasks coverage table for the methods section.",
    )
    parser.add_argument(
        "--out_tex",
        default="latex/table_models_tasks.tex",
        help="Write the table to this path (default: latex/table_models_tasks.tex).",
    )
    args = parser.parse_args(argv)

    def acc_for_step(root: Path, step_value: int, split_value: str = "test") -> Tuple[int, int]:
        n_total = 0
        n_correct = 0
        files = scan_files_step_only(str(root), split_substr=split_value)
        for file_path in files:
            step_hint = nat_step_from_path(file_path)
            if step_hint is not None and int(step_hint) != int(step_value):
                continue
            for record in iter_records_from_file(file_path):
                if split_value and str(record.get("split", "")).lower() != split_value.lower():
                    continue
                step_val = step_from_rec_or_path(record, file_path)
                if int(step_val) != int(step_value):
                    continue
                pass1 = record.get("pass1") or {}
                if not isinstance(pass1, dict):
                    continue
                correct = extract_correct(pass1, record)
                if correct is None:
                    continue
                n_total += 1
                n_correct += int(correct == 1)
        return n_total, n_correct

    def fmt_acc(n_total: int, n_correct: int) -> str:
        if n_total <= 0:
            return "NA"
        return f"{(100.0 * n_correct / n_total):.2f}"

    qwen15b_roots = {
        "Xword": Path("artifacts/results/GRPO-1.5B-xword-temp-0"),
        "Math": Path("artifacts/results/GRPO-1.5B-math-temp-0.0"),
        "RHour": Path("artifacts/results/GRPO-1.5B-carpark-temp-0"),
    }
    qwen15b_step = 950
    q15_total_0 = 0
    q15_correct_0 = 0
    q15_total_f = 0
    q15_correct_f = 0
    for root in qwen15b_roots.values():
        n0, c0 = acc_for_step(root, 0)
        nf, cf = acc_for_step(root, qwen15b_step)
        q15_total_0 += n0
        q15_correct_0 += c0
        q15_total_f += nf
        q15_correct_f += cf
    q15_step0 = fmt_acc(q15_total_0, q15_correct_0)
    q15_after = fmt_acc(q15_total_f, q15_correct_f)
    q15_delta = "NA"
    if q15_total_0 > 0 and q15_total_f > 0:
        delta_val = (q15_correct_f / q15_total_f - q15_correct_0 / q15_total_0) * 100.0
        q15_delta = f"{delta_val:+.2f}"

    qwen_root = Path("artifacts/results/GRPO-7B-math-temp-0")
    llama_root = Path("artifacts/results/GRPO-Llama8B-math-temp-0")
    combo_step = 500
    q0_total, q0_correct = acc_for_step(qwen_root, 0)
    qf_total, qf_correct = acc_for_step(qwen_root, combo_step)
    l0_total, l0_correct = acc_for_step(llama_root, 0)
    lf_total, lf_correct = acc_for_step(llama_root, combo_step)
    combo_total_0 = q0_total + l0_total
    combo_correct_0 = q0_correct + l0_correct
    combo_total_f = qf_total + lf_total
    combo_correct_f = qf_correct + lf_correct
    combo_step0 = fmt_acc(combo_total_0, combo_correct_0)
    combo_after = fmt_acc(combo_total_f, combo_correct_f)
    combo_delta = "NA"
    if combo_total_0 > 0 and combo_total_f > 0:
        delta_val = (combo_correct_f / combo_total_f - combo_correct_0 / combo_total_0) * 100.0
        combo_delta = f"{delta_val:+.2f}"

    table = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{3pt}\n"
        "\\renewcommand{\\arraystretch}{0.95}\n"
        "\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}} ll r r c r@{}}\n"
        "\\toprule\n"
        "\\textbf{Model} & \\textbf{Domain} & \\textbf{Step 0} & \\textbf{After} & \\textbf{Step} & "
        "$\\boldsymbol{\\Delta}$ \\\\\n"
        "\\midrule\n"
        "Qwen2.5-1.5B  & Xwords     & 7.69 & 10.00 & 950 & +2.31 \\\\\n"
        "Qwen2.5-1.5B  & Math      & 31.00 & 35.00 & 950 & +4.00 \\\\\n"
        "Qwen2.5-1.5B  & RHour & 0.00 & 0.01 & 950 & +0.01 \\\\\n"
        f"Qwen2.5-1.5B  & All       & {q15_step0} & {q15_after} & {qwen15b_step} & {q15_delta} \\\\\n"
        "Qwen2.5-7B    & Math      & 61.60 & 66.40 & 500 & +4.80 \\\\\n"
        "Llama\\,3.1-8B & Math      & 40.20 & 48.36 & 500 & +8.16 \\\\\n"
        f"Qwen2.5-7B + Llama\\,3.1-8B & Math      & {combo_step0} & {combo_after} & {combo_step} & {combo_delta} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular*}\n"
        "\\caption{\\textbf{Model coverage and learning progress.}\n"
        "Accuracy at initialization (Step~0) and at the final training checkpoint, along with the absolute gain ($\\Delta$).\n"
        "All results are 1-shot evaluations at temperature $0$ on the fixed test sets described in \\S\\ref{sec:data}.\n"
        "}\n"
        "\\label{tab:models-tasks}\n"
        "\\end{table}\n"
    )

    out_path = Path(args.out_tex)
    _ensure_parent(out_path)
    out_path.write_text(table, encoding="utf-8")
    print(table)
    print(f"[ok] wrote: {out_path}")
    return 0


def cmd_paper_body(argv: Optional[List[str]] = None) -> int:
    """
    Emit the full paper body LaTeX with updated numbers inserted.
    """
    parser = argparse.ArgumentParser(
        prog="master_analysis.py paper-body",
        description="Write the full paper body LaTeX with updated numbers.",
    )
    parser.add_argument(
        "--template",
        default="latex/paper_body_template.tex",
        help="Template LaTeX file with placeholders (default: latex/paper_body_template.tex).",
    )
    parser.add_argument(
        "--out_tex",
        default="latex/paper_body.tex",
        help="Write the full paper body to this path (default: latex/paper_body.tex).",
    )
    parser.add_argument(
        "--shift_accuracy_json",
        default="artifacts/analysis/shift_accuracy_summary.json",
        help="Shift-accuracy summary JSON (default: artifacts/analysis/shift_accuracy_summary.json).",
    )
    parser.add_argument(
        "--section_rq1",
        default="latex/section_rq1.tex",
        help="RQ1 section file (default: latex/section_rq1.tex).",
    )
    parser.add_argument(
        "--section_rq2",
        default="latex/section_rq2.tex",
        help="RQ2 section file (default: latex/section_rq2.tex).",
    )
    parser.add_argument(
        "--section_rq3",
        default="latex/section_rq3.tex",
        help="RQ3 section file (default: latex/section_rq3.tex).",
    )
    parser.add_argument(
        "--models_tasks_table",
        default="latex/table_models_tasks.tex",
        help="Models/tasks table file (default: latex/table_models_tasks.tex).",
    )
    args = parser.parse_args(argv)

    template = _read_text_or_die(str(args.template))
    section_rq1 = _read_text_or_die(str(args.section_rq1))
    section_rq2 = _read_text_or_die(str(args.section_rq2))
    section_rq3 = _read_text_or_die(str(args.section_rq3))
    models_tasks_table = _read_text_or_die(str(args.models_tasks_table))

    import json

    acc_path = Path(args.shift_accuracy_json)
    payload = json.loads(acc_path.read_text(encoding="utf-8"))
    rows = payload.get("summary") or []
    overall = next((r for r in rows if r.get("experiment") == "OVERALL"), None)
    if overall is None:
        raise SystemExit(f"Missing OVERALL row in {acc_path}")

    total_n = int(overall.get("n_total_with_correct") or overall.get("n_labeled") or 0)
    total_million = (float(total_n) / 1_000_000.0) if total_n else 0.0
    total_traces_million = f"{total_million:.2f}"
    total_traces_mplus = f"{total_million:.2f}M+"
    overall_shift_pct = f"{float(overall.get('shift_pct') or 0.0):.2f}"

    forced_roots = {
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
    forced_summary = _compute_forced_aha_summary(forced_roots)

    def fmt_delta(value: float, digits: int) -> str:
        sign = "+" if value > 0 else ""
        return f"{sign}{value:.{digits}f}pp"

    forced_delta_math = fmt_delta(forced_summary["Math"]["delta_pp"], 2)
    forced_delta_xword = fmt_delta(forced_summary["Xword"]["delta_pp"], 2)
    forced_delta_rhour = fmt_delta(forced_summary["RHour"]["delta_pp"], 3)

    replacements = {
        "{{TOTAL_TRACES_MILLION}}": total_traces_million,
        "{{TOTAL_TRACES_MPLUS}}": total_traces_mplus,
        "{{OVERALL_SHIFT_PCT}}": overall_shift_pct,
        "{{FORCED_DELTA_MATH_PP}}": forced_delta_math,
        "{{FORCED_DELTA_XWORD_PP}}": forced_delta_xword,
        "{{FORCED_DELTA_RHOUR_PP}}": forced_delta_rhour,
        "{{MODELS_TASKS_TABLE}}": models_tasks_table.strip(),
        "{{SECTION_RQ1}}": section_rq1.strip(),
        "{{SECTION_RQ2}}": section_rq2.strip(),
        "{{SECTION_RQ3}}": section_rq3.strip(),
    }

    for key, value in replacements.items():
        if key not in template:
            raise SystemExit(f"Missing placeholder {key} in {args.template}")
        template = template.replace(key, value)

    if "{{" in template:
        raise SystemExit("Unexpanded placeholders remain in paper body output.")

    out_path = Path(args.out_tex)
    _ensure_parent(out_path)
    out_path.write_text(template + "\n", encoding="utf-8")
    print(template)
    print(f"[ok] wrote: {out_path}")
    return 0


def cmd_all(argv: Optional[List[str]] = None) -> int:
    """
    Convenience wrapper to run multiple analyses with one command.
    """
    def _print_file(label: str, path: Path) -> None:
        if not path.exists():
            print(f"[warn] missing {path}")
            return
        content = path.read_text(encoding="utf-8").strip()
        print(f"\n--- {label}: {path} ---")
        if content:
            print(content)
        else:
            print("[warn] file is empty")

    parser = argparse.ArgumentParser(
        prog="master_analysis.py all",
        description="Run shift-prevalence, shift-accuracy, and downstream tables/sections.",
    )
    parser.add_argument(
        "--results_base",
        default="artifacts/results",
        help="Base directory containing results roots (default: artifacts/results).",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Record-level split filter (default: test).",
    )
    parser.add_argument(
        "--min_step",
        type=int,
        default=0,
        help="Minimum checkpoint step to include (inclusive, default: 0).",
    )
    parser.add_argument(
        "--max_step_cap",
        type=int,
        default=None,
        help="Optional global cap on checkpoint steps (inclusive).",
    )
    parser.add_argument(
        "--temps",
        nargs="+",
        type=float,
        default=list(DEFAULT_TEMPS),
        help="Temperatures to include (default: 0 0.05 0.3 0.7).",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=[],
        help="Optional subset of experiment keys to run (default: all).",
    )
    parser.add_argument(
        "--pass_key",
        default="pass1",
        help="Which pass dict to read labels/correctness from (default: pass1).",
    )
    parser.add_argument(
        "--label_key",
        default="shift_in_reasoning_v1",
        help="Shift label field name inside the pass dict (default: shift_in_reasoning_v1).",
    )
    parser.add_argument(
        "--no_summary",
        action="store_true",
        help="Disable printing summary tables to stdout.",
    )
    parser.add_argument(
        "--print_summary_only",
        action="store_true",
        help="Only print summaries (skip writing JSON/CSV).",
    )
    parser.add_argument(
        "--skip_rq2",
        action="store_true",
        help="Skip rq2-section (heavy).",
    )
    parser.add_argument(
        "--rq2_min_step",
        type=int,
        default=None,
        help="Optional min_step override for rq2-section.",
    )
    parser.add_argument(
        "--rq2_max_step",
        type=int,
        default=None,
        help="Optional max_step override for rq2-section.",
    )
    parser.add_argument(
        "--prevalence_out_json",
        default="artifacts/analysis/shift_prevalence_by_step.json",
        help="Output JSON for shift-prevalence.",
    )
    parser.add_argument(
        "--prevalence_out_csv",
        default="artifacts/analysis/shift_prevalence_by_step.csv",
        help="Output CSV for shift-prevalence.",
    )
    parser.add_argument(
        "--log_file",
        default="artifacts/analysis/master_analysis_run.log",
        help="Write a full stdout/stderr transcript to this path.",
    )
    parser.add_argument(
        "--log_append",
        action="store_true",
        help="Append to --log_file instead of overwriting.",
    )
    parser.add_argument(
        "--accuracy_out_json",
        default="artifacts/analysis/shift_accuracy_summary.json",
        help="Output JSON for shift-accuracy.",
    )
    parser.add_argument(
        "--accuracy_out_csv",
        default="artifacts/analysis/shift_accuracy_summary.csv",
        help="Output CSV for shift-accuracy.",
    )
    args = parser.parse_args(argv)

    log_path = Path(args.log_file) if args.log_file else None
    log_handle = None
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    if log_path is not None:
        _ensure_parent(log_path)
        mode = "a" if args.log_append else "w"
        log_handle = log_path.open(mode, encoding="utf-8", buffering=1)

        class _Tee:
            def __init__(self, *streams: Any) -> None:
                self._streams = streams

            def write(self, data: str) -> int:
                for stream in self._streams:
                    stream.write(data)
                    stream.flush()
                return len(data)

            def flush(self) -> None:
                for stream in self._streams:
                    stream.flush()

        sys.stdout = _Tee(orig_stdout, log_handle)
        sys.stderr = _Tee(orig_stderr, log_handle)

    base_args: List[str] = [
        "--results_base",
        str(args.results_base),
        "--split",
        str(args.split),
        "--min_step",
        str(int(args.min_step)),
        "--pass_key",
        str(args.pass_key),
        "--label_key",
        str(args.label_key),
    ]
    if args.max_step_cap is not None:
        base_args += ["--max_step_cap", str(int(args.max_step_cap))]
    if args.temps:
        base_args += ["--temps", *[str(float(t)) for t in args.temps]]
    if args.only:
        base_args += ["--only", *[str(x) for x in args.only]]
    if args.no_summary:
        base_args += ["--no_summary"]
    if args.print_summary_only:
        base_args += ["--print_summary_only"]

    print("\n=== shift-prevalence ===")
    rc1 = cmd_shift_prevalence(
        base_args
        + [
            "--out_json",
            str(args.prevalence_out_json),
            "--out_csv",
            str(args.prevalence_out_csv),
        ],
    )
    print("\n=== shift-accuracy ===")
    rc2 = cmd_shift_accuracy(
        base_args
        + [
            "--out_json",
            str(args.accuracy_out_json),
            "--out_csv",
            str(args.accuracy_out_csv),
        ],
    )
    if args.print_summary_only:
        print("\n=== pooled-logit ===")
        print("[warn] --print_summary_only skips JSON/CSV writing; pooled-logit needs shift-accuracy JSON output.")
        rc_logit = 0
    else:
        print("\n=== pooled-logit ===")
        rc_logit = cmd_pooled_logit(
            [
                "--accuracy_json",
                str(args.accuracy_out_json),
            ],
        )
    if not args.print_summary_only:
        _print_file("shift-prevalence JSON", Path(args.prevalence_out_json))
        _print_file("shift-prevalence CSV", Path(args.prevalence_out_csv))
        _print_file("shift-accuracy JSON", Path(args.accuracy_out_json))
        _print_file("shift-accuracy CSV", Path(args.accuracy_out_csv))
    rc3 = 0
    if args.print_summary_only:
        print("\n=== table2 (latex) ===")
        print("[warn] --print_summary_only skips JSON/CSV writing, so Table 2 cannot be emitted in this mode.")
    else:
        print("\n=== table2 (latex) ===")
        rc3 = cmd_table2(
            [
                "--accuracy_json",
                str(args.accuracy_out_json),
            ],
        )
        table2_path = Path("latex/table2_shift_accuracy.tex")
        if table2_path.exists():
            print(table2_path.read_text(encoding="utf-8").strip())
        else:
            print(f"[warn] missing {table2_path}")

    if args.print_summary_only:
        print("\n=== external-models ===")
        print("[warn] --print_summary_only: external-models still runs, but does not depend on previous outputs.")
    else:
        print("\n=== external-models ===")
    rc4 = cmd_external_models([])
    if not args.print_summary_only:
        ext_table = Path("latex/table_external_models.tex")
        if ext_table.exists():
            print(ext_table.read_text(encoding="utf-8").strip())
        else:
            print(f"[warn] missing {ext_table}")

    rc5 = 0
    if args.print_summary_only:
        print("\n=== rq1-section ===")
        print("[warn] --print_summary_only skips JSON/CSV writing, so RQ1 section cannot be emitted in this mode.")
    else:
        print("\n=== rq1-section ===")
        rc5 = cmd_rq1_section([])

    rc6 = 0
    if args.skip_rq2:
        print("\n=== rq2-section ===")
        print("[warn] --skip_rq2: rq2-section skipped.")
    elif args.print_summary_only:
        print("\n=== rq2-section ===")
        print("[warn] --print_summary_only skips JSON/CSV writing; rq2-section is disabled in this mode.")
    else:
        print("\n=== rq2-section ===")
        rq2_args: List[str] = []
        if args.rq2_min_step is not None:
            rq2_args += ["--min_step", str(int(args.rq2_min_step))]
        if args.rq2_max_step is not None:
            rq2_args += ["--max_step", str(int(args.rq2_max_step))]
        rc6 = cmd_rq2_section(rq2_args)
        table_rq2 = Path("latex/table_rq2_rs.tex")
        if table_rq2.exists():
            print(table_rq2.read_text(encoding="utf-8").strip())
        else:
            print(f"[warn] missing {table_rq2}")

    rc7 = 0
    if args.print_summary_only:
        print("\n=== rq3-section ===")
        print("[warn] --print_summary_only skips JSON/CSV writing; rq3-section is disabled in this mode.")
    else:
        print("\n=== rq3-section ===")
        rc7 = cmd_rq3_section([])
        table_rq3_entropy = Path("latex/table_rq3_shift_entropy_strata.tex")
        if table_rq3_entropy.exists():
            print(table_rq3_entropy.read_text(encoding="utf-8").strip())
        else:
            print(f"[warn] missing {table_rq3_entropy}")
        table_rq3_forced = Path("latex/table_rq3_forced_aha.tex")
        if table_rq3_forced.exists():
            print(table_rq3_forced.read_text(encoding="utf-8").strip())
        else:
            print(f"[warn] missing {table_rq3_forced}")

    if args.print_summary_only:
        print("\n=== rq2-stage-by-temp-tables ===")
        print("[warn] --print_summary_only: rq2-stage-by-temp-tables still runs, but does not depend on previous outputs.")
    else:
        print("\n=== rq2-stage-by-temp-tables ===")
    rc_stage = cmd_rq2_stage_by_temp_tables([])
    if not args.print_summary_only:
        stage_tables = [
            Path("latex/table_rq2_stage_by_temp.tex"),
            Path("latex/table_rq2_stage_by_temp_7b8b.tex"),
        ]
        for table_path in stage_tables:
            if table_path.exists():
                print(table_path.read_text(encoding="utf-8").strip())
            else:
                print(f"[warn] missing {table_path}")

    if args.print_summary_only:
        print("\n=== formal-aha-temp-tables ===")
        print("[warn] --print_summary_only: formal-aha-temp-tables still runs, but does not depend on previous outputs.")
    else:
        print("\n=== formal-aha-temp-tables ===")
    rc_formal = cmd_formal_aha_temp_tables([])
    if not args.print_summary_only:
        formal_tables = [
            Path("latex/table_formal_aha_temp.tex"),
            Path("latex/table_formal_aha_temp_7b8b.tex"),
        ]
        for table_path in formal_tables:
            if table_path.exists():
                print(table_path.read_text(encoding="utf-8").strip())
            else:
                print(f"[warn] missing {table_path}")

    if args.print_summary_only:
        print("\n=== pass2-entropy-gate ===")
        print("[warn] --print_summary_only: pass2-entropy-gate still runs, but does not depend on previous outputs.")
    else:
        print("\n=== pass2-entropy-gate ===")
    rc_gate = cmd_pass2_entropy_gate([])
    if not args.print_summary_only:
        _print_file("pass2-entropy-gate JSON", Path("artifacts/analysis/pass2_entropy_gate.json"))
        _print_file("pass2-entropy-gate CSV", Path("artifacts/analysis/pass2_entropy_gate.csv"))

    if args.print_summary_only:
        print("\n=== pass2-entropy-regression ===")
        print("[warn] --print_summary_only: pass2-entropy-regression still runs, but does not depend on previous outputs.")
    else:
        print("\n=== pass2-entropy-regression ===")
    rc_gate_reg = cmd_pass2_entropy_regression([])
    if not args.print_summary_only:
        _print_file("pass2-entropy-regression JSON", Path("artifacts/analysis/pass2_entropy_regression.json"))
        _print_file("pass2-entropy-regression CSV", Path("artifacts/analysis/pass2_entropy_regression.csv"))
        reg_tables = [
            Path("latex/table_pass2_entropy_regression.tex"),
            Path("latex/table_pass2_entropy_regression_7b8b.tex"),
        ]
        for table_path in reg_tables:
            if table_path.exists():
                print(table_path.read_text(encoding="utf-8").strip())
            else:
                print(f"[warn] missing {table_path}")

    if args.print_summary_only:
        print("\n=== models-tasks-table ===")
        print("[warn] --print_summary_only: models-tasks-table still runs, but does not depend on previous outputs.")
    else:
        print("\n=== models-tasks-table ===")
    rc8 = cmd_models_tasks_table([])

    if args.print_summary_only:
        print("\n=== appendix-qwen-llama ===")
        print("[warn] --print_summary_only: appendix-qwen-llama still runs, but does not depend on previous outputs.")
    else:
        print("\n=== appendix-qwen-llama ===")
    rc9 = cmd_appendix_qwen_llama([])
    if not args.print_summary_only:
        appendix_tables = [
            Path("latex/table_rs_7b8b.tex"),
            Path("latex/table_shift_entropy_strata_7b8b.tex"),
            Path("latex/table_forced_aha_math_7b8b.tex"),
            Path("latex/table_external_models_appendix.tex"),
        ]
        for table_path in appendix_tables:
            if table_path.exists():
                print(table_path.read_text(encoding="utf-8").strip())
            else:
                print(f"[warn] missing {table_path}")

    rc_final = int(
        rc1 != 0
        or rc2 != 0
        or rc_logit != 0
        or rc3 != 0
        or rc4 != 0
        or rc5 != 0
        or rc6 != 0
        or rc7 != 0
        or rc_stage != 0
        or rc_formal != 0
        or rc_gate != 0
        or rc_gate_reg != 0
        or rc8 != 0
        or rc9 != 0
    )
    if log_handle is not None:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        log_handle.flush()
        log_handle.close()
        print(f"[ok] log: {log_path}")
    return rc_final


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="master_analysis.py")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_shift = sub.add_parser(
        "shift-prevalence",
        help="Compute per-checkpoint prevalence of shift_in_reasoning_v1.",
    )
    # Parse only to select the subcommand; each subcommand reparses its own args.
    p_shift.set_defaults(_handler="shift-prevalence")

    p_acc = sub.add_parser(
        "shift-accuracy",
        help="Compute accuracy for shifted vs non-shifted traces.",
    )
    p_acc.set_defaults(_handler="shift-accuracy")

    p_all = sub.add_parser(
        "all",
        help="Run shift-prevalence, shift-accuracy, and table2.",
    )
    p_all.set_defaults(_handler="all")

    p_tab2 = sub.add_parser(
        "table2",
        help="Emit Table 2 LaTeX from shift-accuracy output.",
    )
    p_tab2.set_defaults(_handler="table2")

    p_logit = sub.add_parser(
        "pooled-logit",
        help="Emit pooled logistic regression sentence for correct ~ shift.",
    )
    p_logit.set_defaults(_handler="pooled-logit")

    p_ext = sub.add_parser(
        "external-models",
        help="Emit external-model table and updated sentence snippet.",
    )
    p_ext.set_defaults(_handler="external-models")

    p_rq1 = sub.add_parser(
        "rq1-section",
        help="Emit the full RQ1 section LaTeX snippet.",
    )
    p_rq1.set_defaults(_handler="rq1-section")

    p_rq2 = sub.add_parser(
        "rq2-section",
        help="Emit the full RQ2 section LaTeX snippet.",
    )
    p_rq2.set_defaults(_handler="rq2-section")

    p_rq3 = sub.add_parser(
        "rq3-section",
        help="Emit the full RQ3 section LaTeX snippet.",
    )
    p_rq3.set_defaults(_handler="rq3-section")

    p_stage = sub.add_parser(
        "rq2-stage-by-temp-tables",
        help="Emit fixed-temperature training-stage tables (RQ2).",
    )
    p_stage.set_defaults(_handler="rq2-stage-by-temp-tables")

    p_formal = sub.add_parser(
        "formal-aha-temp-tables",
        help="Emit formal Aha temperature-sweep tables.",
    )
    p_formal.set_defaults(_handler="formal-aha-temp-tables")

    p_gate = sub.add_parser(
        "pass2-entropy-gate",
        help="Summarize pass-2 accuracy gains by pass-1 entropy buckets.",
    )
    p_gate.set_defaults(_handler="pass2-entropy-gate")

    p_gate_reg = sub.add_parser(
        "pass2-entropy-regression",
        help="Regress pass-2 correctness on high-entropy indicators.",
    )
    p_gate_reg.set_defaults(_handler="pass2-entropy-regression")

    p_app = sub.add_parser(
        "appendix-qwen-llama",
        help="Emit appendix LaTeX for Qwen2.5-7B/Llama3.1-8B analyses.",
    )
    p_app.set_defaults(_handler="appendix-qwen-llama")

    p_methods = sub.add_parser(
        "models-tasks-table",
        help="Emit the methods table for model coverage and learning progress.",
    )
    p_methods.set_defaults(_handler="models-tasks-table")

    p_body = sub.add_parser(
        "paper-body",
        help="Emit the full paper body LaTeX with updated numbers.",
    )
    p_body.set_defaults(_handler="paper-body")

    ns, rest = parser.parse_known_args(argv)
    if ns._handler == "shift-prevalence":
        return cmd_shift_prevalence(rest)
    if ns._handler == "shift-accuracy":
        return cmd_shift_accuracy(rest)
    if ns._handler == "all":
        return cmd_all(rest)
    if ns._handler == "table2":
        return cmd_table2(rest)
    if ns._handler == "pooled-logit":
        return cmd_pooled_logit(rest)
    if ns._handler == "external-models":
        return cmd_external_models(rest)
    if ns._handler == "rq1-section":
        return cmd_rq1_section(rest)
    if ns._handler == "rq2-section":
        return cmd_rq2_section(rest)
    if ns._handler == "rq3-section":
        return cmd_rq3_section(rest)
    if ns._handler == "rq2-stage-by-temp-tables":
        return cmd_rq2_stage_by_temp_tables(rest)
    if ns._handler == "formal-aha-temp-tables":
        return cmd_formal_aha_temp_tables(rest)
    if ns._handler == "pass2-entropy-gate":
        return cmd_pass2_entropy_gate(rest)
    if ns._handler == "pass2-entropy-regression":
        return cmd_pass2_entropy_regression(rest)
    if ns._handler == "appendix-qwen-llama":
        return cmd_appendix_qwen_llama(rest)
    if ns._handler == "models-tasks-table":
        return cmd_models_tasks_table(rest)
    if ns._handler == "paper-body":
        return cmd_paper_body(rest)
    raise SystemExit(f"Unknown command: {ns.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
