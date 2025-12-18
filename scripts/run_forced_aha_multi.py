#!/usr/bin/env python3
"""
Helper to sweep the forced-Aha analysis across multiple result roots.

Example:
    python scripts/run_forced_aha_multi.py \
        --math_roots artifacts/results/GRPO-1.5B-math-temp-0.0 ... \
        --combine_across_temps \
        --combined_out artifacts/results/forced_aha_combined.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, List, Sequence


DatasetRoots = Sequence[str]


@dataclass
class SuiteSpec:
    """Bundle a suite name with the set of roots to sweep."""

    name: str
    roots: DatasetRoots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run forced-Aha analysis for multiple domains / temperatures.",
    )
    parser.add_argument("--math_roots", nargs="+", default=[], help="Math result roots to process.")
    parser.add_argument("--xword_roots", nargs="+", default=[], help="Crossword result roots to process.")
    parser.add_argument("--rush_roots", nargs="+", default=[], help="Rush Hour (car-park) result roots to process.")
    parser.add_argument("--split", default="test", help="Split substring to filter files (default: test).")
    parser.add_argument("--pass2_key", default="pass2", help="Pass-2 key containing the forced reconsideration data.")
    parser.add_argument("--make_plots", action="store_true", help="Generate figure outputs.")
    parser.add_argument("--min_step", type=int, default=None, help="Minimum training step to include.")
    parser.add_argument("--max_step", type=int, default=None, help="Maximum training step to include.")
    parser.add_argument(
        "--combine_across_temps",
        action="store_true",
        help="After running each root, produce a combined CSV over every temperature.",
    )
    parser.add_argument(
        "--combined_out",
        default=os.path.join("artifacts", "results", "forced_aha_combined.csv"),
        help="Destination for the combined CSV when --combine_across_temps is set.",
    )
    parser.add_argument(
        "--combined_summary_out",
        default=os.path.join("artifacts", "results", "forced_aha_combined_summary.csv"),
        help="Destination for the aggregated suite-level summary CSV.",
    )
    return parser.parse_args()


def forced_aha_out_dir(root: str, pass2_key: str) -> str:
    """Mirror forced_aha_effect_impl's default output directory."""
    suffix = f"_{pass2_key}" if pass2_key and pass2_key != "pass2" else ""
    return os.path.join(root, f"forced_aha_effect{suffix}")


def append_combined_rows(
    rows: List[dict],
    suite: str,
    root: str,
    summary_path: str,
) -> None:
    """Load forced_aha_summary.csv and extend with suite/root metadata."""
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Missing summary CSV at {summary_path}")
    temp_match = re.search(r"temp-([0-9.]+)", root)
    temperature = temp_match.group(1) if temp_match else "unknown"
    with open(summary_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_copy = dict(row)
            row_copy["suite"] = suite
            row_copy["root"] = root
            row_copy["temperature"] = temperature
            rows.append(row_copy)


def _coerce_float(value: str) -> float:
    return float(value) if value not in ("", None) else 0.0


def _coerce_optional_float(value: str) -> float | None:
    return float(value) if value not in ("", None) else None


def write_combined_csv(rows: Iterable[dict], destination: str) -> None:
    """Write the concatenated per-root summary rows to destination."""
    rows = list(rows)
    if not rows:
        print("[combine] no rows to write; skipping combined CSV.")
        return
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(destination, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[combine] wrote {len(rows)} rows to {destination}")


def write_combined_summary(rows: Iterable[dict], destination: str) -> None:
    """Aggregate per-suite/per-metric stats and write a summary CSV."""
    rows = list(rows)
    if not rows:
        return
    aggregated = {}
    for row in rows:
        key = (row["suite"], row["metric"])
        bucket = aggregated.setdefault(
            key,
            {
                "suite": row["suite"],
                "metric": row["metric"],
                "n_units": 0.0,
                "sum_acc_pass1": 0.0,
                "sum_acc_pass2": 0.0,
                "wins_pass2": 0.0,
                "wins_pass1": 0.0,
                "both_correct": 0.0,
                "both_wrong": 0.0,
            },
        )
        n_units = _coerce_float(row.get("n_units"))
        acc1 = _coerce_optional_float(row.get("acc_pass1"))
        acc2 = _coerce_optional_float(row.get("acc_pass2"))
        if acc1 is not None:
            bucket["sum_acc_pass1"] += n_units * acc1
        if acc2 is not None:
            bucket["sum_acc_pass2"] += n_units * acc2
        bucket["n_units"] += n_units
        for key_name in ("wins_pass2", "wins_pass1", "both_correct", "both_wrong"):
            bucket[key_name] += _coerce_float(row.get(key_name))

    summary_rows: List[dict] = []
    for bucket in aggregated.values():
        n_units = bucket["n_units"]
        acc_pass1 = bucket["sum_acc_pass1"] / n_units if n_units else None
        acc_pass2 = bucket["sum_acc_pass2"] / n_units if n_units else None
        delta_pp = None
        if acc_pass1 is not None and acc_pass2 is not None:
            delta_pp = (acc_pass2 - acc_pass1) * 100.0
        summary_rows.append(
            {
                "suite": bucket["suite"],
                "metric": bucket["metric"],
                "n_units": n_units,
                "acc_pass1": acc_pass1,
                "acc_pass2": acc_pass2,
                "delta_pp": delta_pp,
                "wins_pass2": bucket["wins_pass2"],
                "wins_pass1": bucket["wins_pass1"],
                "both_correct": bucket["both_correct"],
                "both_wrong": bucket["both_wrong"],
            },
        )

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    fieldnames = list(summary_rows[0].keys())
    with open(destination, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[combine] wrote aggregated summary to {destination}")


def run_for_suite(
    suite: SuiteSpec,
    args: argparse.Namespace,
) -> List[dict]:
    """Run forced_aha_effect over every root in a suite and return summary rows."""
    combined_rows: List[dict] = []
    for root in suite.roots:
        cmd: List[str] = [
            sys.executable,
            "-m",
            "src.analysis.forced_aha_effect",
            root,
        ]
        if args.split:
            cmd += ["--split", args.split]
        if args.pass2_key:
            cmd += ["--pass2_key", args.pass2_key]
        if args.make_plots:
            cmd.append("--make_plots")
        if args.min_step is not None:
            cmd += ["--min_step", str(args.min_step)]
        if args.max_step is not None:
            cmd += ["--max_step", str(args.max_step)]
        print(f"[forced-aha][{suite.name}] running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        summary_dir = forced_aha_out_dir(root, args.pass2_key)
        summary_path = os.path.join(summary_dir, "forced_aha_summary.csv")
        if args.combine_across_temps:
            append_combined_rows(combined_rows, suite.name, root, summary_path)
    return combined_rows


def main() -> None:
    args = parse_args()
    suites = [
        SuiteSpec("math", args.math_roots),
        SuiteSpec("xword", args.xword_roots),
        SuiteSpec("rush", args.rush_roots),
    ]
    combined_rows: List[dict] = []
    for suite in suites:
        if not suite.roots:
            continue
        combined_rows.extend(run_for_suite(suite, args))
    if args.combine_across_temps:
        write_combined_csv(combined_rows, args.combined_out)
        write_combined_summary(combined_rows, args.combined_summary_out)


if __name__ == "__main__":
    main()
