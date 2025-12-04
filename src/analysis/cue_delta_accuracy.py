#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize cue-specific accuracy gains and entropy-quartile effects.

This script reads the flattened cue data produced by
``src.annotate.tasks.math_cue_variants`` and reports
  * overall Δ accuracy (cue - baseline),
  * Δ accuracy per entropy quartile,
  * average output length/tokens per cue (optional).

Typical usage:

    python -m src.analysis.cue_delta_accuracy \\
      --input-jsonl artifacts/results/.../step0950_test.jsonl

If you already ran ``math_cue_variants`` manually, point at the flattened
file with ``--flat-jsonl`` instead.
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from src.analysis.io import iter_records_from_file
from src.annotate.tasks.math_cue_variants import flatten_math_cue_variants


os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "matplotlib"),
)


def _build_stats() -> Dict[str, Any]:
    return {
        "total": 0,
        "correct": 0,
        "output_len_sum": 0,
        "output_len_count": 0,
        "tokens_total_sum": 0,
        "tokens_total_count": 0,
        "baseline_wrong_total": 0,
        "baseline_wrong_correct": 0,
        "quartiles": defaultdict(lambda: {"total": 0, "correct": 0}),
    }


def _make_variant_stats() -> Dict[str, int]:
    return {
        "total": 0,
        "correct": 0,
        "baseline_wrong_total": 0,
        "baseline_wrong_correct": 0,
    }


def _build_table(
    rows: Iterable[Dict[str, Any]],
) -> Tuple[
    Dict[Tuple[str, int, str], Dict[str, int]],
    Tuple[str, ...],
    Dict[str, Tuple[int, ...]],
    Tuple[str, ...],
]:
    table: Dict[Tuple[str, int, str], Dict[str, int]] = defaultdict(_make_variant_stats)
    domains: Dict[str, None] = {}
    quartiles_by_domain: Dict[str, Dict[int, None]] = defaultdict(dict)
    variants: Dict[str, None] = {}

    for row in rows:
        domain = row.get("domain") or "unknown"
        quartile = row.get("entropy_quartile")
        variant = row.get("cue_variant") or "unknown"
        if not isinstance(quartile, int):
            continue
        domains[domain] = None
        quartiles_by_domain[domain][quartile] = None
        variants[variant] = None

        key = (domain, quartile, variant)
        stats = table[key]
        stats["total"] += 1
        if bool(row.get("intervention_correct")):
            stats["correct"] += 1
        if not bool(row.get("baseline_correct")):
            stats["baseline_wrong_total"] += 1
            if bool(row.get("intervention_correct")):
                stats["baseline_wrong_correct"] += 1

    sorted_domains = tuple(sorted(domains.keys()))
    quartile_mapping = {domain: tuple(sorted(quartiles_by_domain[domain].keys())) for domain in quartiles_by_domain}
    sorted_variants = tuple(sorted(variants.keys()))
    return table, sorted_domains, quartile_mapping, sorted_variants


def _table_rows_from_stats(
    table: Dict[Tuple[str, int, str], Dict[str, int]],
    domains: Tuple[str, ...],
    quartile_map: Dict[str, Tuple[int, ...]],
    variants: Tuple[str, ...],
) -> Tuple[Dict[str, Any], ...]:
    rows: list[Dict[str, Any]] = []
    cues = [v for v in variants if v != "baseline"]
    for domain in domains:
        quartiles = quartile_map.get(domain) or ()
        for quartile in quartiles:
            baseline_stats = table.get((domain, quartile, "baseline"))
            if not baseline_stats or baseline_stats["total"] == 0:
                continue
            for cue in cues:
                cue_stats = table.get((domain, quartile, cue)) or _make_variant_stats()
                rows.append(
                    {
                        "domain": domain,
                        "quartile": quartile,
                        "cue_variant": cue,
                        "N": baseline_stats["total"],
                        "baseline_acc": _pct(baseline_stats["correct"], baseline_stats["total"]),
                        "cue_acc": _pct(cue_stats["correct"], cue_stats["total"]),
                        "cue_acc_baseline_wrong": _pct(
                            cue_stats["baseline_wrong_correct"],
                            cue_stats["baseline_wrong_total"],
                        ),
                    },
                )
    return tuple(rows)


def _format_pct_or_na(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _print_table_rows(rows: Tuple[Dict[str, Any], ...]) -> None:
    if not rows:
        print("\n[cue_table] no table rows generated.")
        return
    domains = sorted({row["domain"] for row in rows})
    cues = sorted({row["cue_variant"] for row in rows})
    for domain in domains:
        domain_rows = [r for r in rows if r["domain"] == domain]
        if not domain_rows:
            continue
        print(f"\n[cue_table] domain={domain}")
        for cue in cues:
            cue_rows = [r for r in domain_rows if r["cue_variant"] == cue]
            if not cue_rows:
                continue
            print(f"Cue {cue}")
            print(" Quartile |     N | baseline | cue (all) | cue (baseline wrong)")
            for row in sorted(cue_rows, key=lambda r: r["quartile"]):
                quartile_label = row["quartile"]
                baseline_acc = _format_pct_or_na(row["baseline_acc"])
                cue_acc = _format_pct_or_na(row["cue_acc"])
                cue_wrong = _format_pct_or_na(row["cue_acc_baseline_wrong"])
                print(
                    f"      Q{quartile_label} | {row['N']:5d} | {baseline_acc:>8} | {cue_acc:>9} | {cue_wrong:>21}",
                )


def _write_table_csv(
    rows: Tuple[Dict[str, Any], ...],
    path: Path,
) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "domain",
                "quartile",
                "cue_variant",
                "N",
                "baseline_acc",
                "cue_acc",
                "cue_acc_baseline_wrong",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _update_quartile_bucket(
    bucket: Dict[str, int],
    correct: bool,
) -> None:
    bucket["total"] += 1
    if correct:
        bucket["correct"] += 1


def _update_stats(stats: Dict[str, Any], row: Dict[str, Any]) -> None:
    stats["total"] += 1
    correct = bool(row.get("intervention_correct"))
    if correct:
        stats["correct"] += 1

    quartile = row.get("entropy_quartile")
    if isinstance(quartile, int):
        _update_quartile_bucket(stats["quartiles"][quartile], correct)

    output_len = row.get("output_len")
    if isinstance(output_len, (int, float)):
        stats["output_len_sum"] += output_len
        stats["output_len_count"] += 1

    tokens_total = row.get("tokens_total")
    if isinstance(tokens_total, (int, float)):
        stats["tokens_total_sum"] += tokens_total
        stats["tokens_total_count"] += 1


def _pct(value: float, denom: int) -> Optional[float]:
    if denom == 0:
        return None
    return value / denom


def _format_pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _format_diff(delta: Optional[float]) -> str:
    if delta is None:
        return "n/a"
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta * 100:.1f}pp"


def _collect_quartiles(
    baseline_stats: Dict[str, Any],
    cue_stats: Dict[str, Dict[str, Any]],
) -> Tuple[int, ...]:
    quartile_set = set(baseline_stats["quartiles"].keys())
    for stats in cue_stats.values():
        quartile_set.update(stats["quartiles"].keys())
    return tuple(sorted(quartile_set))


def _print_average_lengths(stats: Dict[str, Any]) -> None:
    length_count = stats["output_len_count"]
    if length_count:
        avg_len = stats["output_len_sum"] / length_count
        print(f"  avg output_len: {avg_len:.0f} chars")
    tokens_count = stats["tokens_total_count"]
    if tokens_count:
        avg_tokens = stats["tokens_total_sum"] / tokens_count
        print(f"  avg tokens_total: {avg_tokens:.1f}")


def _print_quartile_details(
    baseline_stats: Dict[str, Any],
    cue_stats: Dict[str, Any],
    quartiles: Tuple[int, ...],
) -> None:
    baseline_buckets = baseline_stats["quartiles"]
    cue_buckets = cue_stats["quartiles"]
    for quartile in quartiles:
        baseline_bucket = baseline_buckets.get(quartile, {})
        cue_bucket = cue_buckets.get(quartile, {})
        baseline_q_acc = _pct(
            baseline_bucket.get("correct", 0),
            baseline_bucket.get("total", 0),
        )
        cue_q_acc = _pct(
            cue_bucket.get("correct", 0),
            cue_bucket.get("total", 0),
        )
        delta_q = None if baseline_q_acc is None or cue_q_acc is None else cue_q_acc - baseline_q_acc
        print(
            f"  Q{quartile}: cue acc {_format_pct(cue_q_acc)} "
            f"(baseline {_format_pct(baseline_q_acc)}) "
            f"Δ {_format_diff(delta_q)}",
        )


def _print_cue_summary(
    cue_label: str,
    stats: Dict[str, Any],
    baseline_stats: Dict[str, Any],
    baseline_accuracy: Optional[float],
    quartiles: Tuple[int, ...],
) -> None:
    acc = _pct(stats["correct"], stats["total"])
    delta = None if baseline_accuracy is None or acc is None else acc - baseline_accuracy
    print()
    print(
        f"Cue {cue_label}: accuracy {_format_pct(acc)} → Δ {_format_diff(delta)}",
    )

    _print_average_lengths(stats)
    _print_quartile_details(baseline_stats, stats, quartiles)


def _summarize(
    baseline_stats: Dict[str, Any],
    cue_stats: Dict[str, Dict[str, Any]],
) -> None:
    """
    Print baseline and cue-specific accuracy summaries, including quartiles.
    """
    baseline_total = baseline_stats["total"]
    baseline_accuracy = _pct(baseline_stats["correct"], baseline_total)
    print(
        f"Baseline accuracy (pass1): {_format_pct(baseline_accuracy)} [{baseline_total} rows]",
    )

    quartiles = _collect_quartiles(baseline_stats, cue_stats)
    for cue_label in sorted(cue_stats):
        _print_cue_summary(
            cue_label=cue_label,
            stats=cue_stats[cue_label],
            baseline_stats=baseline_stats,
            baseline_accuracy=baseline_accuracy,
            quartiles=quartiles,
        )


def _sorted_row_labels(
    rows: Iterable[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    Split flattened rows into baseline and per-cue aggregates.
    """
    baseline = _build_stats()
    cues: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        variant = row.get("cue_variant")
        if variant == "baseline":
            _update_stats(baseline, row)
        else:
            stats = cues.setdefault(variant or "unknown", _build_stats())
            _update_stats(stats, row)
    return baseline, cues


def _ensure_flat_path(
    args: argparse.Namespace,
) -> Tuple[Path, Optional[Path]]:
    if args.flat_jsonl:
        return Path(args.flat_jsonl), None
    if not args.input_jsonl:
        raise RuntimeError("either --flat-jsonl or --input-jsonl must be provided")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as temp_handle:
        temp_name = temp_handle.name
    try:
        output_path = Path(
            flatten_math_cue_variants(
                str(args.input_jsonl),
                temp_name,
            ),
        )
        return output_path, output_path
    except Exception as exc:
        os.unlink(temp_name)
        raise exc


def _cleanup_temp_path(cleanup_path: Optional[Path]) -> None:
    """
    Remove a temporary file path if it exists.

    This is shared by cue-analysis scripts that optionally materialize a
    flattened JSONL and then clean it up on exit.
    """
    if cleanup_path is not None and cleanup_path.exists():
        cleanup_path.unlink()


def parse_args() -> argparse.Namespace:
    """
    Build and parse command-line arguments for cue analysis.
    """
    parser = argparse.ArgumentParser(
        description="Compute cue-specific Δ accuracy and quartile effects.",
    )
    parser.add_argument(
        "--flat-jsonl",
        type=Path,
        help="Pre-flattened cue JSONL (produced by math_cue_variants).",
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        help="Original multi-cue JSONL (flattens automatically when --flat-jsonl is omitted).",
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="Print per-domain/quartile summary rows for each cue variant.",
    )
    parser.add_argument(
        "--table-csv",
        type=Path,
        help="Optional CSV output path for the table (overwrites).",
    )
    args = parser.parse_args()
    if not args.flat_jsonl and not args.input_jsonl:
        parser.error("either --flat-jsonl or --input-jsonl must be provided")
    return args


def main() -> None:
    """
    Entry point: load rows, aggregate cue stats, and print summaries.
    """
    args = parse_args()
    flat_path, cleanup_path = _ensure_flat_path(args)
    try:
        rows = tuple(iter_records_from_file(flat_path))
        baseline, cue_stats = _sorted_row_labels(rows)
        _summarize(baseline, cue_stats)
        if args.table or args.table_csv:
            table_data, domains, quartile_map, variants = _build_table(rows)
            table_rows = _table_rows_from_stats(
                table_data,
                domains,
                quartile_map,
                variants,
            )
            _print_table_rows(table_rows)
            if args.table_csv:
                _write_table_csv(table_rows, args.table_csv)
    finally:
        _cleanup_temp_path(cleanup_path)


if __name__ == "__main__":
    main()

# Backwards-compatible alias for callers expecting a function named after the module.
cue_delta_accuracy = main
