#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export a flat CSV with one row per (sample, cue_variant) from JSONL results.

This is useful for ad-hoc analysis in pandas/R when you want to compare:
  - pass1 (baseline)
  - pass2 (main reconsideration)
  - pass2a / pass2b / pass2c (per-cue reconsiderations, if present)

The script makes no assumptions about the domain beyond the standard fields
emitted by the math/carpark/crossword cores:
  - top-level: problem, step, split, sample_idx
  - per-pass: is_correct_pred, is_correct_after_reconsideration, entropy*,
              has_reconsider_cue, reconsider_markers, tokens_*, valid_tag_structure,
              shift_in_reasoning_v1 (optional, if gpt_eval was run).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, Iterable, List, Optional

from src.analysis.io import scan_jsonl_files, iter_records_from_file
from src.analysis.utils import nat_step_from_path, get_problem_id, coerce_bool, coerce_float


DEFAULT_PASSES = ["pass1", "pass2", "pass2a", "pass2b", "pass2c"]


def _as_json(value: Any) -> str:
    """Serialize possibly-structured values (lists, dicts) into a compact string."""
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def iter_flat_rows(
    paths: Iterable[str],
    passes: List[str],
) -> Iterable[Dict[str, Any]]:
    """Yield flattened rows for each (record, pass_key) combination."""
    for path in paths:
        step_from_name = nat_step_from_path(path)
        for rec in iter_records_from_file(path):
            if not isinstance(rec, dict):
                continue

            problem = rec.get("problem") or rec.get("clue") or ""
            problem_id = get_problem_id(rec) or ""
            step = rec.get("step", step_from_name)
            split = rec.get("split")
            sample_idx = rec.get("sample_idx")

            for pass_key in passes:
                pass_obj = rec.get(pass_key) or {}
                if not isinstance(pass_obj, dict) or not pass_obj:
                    continue

                row: Dict[str, Any] = {
                    "path": path,
                    "problem_id": problem_id,
                    "problem": problem,
                    "step": step,
                    "split": split,
                    "sample_idx": sample_idx,
                    "variant": pass_key,
                    "is_baseline": int(pass_key == "pass1"),
                    "is_reconsider_variant": int(pass_key != "pass1"),
                }

                # Core correctness / entropy / tokens
                row["is_correct_pred"] = coerce_bool(pass_obj.get("is_correct_pred"))
                row["is_correct_after_reconsideration"] = coerce_bool(
                    pass_obj.get("is_correct_after_reconsideration"),
                )
                row["entropy"] = coerce_float(pass_obj.get("entropy"))
                row["entropy_think"] = coerce_float(pass_obj.get("entropy_think"))
                row["entropy_answer"] = coerce_float(pass_obj.get("entropy_answer"))
                row["entropy_pre_cue"] = coerce_float(pass_obj.get("entropy_pre_cue"))
                row["entropy_reconsider_think"] = coerce_float(
                    pass_obj.get("entropy_reconsider_think"),
                )
                row["entropy_reconsider_full"] = coerce_float(
                    pass_obj.get("entropy_reconsider_full"),
                )

                row["tokens_think"] = pass_obj.get("tokens_think")
                row["tokens_answer"] = pass_obj.get("tokens_answer")
                row["tokens_total"] = pass_obj.get("tokens_total")
                row["valid_tag_structure"] = coerce_bool(pass_obj.get("valid_tag_structure"))

                # Native reconsideration cues
                row["has_reconsider_cue"] = coerce_bool(pass_obj.get("has_reconsider_cue"))
                row["reconsider_markers"] = _as_json(pass_obj.get("reconsider_markers"))
                row["reconsider_pos"] = pass_obj.get("reconsider_pos")
                row["reconsider_excerpt"] = pass_obj.get("reconsider_excerpt") or ""

                # GPT-based shift labels (if present)
                row["shift_in_reasoning_v1"] = coerce_bool(
                    pass_obj.get("shift_in_reasoning_v1"),
                )
                row["shift_markers_v1"] = _as_json(pass_obj.get("shift_markers_v1"))

                yield row


def export_cue_variants(
    results_root: str,
    split_substr: Optional[str],
    out_csv: str,
    passes: List[str],
) -> None:
    """Scan a results root and write a flat CSV with one row per (sample, pass_key)."""
    files = scan_jsonl_files(results_root, split_substr=split_substr)
    if not files:
        raise SystemExit(
            f"No JSONL files found under {results_root!r} (split={split_substr!r}).",
        )

    rows_iter = iter_flat_rows(files, passes)
    first_row = None
    for first_row in rows_iter:
        break

    if first_row is None:
        raise SystemExit("No matching pass sections found to export.")

    fieldnames = list(first_row.keys())
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    with open(out_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(first_row)
        for row in rows_iter:
            writer.writerow(row)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a flat CSV with one row per (sample, cue_variant) "
            "from JSONL results (pass1/pass2/pass2a/pass2b/pass2c)."
        ),
    )
    parser.add_argument(
        "results_root",
        help="Root directory containing step*/.../*.jsonl (e.g., artifacts/results/GRPO-1.5B-math-temp-0.05-3).",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Optional substring filter on filenames (e.g., 'test').",
    )
    parser.add_argument(
        "--out_csv",
        default=None,
        help=(
            "Output CSV path (default: <results_root>/cue_variants.csv "
            "or <results_root>/cue_variants_<split>.csv when --split is set)."
        ),
    )
    parser.add_argument(
        "--passes",
        default=",".join(DEFAULT_PASSES),
        help=(
            "Comma-separated pass keys to export "
            "(e.g., 'pass1,pass2,pass2a,pass2b,pass2c'). "
            f"Default: {','.join(DEFAULT_PASSES)}"
        ),
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    passes = [
        p.strip()
        for p in (args.passes or "").split(",")
        if p.strip()
    ]
    if not passes:
        raise SystemExit("Must specify at least one pass key via --passes.")

    if args.out_csv:
        out_csv = args.out_csv
    else:
        suffix = f"_{args.split}" if args.split else ""
        out_csv = os.path.join(args.results_root, f"cue_variants{suffix}.csv")

    export_cue_variants(
        results_root=args.results_root,
        split_substr=args.split,
        out_csv=out_csv,
        passes=passes,
    )


if __name__ == "__main__":
    main()

