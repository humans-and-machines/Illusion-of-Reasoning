#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aggregate crossword pass-1/pass-2 evaluation JSONLs by step.

This CLI focuses on parsing arguments, wiring filters, and delegating to the
core aggregation helpers defined in ``summarize_inference_core``.

Assumes crossword inference JSONLs where:
  - rec["problem"] holds the clue text
  - rec["gold_answer_canon"] is the canonical gold
  - rec["pass1"]["pred_answer_canon"], rec["pass2"]["pred_answer_canon"] exist
  - rec["pass1"]["soft_reward"] (optional) holds a soft similarity score in [0,1]
  - rec["pass2"]["soft_reward"] (optional) holds a soft similarity score in [0,1]
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

from src.inference.common import iter_jsonl_objects
from src.inference.summarize_inference_core import (
    StepAgg,
    accumulate_prompt_variants,
    build_step_csv_row,
    extract_group,
    nat_step_from_path,
    scan_files,
    should_drop_group,
)


def _compute_prompt_drop_groups(files: List[str], args: argparse.Namespace) -> set[str]:
    """Return the set of group names to drop based on prompt-variant limits."""
    drop_groups: set[str] = set()
    if args.max_prompt_versions is None:
        return drop_groups
    variants_by_group: Dict[str, set] = defaultdict(set)
    counts = {"seen": 0, "with_prompt": 0}

    for path in files:
        for record in iter_jsonl_objects(path):
            accumulate_prompt_variants(
                record=record,
                args=args,
                variants_by_group=variants_by_group,
                counts=counts,
            )

    for group_name, variants in variants_by_group.items():
        if len(variants) > args.max_prompt_versions:
            drop_groups.add(group_name)

    scope = "per-problem" if args.filter_scope == "per_problem" else "global"
    print(
        "[filter] "
        f"Scope={scope} | "
        f"threshold={args.max_prompt_versions} | "
        f"records_seen={counts['seen']} | "
        f"with_prompt={counts['with_prompt']} | "
        f"groups_dropped={len(drop_groups)}",
    )
    return drop_groups


def _maybe_write_filtered_files(
    files: List[str],
    args: argparse.Namespace,
    drop_groups: set[str],
) -> Tuple[bool, List[str]]:
    """
    Optionally write filtered/capped copies of the input JSONLs.

    Returns (wrote_any, files_for_agg), where files_for_agg is the list of
    files that should be used for aggregation.
    """
    if not (args.rewrite_filtered or args.write_filtered_to):
        return False, files

    if args.rewrite_filtered and args.write_filtered_to:
        raise SystemExit("Choose only one of --rewrite_filtered or --write_filtered_to.")

    wrote_any = False

    for src_path in files:
        if args.rewrite_filtered:
            out_path = src_path + ".tmp_filter"
        else:
            out_path = os.path.join(
                args.write_filtered_to,
                os.path.relpath(src_path, args.results_root),
            )
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

        stats = {"total": 0, "kept": 0}
        group_counts: Dict[str, int] = defaultdict(int)
        exceeded_groups: set[str] = set()

        with open(out_path, "w", encoding="utf-8") as dst_file:
            for record in iter_jsonl_objects(src_path):
                stats["total"] += 1
                if should_drop_group(record, drop_groups, args.filter_scope):
                    continue

                if args.max_per_group is not None:
                    group_key_value = extract_group(record, args.group_key)
                    count = group_counts[group_key_value]
                    if count >= args.max_per_group:
                        exceeded_groups.add(group_key_value)
                        continue
                    group_counts[group_key_value] = count + 1

                dst_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["kept"] += 1

        if args.rewrite_filtered:
            os.replace(out_path, src_path)

        wrote_any = True
        print(
            "[filter-write] "
            f"{src_path} -> "
            f"{src_path if args.rewrite_filtered else out_path} | "
            f"kept={stats['kept']} of {stats['total']} "
            f"| unique_groups={len(group_counts)} "
            f"| groups_truncated={len(exceeded_groups)}",
        )

    files_for_agg = files
    if wrote_any and args.write_filtered_to and args.aggregate_from_filtered:
        files_for_agg = [
            os.path.join(
                args.write_filtered_to,
                os.path.relpath(src_path, args.results_root),
            )
            for src_path in files
        ]

    return wrote_any, files_for_agg


def _aggregate_steps(
    files_for_agg: List[str],
    args: argparse.Namespace,
    drop_groups: set[str],
    wrote_any: bool,
) -> List[StepAgg]:
    """Aggregate records from JSONL files into per-step StepAgg objects."""
    steps_by_step: Dict[int, StepAgg] = {}

    for path in files_for_agg:
        group_counts: Dict[str, int] = defaultdict(int)
        step_from_name = nat_step_from_path(path)
        for record in iter_jsonl_objects(path):
            if should_drop_group(record, drop_groups, args.filter_scope):
                continue

            if args.max_per_group is not None and not wrote_any:
                group_key_value = extract_group(record, args.group_key)
                count = group_counts[group_key_value]
                if count >= args.max_per_group:
                    continue
                group_counts[group_key_value] = count + 1

            step_value = record.get(
                "step",
                step_from_name if step_from_name is not None else 0,
            )
            aggregator = steps_by_step.setdefault(step_value, StepAgg(step_value))
            aggregator.add_record(record, args.recompute_correctness)

    for aggregator in steps_by_step.values():
        aggregator.finalize()

    return [steps_by_step[key] for key in sorted(steps_by_step)]


def _print_step_summaries(aggregates: List[StepAgg]) -> None:
    """Print the main table and per-step footer summaries."""
    header = (
        "  step   n1S  acc1S  acc1E    ent1      t1      a1  soft1S soft1E  "
        "n2S  acc2S  acc2E    ent2      t2      a2  soft2S soft2E  "
        "impS  impE  tag1  tag2"
    )
    print(header)
    print("-" * len(header))
    for aggregator in aggregates:
        print(aggregator.row_text())
    print()

    for aggregator in aggregates:
        print(f"[step {aggregator.step}]")
        print(aggregator.footer_text())
        print()


def _write_csv_outputs(args: argparse.Namespace, aggregates: List[StepAgg]) -> None:
    """Write optional CSV outputs summarizing per-step and per-example metrics."""
    if args.save_csv:
        with open(args.save_csv, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "step",
                    "n1S",
                    "acc1S_pct",
                    "acc1E_pct",
                    "ent1",
                    "t1",
                    "a1",
                    "soft1S",
                    "soft1E",
                    "n2S",
                    "acc2S_pct",
                    "acc2E_pct",
                    "ent2",
                    "t2",
                    "a2",
                    "soft2S",
                    "soft2E",
                    "impS_pct",
                    "impE_pct",
                    "tag1_pct",
                    "tag2_pct",
                ],
            )
            for aggregator in aggregates:
                writer.writerow(build_step_csv_row(aggregator))

    if args.per_example_csv:
        with open(args.per_example_csv, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["step", "problem", "p1_correct", "p2_correct", "improved"])
            for aggregator in aggregates:
                for problem in sorted(aggregator.examples):
                    p1_ok = aggregator.pass1.correct_by_problem.get(problem, False)
                    p2_ok = aggregator.pass2.correct_by_problem.get(problem, False)
                    improved = p2_ok and not p1_ok
                    writer.writerow(
                        [
                            aggregator.step,
                            problem,
                            int(bool(p1_ok)),
                            int(bool(p2_ok)),
                            int(bool(improved)),
                        ],
                    )


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create and return the CLI argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results_root",
        help="Root directory containing step*/.../*.jsonl",
    )
    parser.add_argument(
        "--split",
        default=None,
        help=(
            "Filter filenames containing this split substring "
            "(e.g., 'test')."
        ),
    )

    # Prompt-variant filtering
    parser.add_argument(
        "--max_prompt_versions",
        type=int,
        default=None,
        help=(
            "Drop any group that has more than this many distinct "
            "prompt variants."
        ),
    )
    parser.add_argument(
        "--prompt_key",
        default="prompt",
        help="Dot-path to the prompt field (default: 'prompt').",
    )
    parser.add_argument(
        "--strict_prompt_key",
        action="store_true",
        help="Only use --prompt_key; do not fall back to alternatives.",
    )
    parser.add_argument(
        "--filter_scope",
        choices=["per_problem", "global"],
        default="per_problem",
        help="Count prompt variants per problem (default) or globally.",
    )
    parser.add_argument(
        "--prompt_family_regex",
        default=None,
        help="Regex to strip version markers before counting variants.",
    )

    # Per-group hard cap
    parser.add_argument(
        "--max_per_group",
        type=int,
        default=None,
        help=(
            "Keep at most this many records per group "
            "(grouping uses --group_key, default 'problem')."
        ),
    )
    parser.add_argument(
        "--group_key",
        default="problem",
        help="Dot-path field to group by for capping (default: 'problem').",
    )

    # Write filtered copies or rewrite in place
    parser.add_argument(
        "--rewrite_filtered",
        action="store_true",
        help="Rewrite input JSONLs IN PLACE (dangerous).",
    )
    parser.add_argument(
        "--write_filtered_to",
        default=None,
        help=(
            "Write filtered JSONLs under this output root, "
            "mirroring the input tree."
        ),
    )
    parser.add_argument(
        "--aggregate_from_filtered",
        action="store_true",
        help="Aggregate from filtered copies instead of original JSONLs.",
    )

    # Crossword-specific: recompute correctness
    parser.add_argument(
        "--recompute_correctness",
        choices=["none", "substring", "exact"],
        default="none",
        help=(
            "Override recorded correctness using pred_answer_canon vs "
            "gold_answer_canon."
        ),
    )

    # CSV outputs
    parser.add_argument(
        "--save_csv",
        default=None,
        help="Optional CSV output path for the step table.",
    )
    parser.add_argument(
        "--per_example_csv",
        default=None,
        help="Optional CSV with per-example correctness (p1, p2, improved).",
    )
    return parser


# ------------------------------ Main -----------------------------------------

def main() -> None:
    """
    CLI entrypoint that filters and aggregates crossword inference JSONL results.

    :returns: ``None``. The function parses arguments, performs aggregation, and prints summaries.
    """
    parser = _build_arg_parser()
    args = parser.parse_args()

    files = scan_files(args.results_root, args.split)
    if not files:
        print("No JSONL files found. Check the path or split filter.")
        return

    drop_groups = _compute_prompt_drop_groups(files, args)
    wrote_any, files_for_agg = _maybe_write_filtered_files(files, args, drop_groups)
    aggregates = _aggregate_steps(files_for_agg, args, drop_groups, wrote_any)
    _print_step_summaries(aggregates)
    _write_csv_outputs(args, aggregates)


if __name__ == "__main__":
    main()
