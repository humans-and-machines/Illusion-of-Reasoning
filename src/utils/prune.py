#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prune per-problem oversampling in results JSONLs (group by 'problem').

- Groups records by the JSON field 'problem'. If absent, falls back to a stable
  problem-like key (excluding 'example_id' / idx_*): problem_id, question, clue,
  title, id, uid. If none exist, the line is treated as its own group.
- For each problem group, keep at most K (=8 by default) records, preferring
  smaller sample_idx (0..7). If sample_idx is missing/duplicated, falls back to
  file order.

Usage
-----
python prune_by_problem.py --root results
python prune_by_problem.py --root results --glob 'step*_test.jsonl' --dry-run
python prune_by_problem.py --root results --glob 'step*_test.jsonl' --max-per 8 --no-backup
"""

import argparse
import fnmatch
from typing import Dict, List, Optional, Tuple

from _prune_common import (
    PruneSettings,
    add_common_prune_args,
    aggregate_and_report,
    apply_prune,
    group_key_problem_like,
    group_lines,
    load_lines,
    log_found_files,
    make_prune_settings,
    walk_jsonl_files,
)


def should_consider(filename: str, globs: List[str]) -> bool:
    """Return True if the filename should be considered for pruning."""
    if not globs:
        return filename.lower().endswith(".jsonl")
    return any(fnmatch.fnmatch(filename, pattern) for pattern in globs)


def prune_file(path: str, settings: PruneSettings) -> Tuple[int, int]:
    """Prune a single JSONL file and return (kept_lines, removed_lines)."""
    lines = load_lines(path)
    by_group: Dict[str, List[Tuple[int, Optional[int]]]] = group_lines(
        lines,
        group_key_problem_like,
    )
    return apply_prune(
        path,
        lines,
        by_group,
        settings,
    )


def main() -> None:
    """CLI entrypoint for pruning oversampled per-problem JSONL results."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        required=True,
        help="Root directory (e.g., results)",
    )
    add_common_prune_args(
        parser,
        max_per_help="Max records to keep per problem (default: 8)",
    )
    parser.add_argument(
        "--strategy",
        choices=["sample_idx", "order", "random"],
        default="sample_idx",
        help="Tie-breaker when > max-per (default: sample_idx asc)",
    )
    args = parser.parse_args()

    files = walk_jsonl_files(args.root, args.globs, should_consider)
    log_found_files(args.root, args.globs, files, args.verbose)

    settings = make_prune_settings(
        max_per=args.max_per,
        strategy=args.strategy,
        dry_run=args.dry_run,
        no_backup=args.no_backup,
        verbose=args.verbose,
    )

    aggregate_and_report(
        files,
        lambda path: prune_file(path, settings),
        args.dry_run,
    )


if __name__ == "__main__":
    main()
