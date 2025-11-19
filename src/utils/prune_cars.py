#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prune Carpark oversampling by idx (example_id), skip everything else.

Default behavior
----------------
- Only touches files whose *path* contains "carpark" (case-insensitive).
- For those files, groups by JSON field 'example_id' (e.g., "idx_0", "idx_1").
- Keeps at most K (=8 by default) per example_id, preferring unique sample_idx
  (i.e., 0..7 once each when available). Ties → lower sample_idx, then file order.
- For non-matching files (not "carpark" in path), **no changes** are made.

You can change what counts as "carpark" via --only-path-contains (repeatable),
and you can switch back to the old "problem"-style pruning for all domains with
--all-domains (but by default we DON'T touch non-carpark).

Examples
--------
# Carpark-only (default), limit to 8 per example_id, look at step*.jsonl
python prune_by_problem.py --root results

# Dry run, verbose
python prune_by_problem.py --root results --dry-run --verbose

# If your path doesn't include the word 'carpark', customize the substring(s):
python prune_by_problem.py --root results --only-path-contains RushHour --only-path-contains parking

# (Optional) Force prune all domains using problem-like keys (legacy behavior):
python prune_by_problem.py --root results --all-domains
"""

import argparse
import fnmatch
import random
from dataclasses import dataclass
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


# ---------------------------
# Utility: file + JSON helpers
# ---------------------------

def should_consider_filename(filename: str, globs: List[str]) -> bool:
    """Return True if the filename should be considered for pruning."""
    if not globs:
        return filename.lower().endswith(".jsonl")
    return any(fnmatch.fnmatch(filename, pattern) for pattern in globs)


def is_path_matched(path: str, needles: List[str]) -> bool:
    """Return True if any of the needles appears in the (lowercased) path."""
    if not needles:
        return True
    lower_path = path.lower()
    return any(needle.lower() in lower_path for needle in needles)


# ---------------------------
# Grouping key strategies
# ---------------------------


def group_key_by_field(obj: dict, line_idx: int, field: str) -> str:
    """Carpark: group by 'example_id' (idx). Unknown/missing → singleton line."""
    field_value = obj.get(field, None)
    if field_value is not None and not isinstance(field_value, (dict, list)):
        return f"{field}:{str(field_value)}"
    return f"__LINE__:{line_idx}"


@dataclass
class CarparkOptions:
    """Options specific to Carpark-style pruning."""

    settings: PruneSettings
    path_hints: List[str]
    idx_field: str = "example_id"


# ---------------------------
# Core prune routines
# ---------------------------

def prune_file_carpark_by_idx(
    path: str,
    options: CarparkOptions,
) -> Tuple[int, int]:
    """Touch ONLY if path matches the carpark hints; group by example_id."""
    lines = load_lines(path)

    # Skip if not a carpark file (by path substring)
    if not is_path_matched(path, options.path_hints):
        if options.settings.verbose:
            print(f"[SKIP] {path} — not matched by {options.path_hints}")
        return (len(lines), 0)

    by_group: Dict[str, List[Tuple[int, Optional[int]]]] = group_lines(
        lines,
        lambda obj, line_idx: group_key_by_field(obj, line_idx, options.idx_field),
    )
    return apply_prune(path, lines, by_group, options.settings)


def prune_file_problem_like(
    path: str,
    settings: PruneSettings,
) -> Tuple[int, int]:
    """Legacy mode: group by 'problem' or fallbacks, for ALL files."""
    lines = load_lines(path)
    by_group: Dict[str, List[Tuple[int, Optional[int]]]] = group_lines(
        lines,
        group_key_problem_like,
    )
    return apply_prune(path, lines, by_group, settings)


def main() -> None:
    """CLI entrypoint for carpark/problem-like pruning."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root directory (e.g., results)")
    add_common_prune_args(
        parser,
        max_per_help="Max records to keep per group (default: 8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for --strategy=random",
    )

    # Domain mode switches
    domain_group = parser.add_mutually_exclusive_group()
    domain_group.add_argument(
        "--all-domains",
        action="store_true",
        help="Apply legacy 'problem'-style pruning to ALL files (disables carpark-only).",
    )
    domain_group.add_argument(
        "--carpark-only",
        action="store_true",
        default=True,
        help=(
            "Touch only files whose path contains any --only-path-contains "
            "substrings (default)."
        ),
    )

    # Carpark controls
    parser.add_argument(
        "--only-path-contains",
        dest="only_paths",
        action="append",
        default=["carpark"],
        help='Substrings to identify Carpark files by path (repeatable). Default: "carpark"',
    )
    parser.add_argument(
        "--carpark-idx-field",
        default="example_id",
        help='JSON field used as the "idx" for Carpark grouping. Default: example_id',
    )
    parser.add_argument(
        "--carpark-strategy",
        choices=["sample_idx_unique", "sample_idx", "order", "random"],
        default="sample_idx_unique",
        help="Tie-breaker within each idx group (default: sample_idx_unique)",
    )

    args = parser.parse_args()
    random.seed(args.seed)

    files = walk_jsonl_files(args.root, args.globs, should_consider_filename)
    log_found_files(args.root, args.globs, files, args.verbose)

    base_settings = make_prune_settings(
        max_per=args.max_per,
        strategy="sample_idx",
        dry_run=args.dry_run,
        no_backup=args.no_backup,
        verbose=args.verbose,
    )

    if args.all_domains:
        # Legacy behavior: prune everything by problem-like keys
        aggregate_and_report(
            files,
            lambda path: prune_file_problem_like(path, base_settings),
            args.dry_run,
        )
    else:
        # Default: only carpark paths (by substring), grouped by example_id (idx)
        carpark_options = CarparkOptions(
            settings=make_prune_settings(
                max_per=args.max_per,
                strategy=args.carpark_strategy,
                dry_run=args.dry_run,
                no_backup=args.no_backup,
                verbose=args.verbose,
            ),
            path_hints=args.only_paths,
            idx_field=args.carpark_idx_field,
        )
        aggregate_and_report(
            files,
            lambda path: prune_file_carpark_by_idx(path, carpark_options),
            args.dry_run,
        )


if __name__ == "__main__":
    main()
