#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for JSONL pruning scripts."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple


JsonGroups = Dict[str, List[Tuple[int, Optional[int]]]]


FALLBACK_KEYS = ["problem_id", "question", "clue", "title", "id", "uid"]


@dataclass
class PruneSettings:
    """Configuration controlling how pruning is applied."""

    keep_k: int
    strategy: str
    dry_run: bool
    backup: bool
    verbose: bool


def load_lines(path: str) -> List[str]:
    """Return all lines from a UTF-8 JSONL file."""
    with open(path, "r", encoding="utf-8") as file_handle:
        return file_handle.readlines()


def parse_json_safe(line: str) -> Optional[dict]:
    """Parse a single JSONL line, returning None on failure."""
    stripped_line = line.strip()
    if not stripped_line:
        return None
    try:
        return json.loads(stripped_line)
    except (json.JSONDecodeError, TypeError):
        return None


def group_key_problem_like(obj: dict, line_idx: int) -> str:
    """Return a stable grouping key for a JSONL record (problem-like fields)."""
    problem_value = obj.get("problem", None)
    if problem_value is not None and not isinstance(problem_value, (dict, list)):
        return f"problem:{str(problem_value)}"
    for key in FALLBACK_KEYS:
        fallback_value = obj.get(key, None)
        if fallback_value is not None and not isinstance(fallback_value, (dict, list)):
            return f"{key}:{str(fallback_value)}"
    return f"__LINE__:{line_idx}"


def _strategy_sample_idx(
    items: List[Tuple[int, Optional[int]]],
    items_sorted: List[Tuple[int, Optional[int]]],
    keep_k: int,
) -> List[Tuple[int, Optional[int]]]:
    """Keep the first K items after sorting by sample index and file order."""
    del items  # unused; kept for uniform signature
    return items_sorted[:keep_k]


def _strategy_sample_idx_unique(
    items: List[Tuple[int, Optional[int]]],
    items_sorted: List[Tuple[int, Optional[int]]],
    keep_k: int,
) -> List[Tuple[int, Optional[int]]]:
    """
    Prefer unique sample_idx values; fall back to remaining items to reach keep_k.
    """
    del items  # unused; kept for uniform signature
    kept_items: List[Tuple[int, Optional[int]]] = []
    seen_sample_indices: Set[int] = set()

    for line_index, sample_idx in items_sorted:
        if sample_idx is None:
            continue
        if sample_idx not in seen_sample_indices:
            kept_items.append((line_index, sample_idx))
            seen_sample_indices.add(sample_idx)
            if len(kept_items) >= keep_k:
                break

    if len(kept_items) < keep_k:
        kept_set = set(kept_items)
        for line_index, sample_idx in items_sorted:
            if (line_index, sample_idx) in kept_set:
                continue
            kept_items.append((line_index, sample_idx))
            kept_set.add((line_index, sample_idx))
            if len(kept_items) >= keep_k:
                break
    return kept_items


def _strategy_order(
    items: List[Tuple[int, Optional[int]]],
    items_sorted: List[Tuple[int, Optional[int]]],
    keep_k: int,
) -> List[Tuple[int, Optional[int]]]:
    """Keep the first K items in original file order."""
    del items_sorted  # unused; kept for uniform signature
    return items[:keep_k]


def _strategy_random(
    items: List[Tuple[int, Optional[int]]],
    items_sorted: List[Tuple[int, Optional[int]]],
    keep_k: int,
) -> List[Tuple[int, Optional[int]]]:
    """Sample K items uniformly at random."""
    del items_sorted  # unused; kept for uniform signature
    return random.sample(items, keep_k)


def choose_kept_indices(
    items: List[Tuple[int, Optional[int]]],
    keep_k: int,
    strategy: str,
) -> Set[int]:
    """
    items: list of (line_index, sample_idx or None) for a single group.
    Returns set of line_index to KEEP.
    """
    if len(items) <= keep_k:
        return {line_index for (line_index, _) in items}

    # Stable sort by: (missing sample_idx last) → sample_idx → original file order
    items_sorted = sorted(
        items,
        key=lambda item: (
            item[1] is None,
            item[1] if item[1] is not None else 10**9,
            item[0],
        ),
    )

    strategy_funcs = {
        "sample_idx": _strategy_sample_idx,
        "sample_idx_unique": _strategy_sample_idx_unique,
        "order": _strategy_order,
        "random": _strategy_random,
    }
    try:
        kept_items = strategy_funcs[strategy](items, items_sorted, keep_k)
    except KeyError as exc:
        raise ValueError(f"Unknown strategy: {strategy}") from exc

    return {line_index for (line_index, _) in kept_items}


def group_lines(
    lines: List[str],
    group_key_func: Callable[[dict, int], str],
) -> JsonGroups:
    """Group JSONL lines by the provided key function."""
    by_group: JsonGroups = defaultdict(list)
    for idx, line in enumerate(lines):
        obj = parse_json_safe(line)
        if obj is None:
            # Preserve raw/invalid JSON lines as their own singleton groups.
            by_group[f"__RAW__:{idx}"].append((idx, None))
            continue
        group_key = group_key_func(obj, idx)
        sample_idx = obj.get("sample_idx", None)
        try:
            sample_idx = int(sample_idx) if sample_idx is not None else None
        except (TypeError, ValueError):
            sample_idx = None
        by_group[group_key].append((idx, sample_idx))
    return by_group


def apply_prune(
    path: str,
    lines: List[str],
    by_group: JsonGroups,
    settings: PruneSettings,
) -> Tuple[int, int]:
    """Apply pruning to grouped lines and write the result back."""
    keep_indices: Set[int] = set()
    for _, items in by_group.items():
        keep_indices |= choose_kept_indices(
            items,
            settings.keep_k,
            settings.strategy,
        )

    kept_lines: List[str] = []
    removed = 0
    for index, line in enumerate(lines):
        if index in keep_indices:
            kept_lines.append(line)
        else:
            removed += 1

    if removed == 0:
        if settings.verbose:
            print(f"[SKIP] {path} — nothing to prune.")
        return len(lines), 0

    if settings.dry_run:
        print(
            f"[DRY]  {path} → would remove {removed} / {len(lines)} lines",
        )
        return len(kept_lines), removed

    _write_pruned_file(path, kept_lines, settings)

    if settings.verbose:
        print(
            f"[DONE] {path} → removed {removed} / {len(lines)}; kept {len(kept_lines)}.",
        )
    return len(kept_lines), removed


def _safe_remove(path: str) -> None:
    """Best-effort removal of a temporary file."""
    if not path:
        return
    if not os.path.exists(path):
        return
    try:
        os.remove(path)
    except OSError:
        # Ignore failures removing the temp file.
        pass


def _write_pruned_file(path: str, kept_lines: List[str], settings: PruneSettings) -> None:
    """Atomically write pruned lines back to disk, with optional backup."""
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=".prune_tmp_",
        dir=os.path.dirname(path),
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as output_file:
            output_file.writelines(kept_lines)
        if settings.backup:
            shutil.copy2(path, path + ".bak")
        os.replace(tmp_path, path)
    finally:
        _safe_remove(tmp_path)


def walk_jsonl_files(
    root: str,
    globs: List[str],
    should_consider: Callable[[str, List[str]], bool],
) -> List[str]:
    """Walk the directory tree under root and collect matching JSONL files."""
    files: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if should_consider(filename, globs):
                files.append(os.path.join(dirpath, filename))
    files.sort()
    return files


def print_summary(
    files_count: int,
    kept_total: int,
    removed_total: int,
    dry_run: bool,
) -> None:
    """Print a short summary of pruning results."""
    print(
        f"\n[SUMMARY] files: {files_count} | lines kept: {kept_total} | lines removed: {removed_total}",
    )
    if dry_run:
        print("[NOTE] Dry run only. Re-run without --dry-run to apply changes.")


def add_common_prune_args(
    parser: argparse.ArgumentParser,
    max_per_help: str,
) -> None:
    """Add CLI arguments shared by prune scripts."""
    parser.add_argument(
        "--glob",
        dest="globs",
        action="append",
        default=["step*.jsonl"],
        help="Filename glob (repeatable). Default: step*.jsonl",
    )
    parser.add_argument(
        "--max-per",
        type=int,
        default=8,
        help=max_per_help,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report changes",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .bak backups",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file info",
    )


def log_found_files(
    root: str,
    globs: List[str],
    files: List[str],
    verbose: bool,
) -> None:
    """Log a short summary of discovered files when verbose is enabled."""
    if not verbose:
        return
    print(
        f"[INFO] Found {len(files)} files under {root} matching {globs or ['*.jsonl']}",
    )


def make_prune_settings(
    max_per: int,
    strategy: str,
    dry_run: bool,
    no_backup: bool,
    verbose: bool,
) -> PruneSettings:
    """Construct a PruneSettings instance from CLI-style flags."""
    return PruneSettings(
        keep_k=max_per,
        strategy=strategy,
        dry_run=dry_run,
        backup=not no_backup,
        verbose=verbose,
    )


def aggregate_prune_over_files(
    files: List[str],
    prune_func: Callable[[str], Tuple[int, int]],
) -> Tuple[int, int]:
    """Apply a prune function across files and aggregate totals."""
    kept_total = 0
    removed_total = 0
    for path in files:
        kept, removed = prune_func(path)
        kept_total += kept
        removed_total += removed
    return kept_total, removed_total


def aggregate_and_report(
    files: List[str],
    prune_func: Callable[[str], Tuple[int, int]],
    dry_run: bool,
) -> None:
    """Aggregate pruning over files and print a standard summary."""
    kept_total, removed_total = aggregate_prune_over_files(files, prune_func)
    print_summary(len(files), kept_total, removed_total, dry_run)
