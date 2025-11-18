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
import json
import os
import random
import shutil
import sys
import tempfile
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# Ordered fallbacks for legacy "problem"-style grouping (NO example_id here)
FALLBACK_KEYS = ["problem_id", "question", "clue", "title", "id", "uid"]


# ---------------------------
# Utility: file + JSON helpers
# ---------------------------

def load_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()

def parse_json_safe(line: str) -> Optional[dict]:
    s = line.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None

def should_consider_filename(fn: str, globs: List[str]) -> bool:
    if not globs:
        return fn.lower().endswith(".jsonl")
    return any(fnmatch.fnmatch(fn, pat) for pat in globs)

def is_path_matched(path: str, needles: List[str]) -> bool:
    if not needles:
        return True
    lp = path.lower()
    return any(n.lower() in lp for n in needles)


# ---------------------------
# Grouping key strategies
# ---------------------------

def group_key_problem_like(obj: dict, line_idx: int) -> str:
    """Legacy: primary 'problem', else fallbacks; otherwise singleton line."""
    v = obj.get("problem", None)
    if v is not None and not isinstance(v, (dict, list)):
        return f"problem:{str(v)}"
    for k in FALLBACK_KEYS:
        vv = obj.get(k, None)
        if vv is not None and not isinstance(vv, (dict, list)):
            return f"{k}:{str(vv)}"
    return f"__LINE__:{line_idx}"

def group_key_by_field(obj: dict, line_idx: int, field: str) -> str:
    """Carpark: group by 'example_id' (idx). Unknown/missing → singleton line."""
    v = obj.get(field, None)
    if v is not None and not isinstance(v, (dict, list)):
        return f"{field}:{str(v)}"
    return f"__LINE__:{line_idx}"


# ---------------------------
# Selection (keep) strategies
# ---------------------------

def choose_kept_indices(
    items: List[Tuple[int, Optional[int]]],
    keep_k: int,
    strategy: str
) -> set:
    """
    items: list of (line_index_in_file, sample_idx or None) belonging to a group.
    Returns set of line indices to KEEP.
    """
    if len(items) <= keep_k:
        return {i for (i, _) in items}

    # Helper: stable sort by (missing last) → sample_idx asc → original order
    items_sorted = sorted(
        items,
        key=lambda t: (t[1] is None, t[1] if t[1] is not None else 10**9, t[0])
    )

    if strategy == "sample_idx":
        kept = items_sorted[:keep_k]

    elif strategy == "order":
        kept = items[:keep_k]

    elif strategy == "random":
        kept = random.sample(items, keep_k)

    elif strategy == "sample_idx_unique":
        # Prefer up to one of each unique sample_idx (0..7), then fill leftovers
        kept = []
        seen = set()
        # First pass: unique sample_idx
        for (i, sidx) in items_sorted:
            if sidx is None:
                continue
            if sidx not in seen:
                kept.append((i, sidx))
                seen.add(sidx)
                if len(kept) >= keep_k:
                    break
        # Second pass: fill with anything else if needed
        if len(kept) < keep_k:
            for (i, sidx) in items_sorted:
                if (i, sidx) in kept:
                    continue
                kept.append((i, sidx))
                if len(kept) >= keep_k:
                    break
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return {i for (i, _) in kept}


# ---------------------------
# Core prune routines
# ---------------------------

def prune_file_carpark_by_idx(
    path: str,
    keep_k: int,
    strategy: str,
    dry: bool,
    backup: bool,
    verbose: bool,
    path_hints: List[str],
    idx_field: str = "example_id",
) -> Tuple[int, int]:
    """Touch ONLY if path matches the carpark hints; group by example_id."""
    lines = load_lines(path)

    # Skip if not a carpark file (by path substring)
    if not is_path_matched(path, path_hints):
        if verbose:
            print(f"[SKIP] {path} — not matched by {path_hints}")
        return (len(lines), 0)

    by_group: Dict[str, List[Tuple[int, Optional[int]]]] = defaultdict(list)

    for idx, line in enumerate(lines):
        obj = parse_json_safe(line)
        if obj is None:
            # Preserve raw/invalid JSON lines
            by_group[f"__RAW__:{idx}"].append((idx, None))
            continue

        gkey = group_key_by_field(obj, idx, idx_field)

        sidx = obj.get("sample_idx", None)
        try:
            sidx = int(sidx) if sidx is not None else None
        except Exception:
            sidx = None

        by_group[gkey].append((idx, sidx))

    keep_idx: set = set()
    for gkey, items in by_group.items():
        keep_idx |= choose_kept_indices(items, keep_k, strategy)

    # Build output
    kept_lines: List[str] = []
    removed = 0
    for i, line in enumerate(lines):
        if i in keep_idx:
            kept_lines.append(line)
        else:
            removed += 1

    if removed == 0:
        if verbose:
            print(f"[SKIP] {path} — nothing to prune.")
        return (len(lines), 0)

    if dry:
        print(f"[DRY]  {path} → would remove {removed} / {len(lines)} lines")
        return (len(kept_lines), removed)

    # Atomic write with optional backup
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=".prune_tmp_", dir=os.path.dirname(path))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as w:
            w.writelines(kept_lines)
        if backup:
            shutil.copy2(path, path + ".bak")
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

    print(f"[DONE] {path} → removed {removed} / {len(lines)}; kept {len(kept_lines)}.")
    return (len(kept_lines), removed)


def prune_file_problem_like(
    path: str,
    keep_k: int,
    strategy: str,
    dry: bool,
    backup: bool,
    verbose: bool,
) -> Tuple[int, int]:
    """Legacy mode: group by 'problem' or fallbacks, for ALL files."""
    lines = load_lines(path)
    by_group: Dict[str, List[Tuple[int, Optional[int]]]] = defaultdict(list)

    for idx, line in enumerate(lines):
        obj = parse_json_safe(line)
        if obj is None:
            by_group[f"__RAW__:{idx}"].append((idx, None))
            continue

        gkey = group_key_problem_like(obj, idx)
        sidx = obj.get("sample_idx", None)
        try:
            sidx = int(sidx) if sidx is not None else None
        except Exception:
            sidx = None
        by_group[gkey].append((idx, sidx))

    keep_idx: set = set()
    for gkey, items in by_group.items():
        keep_idx |= choose_kept_indices(items, keep_k, strategy)

    kept_lines: List[str] = []
    removed = 0
    for i, line in enumerate(lines):
        if i in keep_idx:
            kept_lines.append(line)
        else:
            removed += 1

    if removed == 0:
        if verbose:
            print(f"[SKIP] {path} — nothing to prune.")
        return (len(lines), 0)

    if dry:
        print(f"[DRY]  {path} → would remove {removed} / {len(lines)} lines")
        return (len(kept_lines), removed)

    tmp_fd, tmp_path = tempfile.mkstemp(prefix=".prune_tmp_", dir=os.path.dirname(path))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as w:
            w.writelines(kept_lines)
        if backup:
            shutil.copy2(path, path + ".bak")
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

    print(f"[DONE] {path} → removed {removed} / {len(lines)}; kept {len(kept_lines)}.")
    return (len(kept_lines), removed)


# ---------------------------
# Walk and main
# ---------------------------

def walk(root: str, globs: List[str]) -> List[str]:
    files = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if should_consider_filename(fn, globs):
                files.append(os.path.join(dp, fn))
    files.sort()
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root directory (e.g., results)")
    ap.add_argument(
        "--glob", dest="globs", action="append", default=["step*.jsonl"],
        help="Filename glob (repeatable). Default: step*.jsonl"
    )
    ap.add_argument("--max-per", type=int, default=8,
                    help="Max records to keep per group (default: 8)")

    ap.add_argument("--dry-run", action="store_true", help="Only report changes")
    ap.add_argument("--no-backup", action="store_true", help="Do not create .bak backups")
    ap.add_argument("--verbose", action="store_true", help="Print per-file info")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for --strategy=random")

    # Domain mode switches
    mx = ap.add_mutually_exclusive_group()
    mx.add_argument(
        "--all-domains", action="store_true",
        help="Apply legacy 'problem'-style pruning to ALL files (disables carpark-only)."
    )
    mx.add_argument(
        "--carpark-only", action="store_true", default=True,
        help="Touch only files whose path contains any --only-path-contains substrings (default)."
    )

    # Carpark controls
    ap.add_argument(
        "--only-path-contains", dest="only_paths", action="append", default=["carpark"],
        help='Substrings to identify Carpark files by path (repeatable). Default: "carpark"'
    )
    ap.add_argument(
        "--carpark-idx-field", default="example_id",
        help='JSON field used as the "idx" for Carpark grouping. Default: example_id'
    )
    ap.add_argument(
        "--carpark-strategy", choices=["sample_idx_unique", "sample_idx", "order", "random"],
        default="sample_idx_unique",
        help="Tie-breaker within each idx group (default: sample_idx_unique)"
    )

    args = ap.parse_args()
    random.seed(args.seed)

    files = walk(args.root, args.globs)
    if args.verbose:
        print(f"[INFO] Found {len(files)} files under {args.root} matching {args.globs or ['*.jsonl']}")

    kept_total = 0
    removed_total = 0

    if args.all_domains:
        # Legacy behavior: prune everything by problem-like keys
        for path in files:
            k, r = prune_file_problem_like(
                path, args.max_per, "sample_idx", args.dry_run, not args.no_backup, args.verbose
            )
            kept_total += k
            removed_total += r
    else:
        # Default: only carpark paths (by substring), grouped by example_id (idx)
        for path in files:
            k, r = prune_file_carpark_by_idx(
                path,
                args.max_per,
                args.carpark_strategy,
                args.dry_run,
                not args.no_backup,
                args.verbose,
                args.only_paths,
                idx_field=args.carpark_idx_field,
            )
            kept_total += k
            removed_total += r

    print(f"\n[SUMMARY] files: {len(files)} | lines kept: {kept_total} | lines removed: {removed_total}")
    if args.dry_run:
        print("[NOTE] Dry run only. Re-run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
