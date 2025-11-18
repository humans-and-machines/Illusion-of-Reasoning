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
import json
import os
import random
import shutil
import sys
import tempfile
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# Ordered fallbacks (NO example_id here; user asked "not idx")
FALLBACK_KEYS = ["problem_id", "question", "clue", "title", "id", "uid"]

def group_key_for(obj: dict, line_idx: int) -> str:
    # Primary: 'problem'
    v = obj.get("problem", None)
    if v is not None and not isinstance(v, (dict, list)):
        return f"problem:{str(v)}"
    # Fallbacks (still problem-like)
    for k in FALLBACK_KEYS:
        vv = obj.get(k, None)
        if vv is not None and not isinstance(vv, (dict, list)):
            return f"{k}:{str(vv)}"
    # Ungroupable: treat as its own singleton to avoid accidental pruning
    return f"__LINE__:{line_idx}"

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

def choose_kept_indices(items: List[Tuple[int, Optional[int]]], keep_k: int, strategy: str) -> set:
    """
    items: list of (line_index, sample_idx or None) for a single problem group.
    Returns set of line_index to KEEP.
    """
    if len(items) <= keep_k:
        return {i for (i, _) in items}

    if strategy == "sample_idx":
        # Sort by: (missing sample_idx last) → sample_idx → original file order
        items_sorted = sorted(items, key=lambda t: (t[1] is None, t[1] if t[1] is not None else 10**9, t[0]))
        kept = items_sorted[:keep_k]
    elif strategy == "order":
        kept = items[:keep_k]
    elif strategy == "random":
        kept = random.sample(items, keep_k)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return {i for (i, _) in kept}

def should_consider(fn: str, globs: List[str]) -> bool:
    if not globs:
        return fn.lower().endswith(".jsonl")
    return any(fnmatch.fnmatch(fn, pat) for pat in globs)

def prune_file(path: str, keep_k: int, strategy: str, dry: bool, backup: bool, verbose: bool) -> Tuple[int,int]:
    lines = load_lines(path)
    by_group: Dict[str, List[Tuple[int, Optional[int]]]] = defaultdict(list)

    for idx, line in enumerate(lines):
        obj = parse_json_safe(line)
        if obj is None:
            # keep raw lines; one-off group so they won't be pruned
            by_group[f"__RAW__:{idx}"].append((idx, None))
            continue
        gkey = group_key_for(obj, idx)
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

def walk(root: str, globs: List[str]) -> List[str]:
    files = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if should_consider(fn, globs):
                files.append(os.path.join(dp, fn))
    files.sort()
    return files

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root directory (e.g., results)")
    ap.add_argument("--glob", dest="globs", action="append", default=["step*.jsonl"],
                    help="Filename glob (repeatable). Default: step*.jsonl")
    ap.add_argument("--max-per", type=int, default=8, help="Max records to keep per problem (default: 8)")
    ap.add_argument("--strategy", choices=["sample_idx","order","random"], default="sample_idx",
                    help="Tie-breaker when > max-per (default: sample_idx asc)")
    ap.add_argument("--dry-run", action="store_true", help="Only report changes")
    ap.add_argument("--no-backup", action="store_true", help="Do not create .bak backups")
    ap.add_argument("--verbose", action="store_true", help="Print per-file info")
    args = ap.parse_args()

    files = walk(args.root, args.globs)
    if args.verbose:
        print(f"[INFO] Found {len(files)} files under {args.root} matching {args.globs or ['*.jsonl']}")

    kept_total = 0
    removed_total = 0
    for path in files:
        k, r = prune_file(path, args.max_per, args.strategy, args.dry_run, not args.no_backup, args.verbose)
        kept_total += k
        removed_total += r

    print(f"\n[SUMMARY] files: {len(files)} | lines kept: {kept_total} | lines removed: {removed_total}")
    if args.dry_run:
        print("[NOTE] Dry run only. Re-run without --dry-run to apply changes.")

if __name__ == "__main__":
    main()
