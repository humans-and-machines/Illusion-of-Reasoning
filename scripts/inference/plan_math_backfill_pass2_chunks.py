#!/usr/bin/env python3
"""
Plan chunked backfill runs for missing pass-2 in existing MATH result JSONLs.

Why this exists
---------------
`src.inference.cli.backfill_math_pass2` can rewrite an entire results JSONL,
but a full-file backfill can be too long for strict walltime limits.

This planner:
  - scans each `step????_test.jsonl` under `artifacts/results/*math-temp-*`,
  - counts how many distinct `problem` values still have missing pass2,
  - expands each file into `ceil(missing_problems / chunk_problems)` identical
    manifest lines (so each Slurm task can backfill only a small chunk),
  - prints an `sbatch --array=...%<cap>` command.

Each array task should be run with:
  - `MAX_PROBLEMS=<chunk_problems>` so the backfill script fills at most that
    many missing problems per run, and
  - `BATCH_SIZE=1` if you need to keep CUDA memory low (one row generated at a time).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Set, Tuple


@dataclass(frozen=True)
class FileStats:
    family: str
    temp: str
    step: int
    jsonl_path: Path
    missing_rows: int
    missing_problems: int
    skipped_no_pass1: int


def _iter_temp_roots(results_root: Path) -> Iterable[Path]:
    for pattern in (
        "GRPO-1.5B-math-temp-*",
        "GRPO-7B-math-temp-*",
        "GRPO-Llama8B-math-temp-*",
    ):
        yield from sorted(results_root.glob(pattern))


def _family_for_root_name(name: str) -> Optional[str]:
    if name.startswith("GRPO-1.5B-math-temp-"):
        return "1.5B"
    if name.startswith("GRPO-7B-math-temp-"):
        return "7B"
    if name.startswith("GRPO-Llama8B-math-temp-"):
        return "Llama8B"
    return None


def _temp_for_root_name(name: str) -> str:
    return name.split("temp-", 1)[-1]


def _iter_step_jsonls(temp_root: Path) -> Iterable[Path]:
    yield from temp_root.glob("step*/step*_test.jsonl")
    yield from temp_root.glob("step*_test.jsonl")


def _parse_step_from_name(filename: str) -> Optional[int]:
    if not filename.startswith("step") or "_test.jsonl" not in filename:
        return None
    step_str = filename[4:8]
    if not step_str.isdigit():
        return None
    return int(step_str)


def _is_missing_pass(obj: dict, pass_key: str) -> bool:
    if pass_key not in obj:
        return True
    value = obj.get(pass_key)
    return value is None or value == {}


def _pass1_output(obj: dict) -> Optional[str]:
    pass1_obj = obj.get("pass1") or {}
    if not isinstance(pass1_obj, dict):
        return None
    output = pass1_obj.get("output")
    if isinstance(output, str) and output.strip():
        return output
    return None


def _scan_file(path: Path, *, pass_key: str) -> Tuple[int, Set[str], int]:
    missing_rows = 0
    missing_problems: Set[str] = set()
    skipped_no_pass1 = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            problem = obj.get("problem")
            if not isinstance(problem, str) or not problem:
                continue
            if not _is_missing_pass(obj, pass_key):
                continue
            if _pass1_output(obj) is None:
                skipped_no_pass1 += 1
                continue
            missing_rows += 1
            missing_problems.add(problem)
    return missing_rows, missing_problems, skipped_no_pass1


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", default="artifacts/results")
    parser.add_argument("--out", required=True, help="Output manifest path.")
    parser.add_argument("--only_family", default=None, help="Optional: 1.5B | 7B | Llama8B")
    parser.add_argument("--only_temp", default=None, help="Optional temperature suffix (e.g. 0.3)")
    parser.add_argument("--min_step", type=int, default=None)
    parser.add_argument("--max_step", type=int, default=None)
    parser.add_argument("--pass2_key", default="pass2")
    parser.add_argument(
        "--chunk_problems",
        type=int,
        default=1,
        help="How many distinct problems to backfill per Slurm task (default: 1).",
    )
    parser.add_argument("--array_cap", type=int, default=2000, help="Max concurrent Slurm array tasks (default: 2000).")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.chunk_problems <= 0:
        raise ValueError("--chunk_problems must be >= 1")

    results_root = Path(args.results_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats: list[FileStats] = []
    lines: list[str] = []

    for temp_root in _iter_temp_roots(results_root):
        if not temp_root.is_dir():
            continue
        family = _family_for_root_name(temp_root.name)
        if family is None:
            continue
        temp = _temp_for_root_name(temp_root.name)

        if args.only_family and family != args.only_family:
            continue
        if args.only_temp and temp != args.only_temp:
            continue

        for jsonl in sorted(_iter_step_jsonls(temp_root)):
            step = _parse_step_from_name(jsonl.name)
            if step is None:
                continue
            if args.min_step is not None and step < args.min_step:
                continue
            if args.max_step is not None and step > args.max_step:
                continue
            if not jsonl.is_file():
                continue

            missing_rows, missing_problems, skipped_no_pass1 = _scan_file(jsonl, pass_key=args.pass2_key)
            if not missing_problems:
                continue

            missing_problem_count = len(missing_problems)
            chunks = int(math.ceil(missing_problem_count / float(args.chunk_problems)))
            for _ in range(chunks):
                lines.append(f"{family}\t{temp}\t{step}\t{jsonl.as_posix()}\n")

            stats.append(
                FileStats(
                    family=family,
                    temp=temp,
                    step=step,
                    jsonl_path=jsonl,
                    missing_rows=missing_rows,
                    missing_problems=missing_problem_count,
                    skipped_no_pass1=skipped_no_pass1,
                )
            )

    with out_path.open("w", encoding="utf-8") as handle:
        handle.writelines(lines)

    n = len(lines)
    print(f"Wrote {n} chunk tasks → {out_path}")
    if not stats:
        return

    # Brief summary: show the worst offenders.
    stats_sorted = sorted(stats, key=lambda s: (s.missing_problems, s.missing_rows), reverse=True)
    top = stats_sorted[:10]
    print("Top missing (by distinct problems):")
    for s in top:
        print(
            f"- {s.jsonl_path}: missing_problems={s.missing_problems}, missing_rows={s.missing_rows}, "
            f"skipped_no_pass1={s.skipped_no_pass1}"
        )

    if n > 0:
        print("\nSuggested Slurm command:")
        print(
            f"  N=$(wc -l < {out_path})\n"
            f"  sbatch --array=0-$((N-1))%{args.array_cap} scripts/inference/math-backfill-pass2-all.slurm "
            f"MANIFEST={out_path} MAX_PROBLEMS={args.chunk_problems} BATCH_SIZE=1"
        )


if __name__ == "__main__":
    main()

