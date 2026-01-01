#!/usr/bin/env python3
"""
Write a manifest of math result JSONLs that need pass2 backfill.

The manifest is consumed by `scripts/inference/math-backfill-pass2-all.slurm`
in Slurm array mode (one file per task).

Each output line is tab-separated:
  family<TAB>temp<TAB>step<TAB>jsonl_path
where:
  family ∈ {1.5B, 7B, Llama8B}
  temp is the suffix after `...-temp-`
  step is parsed from the JSONL filename `step####_*.jsonl`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional


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


def _file_has_missing_pass2(path: Path) -> bool:
    try:
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
                # Treat missing, null, and `{}` as "missing pass2" (consistent with the backfill CLI).
                if "pass2" not in obj:
                    return True
                value = obj.get("pass2")
                if value is None or value == {}:
                    return True
    except OSError:
        return False
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_root",
        default="artifacts/results",
        help="Root directory containing GRPO-*-math-temp-* folders.",
    )
    parser.add_argument("--out", required=True, help="Output manifest path.")
    parser.add_argument("--only_family", default=None, help="Optional: 1.5B | 7B | Llama8B")
    parser.add_argument("--only_temp", default=None, help="Optional temperature suffix (e.g. 0.3)")
    parser.add_argument("--min_step", type=int, default=None, help="Optional minimum step (inclusive).")
    parser.add_argument("--max_step", type=int, default=None, help="Optional maximum step (inclusive).")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
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
            if _file_has_missing_pass2(jsonl):
                lines.append(f"{family}\t{temp}\t{step}\t{jsonl.as_posix()}\n")

    with out_path.open("w", encoding="utf-8") as handle:
        handle.writelines(lines)

    print(f"Wrote {len(lines)} targets → {out_path}")


if __name__ == "__main__":
    main()
