#!/usr/bin/env python3
"""Audit inference logs for missing rows per checkpoint.

This utility counts rows in each ``step*_test.jsonl`` under the
predefined results roots and reports any files that fall short of the
expected number of samples. The default configuration matches the
paper's checkpoint selection (steps ≤950 for 1.5B runs, ≤500 for the
7B/8B math runs).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class DatasetSpec:
    """Description of a dataset family to audit."""

    label: str
    target_rows: int
    roots: List[str]
    step_regex: re.Pattern[str]


@dataclass(frozen=True)
class FileAudit:
    """Audit stats for one JSONL results file."""

    path: Path
    rows: int
    rows_missing_vs_target: int
    pass1_filled: int
    pass2_filled: int
    pass2_missing_in_file: int
    pass2_missing_vs_target: int


DATASETS: List[DatasetSpec] = [
    DatasetSpec(
        label="Llama-8B math (≤500)",
        target_rows=4000,
        roots=[
            "artifacts/results/GRPO-Llama8B-math-temp-0",
            "artifacts/results/GRPO-Llama8B-math-temp-0.05",
            "artifacts/results/GRPO-Llama8B-math-temp-0.3",
            "artifacts/results/GRPO-Llama8B-math-temp-0.7",
        ],
        step_regex=re.compile(
            r".*/step-?0?[0-4]\d\d(?:/step-?0?[0-4]\d\d)?_test\.jsonl"
        ),
    ),
    DatasetSpec(
        label="Qwen-7B math (≤500)",
        target_rows=4000,
        roots=[
            "artifacts/results/GRPO-7B-math-temp-0",
            "artifacts/results/GRPO-7B-math-temp-0.05",
            "artifacts/results/GRPO-7B-math-temp-0.3",
            "artifacts/results/GRPO-7B-math-temp-0.7",
        ],
        step_regex=re.compile(
            r".*/step-?0?[0-4]\d\d(?:/step-?0?[0-4]\d\d)?_test\.jsonl"
        ),
    ),
    DatasetSpec(
        label="Qwen-1.5B math (≤950)",
        target_rows=4000,
        roots=[
            "artifacts/results/GRPO-1.5B-math-temp-0.0",
            "artifacts/results/GRPO-1.5B-math-temp-0.05",
            "artifacts/results/GRPO-1.5B-math-temp-0.3",
            "artifacts/results/GRPO-1.5B-math-temp-0.7",
        ],
        step_regex=re.compile(
            r".*/step-?0?[0-9]{3}(?:/step-?0?[0-9]{3})?_test\.jsonl"
        ),
    ),
    DatasetSpec(
        label="Crossword (≤950)",
        target_rows=1040,
        roots=[
            "artifacts/results/GRPO-1.5B-xword-temp-0",
            "artifacts/results/GRPO-1.5B-xword-temp-0.05",
            "artifacts/results/GRPO-1.5B-xword-temp-0.3",
            "artifacts/results/GRPO-1.5B-xword-temp-0.7",
        ],
        step_regex=re.compile(
            r".*/step-?0?[0-9]{3}(?:/step-?0?[0-9]{3})?_test\.jsonl"
        ),
    ),
    DatasetSpec(
        label="Carpark (≤950)",
        target_rows=4000,
        roots=[
            "artifacts/results/GRPO-1.5B-carpark-temp-0",
            "artifacts/results/GRPO-1.5B-carpark-temp-0.05",
            "artifacts/results/GRPO-1.5B-carpark-temp-0.3",
            "artifacts/results/GRPO-1.5B-carpark-temp-0.7",
        ],
        step_regex=re.compile(
            r".*/step-?0?[0-9]{3}(?:/step-?0?[0-9]{3})?_test\.jsonl"
        ),
    ),
]


def count_rows(path: Path) -> int:
    """Return the line count for a JSONL file."""

    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _is_pass_filled(obj: dict, key: str) -> bool:
    """
    Return whether a pass dict exists and appears populated.

    We treat missing/None/empty dict as "not filled"; otherwise filled.
    """
    value = obj.get(key)
    if not isinstance(value, dict) or not value:
        return False
    output = value.get("output")
    if output is None:
        return True
    return not isinstance(output, str) or bool(output.strip())


def count_rows_and_passes(path: Path, *, pass2_key: str = "pass2") -> tuple[int, int, int]:
    """
    Return (rows, pass1_filled, pass2_filled) for a JSONL file.

    ``pass1_filled`` is based on ``pass1.output`` being present and non-empty.
    ``pass2_filled`` checks the provided ``pass2_key`` (default: ``pass2``).
    """
    rows = 0
    pass1_filled = 0
    pass2_filled = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                if _is_pass_filled(obj, "pass1"):
                    pass1_filled += 1
                if _is_pass_filled(obj, pass2_key):
                    pass2_filled += 1
    return rows, pass1_filled, pass2_filled


def iter_step_files(root: Path) -> Iterable[Path]:
    """Yield all checkpoint JSONL paths under ``root``.

    Handles both ``step-XXX/step0XXX_test.jsonl`` and flattened
    ``step0XXX_test.jsonl`` layouts.
    """

    yield from root.glob("step*/step*_test.jsonl")
    yield from root.glob("step*_test.jsonl")


def audit_dataset(spec: DatasetSpec) -> List[tuple[Path, int, int]]:
    """Return audits for all matching files under the dataset roots."""

    audits: List[tuple[Path, int, int]] = []
    for root_str in spec.roots:
        root = Path(root_str)
        if not root.exists():
            continue
        for path in iter_step_files(root):
            if not spec.step_regex.match(path.as_posix()):
                continue
            try:
                rows = count_rows(path)
            except OSError:
                continue
            deficit = max(spec.target_rows - rows, 0)
            audits.append((path, rows, deficit))
    return audits


def _render_table(rows: Sequence[Sequence[str]], headers: Sequence[str]) -> str:
    columns = list(zip(*([headers] + list(rows))))
    widths = [max(len(str(cell)) for cell in col) for col in columns]
    out_lines: List[str] = []
    out_lines.append("  " + "  ".join(str(h).ljust(widths[idx]) for idx, h in enumerate(headers)))
    out_lines.append("  " + "  ".join("-" * w for w in widths))
    for row in rows:
        out_lines.append("  " + "  ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(row)))
    return "\n".join(out_lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show-ok",
        action="store_true",
        help="Also print files that meet the target row count.",
    )
    parser.add_argument(
        "--show-pass2",
        action="store_true",
        help="Also report pass2 fill counts (defaults to pass1-only).",
    )
    parser.add_argument(
        "--pass2-key",
        default="pass2",
        help="Field name for the second pass (default: pass2).",
    )
    parser.add_argument(
        "--max-path",
        type=int,
        default=90,
        help="Max characters to show for the path column (default: 90).",
    )
    args = parser.parse_args(argv)

    try:
        for spec in DATASETS:
            audits = audit_dataset(spec)
            print(f"=== {spec.label} ===")
            if not audits:
                print("  No matching files found")
                continue

            rows_out: List[List[str]] = []
            total_rows_missing = 0
            total_pass2_missing = 0

            for path, _rows, rows_deficit in sorted(audits):
                try:
                    rows, pass1_filled, pass2_filled = count_rows_and_passes(path, pass2_key=args.pass2_key)
                except OSError:
                    continue

                pass2_missing_in_file = max(rows - pass2_filled, 0)
                pass2_missing_vs_target = max(spec.target_rows - pass2_filled, 0)

                needs_attention = rows_deficit > 0 or (args.show_pass2 and pass2_missing_in_file > 0)
                if not args.show_ok and not needs_attention:
                    continue

                total_rows_missing += rows_deficit
                if args.show_pass2:
                    total_pass2_missing += pass2_missing_vs_target

                path_str = path.as_posix()
                if args.max_path and len(path_str) > args.max_path:
                    path_str = "…" + path_str[-(args.max_path - 1) :]

                row = [
                    path_str,
                    f"{rows:5d}/{spec.target_rows}",
                    f"{rows_deficit:5d}",
                    f"{pass1_filled:5d}",
                ]
                if args.show_pass2:
                    row.extend(
                        [
                            f"{pass2_filled:5d}",
                            f"{pass2_missing_in_file:5d}",
                            f"{pass2_missing_vs_target:5d}",
                        ]
                    )
                rows_out.append(row)

            if not rows_out:
                if args.show_pass2:
                    print("  All files at target row count and pass2 filled")
                else:
                    print("  All files at target row count")
                continue

            headers = ["path", "rows", "miss", "pass1"]
            if args.show_pass2:
                headers.extend([args.pass2_key, "p2_miss", "p2_vs_tgt"])
            print(_render_table(rows_out, headers=headers))
            if args.show_pass2:
                print(
                    f"  Totals: rows_missing={total_rows_missing}, "
                    f"{args.pass2_key}_missing_vs_target={total_pass2_missing}\n"
                )
            else:
                print(f"  Totals: rows_missing={total_rows_missing}\n")
    except BrokenPipeError:
        # Avoid "Exception ignored in: <stdout>" on interpreter shutdown when piped (e.g. to `head`).
        try:
            sys.stdout = open(os.devnull, "w", encoding="utf-8")
        except OSError:
            pass
        raise SystemExit(0)


if __name__ == "__main__":
    try:
        import signal

        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except Exception:
        pass
    main()
