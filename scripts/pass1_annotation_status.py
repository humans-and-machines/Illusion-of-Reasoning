#!/usr/bin/env python3
"""
Summarize GPT-4o pass1 annotation coverage for common GRPO results roots.

Usage examples (from repository root):

  # Run the built-in sweep covering GRPO math/xword/carpark roots
  python scripts/pass1_annotation_status.py

  # Limit to just the Llama-8B math runs
  python scripts/pass1_annotation_status.py --dataset "Llama-8B math"

  # Inspect an arbitrary directory, providing your own label
  python scripts/pass1_annotation_status.py --root "My Run=artifacts/results/custom-run"
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

DEFAULT_DATASETS: Dict[str, Sequence[str]] = {
    "Qwen-1.5B math": [
        "artifacts/results/GRPO-1.5B-math-temp-0.0",
        "artifacts/results/GRPO-1.5B-math-temp-0.05",
        "artifacts/results/GRPO-1.5B-math-temp-0.3",
        "artifacts/results/GRPO-1.5B-math-temp-0.7",
    ],
    "Qwen-7B math": [
        "artifacts/results/GRPO-7B-math-temp-0",
        "artifacts/results/GRPO-7B-math-temp-0.05",
        "artifacts/results/GRPO-7B-math-temp-0.3",
        "artifacts/results/GRPO-7B-math-temp-0.7",
    ],
    "Llama-8B math": [
        "artifacts/results/GRPO-Llama8B-math-temp-0",
        "artifacts/results/GRPO-Llama8B-math-temp-0.05",
        "artifacts/results/GRPO-Llama8B-math-temp-0.3",
        "artifacts/results/GRPO-Llama8B-math-temp-0.7",
    ],
    "Qwen-1.5B xword": [
        "artifacts/results/GRPO-1.5B-xword-temp-0",
        "artifacts/results/GRPO-1.5B-xword-temp-0.05",
        "artifacts/results/GRPO-1.5B-xword-temp-0.3",
        "artifacts/results/GRPO-1.5B-xword-temp-0.7",
    ],
    "Qwen-1.5B carpark": [
        "artifacts/results/GRPO-1.5B-carpark-temp-0",
        "artifacts/results/GRPO-1.5B-carpark-temp-0.05",
        "artifacts/results/GRPO-1.5B-carpark-temp-0.3",
        "artifacts/results/GRPO-1.5B-carpark-temp-0.7",
    ],
}


@dataclass
class FileStats:
    path: Path
    total_rows: int
    annotated_rows: int

    @property
    def pending(self) -> int:
        return self.total_rows - self.annotated_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count pass1 GPT-4o annotations (shift_rationale_gpt_model) "
            "for GRPO step*_test.jsonl files."
        ),
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DEFAULT_DATASETS.keys()),
        help="Limit to the named dataset(s); defaults to all known datasets.",
    )
    parser.add_argument(
        "--root",
        action="append",
        metavar="LABEL=PATH",
        help="Additional roots to scan (label=path). Can be given multiple times.",
    )
    parser.add_argument(
        "--show-complete",
        action="store_true",
        help="Include files that already have every row annotated.",
    )
    parser.add_argument(
        "--require-existing",
        action="store_true",
        help="Error if any configured root directory is missing.",
    )
    return parser.parse_args()


def iter_step_files(root: Path) -> Iterable[Path]:
    return sorted(root.rglob("step*_test.jsonl"))


def file_stats(path: Path) -> FileStats:
    total = 0
    annotated = 0
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            pass1 = record.get("pass1")
            if isinstance(pass1, dict) and pass1.get("shift_rationale_gpt_model"):
                annotated += 1
    return FileStats(path=path, total_rows=total, annotated_rows=annotated)


def collect_roots(args: argparse.Namespace) -> Dict[str, List[Path]]:
    datasets: Dict[str, List[Path]] = {}
    selected = args.dataset or list(DEFAULT_DATASETS.keys())
    for label in selected:
        datasets[label] = [Path(p) for p in DEFAULT_DATASETS[label]]
    if args.root:
        for entry in args.root:
            if "=" not in entry:
                raise ValueError(f"--root entries must be LABEL=PATH, got {entry!r}")
            label, path = entry.split("=", 1)
            datasets.setdefault(label.strip(), []).append(Path(path.strip()))
    return datasets


def summarize_dataset(
    label: str,
    roots: Sequence[Path],
    show_complete: bool,
    require_existing: bool,
) -> Tuple[int, int, List[FileStats]]:
    dataset_total = 0
    dataset_annotated = 0
    incomplete: List[FileStats] = []

    for root in roots:
        if not root.exists():
            if require_existing:
                raise FileNotFoundError(f"{root} does not exist")
            continue
        for jsonl_path in iter_step_files(root):
            stats = file_stats(jsonl_path)
            if stats.total_rows == 0:
                continue
            dataset_total += stats.total_rows
            dataset_annotated += stats.annotated_rows
            if show_complete or stats.pending:
                incomplete.append(stats)

    return dataset_total, dataset_annotated, incomplete


def main() -> None:
    args = parse_args()
    dataset_roots = collect_roots(args)
    grand_totals: List[Tuple[str, int, int]] = []
    dataset_details: Dict[str, List[FileStats]] = {}

    for label, roots in dataset_roots.items():
        total_rows, annotated_rows, files = summarize_dataset(
            label, roots, args.show_complete, args.require_existing
        )
        if total_rows == 0:
            continue
        grand_totals.append((label, annotated_rows, total_rows))
        dataset_details[label] = files

    if not grand_totals:
        print("No matching datasets or JSONL files found.")
        return

    width_label = max(len(label) for label, _, _ in grand_totals)
    header = f"{'dataset'.ljust(width_label)}   annotated/total   pending"
    print(header)
    for label, annotated_rows, total_rows in sorted(grand_totals):
        pending = total_rows - annotated_rows
        print(
            f"{label.ljust(width_label)}   {annotated_rows:,}/{total_rows:,}   {pending:,}"
        )
    print()

    for label in sorted(dataset_details):
        files = dataset_details[label]
        if not files:
            continue
        print(f"=== {label} outstanding files ===")
        for stats in files:
            if stats.pending == 0 and not args.show_complete:
                continue
            print(
                f"  {stats.annotated_rows:4d}/{stats.total_rows:<4d} annotated → {stats.path}"
            )
        print()


if __name__ == "__main__":  # pragma: no cover
    main()
