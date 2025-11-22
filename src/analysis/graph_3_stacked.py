#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
graph_3_stacked.py (PASS1 ONLY)
--------------------------------
Combine Carpark, Crossword, Math. Bin by PASS1 ANSWER ENTROPY only.
Plot a STACKED histogram per bin split into No Aha vs Aha.
Use --normalize to make each bin sum to 1.0 (proportions).

Usage
-----
python graph_3_stacked.py \
  --root_crossword artifacts/results/GRPO-1.5B-xword-temp-0.7 \
  --root_math      artifacts/results/GRPO-1.5B-math-temp-0.7 \
  --root_carpark   artifacts/results/GRPO-1.5B-carpark-temp-0.7 \
  --split test \
  --gpt_mode canonical \
  --bins 10 \
  --binning quantile \
  --outdir graphs \
  --outfile_tag combined \
  --normalize
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from src.analysis.io import iter_records_from_file
from src.analysis.labels import aha_gpt
from src.analysis.common.parser_helpers import add_binning_argument
from src.analysis.utils import (
    add_split_and_gpt_mode_args,
    step_from_record_if_within_bounds,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the stacked PASS1 entropy plot."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_crossword", type=str, default=None)
    parser.add_argument("--root_math", type=str, default=None)
    parser.add_argument("--root_carpark", type=str, default=None)
    add_split_and_gpt_mode_args(parser)

    parser.add_argument("--bins", type=int, default=20)
    add_binning_argument(parser)
    parser.add_argument("--min_step", type=int, default=0)
    parser.add_argument("--max_step", type=int, default=1000)

    parser.add_argument("--outdir", type=str, default="graphs")
    parser.add_argument("--outfile_tag", type=str, default=None)
    parser.add_argument(
        "--title",
        type=str,
        default="Counts by PASS1 Answer Entropy (Stacked No Aha vs Aha)",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--width_in", type=float, default=10.0)
    parser.add_argument("--height_in", type=float, default=5.5)
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="stack to proportions (each bin sums to 1)",
    )
    return parser.parse_args()


def detect_aha_pass1(rec: Dict[str, Any], mode: str) -> bool:
    """Detect a PASS1 Aha flag according to the configured mode."""
    pass1_data = rec.get("pass1", {})
    if not isinstance(pass1_data, dict):
        return False
    return bool(aha_gpt(pass1_data, rec, mode=mode, gate_by_words=False))


def extract_step(rec: Dict[str, Any], src_path: str) -> int:
    """Best-effort step extraction from record or filename."""
    if isinstance(rec.get("step"), (int, float)):
        return int(rec["step"])
    match = re.search(r"step[-_]?(\d{1,5})", src_path)
    if not match:
        match = re.search(r"global_step[-_]?(\d{1,5})", src_path)
    if match:
        return int(match.group(1))
    return 0


def extract_entropy_pass1(rec: Dict[str, Any]) -> Optional[float]:
    """Extract a scalar PASS1 answer entropy from a record, if present."""
    pass1_data = rec.get("pass1", {})
    if not isinstance(pass1_data, dict):
        return None
    for key in ("answer_entropy", "entropy_answer"):
        value = pass1_data.get(key, None)
        if isinstance(value, (int, float)):
            return float(value)
    token_entropies = (
        pass1_data.get("answer_token_entropies")
        or pass1_data.get("token_entropies")
        or pass1_data.get("entropies")
    )
    if isinstance(token_entropies, list) and token_entropies:
        try:
            values = [float(entry) for entry in token_entropies]
            if values:
                return float(np.mean(values))
        except (TypeError, ValueError):
            pass
    return None


def iter_jsonl_files(root: str) -> Iterable[str]:
    """Yield JSONL file paths under ``root`` (recursing into subdirectories)."""
    if not root:
        return []
    root_path = Path(root)
    if not root_path.exists():
        return []
    generator = (str(path) for path in root_path.rglob("*.jsonl"))
    return generator


def load_pass1_entropy_and_aha(
    root: str,
    split: str,
    min_step: int,
    max_step: int,
    gpt_mode: str,
) -> List[tuple]:
    """
    Load PASS1 answer entropy and Aha flags from JSONL logs under ``root``.

    Returns a list of ``(entropy, aha_flag)`` tuples.
    """
    outputs: List[tuple] = []
    if not root:
        return outputs
    for file_path in iter_jsonl_files(root):
        for rec in iter_records_from_file(file_path):
            step = step_from_record_if_within_bounds(
                rec,
                file_path,
                split_value=split,
                min_step=min_step,
                max_step=max_step,
            )
            if step is None:
                continue
            entropy_value = extract_entropy_pass1(rec)
            if entropy_value is None:
                continue
            aha = 1 if detect_aha_pass1(rec, gpt_mode) else 0
            outputs.append((float(entropy_value), aha))
    return outputs


def _rows_to_arrays(rows: List[tuple]) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw (entropy, aha_flag) tuples into NumPy arrays."""
    entropies = np.array([row[0] for row in rows], dtype=float)
    aha_flags = np.array([row[1] for row in rows], dtype=int)
    return entropies, aha_flags


def _compute_binned_counts(
    entropies: np.ndarray,
    aha_flags: np.ndarray,
    *,
    num_bins: int,
    binning: str,
    normalize: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    """
    Compute per-bin No-Aha/Aha counts plus bin centers and width.
    """
    if binning == "quantile":
        edges = np.unique(
            np.quantile(entropies, np.linspace(0.0, 1.0, num_bins + 1)),
        )
        if len(edges) < 3:
            edges = np.linspace(entropies.min(), entropies.max(), num_bins + 1)
    else:
        edges = np.linspace(entropies.min(), entropies.max(), num_bins + 1)

    bin_indices = np.digitize(entropies, edges) - 1
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = centers[1] - centers[0] if len(centers) > 1 else 0.1

    counts_no_aha = np.zeros(len(edges) - 1, dtype=float)
    counts_aha = np.zeros(len(edges) - 1, dtype=float)
    for bin_index in range(len(edges) - 1):
        mask = bin_indices == bin_index
        if not mask.any():
            continue
        counts_aha[bin_index] = float((aha_flags[mask] == 1).sum())
        counts_no_aha[bin_index] = float((aha_flags[mask] == 0).sum())

    if normalize:
        totals = counts_no_aha + counts_aha
        totals[totals == 0] = 1.0
        counts_no_aha = counts_no_aha / totals
        counts_aha = counts_aha / totals
        ylabel = "Proportion (per bin)"
    else:
        ylabel = "Count"

    return centers, counts_no_aha, counts_aha, bin_width, ylabel


@dataclass
class BinnedHistogram:
    """Container for binned entropy histogram data."""

    centers: np.ndarray
    counts_no_aha: np.ndarray
    counts_aha: np.ndarray
    bin_width: float
    ylabel: str


def _compute_binned_from_rows(
    rows: List[tuple],
    args: argparse.Namespace,
) -> BinnedHistogram:
    """Helper to go from raw rows and CLI args to binned histogram stats."""
    entropies, aha_flags = _rows_to_arrays(rows)
    centers, counts_no_aha, counts_aha, bin_width, ylabel = _compute_binned_counts(
        entropies,
        aha_flags,
        num_bins=args.bins,
        binning=args.binning,
        normalize=args.normalize,
    )
    return BinnedHistogram(
        centers=centers,
        counts_no_aha=counts_no_aha,
        counts_aha=counts_aha,
        bin_width=bin_width,
        ylabel=ylabel,
    )


def _plot_stacked_histogram(
    args: argparse.Namespace,
    histogram: BinnedHistogram,
) -> None:
    """Render and save the stacked PASS1 entropy histogram."""
    _, axes = plt.subplots(
        figsize=(args.width_in, args.height_in),
        constrained_layout=True,
    )
    axes.bar(
        histogram.centers,
        histogram.counts_no_aha,
        width=histogram.bin_width * 0.9,
        label="No Aha",
    )
    axes.bar(
        histogram.centers,
        histogram.counts_aha,
        width=histogram.bin_width * 0.9,
        bottom=histogram.counts_no_aha,
        label="Aha",
    )
    axes.set_xlabel("PASS1 answer entropy (binned)")
    axes.set_ylabel(histogram.ylabel)
    axes.set_title(args.title)
    if args.normalize:
        axes.set_ylim(0, 1.0)
    axes.legend(loc="best")
    axes.grid(True, linestyle="--", alpha=0.3)

    out_path = os.path.join(
        args.outdir,
        f"graph_3_pass1_stacked_{args.outfile_tag or 'combined'}"
        f"{'_normalized' if args.normalize else ''}.png",
    )
    plt.savefig(out_path, dpi=args.dpi)
    print(f"[ok] wrote {out_path}")


def main() -> None:
    """Entry point: build and save the stacked PASS1 entropy histogram."""
    args = parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    rows: List[tuple] = []
    rows += load_pass1_entropy_and_aha(
        args.root_carpark,
        args.split,
        args.min_step,
        args.max_step,
        args.gpt_mode,
    )
    rows += load_pass1_entropy_and_aha(
        args.root_crossword,
        args.split,
        args.min_step,
        args.max_step,
        args.gpt_mode,
    )
    rows += load_pass1_entropy_and_aha(
        args.root_math,
        args.split,
        args.min_step,
        args.max_step,
        args.gpt_mode,
    )

    if not rows:
        print("[error] No PASS1 records with answer entropy found.", file=sys.stderr)
        sys.exit(1)

    histogram = _compute_binned_from_rows(rows, args)
    _plot_stacked_histogram(args, histogram)

if __name__ == "__main__":
    main()
