#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extra figures for artificial recheck analysis.

Reads raw logs (same format as compare_artificial_recheck.py expects) and emits:
  • accuracy_by_step_pass1_pass2.png
  • entropy_before_after_by_step.png
  • heatmap_step_entropy_improvement.png  (P(correct after recheck | p1 wrong, artificial))
  • p1_entropy_hist.png, p2_entropy_hist.png
  • delta_correct_scatter.png  (p1_entropy vs (p2_correct - p1_correct))

Usage:
  python -m src.analysis.h4_analysis /path/to/results_root --split test --out_dir /tmp/recheck_extra
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.io import iter_records_from_file, scan_jsonl_files
from src.analysis.utils import coerce_bool, nat_step_from_path


if hasattr(plt, "switch_backend"):
    plt.switch_backend("Agg")


def scan_files(root: str, split_substr: Optional[str]) -> List[str]:
    """Recursively scan for JSONL result files under ``root``."""
    if split_substr:
        # Restrict to filenames that contain the requested split substring
        all_paths = scan_jsonl_files(root, split_substr=None)
        return [path for path in all_paths if split_substr in os.path.basename(path)]
    return scan_jsonl_files(root, split_substr=None)


def get_correct(prediction: Dict[str, Any]) -> Optional[int]:
    """Extract a 0/1 correctness flag from a prediction dictionary."""
    for key in ("is_correct_after_reconsideration", "is_correct_pred"):
        value = prediction.get(key)
        coerced_bool = coerce_bool(value)
        if coerced_bool is not None:
            return int(coerced_bool)
    return None


def get_entropy(prediction: Dict[str, Any]) -> Optional[float]:
    """Convert an entropy value to ``float`` if present and valid."""
    value = prediction.get("entropy")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def is_artificial_recheck(pass2_data: Dict[str, Any]) -> int:
    """Return 1 if the second pass used an artificial reconsideration cue."""
    markers = pass2_data.get("reconsider_markers") or []
    if isinstance(markers, list) and "injected_cue" in markers:
        return 1
    return int(bool(coerce_bool(pass2_data.get("has_reconsider_cue"))))


def load_pairs(files: List[str]) -> pd.DataFrame:
    """Load paired PASS-1 / PASS-2 records from a collection of JSONL files."""
    rows: List[Dict[str, Any]] = []
    for path in files:
        step_from_name = nat_step_from_path(path)
        for rec in iter_records_from_file(path):
            step = rec.get(
                "step",
                step_from_name if step_from_name is not None else None,
            )
            if step is None:
                continue
            problem_key = (
                rec.get("problem") or rec.get("clue") or rec.get("row_key") or f"idx:{rec.get('dataset_index')}"
            )
            pass1_data = rec.get("pass1") or {}
            pass2_data = rec.get("pass2") or {}
            pass1_correct = get_correct(pass1_data)
            pass2_correct = get_correct(pass2_data)
            if pass1_correct is None or pass2_correct is None:
                continue
            rows.append(
                {
                    "step": int(step),
                    "problem": str(problem_key),
                    "sample_idx": rec.get("sample_idx"),
                    "p1_correct": int(pass1_correct),
                    "p2_correct": int(pass2_correct),
                    "p1_entropy": get_entropy(pass1_data),
                    "p2_entropy": get_entropy(pass2_data),
                    "artificial": is_artificial_recheck(pass2_data),
                    "source_file": path,
                },
            )
    pairs_df = pd.DataFrame(rows)
    for column in ["p1_entropy", "p2_entropy"]:
        if column in pairs_df.columns:
            pairs_df[column] = pd.to_numeric(pairs_df[column], errors="coerce")
    return pairs_df


def plot_accuracy_by_step(pairs_df: pd.DataFrame, out_png: str) -> None:
    """Plot PASS-1 and PASS-2 accuracy by training step."""
    grouped = (
        pairs_df.groupby("step", as_index=False)
        .agg(
            n=("p1_correct", "size"),
            acc1=("p1_correct", "mean"),
            acc2=("p1_correct", "mean"),
        )
        .sort_values("step")
    )
    figure, axes = plt.subplots(figsize=(7.8, 4.6), dpi=140)
    axes.plot(grouped["step"], grouped["acc1"], marker="o", label="PASS-1")
    axes.plot(grouped["step"], grouped["acc2"], marker="o", label="PASS-2")
    axes.set_xlabel("Training step")
    axes.set_ylabel("Accuracy")
    axes.set_title("Accuracy by step (PASS-1 vs PASS-2)")
    axes.grid(True, alpha=0.3)
    axes.legend(loc="best")
    figure.tight_layout()
    figure.savefig(out_png)
    plt.close(figure)


def plot_entropy_by_step(pairs_df: pd.DataFrame, out_png: str) -> None:
    """Plot mean PASS-1 and PASS-2 entropy by training step."""
    grouped = (
        pairs_df.groupby("step", as_index=False)
        .agg(
            entropy_pass1=("p1_entropy", "mean"),
            entropy_pass2=("p2_entropy", "mean"),
        )
        .sort_values("step")
    )
    figure, axes = plt.subplots(figsize=(7.8, 4.6), dpi=140)
    axes.plot(
        grouped["step"],
        grouped["entropy_pass1"],
        marker="o",
        label="PASS-1 entropy",
    )
    axes.plot(
        grouped["step"],
        grouped["entropy_pass2"],
        marker="o",
        label="PASS-2 entropy",
    )
    axes.set_xlabel("Training step")
    axes.set_ylabel("Mean entropy")
    axes.set_title("Entropy before vs after (by step)")
    axes.grid(True, alpha=0.3)
    axes.legend(loc="best")
    figure.tight_layout()
    figure.savefig(out_png)
    plt.close(figure)


def _build_heatmap_matrix(
    grouped: pd.DataFrame,
    steps: np.ndarray,
    num_bins: int,
) -> np.ndarray:
    """Construct a (step, entropy-bin) heatmap matrix from grouped data."""
    heatmap_matrix = np.full((len(steps), num_bins), np.nan)
    for row_index, step_value in enumerate(steps):
        row = grouped[grouped["step"] == step_value]
        for _, record in row.iterrows():
            heatmap_matrix[row_index, int(record["bin"])] = record["rate"]
    return heatmap_matrix


def plot_heatmap_step_entropy_improve(
    pairs_df: pd.DataFrame,
    out_png: str,
    bins: int = 8,
) -> None:
    """Plot a heatmap of correction rate vs. step and PASS-1 entropy bin."""
    sub = pairs_df[(pairs_df["artificial"] == 1) & (pairs_df["p1_correct"] == 0)].copy()
    sub = sub[np.isfinite(sub["p1_entropy"])]
    if sub.empty:
        return
    # Bin entropy into quantiles
    sub["q"] = pd.qcut(sub["p1_entropy"], q=bins, duplicates="drop")
    # Map bins to indices and midpoints
    cats = sub["q"].cat.categories
    mids: List[float] = []
    for category in cats:
        mids.append(0.5 * (category.left + category.right))
    bin_index = {cats[i]: i for i in range(len(cats))}
    sub["bin"] = sub["q"].map(bin_index)
    grouped_stats = sub.groupby(["step", "bin"], as_index=False, observed=False).agg(
        rate=("p2_correct", "mean"),
        n=("p2_correct", "size"),
    )
    steps = np.sort(grouped_stats["step"].unique())
    num_bins = len(cats)
    heatmap_matrix = _build_heatmap_matrix(grouped_stats, steps, num_bins)
    figure, axes = plt.subplots(figsize=(8.2, 4.8), dpi=150)
    image = axes.imshow(
        heatmap_matrix,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )
    axes.set_yticks(np.arange(len(steps)))
    axes.set_yticklabels(steps)
    axes.set_xticks(np.arange(num_bins))
    axes.set_xticklabels([f"{midpoint:.2f}" for midpoint in mids], rotation=45, ha="right")
    axes.set_xlabel("PASS-1 entropy (bin center)")
    axes.set_ylabel("Training step")
    axes.set_title("Correction rate after artificial recheck (p1 wrong)")
    figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04, label="P(correct)")
    figure.tight_layout()
    figure.savefig(out_png)
    plt.close(figure)


def plot_entropy_hists(pairs_df: pd.DataFrame, out_dir: str) -> None:
    """Plot histograms of PASS-1 and PASS-2 entropy."""
    for column, filename in [
        ("p1_entropy", "p1_entropy_hist.png"),
        ("p2_entropy", "p2_entropy_hist.png"),
    ]:
        entropy_values = pd.to_numeric(pairs_df[column], errors="coerce").to_numpy()
        entropy_values = entropy_values[np.isfinite(entropy_values)]
        if entropy_values.size == 0:
            continue
        figure, axes = plt.subplots(figsize=(6.0, 4.2), dpi=140)
        axes.hist(entropy_values, bins=40, alpha=0.9)
        axes.set_xlabel(column)
        axes.set_ylabel("count")
        axes.set_title(f"{column} distribution")
        axes.grid(True, alpha=0.2)
        figure.tight_layout()
        figure.savefig(os.path.join(out_dir, filename))
        plt.close(figure)


def plot_delta_correct_scatter(pairs_df: pd.DataFrame, out_png: str) -> None:
    """Scatter plot of Δ correctness vs. PASS-1 entropy."""
    data_frame = pairs_df.copy()
    data_frame["delta"] = data_frame["p2_correct"] - data_frame["p1_correct"]
    entropy_values = pd.to_numeric(data_frame["p1_entropy"], errors="coerce").to_numpy()
    delta_values = pd.to_numeric(data_frame["delta"], errors="coerce").to_numpy()
    valid_mask = np.isfinite(entropy_values) & np.isfinite(delta_values)
    entropy_values = entropy_values[valid_mask]
    delta_values = delta_values[valid_mask]
    if entropy_values.size == 0:
        return
    figure, axes = plt.subplots(figsize=(6.4, 4.2), dpi=140)
    axes.scatter(entropy_values, delta_values, s=12, alpha=0.6)
    axes.set_xlabel("PASS-1 entropy")
    axes.set_ylabel("Δ correctness (PASS-2 - PASS-1)")
    axes.set_title("Sample-level change vs PASS-1 entropy")
    axes.grid(True, alpha=0.2)
    figure.tight_layout()
    figure.savefig(out_png)
    plt.close(figure)


def main() -> None:
    """Entry point for generating artificial recheck analysis figures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root")
    parser.add_argument("--split", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--bins", type=int, default=8)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(
        args.results_root,
        "artificial_recheck_extra",
    )
    os.makedirs(out_dir, exist_ok=True)

    files = scan_files(args.results_root, args.split)
    if not files:
        raise SystemExit("No JSONL files found.")

    pairs_df = load_pairs(files)
    if pairs_df.empty:
        raise SystemExit("No comparable rows parsed.")

    plot_accuracy_by_step(
        pairs_df,
        os.path.join(out_dir, "accuracy_by_step_pass1_pass2.png"),
    )
    plot_entropy_by_step(
        pairs_df,
        os.path.join(out_dir, "entropy_before_after_by_step.png"),
    )
    plot_heatmap_step_entropy_improve(
        pairs_df,
        os.path.join(out_dir, "heatmap_step_entropy_improvement.png"),
        bins=args.bins,
    )
    plot_entropy_hists(pairs_df, out_dir)
    plot_delta_correct_scatter(
        pairs_df,
        os.path.join(out_dir, "delta_correct_scatter.png"),
    )

    # Bonus: write the parsed pairs for downstream use
    pairs_df.to_csv(os.path.join(out_dir, "pairs_table.csv"), index=False)
    print("Wrote figures to", out_dir)


if __name__ == "__main__":
    main()
