#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=too-many-lines

"""
graph_3.py  (PASS1 ONLY, per-metric variants; accuracy + Reasoning Shift counts)
--------------------------------------------------------------------------------
Produces the same 3-panel plot (Carpark/Crossword/Math) for EACH of:
  • Answer Entropy (pass1)
  • Think  Entropy (pass1)
  • Answer+Think   (pass1; sum or mean via --combined_mode)

For each metric, writes TWO figures:
  1) Accuracy (%) vs entropy bins (Aha vs No Aha, side-by-side bars)
  2) Reasoning Shift COUNT vs entropy bins (Aha only)

NEW: Also writes numeric tables (CSV):
  • graphs/tables/graph_3_pass1_table_<metric>_<tag>__per_domain.csv
  • graphs/tables/graph_3_pass1_table_<metric>_<tag>__overall.csv
"""

import argparse
import glob as _glob
import os
import re
import sys
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.common.parser_helpers import (
    add_carpark_softscore_args,
    add_entropy_range_args,
)
from src.analysis.io import iter_records_from_file
from src.analysis.labels import AHA_KEYS_BROAD, AHA_KEYS_CANONICAL
from src.analysis.plotting import apply_entropy_plot_style
from src.analysis.utils import (
    add_domain_root_args,
    add_split_and_gpt_mode_args,
    step_from_record_if_within_bounds,
    truthy_flag,
)

# ---------- Global typography (Times, size 14) & nice PDFs ----------
apply_entropy_plot_style(
    {
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    },
)

DOMAINS = ["Carpark", "Crossword", "Math"]

# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for PASS-1 entropy bucket plots.
    """
    parser = argparse.ArgumentParser()
    add_domain_root_args(parser)
    parser.add_argument(
        "--ignore_glob",
        nargs="*",
        default=["*compare-1shot*"],
        help="Glob patterns to exclude",
    )

    # Optional: include only certain temperatures by path match 'temp-<value>'
    parser.add_argument(
        "--only_temps",
        nargs="*",
        default=None,
        help="Keep files whose path contains temp-<value> (e.g., 0.7)",
    )

    add_split_and_gpt_mode_args(parser)
    add_carpark_softscore_args(parser, op_default="ge", threshold_default=0.1)
    add_entropy_range_args(parser)

    # Step filter
    parser.add_argument("--min_step", type=int, default=0)
    parser.add_argument("--max_step", type=int, default=1000)

    # Figure & output
    parser.add_argument("--outdir", type=str, default="graphs")
    parser.add_argument("--outfile_tag", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--width_in", type=float, default=5.5)
    parser.add_argument("--height_in", type=float, default=8.0)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument(
        "--y_pad",
        type=float,
        default=6.0,
        help="headroom above local ymax for Accuracy plots, in percentage points",
    )

    # Colors
    parser.add_argument(
        "--cmap",
        type=str,
        default="YlGnBu",
        help="Matplotlib colormap (e.g., YlGnBu, PuBuGn, cividis, magma).",
    )

    # Which metrics to render & combine rule for answer+think
    parser.add_argument(
        "--which_metrics",
        nargs="+",
        choices=["answer", "think", "answer_plus"],
        default=["answer", "think", "answer_plus"],
        help="Which plots to render",
    )
    parser.add_argument(
        "--combined_mode",
        type=str,
        default="sum",
        choices=["sum", "mean"],
        help="For answer_plus: sum or mean of pass1 answer & think entropies",
    )

    return parser.parse_args()

def detect_aha_pass1(rec: Dict[str, Any], mode: str) -> int:
    """
    Determine whether a PASS-1 record contains an Aha-style reasoning shift.
    """
    keys = (
        AHA_KEYS_CANONICAL
        if mode == "canonical"
        else AHA_KEYS_BROAD
    )
    pass1_data = rec.get("pass1", {})
    if not isinstance(pass1_data, dict):
        return 0
    return 1 if any(truthy_flag(pass1_data.get(key, False)) for key in keys) else 0


def extract_step(rec: Dict[str, Any], src_path: str) -> int:
    """
    Extract a numeric step from a record or its source path.
    """
    if isinstance(rec.get("step"), (int, float)):
        return int(rec["step"])
    match = re.search(r"step[-_]?(\d{1,5})", src_path) or re.search(
        r"global_step[-_]?(\d{1,5})",
        src_path,
    )
    return int(match.group(1)) if match else 0

# ----- PASS1 entropy extractors -----
def _num(mapping: Dict[str, Any], names: List[str]) -> Optional[float]:
    """
    Extract the first numeric value from ``mapping`` whose key is in ``names``.
    """
    if not isinstance(mapping, dict):
        return None
    for name in names:
        raw_value = mapping.get(name)
        if isinstance(raw_value, (int, float)):
            return float(raw_value)
    return None


def extract_pass1_answer_entropy(rec: Dict[str, Any]) -> Optional[float]:
    """
    Extract a scalar answer entropy from PASS1 fields, falling back to token entropies.
    """
    pass1_data = rec.get("pass1", {})
    base_entropy = _num(pass1_data, ["answer_entropy", "entropy_answer"])
    if base_entropy is not None:
        return base_entropy
    token_entropies = (
        pass1_data.get("answer_token_entropies")
        or pass1_data.get("token_entropies")
    )
    if isinstance(token_entropies, list) and token_entropies:
        try:
            values = [float(token) for token in token_entropies]
        except (TypeError, ValueError):
            return None
        if values:
            return float(np.mean(values))
    return None


def extract_pass1_think_entropy(rec: Dict[str, Any]) -> Optional[float]:
    """
    Extract a scalar think entropy from PASS1 fields, falling back to token entropies.
    """
    pass1_data = rec.get("pass1", {})
    base_entropy = _num(pass1_data, ["entropy_think", "think_entropy"])
    if base_entropy is not None:
        return base_entropy
    token_entropies = pass1_data.get("think_token_entropies")
    if isinstance(token_entropies, list) and token_entropies:
        try:
            values = [float(token) for token in token_entropies]
        except (TypeError, ValueError):
            return None
        if values:
            return float(np.mean(values))
    return None


def extract_pass1_answer_plus(rec: Dict[str, Any], mode: str) -> Optional[float]:
    """
    Combine answer and think entropies according to ``mode`` (\"sum\" or \"mean\").
    """
    answer_entropy = extract_pass1_answer_entropy(rec)
    think_entropy = extract_pass1_think_entropy(rec)
    if answer_entropy is None and think_entropy is None:
        return None
    if answer_entropy is None:
        return think_entropy
    if think_entropy is None:
        return answer_entropy
    if mode == "sum":
        return answer_entropy + think_entropy
    return (answer_entropy + think_entropy) / 2.0

# ----- correctness (PASS1) -----
def carpark_correct_pass1(
    rec: Dict[str, Any],
    comparison_op: str,
    threshold: float,
) -> bool:
    """
    Determine pass-1 correctness for Carpark using a soft-reward threshold and fallbacks.
    """
    pass1_data = rec.get("pass1", {})
    if not isinstance(pass1_data, dict):
        return False

    def decide(value: float) -> bool:
        if comparison_op == "ge":
            return value >= threshold
        if comparison_op == "gt":
            return value > threshold
        if comparison_op == "le":
            return value <= threshold
        if comparison_op == "lt":
            return value < threshold
        # Default fallback: treat like "gt"
        return value > threshold

    for soft_key in ("soft_1", "soft_reward"):
        soft_value = pass1_data.get(soft_key)
        if isinstance(soft_value, (int, float)):
            return decide(float(soft_value))
    for field in ("is_correct", "correct", "correct_exact", "is_correct_pred"):
        if field in pass1_data:
            return truthy_flag(pass1_data[field])
    return False


def general_correct_pass1(rec: Dict[str, Any]) -> bool:
    """
    Determine pass-1 correctness for non-Carpark domains using common boolean flags.
    """
    pass1_data = rec.get("pass1", {})
    if not isinstance(pass1_data, dict):
        return False
    for field in ("is_correct", "correct", "correct_exact", "is_correct_pred"):
        if field in pass1_data:
            return truthy_flag(pass1_data[field])
    return False

# ----- file discovery helpers -----
def expand_paths(paths: List[str]) -> List[Path]:
    """
    Expand a list of directory paths or globs into concrete directories.
    """
    expanded_paths: List[Path] = []
    for raw_path in paths:
        if any(ch in raw_path for ch in "*?[]"):
            expanded_paths.extend(Path(path_str) for path_str in _glob.glob(raw_path))
        else:
            expanded_paths.append(Path(raw_path))
    return [path for path in expanded_paths if path.exists() and path.is_dir()]


def ignored(path: str, ignore_globs: List[str]) -> bool:
    """
    Return True if ``path`` matches any of the provided ignore patterns.
    """
    return any(fnmatch(path, pattern) for pattern in ignore_globs)

def temp_match(path: str, only_temps: Optional[List[str]]) -> bool:
    """
    Check whether ``path`` matches one of the requested temperature substrings.

    We look for fragments like ``temp-0.3`` built from the provided values.
    """
    if not only_temps:
        return True
    candidates: List[str] = []
    for temp_value in only_temps:
        temp_str = str(temp_value).strip()
        try:
            numeric_value = float(temp_str)
        except (TypeError, ValueError):
            candidates.append(temp_str)
            continue
        candidates.extend(
            [
                f"{numeric_value:g}",
                f"{numeric_value:.2f}",
                f"{numeric_value:.1f}",
            ],
        )
    return any(f"temp-{candidate}" in path for candidate in candidates)

def iter_jsonl_files_many(
    roots: List[str],
    ignore_globs: List[str],
    only_temps: Optional[List[str]],
) -> Iterable[str]:
    """
    Yield JSONL file paths under the given roots, honoring ignore and temp filters.
    """
    for root in expand_paths(roots):
        for jsonl_path in root.rglob("*.jsonl"):
            file_path = str(jsonl_path)
            if ignored(file_path, ignore_globs):
                continue
            if not temp_match(file_path, only_temps):
                continue
            yield file_path


# ----- loading per metric -----
def load_rows_from_roots_metric(
    roots: List[str],
    domain: str,
    metric: str,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    """
    Load flat PASS1 rows for a given metric and domain using CLI-style arguments.
    """
    rows: List[Dict[str, Any]] = []
    if not roots:
        return rows
    for file_path in iter_jsonl_files_many(roots, args.ignore_glob, args.only_temps):
        try:
            for rec in iter_records_from_file(file_path):
                step = step_from_record_if_within_bounds(
                    rec,
                    file_path,
                    split_value=args.split,
                    min_step=args.min_step,
                    max_step=args.max_step,
                )
                if step is None:
                    continue

                if metric == "answer":
                    entropy_value = extract_pass1_answer_entropy(rec)
                elif metric == "think":
                    entropy_value = extract_pass1_think_entropy(rec)
                else:
                    entropy_value = extract_pass1_answer_plus(
                        rec,
                        args.combined_mode,
                    )

                if entropy_value is None:
                    continue
                aha_flag = detect_aha_pass1(rec, args.gpt_mode)
                if domain == "Carpark":
                    is_correct = carpark_correct_pass1(
                        rec,
                        args.carpark_success_op,
                        args.carpark_soft_threshold,
                    )
                else:
                    is_correct = general_correct_pass1(rec)
                rows.append(
                    {
                        "domain": domain,
                        "entropy": float(entropy_value),
                        "aha": int(aha_flag),
                        "correct": int(is_correct),
                    },
                )
        except (OSError, ValueError, TypeError):
            continue
    return rows

# ----- binning -----
def compute_edges(
    entropies: np.ndarray,
    bins: int,
    binning: str,
    entropy_min: Optional[float],
    entropy_max: Optional[float],
) -> np.ndarray:
    """
    Compute bin edges for entropy values using either fixed or quantile binning.
    """
    if (entropy_min is not None) and (entropy_max is not None):
        if entropy_max <= entropy_min:
            raise ValueError("--entropy_max must be > --entropy_min")
        return np.linspace(float(entropy_min), float(entropy_max), bins + 1)
    if entropies.size == 0:
        return np.array([0.0, 1.0])
    if binning == "quantile":
        quantiles = np.linspace(0.0, 1.0, bins + 1)
        edges = np.unique(np.quantile(entropies, quantiles))
        if len(edges) < 3:
            entropy_min_val = float(entropies.min()) if entropy_min is None else float(entropy_min)
            entropy_max_val = float(entropies.max()) if entropy_max is None else float(entropy_max)
            if entropy_max_val <= entropy_min_val:
                entropy_max_val = entropy_min_val + 1e-6
            edges = np.linspace(entropy_min_val, entropy_max_val, bins + 1)
    else:
        entropy_min_val = float(entropies.min()) if entropy_min is None else float(entropy_min)
        entropy_max_val = float(entropies.max()) if entropy_max is None else float(entropy_max)
        if entropy_max_val <= entropy_min_val:
            entropy_max_val = entropy_min_val + 1e-6
        edges = np.linspace(entropy_min_val, entropy_max_val, bins + 1)
    return edges


def binned_accuracy(
    ent: np.ndarray,
    aha: np.ndarray,
    corr: np.ndarray,
    edges: np.ndarray,
    aha_flag: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-bin accuracy for samples with ``aha == aha_flag``.
    """
    ent_clip = np.clip(ent, edges[0], np.nextafter(edges[-1], -np.inf))
    bin_index = np.digitize(ent_clip, edges) - 1
    num_bins = len(edges) - 1
    acc = np.full(num_bins, np.nan, dtype=float)
    for current_bin in range(num_bins):
        mask = (bin_index == current_bin) & (aha == aha_flag)
        n_samples = int(mask.sum())
        if n_samples > 0:
            acc[current_bin] = float(corr[mask].sum()) / n_samples
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, acc


def binned_aha_counts(
    ent: np.ndarray,
    aha: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-bin counts of samples with ``aha == 1``.
    """
    ent_clip = np.clip(ent, edges[0], np.nextafter(edges[-1], -np.inf))
    bin_index = np.digitize(ent_clip, edges) - 1
    num_bins = len(edges) - 1
    counts = np.zeros(num_bins, dtype=int)
    for current_bin in range(num_bins):
        mask = (bin_index == current_bin) & (aha == 1)
        counts[current_bin] = int(mask.sum())
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts

@dataclass
class BinInputs:
    """Inputs required to compute per-bin accuracy and counts."""

    entropies: np.ndarray
    aha_flags: np.ndarray
    correctness: np.ndarray
    edges: np.ndarray


# ----- build per-bin table (NEW) -----
def compute_bin_table(
    metric_name: str,
    scope: str,
    domain: str,
    bin_inputs: BinInputs,
) -> pd.DataFrame:
    """
    Build a per-bin accuracy/counts table plus a pooled TOTAL row.
    """
    if bin_inputs.entropies.size == 0:
        return pd.DataFrame(
            columns=[
                "metric",
                "scope",
                "domain",
                "bin_lo",
                "bin_hi",
                "bin_center",
                "n_total",
                "n_aha",
                "n_noaha",
                "pct_aha",
                "acc_noshift",
                "acc_shift",
            ],
        )

    entropies_clipped = np.clip(
        bin_inputs.entropies,
        bin_inputs.edges[0],
        np.nextafter(bin_inputs.edges[-1], -np.inf),
    )
    bin_index = np.digitize(entropies_clipped, bin_inputs.edges) - 1
    n_total_per_bin = np.bincount(
        bin_index,
        minlength=len(bin_inputs.edges) - 1,
    )
    aha_mask = bin_inputs.aha_flags == 1
    n_aha_per_bin = np.bincount(
        bin_index[aha_mask],
        minlength=len(bin_inputs.edges) - 1,
    )
    correct_shift_per_bin = np.bincount(
        bin_index[aha_mask],
        weights=bin_inputs.correctness[aha_mask],
        minlength=len(bin_inputs.edges) - 1,
    )
    correct_noshift_per_bin = np.bincount(
        bin_index[~aha_mask],
        weights=bin_inputs.correctness[~aha_mask],
        minlength=len(bin_inputs.edges) - 1,
    )

    bin_centers = 0.5 * (bin_inputs.edges[:-1] + bin_inputs.edges[1:])
    bin_table = pd.DataFrame(
        {
            "metric": metric_name,
            "scope": scope,
            "domain": domain,
            "bin_lo": bin_inputs.edges[:-1],
            "bin_hi": bin_inputs.edges[1:],
            "bin_center": bin_centers,
            "n_total": n_total_per_bin,
            "n_aha": n_aha_per_bin,
            "n_noaha": n_total_per_bin - n_aha_per_bin,
            "pct_aha": np.divide(
                n_aha_per_bin,
                n_total_per_bin,
                out=np.full(n_aha_per_bin.shape, np.nan, dtype=float),
                where=n_total_per_bin > 0,
            ),
            "acc_noshift": np.divide(
                correct_noshift_per_bin,
                n_total_per_bin - n_aha_per_bin,
                out=np.full(correct_noshift_per_bin.shape, np.nan, dtype=float),
                where=(n_total_per_bin - n_aha_per_bin) > 0,
            ),
            "acc_shift": np.divide(
                correct_shift_per_bin,
                n_aha_per_bin,
                out=np.full(correct_shift_per_bin.shape, np.nan, dtype=float),
                where=n_aha_per_bin > 0,
            ),
        },
    )

    total_row = pd.DataFrame(
        [
            [
                metric_name,
                scope,
                domain,
                np.nan,
                np.nan,
                np.nan,
                int(bin_inputs.entropies.size),
                int((bin_inputs.aha_flags == 1).sum()),
                int(
                    bin_inputs.entropies.size
                    - (bin_inputs.aha_flags == 1).sum()
                ),
                (
                    (bin_inputs.aha_flags == 1).sum()
                    / float(bin_inputs.entropies.size)
                    if bin_inputs.entropies.size > 0
                    else np.nan
                ),
                (
                    float(
                        bin_inputs.correctness[bin_inputs.aha_flags == 0].mean(),
                    )
                    if (bin_inputs.aha_flags == 0).sum() > 0
                    else np.nan
                ),
                (
                    float(
                        bin_inputs.correctness[bin_inputs.aha_flags == 1].mean(),
                    )
                    if (bin_inputs.aha_flags == 1).sum() > 0
                    else np.nan
                ),
            ],
        ],
        columns=bin_table.columns,
    )
    total_row["bin_label"] = "TOTAL"
    bin_table["bin_label"] = bin_table.apply(
        lambda row: f"[{row.bin_lo:.2f},{row.bin_hi:.2f}]",
        axis=1,
    )
    return pd.concat([bin_table, total_row], ignore_index=True)

# ----- aesthetics & saving -----
def minimal_axes(axes_obj):
    """Apply a minimal style: no top/right spines and a light grid."""
    axes_obj.spines["top"].set_visible(False)
    axes_obj.spines["right"].set_visible(False)
    axes_obj.grid(True, linestyle="--", alpha=0.25)


def save_all_formats(fig, out_base: str, dpi: int = 300):
    """Save a figure as both PNG and PDF with common settings."""
    png_path = f"{out_base}.png"
    pdf_path = f"{out_base}.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"[ok] wrote {png_path}")
    print(f"[ok] wrote {pdf_path}")


def _build_per_domain_accuracy(
    rows: List[Dict[str, Any]],
    domains: List[str],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Convert flat row dicts into per-domain entropy/aha/correct arrays."""
    per_domain_lists: Dict[str, Dict[str, List[Any]]] = {
        domain: {"entropy": [], "aha": [], "correct": []} for domain in domains
    }
    for row in rows:
        domain = row["domain"]
        per_domain_lists[domain]["entropy"].append(row["entropy"])
        per_domain_lists[domain]["aha"].append(row["aha"])
        per_domain_lists[domain]["correct"].append(row["correct"])

    per_domain_arrays: Dict[str, Dict[str, np.ndarray]] = {}
    for domain in domains:
        domain_data = per_domain_lists[domain]
        per_domain_arrays[domain] = {
            "entropy": np.asarray(domain_data["entropy"], dtype=float),
            "aha": np.asarray(domain_data["aha"], dtype=int),
            "correct": np.asarray(domain_data["correct"], dtype=int),
        }
    return per_domain_arrays


def _compute_edges_for_accuracy(
    per_domain: Dict[str, Dict[str, np.ndarray]],
    domains: List[str],
    args,
) -> tuple[Dict[str, np.ndarray], np.ndarray]:
    """Compute per-domain and overall bin edges for accuracy plots."""
    if args.share_bins == "global":
        all_entropies = np.concatenate(
            [
                per_domain[domain]["entropy"]
                for domain in domains
                if per_domain[domain]["entropy"].size
            ],
        )
        global_edges = compute_edges(
            all_entropies,
            args.bins,
            args.binning,
            args.entropy_min,
            args.entropy_max,
        )
        edges_by_domain = {domain: global_edges for domain in domains}
        overall_edges = global_edges
    else:
        edges_by_domain = {
            domain: compute_edges(
                per_domain[domain]["entropy"],
                args.bins,
                args.binning,
                args.entropy_min,
                args.entropy_max,
            )
            for domain in domains
        }
        all_entropies = np.concatenate(
            [
                per_domain[domain]["entropy"]
                for domain in domains
                if per_domain[domain]["entropy"].size
            ],
        )
        overall_edges = compute_edges(
            all_entropies,
            args.bins,
            args.binning,
            args.entropy_min,
            args.entropy_max,
        )
    return edges_by_domain, overall_edges


def _metric_labels(metric: str) -> tuple[str, str]:
    """Return x-axis label and file tag for a metric name."""
    x_label_map = {
        "answer": "Answer Entropy (Binned)",
        "think": "Think Entropy (Binned)",
        "answer_plus": "Answer+Think Entropy (Binned)",
    }
    file_tag_map = {
        "answer": "answer",
        "think": "think",
        "answer_plus": "answer_plus",
    }
    return x_label_map[metric], file_tag_map[metric]


@dataclass
class AccuracyTablesConfig:
    """Configuration for writing accuracy tables for a metric."""

    per_domain: Dict[str, Dict[str, np.ndarray]]
    domains: List[str]
    edges_by_domain: Dict[str, np.ndarray]
    overall_edges: np.ndarray
    metric: str
    file_tag: str
    args: Any


def _write_accuracy_tables(config: AccuracyTablesConfig) -> None:
    """Write per-domain and overall accuracy tables for a metric."""
    tag = config.args.outfile_tag or "combined"
    tables_dir = Path(config.args.outdir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    per_domain_tables: List[pd.DataFrame] = []
    for domain in config.domains:
        bin_inputs = BinInputs(
            entropies=config.per_domain[domain]["entropy"],
            aha_flags=config.per_domain[domain]["aha"],
            correctness=config.per_domain[domain]["correct"],
            edges=config.edges_by_domain[domain],
        )
        table = compute_bin_table(
            metric_name=config.metric,
            scope="domain",
            domain=domain,
            bin_inputs=bin_inputs,
        )
        per_domain_tables.append(table)

    if per_domain_tables:
        per_domain_df = pd.concat(per_domain_tables, ignore_index=True)
        per_domain_df.to_csv(
            tables_dir
            / f"graph_3_pass1_table_{config.file_tag}_{tag}__per_domain.csv",
            index=False,
        )

    ent_all = np.concatenate(
        [
            config.per_domain[domain]["entropy"]
            for domain in config.domains
            if config.per_domain[domain]["entropy"].size
        ],
    )
    aha_all = np.concatenate(
        [
            config.per_domain[domain]["aha"]
            for domain in config.domains
            if config.per_domain[domain]["aha"].size
        ],
    )
    corr_all = np.concatenate(
        [
            config.per_domain[domain]["correct"]
            for domain in config.domains
            if config.per_domain[domain]["correct"].size
        ],
    )
    overall_inputs = BinInputs(
        entropies=ent_all,
        aha_flags=aha_all,
        correctness=corr_all,
        edges=config.overall_edges,
    )
    overall_table = compute_bin_table(
        metric_name=config.metric,
        scope="overall",
        domain="ALL",
        bin_inputs=overall_inputs,
    )
    overall_table.to_csv(
        tables_dir
        / f"graph_3_pass1_table_{config.file_tag}_{tag}__overall.csv",
        index=False,
    )


@dataclass
class AccuracyFigureConfig:
    """Configuration for rendering the 3-panel accuracy figure."""

    per_domain: Dict[str, Dict[str, np.ndarray]]
    domains: List[str]
    edges_by_domain: Dict[str, np.ndarray]
    x_label: str
    args: Any
    color_noaha: Any
    color_aha: Any


@dataclass
class CountsFigureConfig:
    """Configuration for rendering binned aha-count histograms."""

    per_domain: Dict[str, Dict[str, np.ndarray]]
    domains: List[str]
    edges_by_domain: Dict[str, np.ndarray]
    x_label: str
    args: Any
    color_aha: Any


def _render_counts_panel(
    axis: Any,
    domain_name: str,
    config: CountsFigureConfig,
) -> None:
    minimal_axes(axis)
    entropy_values = config.per_domain[domain_name]["entropy"]
    aha_values = config.per_domain[domain_name]["aha"]
    if entropy_values.size == 0:
        axis.text(
            0.5,
            0.5,
            f"No data for {domain_name}",
            ha="center",
            va="center",
        )
        return

    edges = config.edges_by_domain[domain_name]
    centers, counts = binned_aha_counts(
        entropy_values,
        aha_values,
        edges,
    )

    width = (centers[1] - centers[0]) * 0.80 if centers.size > 1 else 0.5
    axis.bar(
        centers,
        counts,
        width=width,
        label="Aha",
        color=config.color_aha,
    )

    if counts.size:
        ymax = float(counts.max())
        axis.set_ylim(0, ymax * 1.06 + (1.0 if ymax < 10 else 0.0))
    label_obj = axis.set_ylabel("Reasoning Shift\n(Count)")
    label_obj.set_multialignment("center")
    axis.set_title(
        domain_name,
        loc="left",
        fontsize=14,
        fontweight="bold",
    )


def _render_accuracy_panel(
    axis: Any,
    domain_name: str,
    config: AccuracyFigureConfig,
) -> None:
    minimal_axes(axis)
    entropy_values = config.per_domain[domain_name]["entropy"]
    if entropy_values.size == 0:
        axis.text(
            0.5,
            0.5,
            f"No data for {domain_name}",
            ha="center",
            va="center",
        )
        return

    aha_values = config.per_domain[domain_name]["aha"]
    correct_values = config.per_domain[domain_name]["correct"]
    edges = config.edges_by_domain[domain_name]
    x_values, acc_no = binned_accuracy(
        entropy_values,
        aha_values,
        correct_values,
        edges,
        aha_flag=0,
    )
    _, acc_yes = binned_accuracy(
        entropy_values,
        aha_values,
        correct_values,
        edges,
        aha_flag=1,
    )

    acc_no_pct = np.array(acc_no, dtype=float) * 100.0
    acc_yes_pct = np.array(acc_yes, dtype=float) * 100.0

    if len(x_values) > 1:
        bar_width = (x_values[1] - x_values[0]) * 0.80
    else:
        bar_width = 0.5

    axis.bar(
        x_values - bar_width / 4.0,
        acc_no_pct,
        width=bar_width / 2.0,
        label="No Aha",
        color=config.color_noaha,
    )
    axis.bar(
        x_values + bar_width / 4.0,
        acc_yes_pct,
        width=bar_width / 2.0,
        label="Aha",
        color=config.color_aha,
    )

    if np.isfinite(acc_no_pct).any() or np.isfinite(acc_yes_pct).any():
        combined = np.concatenate(
            [
                acc_no_pct[~np.isnan(acc_no_pct)],
                acc_yes_pct[~np.isnan(acc_yes_pct)],
            ]
        )
        if combined.size:
            ymax = float(np.nanmax(combined))
            axis.set_ylim(0, min(100.0, ymax + config.args.y_pad))
    axis.set_ylabel("Accuracy (%)")
    axis.set_title(
        domain_name,
        loc="left",
        fontsize=14,
        fontweight="bold",
    )


def _build_accuracy_figure(config: AccuracyFigureConfig):
    """Create the 3-panel accuracy figure for a metric."""
    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(config.args.width_in, config.args.height_in),
        sharex=True,
        constrained_layout=True,
    )

    for axis, domain_name in zip(axes, config.domains):
        _render_accuracy_panel(axis, domain_name, config)

    # Tick labels from Carpark edges (just to standardize)
    ref_edges = config.edges_by_domain["Carpark"]
    centers = 0.5 * (ref_edges[:-1] + ref_edges[1:])
    tick_labels = [
        f"[{ref_edges[i]:.2f},{ref_edges[i + 1]:.2f}]"
        for i in range(len(ref_edges) - 1)
    ]
    axes[-1].set_xticks(centers)
    axes[-1].set_xticklabels(tick_labels, rotation=45, ha="right")
    axes[-1].set_xlabel(config.x_label)

    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper right",
    )

    if config.args.title:
        fig.suptitle(
            config.args.title,
            y=1.02,
            fontsize=14,
            fontweight="bold",
        )

    return fig

# ----- renderers -----
def render_metric_accuracy(args, metric: str, color_noaha, color_aha):
    """
    Render three-panel accuracy plots and write per-bin tables for a metric.

    Returns True if any rows were rendered.
    """
    rows = _load_metric_rows(args, metric, "accuracy")
    if not rows:
        return False

    per_domain = _build_per_domain_accuracy(rows, DOMAINS)
    edges_by_domain, overall_edges = _compute_edges_for_accuracy(
        per_domain,
        DOMAINS,
        args,
    )
    x_label, file_tag = _metric_labels(metric)
    table_config = AccuracyTablesConfig(
        per_domain=per_domain,
        domains=DOMAINS,
        edges_by_domain=edges_by_domain,
        overall_edges=overall_edges,
        metric=metric,
        file_tag=file_tag,
        args=args,
    )
    _write_accuracy_tables(table_config)

    tag = args.outfile_tag or "combined"
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    fig_config = AccuracyFigureConfig(
        per_domain=per_domain,
        domains=DOMAINS,
        edges_by_domain=edges_by_domain,
        x_label=x_label,
        args=args,
        color_noaha=color_noaha,
        color_aha=color_aha,
    )
    fig = _build_accuracy_figure(fig_config)
    out_base = os.path.join(
        args.outdir,
        f"graph_3_pass1_bins_{file_tag}_{tag}",
    )
    save_all_formats(fig, out_base, dpi=args.dpi)
    return True

def _load_metric_rows(args, metric: str, warn_suffix: str) -> List[Dict[str, Any]]:
    """
    Load flat row dicts for a metric across all domains.
    """
    rows: List[Dict[str, Any]] = []
    rows += load_rows_from_roots_metric(
        args.roots_carpark,
        "Carpark",
        metric,
        args,
    )
    rows += load_rows_from_roots_metric(
        args.roots_crossword,
        "Crossword",
        metric,
        args,
    )
    rows += load_rows_from_roots_metric(
        args.roots_math,
        "Math",
        metric,
        args,
    )
    if not rows:
        print(
            f"[warn] No PASS1 records with {metric} entropy found ({warn_suffix}).",
            file=sys.stderr,
        )
    return rows


def _build_per_domain_counts(
    rows: List[Dict[str, Any]],
    domains: List[str],
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Convert flat row dicts into per-domain entropy/aha arrays for counts plots.
    """
    per_domain: Dict[str, Dict[str, List[Any]]] = {
        domain: {"entropy": [], "aha": []} for domain in domains
    }
    for row in rows:
        domain = row["domain"]
        per_domain[domain]["entropy"].append(row["entropy"])
        per_domain[domain]["aha"].append(row["aha"])
    for domain in domains:
        per_domain[domain]["entropy"] = np.asarray(
            per_domain[domain]["entropy"],
            dtype=float,
        )
        per_domain[domain]["aha"] = np.asarray(
            per_domain[domain]["aha"],
            dtype=int,
        )
    return per_domain


def _compute_edges_for_counts(
    per_domain: Dict[str, Dict[str, np.ndarray]],
    domains: List[str],
    args,
) -> Dict[str, np.ndarray]:
    """
    Compute per-domain bin edges for count plots, honoring the share_bins mode.
    """
    if args.share_bins == "global":
        all_ent = np.concatenate(
            [
                per_domain[domain]["entropy"]
                for domain in domains
                if per_domain[domain]["entropy"].size
            ],
        )
        global_edges = compute_edges(
            all_ent,
            args.bins,
            args.binning,
            args.entropy_min,
            args.entropy_max,
        )
        return {domain: global_edges for domain in domains}

    return {
        domain: compute_edges(
            per_domain[domain]["entropy"],
            args.bins,
            args.binning,
            args.entropy_min,
            args.entropy_max,
        )
        for domain in domains
    }


def _build_counts_figure(config: CountsFigureConfig):
    """
    Build the 3-panel histogram figure for Aha counts.
    """
    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(config.args.width_in, config.args.height_in),
        sharex=True,
        constrained_layout=True,
    )

    for axis, domain_name in zip(axes, config.domains):
        _render_counts_panel(axis, domain_name, config)

    ref_edges = config.edges_by_domain["Carpark"]
    centers = 0.5 * (ref_edges[:-1] + ref_edges[1:])
    tick_labels = [
        f"[{ref_edges[i]:.2f},{ref_edges[i + 1]:.2f}]"
        for i in range(len(ref_edges) - 1)
    ]
    axes[-1].set_xticks(centers)
    axes[-1].set_xticklabels(tick_labels, rotation=45, ha="right")
    axes[-1].set_xlabel(config.x_label)

    if config.args.title:
        fig.suptitle(
            config.args.title,
            y=1.02,
            fontsize=14,
            fontweight="bold",
        )

    return fig


def render_metric_counts(args, metric: str, color_aha):
    """
    Render three-panel Aha count histograms for a metric.

    Uses the same binning configuration as the accuracy plots.
    """
    rows = _load_metric_rows(args, metric, warn_suffix="counts")
    if not rows:
        return False

    domains = ["Carpark", "Crossword", "Math"]
    per_dom = _build_per_domain_counts(rows, domains)
    edges_by_dom = _compute_edges_for_counts(per_dom, domains, args)
    x_label, file_tag = _metric_labels(metric)

    counts_config = CountsFigureConfig(
        per_domain=per_dom,
        domains=domains,
        edges_by_domain=edges_by_dom,
        x_label=x_label,
        args=args,
        color_aha=color_aha,
    )
    fig = _build_counts_figure(counts_config)

    tag = args.outfile_tag or "combined"
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    out_base = os.path.join(
        args.outdir,
        f"graph_3_pass1_counts_{file_tag}_{tag}",
    )
    save_all_formats(fig, out_base, dpi=args.dpi)
    return True


def main():
    """CLI entry point for PASS1 entropy bucket plots (graph_3)."""
    args = parse_args()

    try:
        cmap = plt.get_cmap(args.cmap)
    except (ValueError, TypeError):
        cmap = plt.get_cmap("YlGnBu")
    color_noaha = cmap(0.35)  # lighter
    color_aha   = cmap(0.75)  # darker

    anything = False
    for metric in args.which_metrics:
        a_ok = render_metric_accuracy(args, metric, color_noaha, color_aha)
        c_ok = render_metric_counts(args,   metric, color_aha)
        anything = anything or a_ok or c_ok

    if not anything:
        sys.exit(2)

if __name__ == "__main__":
    main()
