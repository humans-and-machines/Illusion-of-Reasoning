#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Entropy-bucket accuracy histograms (Aha! vs No-Aha)
---------------------------------------------------
For each domain, bucket an entropy measure into N bins and plot accuracy
on the y-axis with two bars per bin: Aha=1 (LLM-detected shift) vs Aha=0.

Panels per domain:
  • Think entropy
  • Answer entropy
  • Joint entropy (uses pass1["entropy"] if available; else mean(think, answer))

Success definition by domain (accuracy):
  • Crossword/Math/Math2: pass1["is_correct_pred"]
  • Carpark: 1[ soft_reward OP threshold ] with OP ∈ {gt, ge, eq} (default gt 0.0)

Outputs (under --out_dir, defaults to <first_root>/entropy_histograms):
  1) CSV:  entropy_hist__<dataset>__<model>.csv
     columns = [
       domain, metric, bin_idx, bin_left, bin_right,
       n_total, n_aha, n_noaha, acc_aha, acc_noaha
     ]

  2) One figure per domain:
     <slug>__<domain>__entropy_hist_panels.[png|pdf]
     Panels: think | answer | joint

Notes
-----
- Aha! (shift) uses the same domain-aware gating as your figure code.
- Step filters supported; hard-capped at step ≤ 1000 by default (can override).
- Binning: default uniform-width over [min, max] per-domain per-metric.
  Optionally use quantile bins with --bucket_mode=quantile.
"""

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib as mpl  # type: ignore[import-error]

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore[import-error]
except ImportError:  # pragma: no cover - optional plotting dependency
    mpl = None
    plt = None

try:
    # Package imports
    from .io import build_files_by_domain_for_args, iter_records_from_file
    from .labels import aha_gpt_for_rec
    from .metrics import carpark_success_from_soft_reward
    from .utils import (
        add_carpark_threshold_args,
        add_gpt_mode_arg,
        add_split_and_out_dir_args,
        add_standard_domain_root_args,
        coerce_bool,
        coerce_float,
        get_problem_id,
        gpt_keys_for_mode,
        nat_step_from_path,
    )
except ImportError:  # pragma: no cover - script fallback
    import sys

    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.append(_ROOT)
    from analysis.io import build_files_by_domain_for_args, iter_records_from_file  # type: ignore
    from analysis.labels import aha_gpt_for_rec  # type: ignore
    from analysis.metrics import carpark_success_from_soft_reward  # type: ignore
    from analysis.utils import (  # type: ignore
        add_carpark_threshold_args,
        add_gpt_mode_arg,
        add_split_and_out_dir_args,
        add_standard_domain_root_args,
        coerce_bool,
        coerce_float,
        get_problem_id,
        gpt_keys_for_mode,
        nat_step_from_path,
    )

# ---------- Style (Times-like) ----------
STYLE_PARAMS = {
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "font.size": 13,
    "legend.fontsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "Times",
        "Nimbus Roman No9 L",
        "STIXGeneral",
        "DejaVu Serif",
    ],
    "mathtext.fontset": "stix",
}
mpl.rcParams.update(STYLE_PARAMS)

# ---------- Carpark success policy ----------
def _extract_soft_reward(
    record: Dict[str, Any],
    pass1_data: Dict[str, Any],
) -> Optional[float]:
    """Extract the soft_reward value from a record/pass1 pair."""
    return coerce_float(record.get("soft_reward", pass1_data.get("soft_reward")))


def _make_carpark_success_fn(comparison_op: str, threshold: float):
    """Build a soft-reward comparator for the carpark domain."""

    def _cmp(value: Any) -> Optional[int]:
        return carpark_success_from_soft_reward(
            {"soft_reward": value},
            {},
            comparison_op,
            threshold,
        )

    return _cmp


@dataclass
class RowLoadConfig:
    """Configuration used when loading per-row metrics."""

    gpt_keys: List[str]
    gpt_subset_native: bool
    min_step: Optional[int]
    max_step: Optional[int]


@dataclass
class RecordContext:
    """Lightweight context for a single JSONL results file."""

    domain: str
    path: str
    step_from_name: Optional[int]


def _extract_step_for_record(
    rec: Dict[str, Any],
    ctx: RecordContext,
    cfg: RowLoadConfig,
) -> Optional[int]:
    """Return the integer step for a record, or None if out of bounds."""
    raw_step = rec.get(
        "step",
        ctx.step_from_name if ctx.step_from_name is not None else None,
    )
    if raw_step is None:
        return None
    try:
        step_value = int(raw_step)
    except (TypeError, ValueError):
        return None
    if cfg.min_step is not None and step_value < cfg.min_step:
        return None
    if cfg.max_step is not None and step_value > cfg.max_step:
        return None
    return step_value


def _compute_success_for_record(
    ctx: RecordContext,
    rec: Dict[str, Any],
    pass1_data: Dict[str, Any],
    carpark_success_fn,
) -> Optional[int]:
    """Compute the success flag for a single record."""
    domain_lower = str(ctx.domain).lower()
    if domain_lower.startswith("carpark"):
        success_flag = carpark_success_fn(_extract_soft_reward(rec, pass1_data))
    else:
        success_flag = coerce_bool(pass1_data.get("is_correct_pred"))
    if success_flag is None:
        return None
    return int(success_flag)


def _compute_entropies(
    pass1_data: Dict[str, Any],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Extract think/answer/joint entropies from a pass1 mapping."""
    ent_think = coerce_float(pass1_data.get("entropy_think"))
    ent_answer = coerce_float(pass1_data.get("entropy_answer"))
    ent_joint = coerce_float(pass1_data.get("entropy"))
    if ent_joint is None:
        values = [value for value in (ent_think, ent_answer) if value is not None]
        ent_joint = float(np.mean(values)) if values else None
    return ent_think, ent_answer, ent_joint


# ---------- Row loader ----------
def _process_single_record(
    ctx: RecordContext,
    rec: Dict[str, Any],
    cfg: RowLoadConfig,
    carpark_success_fn,
) -> Optional[Dict[str, Any]]:
    """Process a single JSONL record into a metrics row if applicable."""
    if not isinstance(rec, dict):
        return None

    pass1_data = rec.get("pass1") or {}
    if not isinstance(pass1_data, dict):
        return None

    step = _extract_step_for_record(rec, ctx, cfg)
    if step is None:
        return None

    success = _compute_success_for_record(
        ctx,
        rec,
        pass1_data,
        carpark_success_fn,
    )
    if success is None:
        return None

    ent_think, ent_answer, ent_joint = _compute_entropies(pass1_data)
    if ent_think is None and ent_answer is None and ent_joint is None:
        return None

    aha = aha_gpt_for_rec(
        pass1_data,
        rec,
        cfg.gpt_subset_native,
        cfg.gpt_keys,
        ctx.domain,
    )
    pid = get_problem_id(rec) or f"unnamed_{hash((ctx.path, repr(rec))) % 10**9}"

    return {
        "domain": str(ctx.domain),
        "problem_id": pid,
        "step": step,
        "correct": int(success),
        "aha": int(aha),
        "entropy_think": ent_think,
        "entropy_answer": ent_answer,
        "entropy_joint": ent_joint,
    }


def load_rows(
    files_by_domain: Dict[str, List[str]],
    cfg: RowLoadConfig,
    carpark_success_fn,
) -> pd.DataFrame:
    """
    Load per-problem rows from JSONL result files into a DataFrame.

    The function applies step filters, domain-specific success policies, and
    derives think/answer/joint entropies plus Aha labels.
    """
    rows: List[Dict[str, Any]] = []
    for dom, files in files_by_domain.items():
        for path in files:
            step_from_name = nat_step_from_path(path)
            ctx = RecordContext(domain=str(dom), path=path, step_from_name=step_from_name)
            for record in iter_records_from_file(path):
                row = _process_single_record(
                    ctx,
                    record,
                    cfg,
                    carpark_success_fn=carpark_success_fn,
                )
                if row is not None:
                    rows.append(row)
    return pd.DataFrame(rows)

# ---------- Binning and aggregation ----------
def make_bins(series: pd.Series, n_bins: int, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute bin edges and centers for a 1D numeric series.
    """
    non_null_values = series.dropna()
    if non_null_values.empty:
        return np.array([]), np.array([])
    if mode == "quantile":
        quantiles = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(non_null_values, quantiles))
        # ensure at least 2 distinct edges
        if len(edges) < 2:
            edges = np.array(
                [non_null_values.min(), non_null_values.max() + 1e-9],
            )
    else:
        min_val, max_val = float(non_null_values.min()), float(non_null_values.max())
        if min_val == max_val:
            min_val, max_val = min_val - 1e-9, max_val + 1e-9
        edges = np.linspace(min_val, max_val, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def _empty_bins_df() -> pd.DataFrame:
    """Return an empty bins DataFrame with the standard columns."""
    return pd.DataFrame(
        columns=[
            "domain",
            "metric",
            "bin_idx",
            "bin_left",
            "bin_right",
            "n_total",
            "n_aha",
            "n_noaha",
            "acc_aha",
            "acc_noaha",
        ],
    )


def _bin_group_stats(tmp: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return pivoted accuracy and count tables indexed by bin and aha."""
    grouped = tmp.groupby(["_bin", "aha"], observed=True)
    acc = grouped["correct"].mean().rename("acc").reset_index()
    cnt = grouped["correct"].size().rename("n").reset_index()
    merged = acc.merge(cnt, on=["_bin", "aha"], how="outer")
    piv_acc = merged.pivot(index="_bin", columns="aha", values="acc")
    piv_cnt = (
        merged.pivot(index="_bin", columns="aha", values="n")
        .fillna(0)
        .astype(int)
    )
    return piv_acc, piv_cnt


def _filter_supported_bins(
    piv_acc: pd.DataFrame,
    piv_cnt: pd.DataFrame,
    min_per_bar: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter bins that lack sufficient support in either Aha-state."""
    support_no = piv_cnt.get(0, 0)
    support_yes = piv_cnt.get(1, 0)
    ok_mask = (support_no >= min_per_bar) & (support_yes >= min_per_bar)
    return piv_acc[ok_mask], piv_cnt[ok_mask]


def _rows_from_pivots(
    metric_col: str,
    piv_acc: pd.DataFrame,
    piv_cnt: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Build a list of bin rows from pivoted stats."""
    rows: List[Dict[str, Any]] = []
    for bin_idx, bin_interval in enumerate(piv_acc.index):
        left = float(bin_interval.left)
        right = float(bin_interval.right)
        support_no = int(piv_cnt.loc[bin_interval, 0]) if 0 in piv_cnt.columns else 0
        support_yes = int(piv_cnt.loc[bin_interval, 1]) if 1 in piv_cnt.columns else 0
        acc_no = float(piv_acc.loc[bin_interval, 0]) if 0 in piv_acc.columns else np.nan
        acc_yes = float(piv_acc.loc[bin_interval, 1]) if 1 in piv_acc.columns else np.nan
        rows.append(
            {
                "metric": metric_col,
                "bin_idx": bin_idx,
                "bin_left": left,
                "bin_right": right,
                "bin_center": 0.5 * (left + right),
                "n_total": support_no + support_yes,
                "n_noaha": support_no,
                "n_aha": support_yes,
                "acc_noaha": acc_no,
                "acc_aha": acc_yes,
            },
        )
    return rows


def aggregate_bins(
    df_dom: pd.DataFrame,
    metric_col: str,
    n_bins: int,
    mode: str,
    min_per_bar: int,
) -> pd.DataFrame:
    """Aggregate accuracy into entropy bins for a single metric/domain."""
    metric_series = df_dom[metric_col]
    edges, _ = make_bins(metric_series, n_bins, mode)
    if edges.size == 0:
        return _empty_bins_df()

    cats = pd.cut(metric_series, bins=edges, include_lowest=True, right=True)
    tmp = df_dom.assign(_bin=cats).dropna(subset=["_bin"])
    if tmp.empty:
        return _empty_bins_df()

    piv_acc, piv_cnt = _bin_group_stats(tmp)
    if piv_acc.empty:
        return _empty_bins_df()

    piv_acc, piv_cnt = _filter_supported_bins(piv_acc, piv_cnt, min_per_bar)
    if piv_acc.empty:
        return _empty_bins_df()

    rows = _rows_from_pivots(metric_col, piv_acc, piv_cnt)
    return pd.DataFrame(rows)

# ---------- Plotting ----------
METRIC_CONFIGS = [
    ("entropy_think", "Think entropy"),
    ("entropy_answer", "Answer entropy"),
    ("entropy_joint", "Joint entropy"),
]


def _add_row_legend(axes) -> None:
    """Attach a shared legend for the Aha vs No-Aha bars."""
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        return
    axes[0].figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )


def plot_domain_panels(
    domain: str,
    stats_by_metric: Dict[str, pd.DataFrame],
    out_path_png: str,
    dpi: int = 600,
    title_prefix: Optional[str] = None,
) -> None:
    """Render a three-panel entropy/accuracy histogram for one domain."""
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.2), sharey=True)
    if title_prefix:
        fig.suptitle(f"{title_prefix} — {domain}", y=1.05)

    for axis, (metric_name, metric_label) in zip(axes, METRIC_CONFIGS):
        stat = stats_by_metric.get(metric_name)
        axis.set_title(metric_label)
        axis.set_xlabel("Entropy bins")
        axis.set_ylim(0.0, 1.0)
        if stat is None or stat.empty:
            axis.text(
                0.5,
                0.5,
                "No bins with sufficient data",
                ha="center",
                va="center",
                transform=axis.transAxes,
                fontsize=11,
            )
            continue

        positions = np.arange(len(stat))
        axis.bar(
            positions - 0.2,
            stat["acc_noaha"],
            width=0.4,
            label="No Aha",
            alpha=0.85,
        )
        axis.bar(
            positions + 0.2,
            stat["acc_aha"],
            width=0.4,
            label="Aha",
            alpha=0.85,
        )

        labels = [
            f"[{left:.2f},{right:.2f}]"
            for left, right in zip(stat["bin_left"], stat["bin_right"])
        ]
        axis.set_xticks(positions, labels, rotation=45, ha="right")
        axis.grid(axis="y", linestyle=":", alpha=0.4)

    axes[0].set_ylabel("Accuracy")
    _add_row_legend(axes)
    fig.tight_layout()
    fig.savefig(out_path_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)

# ---------- Main ----------
def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser for entropy histograms."""
    parser = argparse.ArgumentParser()
    # Domain roots + optional results_root
    add_standard_domain_root_args(parser)
    add_split_and_out_dir_args(
        parser,
        out_dir_help=(
            "Base output directory (default: <first_root>/entropy_histograms)."
        ),
    )
    parser.add_argument("--dataset_name", default="MIXED")
    parser.add_argument("--model_name", default="Qwen2.5-1.5B")

    # GPT label policy
    add_gpt_mode_arg(parser)
    parser.add_argument(
        "--no_gpt_subset_native",
        action="store_true",
        help="Disable domain-aware gate; use raw GPT shift flags",
    )

    # Binning
    parser.add_argument("--n_bins", type=int, default=10)
    parser.add_argument(
        "--bucket_mode",
        choices=["uniform", "quantile"],
        default="uniform",
        help="Bin edges strategy per domain/metric distribution",
    )
    parser.add_argument(
        "--min_per_bar",
        type=int,
        default=25,
        help="Min examples per (bin, Aha-state) to render the bar",
    )

    # Steps (default hard cap 1000 for safety)
    parser.add_argument("--min_step", type=int, default=None)
    parser.add_argument(
        "--max_step",
        type=int,
        default=1000,
        help="Upper bound on step (default 1000). Set higher to disable cap.",
    )

    # Carpark policy
    add_carpark_threshold_args(parser)

    # Rendering
    parser.add_argument("--dpi", type=int, default=600)
    return parser


def _compute_domain_stats_and_plots(
    dataframe: pd.DataFrame,
    args: argparse.Namespace,
    out_dir: str,
) -> None:
    """Compute per-domain entropy histograms and write CSV and figures."""
    csv_rows: List[pd.DataFrame] = []
    slug = f"{args.dataset_name}__{args.model_name}".replace(" ", "_")

    for domain in sorted(dataframe["domain"].unique()):
        sub = dataframe[dataframe["domain"] == domain].copy()
        stats_by_metric: Dict[str, pd.DataFrame] = {}
        for metric_name in ["entropy_think", "entropy_answer", "entropy_joint"]:
            available = sub[metric_name].dropna()
            if available.empty:
                stats_by_metric[metric_name] = pd.DataFrame()
                continue
            stat = aggregate_bins(
                sub,
                metric_name,
                n_bins=args.n_bins,
                mode=args.bucket_mode,
                min_per_bar=args.min_per_bar,
            )

            if "domain" not in stat.columns:
                stat.insert(0, "domain", domain)
            if "metric" not in stat.columns:
                stat.insert(1, "metric", metric_name)
            stats_by_metric[metric_name] = stat
            csv_rows.append(stat)

        fig_path = os.path.join(
            out_dir,
            f"{slug}__{domain}__entropy_hist_panels.png",
        )
        plot_domain_panels(
            domain,
            stats_by_metric,
            fig_path,
            dpi=args.dpi,
            title_prefix="Accuracy vs Entropy (Aha vs No-Aha)",
        )

    if not csv_rows:
        return

    table = pd.concat(csv_rows, axis=0, ignore_index=True)
    out_csv = os.path.join(out_dir, f"entropy_hist__{slug}.csv")
    table.to_csv(out_csv, index=False)
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print("\nPer-bin accuracy (head):")
        print(table.head(12).to_string(index=False))
    print(f"\nSaved CSV  -> {out_csv}")


def main() -> None:
    """CLI entrypoint for entropy-bucket accuracy histograms."""
    args = _build_arg_parser().parse_args()

    files_by_domain, first_root = build_files_by_domain_for_args(args)
    out_dir = args.out_dir or os.path.join(first_root, "entropy_histograms")
    os.makedirs(out_dir, exist_ok=True)

    load_cfg = RowLoadConfig(
        gpt_keys=gpt_keys_for_mode(args.gpt_mode),
        gpt_subset_native=not args.no_gpt_subset_native,
        min_step=args.min_step,
        max_step=args.max_step,
    )
    carpark_success_fn = _make_carpark_success_fn(
        comparison_op=args.carpark_success_op,
        threshold=args.carpark_soft_threshold,
    )
    dataframe = load_rows(
        files_by_domain,
        load_cfg,
        carpark_success_fn=carpark_success_fn,
    )
    if dataframe.empty:
        raise SystemExit("No rows after filtering.")

    _compute_domain_stats_and_plots(dataframe, args, out_dir)
    print(f"Saved figs -> {out_dir}/*__entropy_hist_panels.[png|pdf]")


if __name__ == "__main__":
    main()
