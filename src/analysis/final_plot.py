#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
uncertainty_gated_reconsideration.py  (final-plot with NaN-safe binning)

Figure: (A) Shift prevalence vs entropy (pass-1) → "uncertainty-seeking"
        (B) Gated reconsideration success vs entropy (pass-2 injected cue),
            stratified by pass-1 group baseline: ≥1/8 correct vs 0/8 correct.

Key robustness fixes:
- Uses pandas IntervalIndex from the provided bin edges for consistent bin keys.
- Fills/guards N before int() casts (prevents "cannot convert float NaN to integer").
- Works when some bins/strata are empty.

Inputs
------
- Scans JSON/JSONL(.gz)/JSON files under --scan_root; records should include:
  pass1{ entropy(_think/_answer), ... }, pass2{ has_reconsider_cue, reconsider_markers }, and
  correctness signals (booleans or canon/raw equality).
- For Carpark, you can derive correctness from soft_reward via
  --carpark_success_op/--carpark_soft_threshold.

Outputs
-------
- PNG + PDF: two stacked panels
- CSVs with per-bin aggregates

Example
-------
python -m src.analysis.final_plot_uncertainty_gate \
  --scan_root graphs/_agg_qwen15b_t07 \
  --bins 0 1 2 3 4 \
  --min_step 0 --max_step 1000 \
  --carpark_success_op ge --carpark_soft_threshold 0.1 \
  --out_dir graphs/uncertainty_qwen15b_t07 \
  --dataset_name MIXED \
  --model_name "Qwen2.5-1.5B @ T=0.7" \
  --dpi 600 --make_plot
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


try:  # pragma: no cover - optional dependency for tests
    from src.analysis.io import iter_records_from_file, scan_files_step_only
except ImportError as _IO_IMPORT_ERROR:  # type: ignore[misc]  # pragma: no cover
    ITER_RECORDS_FROM_FILE = None  # type: ignore[assignment]
    SCAN_FILES_STEP_ONLY = None  # type: ignore[assignment]
else:
    _IO_IMPORT_ERROR = None
    ITER_RECORDS_FROM_FILE = iter_records_from_file  # type: ignore[assignment]
    SCAN_FILES_STEP_ONLY = scan_files_step_only  # type: ignore[assignment]
try:  # pragma: no cover - optional dependency for tests
    from src.analysis.metrics import carpark_success_from_soft_reward, extract_correct
    from src.analysis.utils import (
        add_carpark_threshold_args,
        add_common_plot_args,
        coerce_bool,
        entropy_from_pass1,
        step_from_rec_or_path,
    )
except ImportError as _UTILS_IMPORT_ERROR:  # type: ignore[misc]  # pragma: no cover
    CARPARK_SUCCESS_FROM_SOFT_REWARD = None  # type: ignore[assignment]
    EXTRACT_CORRECT = None  # type: ignore[assignment]
    ADD_CARPARK_THRESHOLD_ARGS = None  # type: ignore[assignment]
    ADD_COMMON_PLOT_ARGS = None  # type: ignore[assignment]
    COERCE_BOOL = None  # type: ignore[assignment]
    ENTROPY_FROM_PASS1 = None  # type: ignore[assignment]
    STEP_FROM_REC_OR_PATH = None  # type: ignore[assignment]
else:
    _UTILS_IMPORT_ERROR = None
    CARPARK_SUCCESS_FROM_SOFT_REWARD = carpark_success_from_soft_reward  # type: ignore[assignment]
    EXTRACT_CORRECT = extract_correct  # type: ignore[assignment]
    ADD_CARPARK_THRESHOLD_ARGS = add_carpark_threshold_args  # type: ignore[assignment]
    ADD_COMMON_PLOT_ARGS = add_common_plot_args  # type: ignore[assignment]
    COERCE_BOOL = coerce_bool  # type: ignore[assignment]
    ENTROPY_FROM_PASS1 = entropy_from_pass1  # type: ignore[assignment]
    STEP_FROM_REC_OR_PATH = step_from_rec_or_path  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from src.analysis import rq3_analysis as _rq3_analysis_module
except ImportError as _RQ3_IMPORT_ERROR:  # type: ignore[misc]
    _rq3_analysis_module = None
else:
    _RQ3_IMPORT_ERROR = None

try:  # pragma: no cover - optional dependency
    import matplotlib.pyplot as plt
except ImportError as _PLT_IMPORT_ERROR:  # type: ignore[misc]
    plt = None  # type: ignore[assignment]
else:
    _PLT_IMPORT_ERROR = None


def example_key(rec: Dict[str, Any]) -> Optional[str]:
    """Return a stable per-example key for grouping rows."""
    for k in ("example_id", "problem_id", "id", "uid", "question", "clue", "title"):
        value = rec.get(k)
        if value is not None and not isinstance(value, (dict, list)):
            return f"{k}:{value}"
    sample_idx = rec.get("sample_idx")
    return None if sample_idx is None else f"sample_{sample_idx}"


@dataclass
class CarparkEvalConfig:
    """Configuration for Carpark-style correctness extraction and filtering."""

    carpark_op: str
    carpark_thr: float
    min_step: Optional[int]
    max_step: Optional[int]


def pass2_triggered(pass2_data: Dict[str, Any]) -> int:
    """Return 1 iff pass-2 was explicitly triggered via an injected cue."""
    has_cue = coerce_bool(pass2_data.get("has_reconsider_cue"))
    markers = pass2_data.get("reconsider_markers") or []
    has_injected_cue = (
        ("injected_cue" in markers)
        if isinstance(
            markers,
            (list, tuple),
        )
        else False
    )
    return int(has_cue == 1 and has_injected_cue)


# -------------------------- Aggregation logic --------------------------


def _build_row_for_record(
    path: str,
    rec: Dict[str, Any],
    carpark_cfg: CarparkEvalConfig,
) -> Optional[Dict[str, Any]]:
    """
    Construct a single per-sample row for the main DataFrame, or ``None``.
    """
    pass1_data = rec.get("pass1") or {}
    if not isinstance(pass1_data, dict):
        return None

    # Prefer canonical pass2; fall back to multi-cue variants if needed.
    pass2_data = rec.get("pass2") or rec.get("pass2c") or rec.get("pass2b") or rec.get("pass2a") or {}

    step = step_from_rec_or_path(rec, path)
    if carpark_cfg.min_step is not None and step < carpark_cfg.min_step:
        return None
    if carpark_cfg.max_step is not None and step > carpark_cfg.max_step:
        return None

    group_id = f"{example_key(rec) or f'path:{os.path.basename(path)}:line'}::step{step}"
    entropy_value = entropy_from_pass1(pass1_data, mode="combined")

    # correctness (pass-1), with carpark fallback
    pass1_correct = extract_correct(pass1_data, rec)
    if pass1_correct is None:
        pass1_correct = carpark_success_from_soft_reward(
            rec,
            pass1_data,
            carpark_cfg.carpark_op,
            carpark_cfg.carpark_thr,
        )

    # pass-2
    triggered = 0
    pass2_correct = None
    if isinstance(pass2_data, dict) and pass2_data:
        triggered = pass2_triggered(pass2_data)
        pass2_correct = extract_correct(pass2_data, rec)
        if pass2_correct is None:
            pass2_correct = carpark_success_from_soft_reward(
                rec,
                pass2_data,
                carpark_cfg.carpark_op,
                carpark_cfg.carpark_thr,
            )

    return {
        "group_id": group_id,
        "entropy": entropy_value,
        "p1_correct": None if pass1_correct is None else int(pass1_correct),
        "p2_triggered": int(triggered),
        "p2_correct": None if pass2_correct is None else int(pass2_correct),
        "p1_shift": int(coerce_bool(pass1_data.get("shift_in_reasoning_v1")) == 1),
    }


def load_dataframe(
    files: List[str],
    carpark_op: str,
    carpark_thr: float,
    min_step: Optional[int],
    max_step: Optional[int],
) -> pd.DataFrame:
    """
    Load raw JSONL records into a per-sample DataFrame used by the final plot.

    Row = one sample. Columns:
      group_id = (example_key || sample_idx) + "::stepNNN"
      entropy  = pass-1 entropy (fallback rules)
      p1_correct, p2_triggered, p2_correct, p1_shift
    """
    rows: List[Dict[str, Any]] = []
    carpark_cfg = CarparkEvalConfig(
        carpark_op=carpark_op,
        carpark_thr=carpark_thr,
        min_step=min_step,
        max_step=max_step,
    )
    for path in files:
        for rec in iter_records_from_file(path):
            row = _build_row_for_record(
                path=path,
                rec=rec,
                carpark_cfg=carpark_cfg,
            )
            if row is not None:
                rows.append(row)

    dataframe = pd.DataFrame(rows)
    # drop rows with no entropy or p1 correctness
    dataframe = dataframe[pd.notna(dataframe["entropy"]) & pd.notna(dataframe["p1_correct"])]
    return dataframe


def summarize_for_figure(
    dataframe: pd.DataFrame,
    bins: List[float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate per-sample rows into the two panel DataFrames used for plotting.

    Returns:
      df_top  : per-bin shift prevalence (pass-1 sample-level)
      df_bot  : per-bin, per-stratum (baseline 0/8 vs ≥1/8) pass-2 any-correct rate (triggered only)
    """
    # ---- Top panel: shift prevalence vs entropy (sample-level) ----
    df_top = dataframe.copy()
    df_top["_bin"] = pd.cut(
        df_top["entropy"],
        bins=bins,
        right=False,
        include_lowest=True,
    )
    grp_top = (
        df_top.groupby("_bin", observed=False)
        .agg(
            N=("p1_shift", "size"),
            shift_share=("p1_shift", "mean"),
        )
        .reset_index()
    )

    # ---- Bottom panel: gated reconsideration success vs entropy (group-level) ----
    grouped = (
        dataframe.copy()
        .groupby("group_id", as_index=False)
        .agg(
            entropy_mean=("entropy", "mean"),
            p1_any_correct=("p1_correct", lambda x: int(np.nansum(x) > 0)),
            p2_any_correct_triggered=("p2_correct", lambda x: int(np.nansum(x) > 0)),
            any_trigger=("p2_triggered", lambda x: int(np.nansum(x) > 0)),
        )
    )
    grouped = grouped[grouped["any_trigger"] == 1].copy()
    grouped["_bin"] = pd.cut(
        grouped["entropy_mean"],
        bins=bins,
        right=False,
        include_lowest=True,
    )
    grouped["_stratum"] = np.where(
        grouped["p1_any_correct"] == 1,
        "baseline ≥1/8",
        "baseline 0/8",
    )

    def _agg(subframe: pd.DataFrame) -> pd.Series:
        num_rows = len(subframe)
        if num_rows == 0:
            return pd.Series(
                {
                    "N": 0,
                    "rate": np.nan,
                    "se": np.nan,
                    "lo": np.nan,
                    "hi": np.nan,
                },
            )
        rate = float(np.mean(subframe["p2_any_correct_triggered"]))
        std_err = math.sqrt(rate * (1.0 - rate) / num_rows) if num_rows > 0 else np.nan
        lower = max(0.0, rate - 1.96 * std_err)
        upper = min(1.0, rate + 1.96 * std_err)
        return pd.Series(
            {
                "N": num_rows,
                "rate": rate,
                "se": std_err,
                "lo": lower,
                "hi": upper,
            },
        )

    try:
        df_bot = grouped.groupby(["_bin", "_stratum"], observed=False).apply(_agg, include_groups=False).reset_index()
    except TypeError:  # pragma: no cover - older pandas
        df_bot = grouped.groupby(["_bin", "_stratum"], observed=False).apply(_agg).reset_index()

    return grp_top, df_bot


# ----------------------------- Plotting (FIXED) -----------------------------


@dataclass
class FigureOutputConfig:
    """Configuration for plot output paths and styling."""

    out_png: str
    out_pdf: str
    title_suffix: str
    dpi: int = 300


@dataclass
class BinLayout:
    """Layout for bin-based plots: intervals, tick labels, and x positions."""

    intervals: pd.IntervalIndex
    tick_labels: List[str]
    x_positions: np.ndarray


def _build_interval_index(bins: List[float]) -> BinLayout:
    """
    Construct an IntervalIndex and matching tick labels/positions for entropy bins.
    """
    intervals = pd.IntervalIndex.from_breaks(bins, closed="left")
    tick_labels = [f"[{left:g},{right:g})" for left, right in zip(intervals.left, intervals.right)]
    x_positions = np.arange(len(intervals))
    return BinLayout(intervals=intervals, tick_labels=tick_labels, x_positions=x_positions)


def _plot_top_panel(
    axis,
    df_top: pd.DataFrame,
    layout: BinLayout,
    title_suffix: str,
) -> None:
    """Render panel (A): shift prevalence vs entropy."""
    share_map: Dict[pd.Interval, float] = {}
    for _, row in df_top.iterrows():
        share_map[row["_bin"]] = row["shift_share"]
    y_values = [share_map.get(interval, np.nan) for interval in layout.intervals]

    axis.bar(layout.x_positions, y_values)
    axis.set_xticks(layout.x_positions, layout.tick_labels, rotation=0)
    axis.set_ylabel("Shift share (pass-1)")
    axis.set_title(f"(A) Shifts cluster at high uncertainty — {title_suffix}")
    axis.grid(True, axis="y", alpha=0.25)


def _plot_bottom_panel(
    axis,
    df_bot: pd.DataFrame,
    layout,
) -> Tuple[np.ndarray, np.ndarray]:
    """Render panel (B): gated reconsideration success vs entropy and return per-bin counts."""
    bot = df_bot.copy()
    bot["N"] = pd.to_numeric(bot["N"], errors="coerce").fillna(0)

    def _series_for_stratum(stratum: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        stratum_frame = bot[bot["_stratum"] == stratum].copy()
        rate_map = dict(zip(stratum_frame["_bin"], stratum_frame["rate"]))
        se_map = dict(zip(stratum_frame["_bin"], stratum_frame["se"]))
        count_map = dict(
            zip(
                stratum_frame["_bin"],
                (0 if pd.isna(count) else int(count) for count in stratum_frame["N"]),
            ),
        )

        rates = np.array(
            [rate_map.get(interval, np.nan) for interval in layout.intervals],
            float,
        )
        std_errs = np.array(
            [se_map.get(interval, np.nan) for interval in layout.intervals],
            float,
        )
        counts = np.array(
            [count_map.get(interval, 0) for interval in layout.intervals],
            int,
        )
        return rates, std_errs, counts

    series = {
        "high": _series_for_stratum("baseline ≥1/8"),
        "low": _series_for_stratum("baseline 0/8"),
    }

    axis.errorbar(
        layout.x_positions,
        series["high"][0],
        yerr=series["high"][1],
        fmt="o-",
        capsize=4,
        linewidth=2,
        label="baseline ≥1/8",
    )
    axis.errorbar(
        layout.x_positions,
        series["low"][0],
        yerr=series["low"][1],
        fmt="s--",
        capsize=4,
        linewidth=2,
        label="baseline 0/8",
    )
    axis.set_xticks(layout.x_positions, layout.tick_labels, rotation=0)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("P(pass-2 any-correct | injected)")
    axis.set_title("(B) Triggering reconsideration helps only when prior ≥1/8")
    axis.grid(True, axis="y", alpha=0.25)

    # Annotate sample sizes under each x tick: N_high/N_low
    for x_index, (count_high, count_low) in enumerate(
        zip(series["high"][2], series["low"][2]),
    ):
        axis.text(
            x_index,
            -0.08,
            f"N={count_high}/{count_low}",
            ha="center",
            va="top",
            fontsize=9,
            transform=axis.get_xaxis_transform(),
        )

    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=2,
        frameon=True,
    )


def plot_figure(
    df_top: pd.DataFrame,
    df_bot: pd.DataFrame,
    bins: List[float],
    config: FigureOutputConfig,
) -> None:
    """
    NaN-safe plotting using IntervalIndex keys (prevents cast errors when bins/strata are empty).
    """
    if plt is None:
        raise SystemExit(
            f"matplotlib is required for plotting: {_PLT_IMPORT_ERROR}",
        )

    layout = _build_interval_index(bins)
    fig, axes = plt.subplots(2, 1, figsize=(6.0, 5.0), constrained_layout=False)
    ax1, ax2 = axes

    _plot_top_panel(
        axis=ax1,
        df_top=df_top,
        layout=layout,
        title_suffix=config.title_suffix,
    )
    _plot_bottom_panel(
        axis=ax2,
        df_bot=df_bot,
        layout=layout,
    )

    fig.subplots_adjust(bottom=0.22, hspace=0.35)
    fig.savefig(config.out_png, dpi=config.dpi, bbox_inches="tight")
    fig.savefig(config.out_pdf, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)


# ------------------------------- Main ---------------------------------


def _maybe_run_rq3(args: argparse.Namespace) -> None:
    """
    Optionally run the core RQ3 analysis on the same scan root.
    """
    if not args.run_rq3:
        return

    # Optionally materialize the RQ3 outputs (H3 GLMs and bucket plots) so the
    # core uncertainty×intervention analysis is run alongside this specialized
    # gated reconsideration figure.
    if _rq3_analysis_module is None:
        print(
            f"[warn] Could not import src.analysis.rq3_analysis: {_RQ3_IMPORT_ERROR}",
            file=sys.stderr,
        )
        return

    old_argv = list(sys.argv)
    sys.argv = ["rq3_analysis.py", args.scan_root]
    if args.split:
        sys.argv += ["--split", args.split]
    try:
        _rq3_analysis_module.main()
    finally:
        sys.argv = old_argv


def _save_outputs(
    args: argparse.Namespace,
    grp_top: pd.DataFrame,
    df_bot: pd.DataFrame,
    bins: List[float],
) -> None:
    """
    Save CSV and optional PNG/PDF outputs and print a console summary.
    """
    slug = f"{args.dataset_name}__{args.model_name}".replace(" ", "_")
    out_dir = args.out_dir or os.path.join(args.scan_root, "uncertainty_gated_effect")
    os.makedirs(out_dir, exist_ok=True)
    csv_top = os.path.join(out_dir, f"uncertainty_shift_prevalence__{slug}.csv")
    csv_bot = os.path.join(out_dir, f"uncertainty_gated_success__{slug}.csv")
    grp_top.to_csv(csv_top, index=False)
    df_bot.to_csv(csv_bot, index=False)
    print(f"[saved] {csv_top}")
    print(f"[saved] {csv_bot}")

    if args.make_plot:
        png_path = os.path.join(out_dir, f"uncertainty_gated_effect__{slug}.png")
        pdf_path = os.path.join(out_dir, f"uncertainty_gated_effect__{slug}.pdf")
        plot_figure(
            df_top=grp_top,
            df_bot=df_bot,
            bins=bins,
            config=FigureOutputConfig(
                out_png=png_path,
                out_pdf=pdf_path,
                title_suffix=args.model_name,
                dpi=args.dpi,
            ),
        )
        print(f"[saved] {png_path}")
        print(f"[saved] {pdf_path}")

    # Console preview
    with pd.option_context("display.width", 140):
        print("\n[Top panel] Shift prevalence vs entropy (pass-1 sample-level):")
        print(grp_top.to_string(index=False))
        print("\n[Bottom panel] P(pass-2 any-correct | injected) by entropy bin and stratum:")
        print(df_bot.to_string(index=False))


def main() -> None:
    """
    CLI entry point for uncertainty-gated reconsideration analysis and plotting.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scan_root",
        type=str,
        required=True,
        help="Root directory containing step*/.../*.jsonl|json(.gz)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Only include files whose NAMES contain this substring (e.g., 'test').",
    )
    parser.add_argument(
        "--bins",
        nargs="+",
        type=float,
        default=[0, 1, 2, 3, 4],
        help="Entropy bin edges (closed-open).",
    )
    parser.add_argument("--min_step", type=int, default=None)
    parser.add_argument("--max_step", type=int, default=None)

    add_carpark_threshold_args(parser)

    add_common_plot_args(parser)

    # Optional: run RQ3 (uncertainty × intervention) analysis first
    parser.add_argument(
        "--run_rq3",
        action="store_true",
        help=(
            "Also run the core RQ3 analysis (src.analysis.rq3_analysis) on --scan_root before building this figure."
        ),
    )

    args = parser.parse_args()

    # Scan
    files = scan_files_step_only(
        root=args.scan_root,
        split_substr=args.split,
        skip_substrings=None,
    )
    if not files:
        sys.exit("No files found. Check --scan_root / --split.")

    _maybe_run_rq3(args)

    # Load
    dataframe = load_dataframe(
        files=files,
        carpark_op=args.carpark_success_op,
        carpark_thr=args.carpark_soft_threshold,
        min_step=args.min_step,
        max_step=args.max_step,
    )
    if dataframe.empty:
        sys.exit("No usable rows after parsing (missing entropy or p1 correctness).")

    # Summaries
    bins = list(args.bins)
    grp_top, df_bot = summarize_for_figure(dataframe, bins=bins)

    _save_outputs(args, grp_top, df_bot, bins)


if __name__ == "__main__":
    main()
