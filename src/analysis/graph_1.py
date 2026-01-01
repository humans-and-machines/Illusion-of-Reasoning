#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Raw shift effect per step (scatter only; no trend lines, no legend)
-------------------------------------------------------------------
Computes and plots the *raw* effect of LLM-detected "Aha!" (shift) on success
per training step, hard-capped at step ≤ 1000:

    raw_effect(step) = P(success | shift=1, step) - P(success | shift=0, step)

Success definition by domain:
  • Crossword/Math/Math2: success = is_correct_pred
  • Carpark: success = 1[ soft_reward OP threshold ] (OP ∈ {gt, ge, eq}; default: gt 0.0)

Outputs
-------
1) CSV  : <out_dir>/raw_effect_per_step__<dataset>__<model>.csv
2) PNGs : <out_dir>/raw_effect_per_step_panels_linear.png
          (3 panels; Math overlays Math & Math2)
          <out_dir>/raw_effect_per_step_overlay_linear.png
          (overlay across domains, not forced square)
          (PDFs saved alongside)

Plot units
----------
By default, plots are rendered in percentage points (pp), with ticks at
-20,-10,0,10,20 and axis padded slightly beyond ±20 ([-22,22]) so points/error
bars aren’t clipped. CSV remains in probability units (0..1).
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass
from textwrap import fill
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.plotting import apply_default_style


matplotlib.use("Agg")

try:
    # Package imports
    from .core import LoadRowsConfig, iter_correct_and_shift_samples_for_config
    from .core.plotting_helpers import compute_effective_max_step
    from .io import build_files_by_domain_for_args
    from .metrics import make_carpark_success_fn
    from .utils import add_gpt_step_and_carpark_args, build_mixed_root_arg_parser, get_problem_id, gpt_keys_for_mode
except ImportError:  # pragma: no cover - script fallback
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.append(_ROOT)
    from analysis.core import LoadRowsConfig, iter_correct_and_shift_samples_for_config  # type: ignore
    from analysis.core.plotting_helpers import compute_effective_max_step  # type: ignore
    from analysis.io import build_files_by_domain_for_args  # type: ignore
    from analysis.metrics import make_carpark_success_fn  # type: ignore
    from analysis.utils import (  # type: ignore
        add_gpt_step_and_carpark_args,
        build_mixed_root_arg_parser,
        get_problem_id,
        gpt_keys_for_mode,
    )

apply_default_style(
    {
        "savefig.bbox": "standard",
        "savefig.pad_inches": 0.02,
    },
)


@dataclass
class PlotUnitsConfig:
    """Configuration for axis units and limits."""

    y_scale: float
    ylim: Tuple[float, float]
    yticks: Optional[List[float]]
    ylabel: str


@dataclass
class PanelFigureConfig:
    """Configuration specific to the 3-panel figure."""

    dpi: int
    width_in: float
    height_scale: float
    marker_size: float


@dataclass
class OverlayFigureConfig:
    """Configuration specific to the overlay figure."""

    dpi: int
    width_in: float
    height_scale: float
    marker_size: float
    title: str


# ---------- Load rows ----------
def _iter_rows(
    files_by_domain: Dict[str, List[str]],
    config: LoadRowsConfig,
) -> Iterable[Dict[str, Any]]:
    """
    Yield row dictionaries for each record grouped by domain.
    """
    for dom, step, rec, correct, shift in iter_correct_and_shift_samples_for_config(
        files_by_domain,
        config,
    ):
        pid = get_problem_id(rec)
        if pid is None:
            continue

        yield {
            "domain": str(dom),
            "problem_id": pid,
            "step": int(step),
            "correct": int(correct),
            "shift": int(shift),
        }


def load_rows(
    files_by_domain: Dict[str, List[str]],
    config: LoadRowsConfig,
) -> pd.DataFrame:
    """
    Materialize the per-record rows for all domains into a DataFrame.
    """
    return pd.DataFrame(list(_iter_rows(files_by_domain, config)))


# ---------- Per-step aggregation ----------
def per_step_raw_effect(
    data_frame: pd.DataFrame,
    domain: str,
    min_per_group: int = 20,
) -> pd.DataFrame:
    """
    Aggregate per-step raw effects for a single domain.
    """
    sub_frame = data_frame[data_frame["domain"] == domain].copy()
    if sub_frame.empty:
        return pd.DataFrame(
            columns=[
                "domain",
                "step",
                "n",
                "n_shift",
                "n_noshift",
                "p_correct_shift",
                "p_correct_noshift",
                "raw_effect",
            ],
        )
    rows: List[Dict[str, Any]] = []
    for step, group_df in sub_frame.groupby("step"):
        group_size = len(group_df)
        n_shift = int((group_df["shift"] == 1).sum())
        n_noshift = group_size - n_shift
        if n_shift == 0 or n_noshift == 0 or group_size < min_per_group:
            continue
        p_s = float(group_df.loc[group_df["shift"] == 1, "correct"].mean())
        p_n = float(group_df.loc[group_df["shift"] == 0, "correct"].mean())
        rows.append(
            {
                "domain": domain,
                "step": int(step),
                "n": int(group_size),
                "n_shift": int(n_shift),
                "n_noshift": int(n_noshift),
                "p_correct_shift": p_s,
                "p_correct_noshift": p_n,
                "raw_effect": p_s - p_n,
            },
        )
    return pd.DataFrame(rows).sort_values("step")


# ---------- Plotters ----------
_COLORS = {"Crossword": "#1f77b4", "Math": "#2ca02c", "Math2": "#9467bd", "Carpark": "#d62728"}


def _color_for(dom_key: str) -> str:
    return _COLORS.get(dom_key, "C0")


def plot_panels(
    per_step: Dict[str, pd.DataFrame],
    label_map: Dict[str, str],
    out_png: str,
    fig_config: PanelFigureConfig,
    units: PlotUnitsConfig,
) -> None:
    """
    Plot the 3-panel raw-effect figure for Crossword, Math, and Carpark.
    """
    base_w, base_h = 9.0, 4.2
    height_in = max(
        2.0,
        (fig_config.width_in * (base_h / base_w)) * float(fig_config.height_scale),
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(fig_config.width_in, height_in),
        sharey=True,
        constrained_layout=False,
    )

    for axis, dom in zip(axes, ["Crossword", "Math", "Carpark"]):
        axis.set_title(label_map.get(dom, dom))
        axis.axhline(0.0, lw=1, color="k", alpha=0.35)
        if dom == "Math":
            for key in ["Math", "Math2"]:
                domain_df = per_step.get(key)
                if domain_df is None or domain_df.empty:
                    continue
                yvals = domain_df["raw_effect"].values * units.y_scale
                axis.scatter(
                    domain_df["step"],
                    yvals,
                    s=fig_config.marker_size,
                    alpha=0.55,
                    edgecolor="none",
                    color=_color_for(key),
                )
        else:
            domain_df = per_step.get(dom)
            if domain_df is not None and not domain_df.empty:
                yvals = domain_df["raw_effect"].values * units.y_scale
                axis.scatter(
                    domain_df["step"],
                    yvals,
                    s=fig_config.marker_size,
                    alpha=0.55,
                    edgecolor="none",
                    color=_color_for(dom),
                )
        axis.set_xlabel("Step")
        axis.set_ylim(units.ylim[0], units.ylim[1])
        if units.yticks is not None:
            axis.set_yticks(units.yticks)

    axes[0].set_ylabel(units.ylabel)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.90, bottom=0.3)
    fig.set_size_inches(fig_config.width_in, height_in, forward=True)
    fig.savefig(out_png, dpi=fig_config.dpi)
    fig.savefig(out_png.replace(".png", ".pdf"))
    plt.close(fig)


def _auto_wrap_title_to_two_lines(title: str, width: int = 42) -> str:
    """
    Wrap a figure title into at most two lines.
    """
    wrapped = fill(title, width=width)
    lines = wrapped.splitlines()
    return "\n".join(lines[:2]) if len(lines) > 2 else wrapped


def plot_overlay_all(
    per_step: Dict[str, pd.DataFrame],
    label_map: Dict[str, str],
    out_png: str,
    fig_config: OverlayFigureConfig,
    units: PlotUnitsConfig,
) -> None:
    """
    Plot the overlay figure across all domains.
    """
    height_in = max(
        3.0,
        (fig_config.width_in * (7.0 / 9.0)) * float(fig_config.height_scale),
    )
    fig, axis = plt.subplots(
        figsize=(fig_config.width_in, height_in),
        constrained_layout=False,
    )

    wrapped_title = _auto_wrap_title_to_two_lines(fig_config.title, width=42) if fig_config.title else None
    if wrapped_title:
        axis.set_title(wrapped_title, pad=6)

    fig.subplots_adjust(
        left=0.2,
        right=0.98,
        top=0.86 if wrapped_title and "\n" in wrapped_title else 0.90,
        bottom=0.19,
    )

    axis.axhline(0.0, lw=1, color="k", alpha=0.35)
    # Draw Math after Math2 so the Math (green) dots sit on top when points overlap at the same step.
    for dom in ["Crossword", "Math2", "Math", "Carpark"]:
        domain_df = per_step.get(dom)
        if domain_df is None or domain_df.empty:
            continue
        yvals = domain_df["raw_effect"].values * units.y_scale
        display_label = label_map.get(dom, dom)
        axis.scatter(
            domain_df["step"],
            yvals,
            s=fig_config.marker_size,
            alpha=1,
            edgecolor="none",
            color=_color_for(dom),
            label=display_label,
        )

    axis.set_xlabel("Step")
    axis.set_ylabel(units.ylabel)
    axis.set_ylim(units.ylim[0], units.ylim[1])
    if units.yticks is not None:
        axis.set_yticks(units.yticks)

    fig.set_size_inches(fig_config.width_in, height_in, forward=True)
    fig.savefig(out_png, dpi=fig_config.dpi)
    fig.savefig(out_png.replace(".png", ".pdf"))
    plt.close(fig)


# ---------- Main ----------
def _parse_float_list(value_string: Optional[str]) -> Optional[List[float]]:
    """
    Parse a comma- or whitespace-separated list of floats.
    """
    if value_string is None:
        return None
    value_string = value_string.strip()
    if not value_string:
        return None
    parts = re.split(r"[,\s]+", value_string)
    out: List[float] = []
    for part in parts:
        if not part:
            continue
        try:
            out.append(float(part))
        except (TypeError, ValueError):
            pass
    return out or None


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = build_mixed_root_arg_parser()
    parser.add_argument("--label_math", type=str, default="Qwen1.5B-Math")
    parser.add_argument("--label_math2", type=str, default="Qwen7B-Math")
    parser.add_argument("--min_per_group", type=int, default=20)
    add_gpt_step_and_carpark_args(parser)

    # Figure knobs
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
    )
    parser.add_argument(
        "--width_in",
        type=float,
        default=9.0,
        help="Canvas width (inches) for 3-panel figure",
    )
    parser.add_argument(
        "--height_scale",
        type=float,
        default=0.6667,
        help="Height scale for the 3-panel figure (width fixed)",
    )
    parser.add_argument(
        "--overlay_width_in",
        type=float,
        default=5,
        help="Canvas width (inches) for overlay figure (not forced square)",
    )
    parser.add_argument(
        "--overlay_height_scale",
        type=float,
        default=0.8,
        help="Height scale for overlay figure (relative to overlay_width_in)",
    )
    parser.add_argument(
        "--marker_size",
        type=float,
        default=28.0,
    )
    parser.add_argument(
        "--overlay_title",
        type=str,
        default=None,
    )

    # Units, limits, ticks
    parser.add_argument(
        "--plot_units",
        choices=["pp", "prob"],
        default="pp",
        help=("Axis units: 'pp' = percentage points (×100), 'prob' = raw probabilities."),
    )
    parser.add_argument(
        "--ymin_pp",
        type=float,
        default=-50.0,
    )
    parser.add_argument(
        "--ymax_pp",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--ymin_prob",
        type=float,
        default=-0.2,
    )
    parser.add_argument(
        "--ymax_prob",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--ylim_pad_pp",
        type=float,
        default=2.0,
        help="Extend beyond [ymin_pp,ymax_pp] by this many pp (default 2).",
    )
    parser.add_argument(
        "--ylim_pad_prob",
        type=float,
        default=0.02,
        help="Extend beyond [ymin_prob,ymax_prob] by this amount (default 0.02).",
    )
    parser.add_argument(
        "--yticks_pp",
        type=str,
        default="-40,-20,0",
        help="Comma/space-separated tick values when plot_units=pp.",
    )
    parser.add_argument(
        "--yticks_prob",
        type=str,
        default="-0.2,-0.1,0,0.1,0.2",
        help="Comma/space-separated tick values when plot_units=prob.",
    )
    return parser


def _build_label_map(args: argparse.Namespace) -> Dict[str, str]:
    return {
        "Crossword": "Xword",
        "Math": args.label_math.strip() or "Math",
        "Math2": args.label_math2.strip() or "Qwen7B-Math",
        "Carpark": "Rush Hour",
    }


def _compute_per_step(
    data_frame: pd.DataFrame,
    min_per_group: int,
) -> Tuple[Dict[str, pd.DataFrame], List[pd.DataFrame]]:
    per_step: Dict[str, pd.DataFrame] = {}
    rows_all: List[pd.DataFrame] = []

    for domain_name in ["Crossword", "Math", "Math2", "Carpark"]:
        if domain_name not in data_frame["domain"].unique():
            continue
        domain_df = per_step_raw_effect(
            data_frame,
            domain_name,
            min_per_group=min_per_group,
        )
        per_step[domain_name] = domain_df
        if not domain_df.empty:
            rows_all.append(domain_df)

    return per_step, rows_all


def _determine_plot_config(args: argparse.Namespace) -> PlotUnitsConfig:
    """
    Build the vertical axis configuration based on CLI options.
    """
    if args.plot_units == "pp":
        y_scale = 100.0
        base_lim = (args.ymin_pp, args.ymax_pp)
        pad = args.ylim_pad_pp
        yticks = _parse_float_list(args.yticks_pp) or [
            -20.0,
            -10.0,
            0.0,
            10.0,
            20.0,
        ]
        ylabel = "Raw effect of training \nstep on accuracy (pp)"
    else:
        y_scale = 1.0
        base_lim = (args.ymin_prob, args.ymax_prob)
        pad = args.ylim_pad_prob
        yticks = _parse_float_list(args.yticks_prob) or [
            -0.2,
            -0.1,
            0.0,
            0.1,
            0.2,
        ]
        ylabel = "Raw effect on accuracy"

    ylim = (base_lim[0] - pad, base_lim[1] + pad)
    return PlotUnitsConfig(
        y_scale=y_scale,
        ylim=ylim,
        yticks=yticks,
        ylabel=ylabel,
    )


def main() -> None:
    """
    Command-line entry point for the raw-effect per-step analysis.
    """
    args = _build_arg_parser().parse_args()

    label_map = _build_label_map(args)
    files_by_domain, first_root = build_files_by_domain_for_args(args)

    out_dir = args.out_dir or os.path.join(first_root, "raw_effect_plots")
    os.makedirs(out_dir, exist_ok=True)

    load_config = LoadRowsConfig(
        gpt_keys=gpt_keys_for_mode(args.gpt_mode),
        gpt_subset_native=not args.no_gpt_subset_native,
        min_step=args.min_step,
        max_step=compute_effective_max_step(args, hard_max_step=1000),
        carpark_success_fn=make_carpark_success_fn(
            args.carpark_success_op,
            args.carpark_soft_threshold,
        ),
    )

    rows_df = load_rows(files_by_domain, load_config)
    if rows_df.empty:
        raise SystemExit("No rows after filtering.")

    per_step, rows_all = _compute_per_step(rows_df, args.min_per_group)
    if not rows_all:
        raise SystemExit("No per-step groups met the minimum requirements.")

    table = pd.concat(rows_all, axis=0, ignore_index=True)
    table["domain"] = table["domain"].map(
        lambda domain: label_map.get(domain, domain),
    )
    out_csv = os.path.join(
        out_dir,
        "raw_effect_per_step__" + f"{args.dataset_name}__{args.model_name}".replace(" ", "_") + ".csv",
    )
    table.to_csv(out_csv, index=False)

    plot_units = _determine_plot_config(args)

    # 3-panel figure
    plot_panels(
        per_step,
        label_map,
        os.path.join(out_dir, "raw_effect_per_step_panels_linear.png"),
        PanelFigureConfig(
            dpi=args.dpi,
            width_in=args.width_in,
            height_scale=args.height_scale,
            marker_size=args.marker_size,
        ),
        plot_units,
    )

    # Overlay figure (NOT square unless you set matching width/height)
    overlay_path = os.path.join(
        out_dir,
        "raw_effect_per_step_overlay_linear.png",
    )
    if args.overlay_title is None:
        overlay_title = f"Raw Effect of LLM-Detected Shifts by Training Step ({args.model_name})"
    else:
        overlay_title = args.overlay_title
    plot_overlay_all(
        per_step,
        label_map,
        overlay_path,
        OverlayFigureConfig(
            dpi=args.dpi,
            width_in=args.overlay_width_in,
            height_scale=args.overlay_height_scale,
            marker_size=args.marker_size,
            title=overlay_title,
        ),
        plot_units,
    )

    # Back-compat filename
    plot_overlay_all(
        per_step,
        label_map,
        os.path.join(
            out_dir,
            "raw_effect_per_step_crossword_math_linear.png",
        ),
        OverlayFigureConfig(
            dpi=args.dpi,
            width_in=args.overlay_width_in,
            height_scale=args.overlay_height_scale,
            marker_size=args.marker_size,
            title=overlay_title,
        ),
        plot_units,
    )

    with pd.option_context("display.max_columns", None, "display.width", 120):
        print("\nRaw effect per step (head):")
        print(table.head(12).to_string(index=False))
        print(f"\nSaved CSV -> {out_csv}")
        print(
            f"Saved figs -> {overlay_path} (+ PDF), "
            f"{out_dir}/raw_effect_per_step_panels_linear.[PNG|PDF], "
            f"{out_dir}/raw_effect_per_step_crossword_math_linear.[PNG|PDF]",
        )
        if args.plot_units == "pp":
            print(
                "[info] Plots use percentage points (pp). "
                f"Ticks: {plot_units.yticks}. Axis limits with pad: {plot_units.ylim}. "
                "CSV remains in probabilities (0..1).",
            )


if __name__ == "__main__":
    main()
