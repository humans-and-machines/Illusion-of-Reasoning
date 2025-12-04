#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""CLI wiring for the Figure 1 analysis pipeline."""

from __future__ import annotations

import argparse
import os
import types
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .figure_1_data import (
    bootstrap_problem_ratio,
    build_positive_delta_flags,
    build_problem_step,
    load_pass1_samples_multi,
    mark_formal_pairs,
    parse_float_list,
    scan_files,
    slugify,
)
from .figure_1_export import (
    ExportDestinations,
    FormalExportConfig,
    FormalThresholds,
    GptFilterConfig,
    export_formal_aha_json_with_text,
)
from .figure_1_plotting import FormalSweepPlotConfig, plot_formal_sweep_grid, plot_three_ratios_shared_axes_multi
from .figure_1_style import lighten_hex, set_global_fonts


# Matplotlib stubs used in tests may lack ``cycler``; backfill a tiny version.
if not hasattr(plt, "cycler"):
    try:  # pragma: no cover - best-effort compat
        import matplotlib as _mpl  # type: ignore

        plt.cycler = getattr(_mpl, "cycler")  # type: ignore[assignment]
    except (ImportError, AttributeError, TypeError):

        def _fallback_cycler(**kwargs):
            """Minimal cycler stub returning a by_key() mapping."""
            colors = list(kwargs.get("color", []) or [])
            return types.SimpleNamespace(by_key=lambda: {"color": colors})

        plt.cycler = _fallback_cycler  # type: ignore[assignment]
    if isinstance(getattr(plt, "rcParams", None), dict) and "axes.prop_cycle" not in plt.rcParams:
        plt.rcParams["axes.prop_cycle"] = plt.cycler(color=[])


@dataclass(frozen=True)
class Figure1Context:
    """Runtime configuration/state shared across Figure 1 helpers."""

    args: argparse.Namespace
    files_by_domain: Dict[str, List[str]]
    out_dir: str
    slug: str
    domain_colors: Dict[str, str]
    gpt_keys: List[str]
    gpt_subset_native: bool


@dataclass(frozen=True)
class DomainBootstrapResult:
    """Container for per-domain bootstrap data and highlight metadata."""

    native: Dict[str, pd.DataFrame]
    gpt: Dict[str, pd.DataFrame]
    formal: Dict[str, pd.DataFrame]
    highlights: Dict[str, Any]


@dataclass(frozen=True)
class DeltaGrid:
    """Parsed δ1/δ2 sweep inputs for the Formal panel."""

    delta1: List[float]
    delta2: List[float]


@dataclass(frozen=True)
class SummaryArtifacts:
    """Pointers to saved artifacts produced by the Figure 1 pipeline."""

    main_outputs: Tuple[str, str]
    ratios_csv: str
    trend_csv: str
    sweep_outputs: Tuple[str, str]
    sweep_csv: str
    denom_note: str

    @property
    def main_png(self) -> str:
        """Return the PNG path for the main figure."""
        return self.main_outputs[0]

    @property
    def main_pdf(self) -> str:
        """Return the PDF path for the main figure."""
        return self.main_outputs[1]

    @property
    def sweep_png(self) -> str:
        """Return the PNG path for the sweep plot."""
        return self.sweep_outputs[0]

    @property
    def sweep_pdf(self) -> str:
        """Return the PDF path for the sweep plot."""
        return self.sweep_outputs[1]


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser for the Figure 1 workflow."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_crossword", type=str, default=None)
    parser.add_argument("--root_math", type=str, default=None)
    parser.add_argument("results_root", nargs="?", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--dataset_name", default="MIXED")
    parser.add_argument("--model_name", default="Qwen2.5-1.5B")

    parser.add_argument("--delta1", type=float, default=0.20)
    parser.add_argument("--delta2", type=float, default=0.20)
    parser.add_argument("--delta3", type=float, default=None)
    parser.add_argument("--min_prior_steps", type=int, default=2)

    parser.add_argument("--delta1_list", type=str, default="0.15,0.20,0.25")
    parser.add_argument("--delta2_list", type=str, default="0.15,0.20,0.25")

    parser.add_argument("--min_step", type=int, default=None)
    parser.add_argument("--max_step", type=int, default=None)
    parser.add_argument("--balanced_panel", action="store_true")

    parser.add_argument("--color_crossword", default="#4C78A8")
    parser.add_argument("--color_math", default="#E45756")

    parser.add_argument("--B", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ymax", type=float, default=0.20)
    parser.add_argument("--ci_alpha", type=float, default=0.20)
    parser.add_argument("--ms", type=float, default=5.0)

    parser.add_argument("--font_family", default="Times New Roman")
    parser.add_argument("--font_size", type=int, default=12)

    parser.add_argument("--gpt_mode", choices=["canonical", "broad"], default="canonical")
    parser.add_argument("--no_gpt_subset_native", action="store_true")

    parser.add_argument("--out_basename", default=None)
    parser.add_argument("--a4_pdf", action="store_true")
    parser.add_argument("--a4_orientation", choices=["landscape", "portrait"], default="landscape")
    parser.add_argument("--panel_box_aspect", type=float, default=0.8)
    return parser


def _collect_files(args: argparse.Namespace) -> Tuple[Dict[str, List[str]], str]:
    """Return files grouped by domain and the first root path used."""
    files_by_domain: Dict[str, List[str]] = {}
    first_root = None
    if args.root_crossword:
        files_by_domain["Crossword"] = scan_files(args.root_crossword, args.split)
        first_root = first_root or args.root_crossword
    if args.root_math:
        files_by_domain["Math"] = scan_files(args.root_math, args.split)
        first_root = first_root or args.root_math
    if not files_by_domain:
        if not args.results_root:
            raise SystemExit("Provide --root_crossword/--root_math or a fallback results_root.")
        files_by_domain["All"] = scan_files(args.results_root, args.split)
        first_root = args.results_root

    total_files = sum(len(paths) for paths in files_by_domain.values())
    if total_files == 0:
        raise SystemExit("No JSONL files found. Check roots/--split.")
    return files_by_domain, first_root or ""


def _configure_domain_colors(
    files_by_domain: Dict[str, List[str]],
    args: argparse.Namespace,
) -> Dict[str, str]:
    """Return a color per domain, adding fallbacks for unseen domains."""
    domain_colors: Dict[str, str] = {}
    if "Crossword" in files_by_domain:
        domain_colors["Crossword"] = args.color_crossword
    if "Math" in files_by_domain:
        domain_colors["Math"] = args.color_math
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for domain in files_by_domain:
        if domain in domain_colors:
            continue
        domain_colors[domain] = color_cycle[len(domain_colors) % len(color_cycle)]
    return domain_colors


def _select_gpt_config(args: argparse.Namespace) -> Tuple[List[str], bool]:
    """Return GPT label keys and whether to subset to native cues."""
    gpt_subset_native = not args.no_gpt_subset_native
    if args.gpt_mode == "canonical":
        keys = ["change_way_of_thinking", "shift_in_reasoning_v1"]
    else:
        keys = [
            "change_way_of_thinking",
            "shift_in_reasoning_v1",
            "shift_llm",
            "shift_gpt",
            "pivot_llm",
            "rechecked",
        ]
    return keys, gpt_subset_native


def _build_context(args: argparse.Namespace) -> Figure1Context:
    """Assemble the reusable Figure 1 context (paths, colors, GPT config)."""
    files_by_domain, first_root = _collect_files(args)
    out_dir = args.out_dir or os.path.join(first_root, "aha_ratios_bootstrap")
    os.makedirs(out_dir, exist_ok=True)
    slug = f"{slugify(args.dataset_name)}__{slugify(args.model_name)}"
    domain_colors = _configure_domain_colors(files_by_domain, args)
    gpt_keys, gpt_subset_native = _select_gpt_config(args)
    return Figure1Context(
        args=args,
        files_by_domain=files_by_domain,
        out_dir=out_dir,
        slug=slug,
        domain_colors=domain_colors,
        gpt_keys=gpt_keys,
        gpt_subset_native=gpt_subset_native,
    )


def _load_samples(
    files_by_domain: Dict[str, List[str]],
    gpt_keys: List[str],
    gpt_subset_native: bool,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Load, filter, and optionally balance the pass-1 samples."""
    samples_df = load_pass1_samples_multi(
        files_by_domain,
        gpt_keys=gpt_keys,
        gpt_subset_native=gpt_subset_native,
    )
    if args.min_step is not None:
        samples_df = samples_df[samples_df["step"] >= args.min_step]
    if args.max_step is not None:
        samples_df = samples_df[samples_df["step"] <= args.max_step]
    if samples_df.empty:
        raise SystemExit("No rows left after step filtering.")
    if args.balanced_panel:
        balanced_parts: List[pd.DataFrame] = []
        for _, sub in samples_df.groupby("domain", sort=False):
            step_values = np.sort(sub["step"].unique())
            have = sub.groupby("problem")["step"].nunique() == len(step_values)
            keep_probs = set(have[have].index.tolist())
            balanced_parts.append(sub[sub["problem"].isin(keep_probs)])
        samples_df = pd.concat(balanced_parts, ignore_index=True) if balanced_parts else pd.DataFrame()
        if samples_df.empty:
            raise SystemExit("Balanced panel filter removed all rows; relax filters.")
    return samples_df


def _prepare_formal_sweep_config(
    args: argparse.Namespace,
    out_dir: str,
    slug: str,
) -> FormalSweepPlotConfig:
    """Return the configuration for the δ1/δ2 sweep plot."""
    return FormalSweepPlotConfig(
        min_prior_steps=int(args.min_prior_steps),
        n_bootstrap=int(args.B),
        seed=int(args.seed),
        out_png=os.path.join(out_dir, f"aha_formal_ratio_sweep__{slug}.png"),
        out_pdf=os.path.join(out_dir, f"aha_formal_ratio_sweep__{slug}.pdf"),
        dataset=args.dataset_name,
        model=args.model_name,
        primary_color="#2F5597",
        ci_color=lighten_hex("#2F5597", 0.65),
        ymax=args.ymax,
        alpha_ci=args.ci_alpha,
        a4_pdf=bool(args.a4_pdf),
        orientation=args.a4_orientation,
        delta3=args.delta3,
    )


def _write_domain_ratios_csv(
    out_dir: str,
    slug: str,
    native_by_dom: Dict[str, pd.DataFrame],
    gpt_by_dom: Dict[str, pd.DataFrame],
    formal_by_dom: Dict[str, pd.DataFrame],
) -> str:
    """Persist the per-domain ratio bootstrap tables."""

    def _tag(domain_df: pd.DataFrame, series_label: str, domain_name: str) -> pd.DataFrame:
        tagged = domain_df.copy()
        tagged["series"] = series_label
        tagged["domain"] = domain_name
        return tagged[["domain", "step", "k", "n", "ratio", "lo", "hi", "series"]]

    frames = []
    for domain in sorted(native_by_dom.keys()):
        frames.append(_tag(native_by_dom[domain], "Words/Cue Phrases", domain))
        frames.append(_tag(gpt_by_dom[domain], "LLM-Detected Shifts", domain))
        frames.append(_tag(formal_by_dom[domain], "Formal Shifts", domain))
    csv_path = os.path.join(out_dir, f"aha_ratios_problems_bootstrap__{slug}.csv")
    (pd.concat(frames, ignore_index=True).sort_values(["series", "domain", "step"]).to_csv(csv_path, index=False))
    return csv_path


def _write_trend_csv(out_dir: str, slug: str, trend_rows: List[Dict[str, Any]]) -> str:
    """Persist the trend summary table."""
    trend_path = os.path.join(out_dir, f"aha_trend_summary__{slug}.csv")
    pd.DataFrame(trend_rows).to_csv(trend_path, index=False)
    return trend_path


def _build_domain_bootstraps(
    ps_main: pd.DataFrame,
    args: argparse.Namespace,
) -> DomainBootstrapResult:
    native_by_dom: Dict[str, pd.DataFrame] = {}
    gpt_by_dom: Dict[str, pd.DataFrame] = {}
    formal_by_dom: Dict[str, pd.DataFrame] = {}
    for domain, subset in ps_main.groupby("domain", sort=False):
        native_by_dom[domain] = bootstrap_problem_ratio(subset, "aha_any_native", args.B, args.seed)
        gpt_by_dom[domain] = bootstrap_problem_ratio(subset, "aha_any_gpt", args.B, args.seed)
        formal_by_dom[domain] = bootstrap_problem_ratio(subset, "aha_formal", args.B, args.seed)
    highlights = build_positive_delta_flags(ps_main)
    return DomainBootstrapResult(
        native=native_by_dom,
        gpt=gpt_by_dom,
        formal=formal_by_dom,
        highlights=highlights,
    )


def _write_sweep_csv(
    out_dir: str,
    slug: str,
    ps_base: pd.DataFrame,
    delta_pairs: Tuple[Tuple[float, float], ...],
    args: argparse.Namespace,
) -> str:
    """Persist the per-step sweep ratios."""
    sweep_rows = []
    for delta1_value, delta2_value in delta_pairs:
        formal_pairs = mark_formal_pairs(
            ps_base.copy(),
            delta1=float(delta1_value),
            delta2=float(delta2_value),
            min_prior_steps=args.min_prior_steps,
            delta3=args.delta3,
        )
        sweep_ratio_df = bootstrap_problem_ratio(
            formal_pairs,
            "aha_formal",
            num_bootstrap=args.B,
            seed=args.seed,
        ).assign(
            delta1=float(delta1_value),
            delta2=float(delta2_value),
            delta3=args.delta3 if args.delta3 is not None else np.nan,
        )
        sweep_rows.append(sweep_ratio_df)
    sweep_csv = os.path.join(out_dir, f"aha_formal_ratio_sweep__{slug}.csv")
    (pd.concat(sweep_rows, ignore_index=True).sort_values(["delta1", "delta2", "step"]).to_csv(sweep_csv, index=False))
    return sweep_csv


def _export_formal_events(
    ps_main: pd.DataFrame,
    files_by_domain: Dict[str, List[str]],
    ctx: Figure1Context,
) -> None:
    export_config = FormalExportConfig(
        dataset=ctx.args.dataset_name,
        model=ctx.args.model_name,
        thresholds=FormalThresholds(
            delta1=ctx.args.delta1,
            delta2=ctx.args.delta2,
            delta3=ctx.args.delta3,
            min_prior_steps=ctx.args.min_prior_steps,
        ),
        gpt_filter=GptFilterConfig(
            keys=ctx.gpt_keys,
            subset_native=ctx.gpt_subset_native,
        ),
        destinations=ExportDestinations(
            out_dir=ctx.out_dir,
            slug=ctx.slug,
        ),
        max_chars=4000,
    )
    json_path, jsonl_path, n_events = export_formal_aha_json_with_text(
        ps_main,
        files_by_domain,
        export_config,
    )
    print(f"Formal Aha events: {n_events} written to:\n  {json_path}\n  {jsonl_path}")


def _parse_delta_grid(args: argparse.Namespace) -> DeltaGrid:
    return DeltaGrid(
        delta1=parse_float_list(args.delta1_list),
        delta2=parse_float_list(args.delta2_list),
    )


def _print_summary(artifacts: SummaryArtifacts, trend_rows: List[Dict[str, Any]]) -> None:
    """Emit a console summary of saved artifacts."""
    print("Wrote:")
    print("  3-up figure:", artifacts.main_png, "and", artifacts.main_pdf)
    print("  Ratios CSV :", artifacts.ratios_csv, artifacts.denom_note)
    print("  Trend CSV  :", artifacts.trend_csv)
    print("  Sweep fig  :", artifacts.sweep_png, "and", artifacts.sweep_pdf)
    print("  Sweep CSV  :", artifacts.sweep_csv)
    for row in trend_rows:
        trend_text = (
            f"[Trend] {row['series']} [{row['domain']}]: "
            f"slope/1k={row['slope_per_1k']:.4f}, "
            f"Δ={row['delta_over_range']:.4f}, "
            f"R^2={row['weighted_R2']:.3f}"
        )
        print(trend_text)


def main() -> None:
    """Entry point for the Figure 1 analysis CLI."""
    ctx = _build_context(build_arg_parser().parse_args())
    set_global_fonts(ctx.args.font_family, ctx.args.font_size)

    ps_base = build_problem_step(
        _load_samples(ctx.files_by_domain, ctx.gpt_keys, ctx.gpt_subset_native, ctx.args),
    )
    ps_main = mark_formal_pairs(
        ps_base.copy(),
        delta1=ctx.args.delta1,
        delta2=ctx.args.delta2,
        min_prior_steps=ctx.args.min_prior_steps,
        delta3=ctx.args.delta3,
    )

    bootstrap_result = _build_domain_bootstraps(ps_main, ctx.args)
    _export_formal_events(ps_main, ctx.files_by_domain, ctx)

    out_base = (
        os.path.join(ctx.out_dir, ctx.args.out_basename)
        if ctx.args.out_basename
        else os.path.join(ctx.out_dir, f"aha_ratios_problems_bootstrap__{ctx.slug}")
    )
    out_png_main = out_base + ".png"
    out_pdf_main = out_base + ".pdf"
    plot_config = {
        "domain_colors": ctx.domain_colors,
        "out_png": out_png_main,
        "out_pdf": out_pdf_main,
        "dataset": ctx.args.dataset_name,
        "model": ctx.args.model_name,
        "alpha_ci": ctx.args.ci_alpha,
        "a4_pdf": bool(ctx.args.a4_pdf),
        "a4_orientation": ctx.args.a4_orientation,
        "panel_box_aspect": ctx.args.panel_box_aspect,
        "marker_size": ctx.args.ms,
        "highlight_formal_by_dom": bootstrap_result.highlights,
        "highlight_color": "#2ca02c",
    }
    trend_rows = plot_three_ratios_shared_axes_multi(
        bootstrap_result.native,
        bootstrap_result.gpt,
        bootstrap_result.formal,
        plot_config,
    )

    delta_grid = _parse_delta_grid(ctx.args)
    sweep_config = _prepare_formal_sweep_config(ctx.args, ctx.out_dir, ctx.slug)
    plot_formal_sweep_grid(ps_base, delta_grid.delta1, delta_grid.delta2, sweep_config)

    summary = SummaryArtifacts(
        main_outputs=(out_png_main, out_pdf_main),
        ratios_csv=_write_domain_ratios_csv(
            ctx.out_dir,
            ctx.slug,
            bootstrap_result.native,
            bootstrap_result.gpt,
            bootstrap_result.formal,
        ),
        trend_csv=_write_trend_csv(ctx.out_dir, ctx.slug, trend_rows),
        sweep_outputs=(sweep_config.out_png, sweep_config.out_pdf),
        sweep_csv=_write_sweep_csv(
            ctx.out_dir,
            ctx.slug,
            ps_base,
            tuple(product(delta_grid.delta1, delta_grid.delta2)),
            ctx.args,
        ),
        denom_note=(
            "(balanced panel per-domain; n constant within domain)"
            if ctx.args.balanced_panel
            else "(unbalanced; n may vary by step and domain)"
        ),
    )
    _print_summary(summary, trend_rows)


if __name__ == "__main__":
    main()
