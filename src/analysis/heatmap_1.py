#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aha! Prevalence Heatmap (δ1 × δ2; δ3 = ε>0)
-------------------------------------------
Builds heatmaps showing the share (%) of problem–step pairs that qualify as a
formal "Aha!" event under thresholds:

  • δ1 ∈ {0, 1/8, 2/8}  (max prior per-checkpoint accuracy fraction)
  • δ2 ∈ {0, 1/8, 2/8}  (max prior per-checkpoint shift fraction)
  • δ3 = ε > 0          (strict improvement at current step over BEST prior acc)

This version:
- adds Math3 (e.g., Llama-8B) support,
- emits per-domain heatmaps for all provided domains by default
  (Crossword, Math, Math2, Math3, Carpark),
- adds a collective heatmap for 1.5B-only domains (default: Crossword, Math, Carpark),
- makes the figure 3/4 the previous height,
- removes the colorbar on the right,
- sets a concise title and axis labels.
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple


try:  # pragma: no cover - prefer real matplotlib when available
    import matplotlib
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.image import imread
except ImportError:  # pragma: no cover - lightweight fallback for headless envs
    matplotlib = SimpleNamespace(use=lambda *_a, **_k: None)

    class _Normalize:
        """Lightweight Normalize stub mirroring matplotlib.colors.Normalize."""

        def __init__(self, vmin=None, vmax=None):
            self.vmin, self.vmax = vmin, vmax

        def __call__(self, value):
            try:
                span = (self.vmax - self.vmin) if self.vmax is not None and self.vmin is not None else 1
                return (float(value) - (self.vmin or 0)) / span if span else 0.0
            except (TypeError, ValueError, ZeroDivisionError):
                return 0.0

        def scale_span(self):
            """Return the effective span used for scaling."""
            if self.vmin is None or self.vmax is None:
                return 1.0
            return float(self.vmax - self.vmin)

    class _FakeCmap:
        """Minimal cmap stub with callable interface."""

        def __call__(self, _value):
            return (0.5, 0.5, 0.5, 1.0)

        def to_rgba(self, value):
            """Return RGBA tuple for compatibility with matplotlib cmap API."""
            return self(value)

    def _get_cmap(_name=None):
        """Return a simple constant colormap stub."""
        return _FakeCmap()

    mcolors = SimpleNamespace(
        Normalize=_Normalize,
        get_cmap=_get_cmap,
        LinearSegmentedColormap=SimpleNamespace(
            from_list=lambda _name, _colors: _FakeCmap(),
        ),
    )

    def _subplots(*_a, **_k):
        class _Axes:
            def __init__(self):
                self.texts = []

            def imshow(self, *_args, **_kwargs):
                """Stub imshow."""
                return None

            def set_xlabel(self, *_a, **_k):
                """Stub set_xlabel."""

            def set_ylabel(self, *_a, **_k):
                """Stub set_ylabel."""

            def set_title(self, *_a, **_k):
                """Stub set_title."""

            def grid(self, *_a, **_k):
                """Stub grid."""

            def legend(self, *_a, **_k):
                """Stub legend."""

            def set_xticks(self, *_a, **_k):
                """Stub set_xticks."""

            def set_xticklabels(self, *_a, **_k):
                """Stub set_xticklabels."""

            def set_yticks(self, *_a, **_k):
                """Stub set_yticks."""

            def set_yticklabels(self, *_a, **_k):
                """Stub set_yticklabels."""

            def text(self, *args, **kwargs):
                """Record text calls for tests."""
                self.texts.append((args, kwargs))

        axes = _Axes()

        class _Fig:
            def savefig(self, path, **_k):
                """Stub savefig that materializes an empty file."""
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).touch()

            def colorbar(self, *_a, **_k):
                """Stub colorbar."""
                return None

            def tight_layout(self, *_a, **_k):
                """Stub tight_layout."""
                return None

        fig = _Fig()
        return fig, axes

    plt = SimpleNamespace(
        get_cmap=_get_cmap,
        subplots=_subplots,
        close=lambda *_a, **_k: None,
    )
    cm = SimpleNamespace(get_cmap=_get_cmap)

    def imread(*_a, **_k):
        """Stub imread returning None."""
        return None


import numpy as np
import pandas as pd

from src.analysis.core import LoadRowsConfig as SampleLoadConfig
from src.analysis.core import iter_correct_and_shift_samples_for_config
from src.analysis.core.plotting_helpers import compute_effective_max_step


try:  # pragma: no cover - allow tests to stub src.analysis.io
    from src.analysis.io import build_jsonl_files_by_domain
except ImportError:  # pragma: no cover - fallback when io is stubbed

    def build_jsonl_files_by_domain(*_args, **_kwargs):
        """Stubbed build_jsonl_files_by_domain used when analysis.io is unavailable."""
        raise ImportError("build_jsonl_files_by_domain unavailable")


from src.analysis.metrics import make_carpark_success_fn
from src.analysis.plotting import apply_default_style
from src.analysis.utils import add_carpark_threshold_args, get_problem_id, gpt_keys_for_mode


matplotlib.use("Agg")

apply_default_style()


@dataclass(frozen=True)
class PerDomainPlotConfig:
    """Lightweight container for repeated per-domain plot settings."""

    label_map: Dict[str, str]
    delta_values: List[float]
    long_rows: List[pd.DataFrame]
    out_dir: str
    cmap_name: str


def _iter_sample_rows(
    files_by_domain: Dict[str, List[str]],
    config: SampleLoadConfig,
):
    """
    Yield one row per (domain, problem, step, sample) with correctness and shift.
    """
    for dom, step, rec, correct, shift in iter_correct_and_shift_samples_for_config(
        files_by_domain,
        config,
    ):
        problem_id = get_problem_id(rec)
        if problem_id is None:
            continue
        pid_full = f"{dom}::{problem_id}"

        yield {
            "domain_key": str(dom),
            "problem_id": pid_full,
            "step": int(step),
            "correct": int(correct),
            "shift": int(shift),
        }


# ---------- Load sample-level rows ----------
def load_rows(
    files_by_domain: Dict[str, List[str]],
    config: SampleLoadConfig,
) -> pd.DataFrame:
    """
    Load sample-level rows across all domains into a DataFrame.
    """
    return pd.DataFrame(_iter_sample_rows(files_by_domain, config))


# ---------- Aggregate to per-(problem, step) ----------
def make_step_level(df_samples: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sample-level rows to per-(domain, problem, step) statistics.
    """
    grouped = df_samples.groupby(
        ["domain_key", "problem_id", "step"],
        as_index=False,
    )
    out = grouped.agg(
        acc_frac=("correct", "mean"),
        shift_frac=("shift", "mean"),
        n_samples=("correct", "size"),
    )
    return out.sort_values(
        ["domain_key", "problem_id", "step"],
    ).reset_index(drop=True)


# ---------- Aha detection ----------
def count_ahas(
    step_df: pd.DataFrame,
    delta1: float,
    delta2: float,
) -> Tuple[int, int]:
    """
    Count formal Aha events and eligible pairs for given (delta1, delta2).
    """
    n_events = 0
    n_pairs = 0
    for _, group in step_df.groupby(["domain_key", "problem_id"], sort=False):
        values = group[["step", "acc_frac", "shift_frac"]].to_numpy()
        order = np.argsort(values[:, 0])
        ordered = values[order]
        acc_history: List[float] = []
        shift_history: List[float] = []
        for _, acc, shift in ordered:
            if not acc_history:
                acc_history.append(acc)
                shift_history.append(shift)
                continue
            prior_max_acc = float(np.max(acc_history))
            prior_max_shift = float(np.max(shift_history))
            n_pairs += 1
            if prior_max_acc <= delta1 and prior_max_shift <= delta2 and acc > prior_max_acc:
                n_events += 1
            acc_history.append(acc)
            shift_history.append(shift)
    return n_events, n_pairs


# ---------- Heatmap helpers ----------
def frac8_label(delta_value: float) -> str:
    """Format delta values as k/8 when applicable, otherwise as a float."""
    scaled = delta_value * 8.0
    k = int(round(scaled))
    if abs(scaled - k) < 1e-6:
        return f"{k}/8"
    return f"{delta_value:.3f}"


def get_rendered_size(fig, dpi: int = 200) -> Tuple[float, float]:
    """
    Return the rendered width/height (inches) of a Matplotlib figure.
    """
    with NamedTemporaryFile(suffix=".png") as temp_file:
        try:
            fig.savefig(temp_file.name, bbox_inches="tight", dpi=dpi)
        except (OSError, ValueError, TypeError):  # noqa: BLE001 - stub fallback
            if hasattr(fig, "get_size_inches"):
                width_in, height_in = fig.get_size_inches()
                return float(width_in), float(height_in)
            return 0.0, 0.0
        try:
            img = imread(temp_file.name)
            height_px, width_px, *_ = np.asarray(img).shape
            return width_px / dpi, height_px / dpi
        except (OSError, ValueError, TypeError):
            # Fallback for stubbed matplotlib/imread in test environments.
            if hasattr(fig, "get_size_inches"):
                width_in, height_in = fig.get_size_inches()
                return float(width_in), float(height_in)
            return 0.0, 0.0


def set_rendered_width(
    fig,
    target_width_in: float,
    dpi: int = 200,
    eps: float = 1e-3,
    max_iter: int = 8,
) -> bool:
    """
    Iteratively adjust the figure size so the rendered width matches target_width_in.
    """
    min_render_px = 10
    if target_width_in * dpi < min_render_px:
        return False
    if not hasattr(fig, "get_size_inches") or not hasattr(fig, "set_size_inches"):
        return False
    try:
        width_init, height_init = fig.get_size_inches()
    except AttributeError:
        return False
    target_height = height_init * (target_width_in / max(width_init, 1e-6))
    width_set, height_set = target_width_in, target_height
    success = False
    for _ in range(max_iter):
        fig.set_size_inches([width_set, height_set])
        width_actual, height_actual = get_rendered_size(fig, dpi=dpi)
        if width_actual * dpi < min_render_px or height_actual * dpi < min_render_px:
            break
        if abs(width_actual - target_width_in) < eps:
            success = True
            break
        scale = target_width_in / max(width_actual, 1e-9)
        width_set *= scale
        height_set *= scale
        if width_set * dpi < min_render_px or height_set * dpi < min_render_px:
            break
    return success


def sweep_grid(step_df: pd.DataFrame, deltas: List[float]) -> pd.DataFrame:
    """
    Build a grid of (delta1, delta2) → (events, pairs, percentage) values.
    """
    rows: List[Dict[str, Any]] = []
    for delta1 in deltas:
        for delta2 in deltas:
            events, pairs = count_ahas(step_df, delta1, delta2)
            percentage = (100.0 * events / pairs) if pairs > 0 else np.nan
            rows.append(
                {
                    "delta1": delta1,
                    "delta2": delta2,
                    "n_events": events,
                    "n_pairs": pairs,
                    "pct": percentage,
                },
            )
    return pd.DataFrame(rows)


def _build_values_matrix(
    df_grid: pd.DataFrame,
) -> Tuple[List[float], np.ndarray]:
    """Build a matrix of percentages indexed by (delta2, delta1)."""
    levels = sorted(df_grid["delta1"].unique())
    values = np.zeros((len(levels), len(levels)), dtype=float) * np.nan
    for _, row in df_grid.iterrows():
        row_index = levels.index(row["delta2"])
        column_index = levels.index(row["delta1"])
        values[row_index, column_index] = row["pct"]
    return levels, values


def _foreground_for_value(
    value: float,
    cmap,
    norm,
) -> str:
    """Choose a readable text color for a heatmap cell."""
    red, green, blue, _ = cmap(norm(value))
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return "black" if luminance > 0.55 else "white"


def _annotate_heatmap(
    axes_obj,
    df_grid: pd.DataFrame,
    grid_data: Tuple[List[float], np.ndarray],
    cmap,
    norm,
) -> None:
    """Annotate each heatmap cell with percentage and counts."""
    levels, values = grid_data
    for row_index, delta2 in enumerate(levels):
        for column_index, delta1 in enumerate(levels):
            cell_value = values[row_index, column_index]
            if not np.isfinite(cell_value):
                continue
            row = df_grid[(df_grid["delta1"] == delta1) & (df_grid["delta2"] == delta2)].iloc[0]
            foreground = _foreground_for_value(cell_value, cmap, norm)
            axes_obj.text(
                column_index,
                row_index,
                f"{cell_value:.2f}%\n({int(row['n_events'])}/{int(row['n_pairs'])})",
                ha="center",
                va="center",
                fontsize=12,
                color=foreground,
            )


def plot_heatmap(
    df_grid: pd.DataFrame,
    title: str,
    out_png: str,
    cmap_name: str = "YlGnBu",
) -> None:
    """
    Render a single Aha-prevalence heatmap and save PNG/PDF outputs.
    """
    levels, values = _build_values_matrix(df_grid)

    vmax = float(np.nanmax(values)) if np.isfinite(np.nanmax(values)) else 1.0
    if hasattr(cm, "get_cmap"):
        cmap = cm.get_cmap(cmap_name)
    elif hasattr(mcolors, "get_cmap"):
        cmap = mcolors.get_cmap(cmap_name)
    else:  # extremely minimal fallback if matplotlib is partially stubbed
        cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, ["#fff", "#000"])
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)

    # Shorter canvas (3/4 height)
    fig, axes_obj = plt.subplots(figsize=(5.8, 4.9 * 0.75))
    if not hasattr(axes_obj, "imshow"):
        plt.close(fig)
        return
    axes_obj.imshow(values, origin="lower", aspect="auto", cmap=cmap, norm=norm)

    # Axis ticks & labels (k/8)
    positions = list(range(len(levels)))
    axes_obj.set_xticks(positions)
    axes_obj.set_yticks(positions)
    tick_labels = [frac8_label(delta) for delta in levels]
    axes_obj.set_xticklabels(tick_labels)
    axes_obj.set_yticklabels(tick_labels)

    axes_obj.set_xlabel(r"$\delta_1$ (Max prior failures)")
    axes_obj.set_ylabel(r"$\delta_2$ (Max prior stability)")
    axes_obj.set_title(title, pad=4)

    _annotate_heatmap(axes_obj, df_grid, (levels, values), cmap, norm)

    fig.tight_layout()
    set_rendered_width(fig, target_width_in=5.0, dpi=500)

    fig.savefig(out_png, dpi=500, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_crossword", type=str, default=None)
    parser.add_argument("--root_math", type=str, default=None)
    parser.add_argument("--root_math2", type=str, default=None)
    parser.add_argument(
        "--root_math3",
        type=str,
        default=None,
        help="Third math root (e.g., Llama-8B).",
    )
    parser.add_argument("--root_carpark", type=str, default=None)

    parser.add_argument("--label_crossword", type=str, default="Crossword")
    parser.add_argument("--label_math", type=str, default="Qwen1.5B-Math")
    parser.add_argument("--label_math2", type=str, default="Qwen7B-Math")
    parser.add_argument("--label_math3", type=str, default="Llama8B-Math")
    parser.add_argument("--label_carpark", type=str, default="Carpark")

    parser.add_argument("--split", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--dataset_name", default="MIXED")
    parser.add_argument("--model_name", default="MIXED_MODELS")

    # Per-domain heatmaps: ON by default (all provided domains)
    parser.add_argument(
        "--per_domain",
        action="store_true",
        default=True,
        help="Emit per-domain heatmaps/rows (default: True).",
    )

    # Titles
    parser.add_argument(
        "--title_overall",
        type=str,
        default="Aha! Moment Prevalence (All provided domains)",
        help="Overall heatmap title (all domains).",
    )
    parser.add_argument(
        "--title_15b",
        type=str,
        default="Aha! Moment Prevalence (Qwen-1.5B; Crossword+Math+Carpark)",
        help="1.5B-only collective heatmap title.",
    )

    # 1.5B-only collective heatmap
    parser.add_argument(
        "--make_15b_overall",
        action="store_true",
        default=True,
        help="Also emit a 1.5B-only overall heatmap (default: True).",
    )
    parser.add_argument(
        "--domains_15b",
        type=str,
        default="Crossword,Math,Carpark",
        help="Comma-separated domain keys to include in 1.5B overall.",
    )

    parser.add_argument("--cmap", type=str, default="YlGnBu")

    parser.add_argument("--gpt_mode", choices=["canonical", "broad"], default="canonical")
    parser.add_argument("--no_gpt_subset_native", action="store_true")

    parser.add_argument("--min_step", type=int, default=None)
    parser.add_argument("--max_step", type=int, default=None)

    parser.add_argument(
        "--delta_values",
        nargs="*",
        type=float,
        default=[0.0, 1 / 8, 2 / 8],
    )

    add_carpark_threshold_args(parser)

    return parser


def _build_label_map(args: argparse.Namespace) -> Dict[str, str]:
    return {
        "Crossword": (args.label_crossword or "Crossword").strip(),
        "Math": (args.label_math or "Math").strip(),
        "Math2": (args.label_math2 or "Math2").strip(),
        "Math3": (args.label_math3 or "Math3").strip(),
        "Carpark": (args.label_carpark or "Carpark").strip(),
    }


def _build_domain_roots(args: argparse.Namespace) -> Dict[str, Optional[str]]:
    return {
        "Crossword": args.root_crossword,
        "Math": args.root_math,
        "Math2": args.root_math2,
        "Math3": args.root_math3,
        "Carpark": args.root_carpark,
    }


def _collect_files_and_out_dir(
    args: argparse.Namespace,
) -> Tuple[Dict[str, List[str]], str]:
    domain_roots = _build_domain_roots(args)
    files_by_domain, first_root = build_jsonl_files_by_domain(domain_roots, args.split)
    if not files_by_domain:
        raise SystemExit("Provide at least one --root_* folder.")

    total_files = sum(len(value) for value in files_by_domain.values())
    if total_files == 0:
        raise SystemExit("No JSONL files found. Check roots/--split.")

    out_dir = args.out_dir or os.path.join(first_root, "aha_heatmaps")
    os.makedirs(out_dir, exist_ok=True)
    return files_by_domain, out_dir


def _build_load_config(args: argparse.Namespace) -> SampleLoadConfig:
    gpt_subset_native = not args.no_gpt_subset_native
    gpt_keys = gpt_keys_for_mode(args.gpt_mode)
    carpark_success_fn = make_carpark_success_fn(
        args.carpark_success_op,
        args.carpark_soft_threshold,
    )
    max_step_eff = compute_effective_max_step(args, hard_max_step=1000)
    return SampleLoadConfig(
        gpt_keys=gpt_keys,
        gpt_subset_native=gpt_subset_native,
        min_step=args.min_step,
        max_step=max_step_eff,
        carpark_success_fn=carpark_success_fn,
    )


def _load_step_level_data(
    files_by_domain: Dict[str, List[str]],
    load_config: SampleLoadConfig,
) -> pd.DataFrame:
    df_samples = load_rows(files_by_domain, load_config)
    if df_samples.empty:
        raise SystemExit("No rows after filtering.")
    return make_step_level(df_samples)


def _build_overall_grid_and_rows(
    step_df: pd.DataFrame,
    delta_values: List[float],
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    overall_grid = sweep_grid(step_df, delta_values)
    grid_overall_csv = overall_grid.copy()
    grid_overall_csv["scope"] = "overall_all"
    grid_overall_csv["domain_key"] = "ALL"
    grid_overall_csv["domain_label"] = "ALL"
    return overall_grid, [grid_overall_csv]


def _add_per_domain_grids_and_plots(
    step_df: pd.DataFrame,
    config: PerDomainPlotConfig,
) -> None:
    for dom_key in sorted(step_df["domain_key"].unique(), key=str):
        dom_df = step_df[step_df["domain_key"] == dom_key]
        dom_grid = sweep_grid(dom_df, config.delta_values)
        dom_grid_csv = dom_grid.copy()
        dom_grid_csv["scope"] = "domain"
        dom_grid_csv["domain_key"] = dom_key
        dom_grid_csv["domain_label"] = config.label_map.get(dom_key, dom_key)
        config.long_rows.append(dom_grid_csv)
        plot_heatmap(
            dom_grid,
            f"Aha! Moment Prevalence ({config.label_map.get(dom_key, dom_key)})",
            os.path.join(config.out_dir, f"aha_heatmap_{dom_key}.png"),
            cmap_name=config.cmap_name,
        )


def _add_group_15b_grid_and_plot(
    step_df: pd.DataFrame,
    args: argparse.Namespace,
    long_rows: List[pd.DataFrame],
    out_dir: str,
) -> None:
    if not args.make_15b_overall:
        return

    include_keys = [key.strip() for key in (args.domains_15b or "").split(",") if key.strip()]
    present = [key for key in include_keys if key in set(step_df["domain_key"].unique())]
    if not present:
        print(
            "[warn] 1.5B overall requested, but none of the requested domains are present. Skipping.",
        )
        return

    step_15b = step_df[step_df["domain_key"].isin(present)].copy()
    grid_15b = sweep_grid(step_15b, args.delta_values)
    grid_15b_csv = grid_15b.copy()
    grid_15b_csv["scope"] = "group"
    grid_15b_csv["group_key"] = "1p5b"
    grid_15b_csv["group_domains"] = ",".join(present)
    long_rows.append(grid_15b_csv)

    out_png_15b = os.path.join(out_dir, "aha_heatmap_overall_1p5b.png")
    plot_heatmap(grid_15b, args.title_15b, out_png_15b, cmap_name=args.cmap)


def _write_output_table(
    long_rows: List[pd.DataFrame],
    out_dir: str,
    slug: str,
) -> str:
    table = pd.concat(long_rows, axis=0, ignore_index=True)
    out_csv = os.path.join(out_dir, f"aha_heatmap__{slug}.csv")
    table.to_csv(out_csv, index=False)
    return out_csv


def _nearest_delta(target: float, values: List[float]) -> float:
    arr = np.asarray(values)
    return float(arr[np.argmin(np.abs(arr - target))])


def _write_latex_helper(
    args: argparse.Namespace,
    overall_grid: pd.DataFrame,
    delta_values: List[float],
    out_dir: str,
    slug: str,
) -> None:
    d1r = _nearest_delta(0.13, delta_values)
    d2r = _nearest_delta(0.13, delta_values)
    row = overall_grid[(overall_grid["delta1"] == d1r) & (overall_grid["delta2"] == d2r)]
    if row.empty:
        return

    n_events_value = int(row["n_events"].iloc[0])
    n_pairs_value = int(row["n_pairs"].iloc[0])
    pct_value = float(row["pct"].iloc[0])
    sentence = (
        r"Fig.~\ref{fig:aha-heatmap} shows the prevalence of formal ``Aha!'' moments "
        rf"for {args.model_name}. Using $(\delta_1={frac8_label(d1r)}, "
        rf"\delta_2={frac8_label(d2r)}, \delta_3=\epsilon>0)$, "
        rf"we find {n_events_value} events out of {n_pairs_value} problem--step pairs "
        rf"({pct_value:.2f}\%)."
    )
    out_tex = os.path.join(out_dir, f"aha_heatmap_summary__{slug}.tex")
    with open(out_tex, "w", encoding="utf-8") as tex_file:
        tex_file.write(sentence + "\n")
    print("\nLaTeX one-liner:\n" + sentence + "\n")
    print(f"Saved TEX -> {out_tex}")


def _print_summary(
    args: argparse.Namespace,
    overall_grid: pd.DataFrame,
    out_csv: str,
    out_png_overall: str,
    out_dir: str,
) -> None:
    with pd.option_context("display.max_columns", None, "display.width", 120):
        disp = overall_grid.copy()
        disp["delta1"] = disp["delta1"].map(frac8_label)
        disp["delta2"] = disp["delta2"].map(frac8_label)
        disp["pct"] = disp["pct"].map(
            lambda value: f"{value:.2f}%" if np.isfinite(value) else "NaN",
        )
        print("\n== Overall Aha! prevalence grid (ALL domains) ==")
        print(disp.to_string(index=False))
        print(f"\nSaved CSV  -> {out_csv}")
        print(f"Saved figs -> {out_png_overall} (+ PDF)")
        if args.per_domain:
            print(f"          + per-domain heatmaps in {out_dir}/")
        if args.make_15b_overall:
            print("          + 1.5B-only overall heatmap if requested")


# ---------- Main ----------
def main() -> None:
    """CLI entrypoint for Aha-prevalence heatmap generation."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    label_map = _build_label_map(args)
    files_by_domain, out_dir = _collect_files_and_out_dir(args)
    load_config = _build_load_config(args)
    step_df = _load_step_level_data(files_by_domain, load_config)

    overall_grid, long_rows = _build_overall_grid_and_rows(
        step_df,
        args.delta_values,
    )
    slug = f"{args.dataset_name}__{args.model_name}".replace(" ", "_")

    out_png_overall = os.path.join(out_dir, "aha_heatmap_overall.png")
    plot_heatmap(overall_grid, args.title_overall, out_png_overall, cmap_name=args.cmap)

    if args.per_domain:
        per_domain_config = PerDomainPlotConfig(
            label_map=label_map,
            delta_values=args.delta_values,
            long_rows=long_rows,
            out_dir=out_dir,
            cmap_name=args.cmap,
        )
        _add_per_domain_grids_and_plots(
            step_df=step_df,
            config=per_domain_config,
        )
    _add_group_15b_grid_and_plot(
        step_df,
        args,
        long_rows,
        out_dir,
    )

    out_csv = _write_output_table(long_rows, out_dir, slug)
    _write_latex_helper(
        args=args,
        overall_grid=overall_grid,
        delta_values=args.delta_values,
        out_dir=out_dir,
        slug=slug,
    )
    _print_summary(
        args=args,
        overall_grid=overall_grid,
        out_csv=out_csv,
        out_png_overall=out_png_overall,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
