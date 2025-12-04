#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Math summary plots (PASS-1 vs PASS-2).

Plots accuracy, entropy, improvement rates, tag validity, and sample counts
from the CSVs produced by ``summarize_*_inference.py``.
"""

from __future__ import annotations

import argparse
import importlib
import os
from typing import Callable, Optional

import numpy as np
import pandas as pd


try:  # pragma: no cover - allow lightweight test environments
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
except (ImportError, RuntimeError):  # pragma: no cover - matplotlib fallback
    from src.analysis.common.mpl_stub_helpers import AxisSettersMixin

    class _StubPdfPages:
        def __init__(self, path):
            self.path = path
            self.saved = 0

        def savefig(self, _fig):
            """Stub savefig; count saves for tests."""
            self.saved += 1

        def close(self):
            """Stub close; mirror PdfPages API."""
            return None

    class _StubAxes(AxisSettersMixin):
        """Axis stub mirroring the minimal matplotlib API used in tests."""

    class _StubFigure:
        def __init__(self):
            self.axis = _StubAxes()

        def tight_layout(self):
            """Stub tight layout."""
            return None

        def savefig(self, path, **_k):
            """Stub savefig that writes an empty placeholder file."""
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("stub")

        def __iter__(self):  # pragma: no cover - compatibility with tuple unpacking
            yield self.axis

    class _StubPlt:
        Figure = _StubFigure  # type: ignore[attr-defined]

        def subplots(self, *_args, **_kwargs):
            """Stub subplots; return a single figure/axis pair."""
            fig = _StubFigure()
            return fig, fig.axis

        def close(self, *_a, **_k):
            """Stub close; mirrors plt.close."""
            return None

    plt = _StubPlt()
    PdfPages = _StubPdfPages


def _ensure_dir(path: str) -> None:
    """Create ``path`` if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def _coerce_pct_series(series: pd.Series) -> pd.Series:
    """
    Accept floats or strings like ``\"26.2%\"`` and return float percents in [0, 100].
    """
    if series.dtype.kind in "fciu":
        return series.astype(float)
    return series.astype(str).str.strip().str.replace("%", "", regex=False).replace({"-": np.nan}).astype(float)


def _get_seaborn():
    """
    Import seaborn lazily so that this module does not hard-require it.
    """
    try:
        return importlib.import_module("seaborn")
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "seaborn is required for math summary plots; install it with 'pip install seaborn'.",
        ) from exc


def _load_summary(path: str) -> pd.DataFrame:
    """Load a summary CSV and normalize percentage columns and step ordering."""
    summary_df = pd.read_csv(path)
    for col in [
        "acc1S_pct",
        "acc1E_pct",
        "acc2S_pct",
        "acc2E_pct",
        "impS_pct",
        "impE_pct",
        "tag1_pct",
        "tag2_pct",
    ]:
        if col in summary_df.columns:
            summary_df[col] = _coerce_pct_series(summary_df[col])

    if "step" in summary_df.columns:
        summary_df = summary_df.sort_values("step").reset_index(drop=True)
    return summary_df


def _theme() -> None:
    """Apply a consistent Seaborn style for all plots."""
    seaborn_mod = _get_seaborn()
    seaborn_mod.set_theme(context="paper", style="whitegrid", font_scale=1.1)
    seaborn_mod.set_palette("deep")


def _save(figure: plt.Figure, outpath: str) -> None:
    """Tighten layout and save a figure to ``outpath``."""
    figure.tight_layout()
    figure.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _line(
    axes,
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    *,
    hue: Optional[str] = None,
) -> None:
    """Helper to draw a standardized line plot over training step."""
    seaborn_mod = _get_seaborn()
    seaborn_mod.lineplot(
        ax=axes,
        data=data,
        x=x_column,
        y=y_column,
        hue=hue,
        marker="o",
    )
    axes.set_xlabel("Training step")
    axes.grid(True, alpha=0.3)


def _plot_sample_accuracy(
    summary_df: pd.DataFrame,
    save_page: Callable[[plt.Figure, str], None],
) -> None:
    """Plot sample-level accuracy for Pass 1 vs Pass 2."""
    if not {"step", "acc1S_pct", "acc2S_pct"}.issubset(summary_df.columns):
        return
    plot_df = (
        summary_df[["step", "acc1S_pct", "acc2S_pct"]]
        .melt(id_vars="step", var_name="metric", value_name="accuracy_pct")
        .replace(
            {
                "acc1S_pct": "Pass 1 (sample)",
                "acc2S_pct": "Pass 2 (sample)",
            },
        )
    )
    figure, axes = plt.subplots(figsize=(7.2, 4.2))
    _line(axes, plot_df, "step", "accuracy_pct", hue="metric")
    axes.set_ylabel("Accuracy (sample %)")
    axes.set_title("Sample Accuracy vs Step")
    axes.legend(title="")
    save_page(figure, "accuracy_sample.png")


def _plot_example_accuracy(
    summary_df: pd.DataFrame,
    save_page: Callable[[plt.Figure, str], None],
) -> None:
    """Plot example-level accuracy for Pass 1 vs Pass 2."""
    if not {"step", "acc1E_pct", "acc2E_pct"}.issubset(summary_df.columns):
        return
    plot_df = (
        summary_df[["step", "acc1E_pct", "acc2E_pct"]]
        .melt(id_vars="step", var_name="metric", value_name="accuracy_pct")
        .replace(
            {
                "acc1E_pct": "Pass 1 (example)",
                "acc2E_pct": "Pass 2 (example)",
            },
        )
    )
    figure, axes = plt.subplots(figsize=(7.2, 4.2))
    _line(axes, plot_df, "step", "accuracy_pct", hue="metric")
    axes.set_ylabel("Accuracy (example %)")
    axes.set_title("Example Accuracy vs Step")
    axes.legend(title="")
    save_page(figure, "accuracy_example.png")


def _plot_delta_accuracy(
    summary_df: pd.DataFrame,
    save_page: Callable[[plt.Figure, str], None],
) -> None:
    """Plot Δ accuracy (Pass 2 − Pass 1) for sample- and example-level metrics."""
    have_sample_delta = {"acc1S_pct", "acc2S_pct"}.issubset(summary_df.columns)
    have_example_delta = {"acc1E_pct", "acc2E_pct"}.issubset(summary_df.columns)
    if not (have_sample_delta or have_example_delta):
        return

    delta_frames = []
    if have_sample_delta:
        delta_frames.append(
            pd.DataFrame(
                {
                    "step": summary_df["step"],
                    "delta_pct": summary_df["acc2S_pct"] - summary_df["acc1S_pct"],
                    "which": "Δ Sample (P2−P1)",
                },
            ),
        )
    if have_example_delta:
        delta_frames.append(
            pd.DataFrame(
                {
                    "step": summary_df["step"],
                    "delta_pct": summary_df["acc2E_pct"] - summary_df["acc1E_pct"],
                    "which": "Δ Example (P2−P1)",
                },
            ),
        )
    plot_df = pd.concat(delta_frames, ignore_index=True)
    figure, axes = plt.subplots(figsize=(7.2, 4.2))
    _line(axes, plot_df, "step", "delta_pct", hue="which")
    axes.axhline(0, ls="--", lw=1, color="gray", alpha=0.8)
    axes.set_ylabel("Accuracy Delta (pct pts)")
    axes.set_title("Pass-2 Gain vs Step")
    axes.legend(title="")
    save_page(figure, "delta_accuracy.png")


def _plot_entropy_overall(
    summary_df: pd.DataFrame,
    save_page: Callable[[plt.Figure, str], None],
) -> None:
    """Plot overall entropy for Pass 1 vs Pass 2."""
    if not {"step", "ent1", "ent2"}.issubset(summary_df.columns):
        return
    plot_df = (
        summary_df[["step", "ent1", "ent2"]]
        .melt(id_vars="step", var_name="metric", value_name="entropy")
        .replace({"ent1": "Pass 1 (overall)", "ent2": "Pass 2 (overall)"})
    )
    figure, axes = plt.subplots(figsize=(7.2, 4.2))
    _line(axes, plot_df, "step", "entropy", hue="metric")
    axes.set_ylabel("Mean token entropy")
    axes.set_title("Overall Entropy vs Step")
    axes.legend(title="")
    save_page(figure, "entropy_overall.png")


def _plot_entropy_phase(
    summary_df: pd.DataFrame,
    save_page: Callable[[plt.Figure, str], None],
) -> None:
    """Plot entropy for Think vs Answer phases, highlighting Pass 2 with dashed lines."""
    has_all_phase_columns = {"t1", "a1", "t2", "a2"}.issubset(summary_df.columns)
    if not ({"step"}.issubset(summary_df.columns) and has_all_phase_columns):
        return

    phase_df = _build_phase_entropy_frame(summary_df)

    figure, axes = plt.subplots(figsize=(7.8, 4.6))
    _line(axes, phase_df[phase_df["pass"] == "Pass 1"], "step", "entropy", hue="phase")
    axes.set_ylabel("Mean token entropy")
    axes.set_title("Phase Entropy vs Step (Pass 1 solid, Pass 2 dashed)")

    seaborn_mod = _get_seaborn()
    for phase_label in ("Think", "Answer"):
        selector = (phase_df["pass"] == "Pass 2") & (phase_df["phase"] == phase_label)
        if selector.any():
            seaborn_mod.lineplot(
                data=phase_df[selector],
                x="step",
                y="entropy",
                ax=axes,
                marker="o",
                linestyle="--",
                label=f"{phase_label} (Pass 2)",
            )

    handles, labels = axes.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes.legend(unique.values(), unique.keys(), title="")
    save_page(figure, "entropy_phase.png")


def _build_phase_entropy_frame(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Expand per-step entropy into a long DataFrame over phase and pass."""
    rows: list[pd.DataFrame] = []
    for pass_label, think_col, answer_col in [
        ("Pass 1", "t1", "a1"),
        ("Pass 2", "t2", "a2"),
    ]:
        rows.append(
            pd.DataFrame(
                {
                    "step": summary_df["step"],
                    "phase": "Think",
                    "entropy": summary_df[think_col],
                    "pass": pass_label,
                },
            ),
        )
        rows.append(
            pd.DataFrame(
                {
                    "step": summary_df["step"],
                    "phase": "Answer",
                    "entropy": summary_df[answer_col],
                    "pass": pass_label,
                },
            ),
        )
    return pd.concat(rows, ignore_index=True)


def _plot_improvement_rates(
    summary_df: pd.DataFrame,
    save_page: Callable[[plt.Figure, str], None],
) -> None:
    """Plot improvement rates (sample and example) vs step."""
    if not {"step", "impS_pct", "impE_pct"}.issubset(summary_df.columns):
        return
    plot_df = (
        summary_df[["step", "impS_pct", "impE_pct"]]
        .melt(id_vars="step", var_name="metric", value_name="improve_pct")
        .replace(
            {
                "impS_pct": "Sample improved",
                "impE_pct": "Example improved",
            },
        )
    )
    figure, axes = plt.subplots(figsize=(7.2, 4.2))
    _line(axes, plot_df, "step", "improve_pct", hue="metric")
    axes.set_ylabel("Improved (%)")
    axes.set_title("Pass-2 Improvement Rate vs Step")
    axes.legend(title="")
    save_page(figure, "improvement_rates.png")


def _plot_tag_validity(
    summary_df: pd.DataFrame,
    save_page: Callable[[plt.Figure, str], None],
) -> None:
    """Plot tag validity rates (Pass 1 vs Pass 2)."""
    if not {"step", "tag1_pct", "tag2_pct"}.issubset(summary_df.columns):
        return
    plot_df = (
        summary_df[["step", "tag1_pct", "tag2_pct"]]
        .melt(id_vars="step", var_name="metric", value_name="tag_ok_pct")
        .replace({"tag1_pct": "Pass 1", "tag2_pct": "Pass 2"})
    )
    figure, axes = plt.subplots(figsize=(7.2, 4.2))
    _line(axes, plot_df, "step", "tag_ok_pct", hue="metric")
    axes.set_ylabel("Valid tag structure (%)")
    axes.set_title("Tag Validity vs Step")
    axes.legend(title="")
    save_page(figure, "tag_validity.png")


def _plot_sample_counts(
    summary_df: pd.DataFrame,
    save_page: Callable[[plt.Figure, str], None],
) -> None:
    """Plot number of samples used per step (sanity check)."""
    if not {"step", "n1S", "n2S"}.issubset(summary_df.columns):
        return
    plot_df = (
        summary_df[["step", "n1S", "n2S"]]
        .melt(id_vars="step", var_name="metric", value_name="n")
        .replace({"n1S": "Pass 1 samples", "n2S": "Pass 2 samples"})
    )
    figure, axes = plt.subplots(figsize=(7.2, 4.2))
    _line(axes, plot_df, "step", "n", hue="metric")
    axes.set_ylabel("# samples")
    axes.set_title("Number of Samples vs Step")
    axes.legend(title="")
    save_page(figure, "sample_counts.png")


def make_plots(summary_df: pd.DataFrame, outdir: str, pdf_path: Optional[str] = None) -> None:
    """Create all standard plots for the math summary CSV."""
    _ensure_dir(outdir)
    _theme()

    pages = PdfPages(pdf_path) if pdf_path else None

    def save_page(figure: plt.Figure, name: str) -> None:
        _save(figure, os.path.join(outdir, name))
        if pages:
            pages.savefig(figure)

    _plot_sample_accuracy(summary_df, save_page)
    _plot_example_accuracy(summary_df, save_page)
    _plot_delta_accuracy(summary_df, save_page)
    _plot_entropy_overall(summary_df, save_page)
    _plot_entropy_phase(summary_df, save_page)
    _plot_improvement_rates(summary_df, save_page)
    _plot_tag_validity(summary_df, save_page)
    _plot_sample_counts(summary_df, save_page)

    if pages:
        pages.close()


def main() -> None:
    """CLI entry point to build plots from summarize_*_inference CSV output."""
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_csv", help="CSV from summarize_*_inference.py --save_csv")
    parser.add_argument("--outdir", default=None, help="Directory to write plots")
    parser.add_argument("--pdf", action="store_true", help="Also write a multi-page PDF")
    args = parser.parse_args()

    outdir = args.outdir or os.path.join(os.path.dirname(args.summary_csv), "plots")
    pdf_path = os.path.join(outdir, "summary_plots.pdf") if args.pdf else None

    summary_df = _load_summary(args.summary_csv)
    make_plots(summary_df, outdir, pdf_path=pdf_path)

    print(f"Saved plots to {outdir}")
    if args.pdf:
        print(f"Wrote combined PDF to {pdf_path}")


if __name__ == "__main__":
    main()  # pragma: no cover
