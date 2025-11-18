#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot summary metrics produced by summarize_*_inference.py.

Inputs
------
A CSV with columns like:
  step,n1S,acc1S_pct,acc1E_pct,ent1,t1,a1,n2S,acc2S_pct,acc2E_pct,ent2,t2,a2,
  impS_pct,impE_pct,tag1_pct,tag2_pct

Usage
-----
python plot_summary_metrics.py /path/to/summary.csv --outdir /tmp/plots
"""

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _coerce_pct_series(s: pd.Series) -> pd.Series:
    """Accept floats or strings like '26.2%' and return float percent in [0, 100]."""
    if s.dtype.kind in "fciu":
        return s.astype(float)
    return (
        s.astype(str)
         .str.strip()
         .str.replace("%", "", regex=False)
         .replace({"-": np.nan})
         .astype(float)
    )


def _load_summary(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Standardize expected columns if present
    for col in ["acc1S_pct", "acc1E_pct", "acc2S_pct", "acc2E_pct",
                "impS_pct", "impE_pct", "tag1_pct", "tag2_pct"]:
        if col in df.columns:
            df[col] = _coerce_pct_series(df[col])

    # Sort by step (numeric)
    if "step" in df.columns:
        df = df.sort_values("step").reset_index(drop=True)
    return df


def _theme():
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.1)
    sns.set_palette("deep")


def _save(fig, outpath: str):
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _line(ax, data, x, y, hue=None, marker="o"):
    sns.lineplot(ax=ax, data=data, x=x, y=y, hue=hue, marker=marker)
    ax.set_xlabel("Training step")
    ax.grid(True, alpha=0.3)


def make_plots(df: pd.DataFrame, outdir: str, pdf_path: Optional[str] = None):
    _ensure_dir(outdir)
    _theme()

    pages = PdfPages(pdf_path) if pdf_path else None
    def save_page(fig, name):
        _save(fig, os.path.join(outdir, name))
        if pages:
            pages.savefig(fig)

    # 1) Sample-level accuracy (Pass1 vs Pass2)
    if {"step", "acc1S_pct", "acc2S_pct"}.issubset(df.columns):
        d = (
            df[["step", "acc1S_pct", "acc2S_pct"]]
            .melt(id_vars="step", var_name="metric", value_name="accuracy_pct")
            .replace({"acc1S_pct": "Pass 1 (sample)",
                      "acc2S_pct": "Pass 2 (sample)"})
        )
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        _line(ax, d, "step", "accuracy_pct", hue="metric")
        ax.set_ylabel("Accuracy (sample %)"); ax.set_title("Sample Accuracy vs Step")
        ax.legend(title="")
        save_page(fig, "accuracy_sample.png")

    # 2) Example-level accuracy (Pass1 vs Pass2)
    if {"step", "acc1E_pct", "acc2E_pct"}.issubset(df.columns):
        d = (
            df[["step", "acc1E_pct", "acc2E_pct"]]
            .melt(id_vars="step", var_name="metric", value_name="accuracy_pct")
            .replace({"acc1E_pct": "Pass 1 (example)",
                      "acc2E_pct": "Pass 2 (example)"})
        )
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        _line(ax, d, "step", "accuracy_pct", hue="metric")
        ax.set_ylabel("Accuracy (example %)"); ax.set_title("Example Accuracy vs Step")
        ax.legend(title="")
        save_page(fig, "accuracy_example.png")

    # 3) Δ accuracy (Pass2 − Pass1), sample & example
    have_sample_delta = {"acc1S_pct", "acc2S_pct"}.issubset(df.columns)
    have_example_delta = {"acc1E_pct", "acc2E_pct"}.issubset(df.columns)
    if have_sample_delta or have_example_delta:
        dparts = []
        if have_sample_delta:
            dparts.append(pd.DataFrame({
                "step": df["step"],
                "delta_pct": df["acc2S_pct"] - df["acc1S_pct"],
                "which": "Δ Sample (P2−P1)"
            }))
        if have_example_delta:
            dparts.append(pd.DataFrame({
                "step": df["step"],
                "delta_pct": df["acc2E_pct"] - df["acc1E_pct"],
                "which": "Δ Example (P2−P1)"
            }))
        d = pd.concat(dparts, ignore_index=True)
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        _line(ax, d, "step", "delta_pct", hue="which")
        ax.axhline(0, ls="--", lw=1, color="gray", alpha=0.8)
        ax.set_ylabel("Accuracy Delta (pct pts)"); ax.set_title("Pass-2 Gain vs Step")
        ax.legend(title="")
        save_page(fig, "delta_accuracy.png")

    # 4) Entropy — overall (Pass1 vs Pass2)
    if {"step", "ent1", "ent2"}.issubset(df.columns):
        d = (
            df[["step", "ent1", "ent2"]]
            .melt(id_vars="step", var_name="metric", value_name="entropy")
            .replace({"ent1": "Pass 1 (overall)", "ent2": "Pass 2 (overall)"})
        )
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        _line(ax, d, "step", "entropy", hue="metric")
        ax.set_ylabel("Mean token entropy"); ax.set_title("Overall Entropy vs Step")
        ax.legend(title="")
        save_page(fig, "entropy_overall.png")

    # 5) Entropy — think vs answer phases
    have_phase = {"t1", "a1", "t2", "a2"}.issubset(df.columns)
    if {"step"}.issubset(df.columns) and have_phase:
        d = []
        for pass_id, T, A in [("Pass 1", "t1", "a1"), ("Pass 2", "t2", "a2")]:
            d.append(pd.DataFrame({
                "step": df["step"],
                "phase": "Think",
                "entropy": df[T],
                "pass": pass_id
            }))
            d.append(pd.DataFrame({
                "step": df["step"],
                "phase": "Answer",
                "entropy": df[A],
                "pass": pass_id
            }))
        d = pd.concat(d, ignore_index=True)
        fig, ax = plt.subplots(figsize=(7.8, 4.6))
        _line(ax, d, "step", "entropy", hue="phase")
        ax.set_ylabel("Mean token entropy")
        ax.set_title("Phase Entropy vs Step (Pass 1 shown as solid, Pass 2 dashed)")
        # Style: dashed for pass-2
        for line, label in zip(ax.lines, [ln.get_label() for ln in ax.lines]):
            pass
        # Overlay pass-2 with dashed style
        for p in ("Pass 2",):
            for ph in ("Think", "Answer"):
                sel = (d["pass"] == p) & (d["phase"] == ph)
                if sel.any():
                    sns.lineplot(
                        data=d[sel], x="step", y="entropy",
                        ax=ax, marker="o", linestyle="--", label=f"{ph} (Pass 2)")
        # Reset legend to unique labels
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(), title="")
        save_page(fig, "entropy_phase.png")

    # 6) Improvement rates (sample & example)
    if {"step", "impS_pct", "impE_pct"}.issubset(df.columns):
        d = (
            df[["step", "impS_pct", "impE_pct"]]
            .melt(id_vars="step", var_name="metric", value_name="improve_pct")
            .replace({"impS_pct": "Sample improved",
                      "impE_pct": "Example improved"})
        )
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        _line(ax, d, "step", "improve_pct", hue="metric")
        ax.set_ylabel("Improved (%)"); ax.set_title("Pass-2 Improvement Rate vs Step")
        ax.legend(title="")
        save_page(fig, "improvement_rates.png")

    # 7) Tag validity (structural well-formedness)
    if {"step", "tag1_pct", "tag2_pct"}.issubset(df.columns):
        d = (
            df[["step", "tag1_pct", "tag2_pct"]]
            .melt(id_vars="step", var_name="metric", value_name="tag_ok_pct")
            .replace({"tag1_pct": "Pass 1", "tag2_pct": "Pass 2"})
        )
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        _line(ax, d, "step", "tag_ok_pct", hue="metric")
        ax.set_ylabel("Valid tag structure (%)"); ax.set_title("Tag Validity vs Step")
        ax.legend(title="")
        save_page(fig, "tag_validity.png")

    # 8) Sample counts used per step (sanity)
    if {"step", "n1S", "n2S"}.issubset(df.columns):
        d = (
            df[["step", "n1S", "n2S"]]
            .melt(id_vars="step", var_name="metric", value_name="n")
            .replace({"n1S": "Pass 1 samples", "n2S": "Pass 2 samples"})
        )
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        _line(ax, d, "step", "n", hue="metric")
        ax.set_ylabel("# samples"); ax.set_title("Number of Samples vs Step")
        ax.legend(title="")
        save_page(fig, "sample_counts.png")

    if pages:
        pages.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary_csv", help="CSV from summarize_*_inference.py --save_csv")
    ap.add_argument("--outdir", default=None, help="Directory to write plots")
    ap.add_argument("--pdf", action="store_true", help="Also write a multi-page PDF")
    args = ap.parse_args()

    outdir = args.outdir or os.path.join(os.path.dirname(args.summary_csv), "plots")
    pdf_path = os.path.join(outdir, "summary_plots.pdf") if args.pdf else None

    df = _load_summary(args.summary_csv)
    make_plots(df, outdir, pdf_path=pdf_path)

    print(f"Saved plots to {outdir}")
    if args.pdf:
        print(f"Wrote combined PDF to {pdf_path}")


if __name__ == "__main__":
    main()
