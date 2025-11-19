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
- For Carpark, you can derive correctness from soft_reward via --carpark_success_op/--carpark_soft_threshold.

Outputs
-------
- PNG + PDF: two stacked panels
- CSVs with per-bin aggregates

Example
-------
python final-plot.py \
  --scan_root graphs/_agg_qwen15b_t07 \
  --bins 0 1 2 3 4 \
  --min_step 0 --max_step 1000 \
  --carpark_success_op ge --carpark_soft_threshold 0.1 \
  --out_dir graphs/uncertainty_qwen15b_t07 \
  --dataset_name MIXED \
  --model_name "Qwen2.5-1.5B @ T=0.7" \
  --dpi 600 --make_plot
"""

import os
import sys
import argparse
import math
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd

try:
    # Preferred when analysis is installed as a package
    from .io import iter_records_from_file, scan_files_step_only
    from .metrics import carpark_success_from_soft_reward, extract_correct
    from .utils import coerce_bool, coerce_float, nat_step_from_path
except ImportError:  # pragma: no cover - script fallback
    # Fallback for running this file directly: add project src root and import
    import os as _os
    import sys as _sys

    _ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _ROOT not in _sys.path:
        _sys.path.append(_ROOT)
    from analysis.io import iter_records_from_file, scan_files_step_only  # type: ignore
    from analysis.metrics import carpark_success_from_soft_reward, extract_correct  # type: ignore
    from analysis.utils import coerce_bool, coerce_float, nat_step_from_path  # type: ignore


def example_key(rec: Dict[str, Any]) -> Optional[str]:
    for k in ("example_id", "problem_id", "id", "uid", "question", "clue", "title"):
        v = rec.get(k)
        if v is not None and not isinstance(v, (dict, list)):
            return f"{k}:{v}"
    v = rec.get("sample_idx")
    return None if v is None else f"sample_{v}"


def step_from_rec_or_path(rec: Dict[str, Any], path: str) -> int:
    step = rec.get("step") or rec.get("global_step") or rec.get("training_step")
    if step is None:
        step = nat_step_from_path(path)
    try:
        return int(step) if step is not None else 0
    except Exception:
        return 0


def entropy_from_pass1(p1: Dict[str, Any]) -> Optional[float]:
    # Prefer combined 'entropy' if present; else avg(think, answer) when both exist
    e = coerce_float(p1.get("entropy"))
    et = coerce_float(p1.get("entropy_think"))
    ea = coerce_float(p1.get("entropy_answer"))
    if e is not None:
        return e
    if et is not None and ea is not None:
        return 0.5 * (et + ea)
    return et if et is not None else ea


def pass2_triggered(p2: Dict[str, Any]) -> int:
    h = coerce_bool(p2.get("has_reconsider_cue"))
    marks = p2.get("reconsider_markers") or []
    inj = ("injected_cue" in marks) if isinstance(marks, (list, tuple)) else False
    return int(h == 1 and inj)

# -------------------------- Aggregation logic --------------------------

def load_dataframe(files: List[str],
                   carpark_op: str, carpark_thr: float,
                   min_step: Optional[int], max_step: Optional[int]) -> pd.DataFrame:
    """
    Row = one sample. Columns:
      group_id = (example_key || sample_idx) + "::stepNNN"
      entropy  = pass-1 entropy (fallback rules)
      p1_correct, p2_triggered, p2_correct, p1_shift
    """
    rows = []
    for path in files:
        for rec in iter_records_from_file(path):
            p1 = rec.get("pass1") or {}
            p2 = rec.get("pass2") or {}

            step = step_from_rec_or_path(rec, path)
            if min_step is not None and step < min_step: continue
            if max_step is not None and step > max_step: continue

            gid = example_key(rec) or f"path:{os.path.basename(path)}:line"
            group_id = f"{gid}::step{step}"

            ent = entropy_from_pass1(p1)

            # correctness (pass-1), with carpark fallback
            p1_corr = extract_correct(p1, rec)
            if p1_corr is None:
                cpr = carpark_success_from_soft_reward(rec, p1, carpark_op, carpark_thr)
                p1_corr = cpr

            # pass-2
            trig = 0
            p2_corr = None
            if isinstance(p2, dict) and p2:
                trig = pass2_triggered(p2)
                p2_corr = extract_correct(p2, rec)
                if p2_corr is None:
                    cpr2 = carpark_success_from_soft_reward(rec, p2, carpark_op, carpark_thr)
                    p2_corr = cpr2

            rows.append({
                "group_id": group_id,
                "entropy": ent,
                "p1_correct": None if p1_corr is None else int(p1_corr),
                "p2_triggered": int(trig),
                "p2_correct": None if p2_corr is None else int(p2_corr),
                "p1_shift": int(coerce_bool(p1.get("shift_in_reasoning_v1")) == 1),
            })
    df = pd.DataFrame(rows)
    # drop rows with no entropy or p1 correctness
    df = df[pd.notna(df["entropy"]) & pd.notna(df["p1_correct"])]
    return df

def summarize_for_figure(df: pd.DataFrame, bins: List[float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      df_top  : per-bin shift prevalence (pass-1 sample-level)
      df_bot  : per-bin, per-stratum (baseline 0/8 vs ≥1/8) pass-2 any-correct rate (triggered only)
    """
    # ---- Top panel: shift prevalence vs entropy (sample-level) ----
    df_top = df.copy()
    df_top["_bin"] = pd.cut(df_top["entropy"], bins=bins, right=False, include_lowest=True)
    grp_top = df_top.groupby("_bin", observed=False).agg(
        N=("p1_shift", "size"),
        shift_share=("p1_shift", "mean"),
    ).reset_index()

    # ---- Bottom panel: gated reconsideration success vs entropy (group-level) ----
    g = df.copy().groupby("group_id", as_index=False).agg(
        entropy_mean=("entropy", "mean"),
        p1_any_correct=("p1_correct", lambda x: int(np.nansum(x) > 0)),
        p2_any_correct_triggered=("p2_correct", lambda x: int(np.nansum(x) > 0)),
        any_trigger=("p2_triggered", lambda x: int(np.nansum(x) > 0)),
    )
    g = g[g["any_trigger"] == 1].copy()
    g["_bin"] = pd.cut(g["entropy_mean"], bins=bins, right=False, include_lowest=True)
    g["_stratum"] = np.where(g["p1_any_correct"] == 1, "baseline ≥1/8", "baseline 0/8")

    def _agg(sub):
        n = len(sub)
        if n == 0:
            return pd.Series(dict(N=0, rate=np.nan, se=np.nan, lo=np.nan, hi=np.nan))
        p = float(np.mean(sub["p2_any_correct_triggered"]))
        se = math.sqrt(p*(1.0-p)/n) if n > 0 else np.nan
        lo = max(0.0, p - 1.96*se)
        hi = min(1.0, p + 1.96*se)
        return pd.Series(dict(N=n, rate=p, se=se, lo=lo, hi=hi))

    df_bot = g.groupby(["_bin","_stratum"], observed=False).apply(_agg).reset_index()

    return grp_top, df_bot

# ----------------------------- Plotting (FIXED) -----------------------------

def plot_figure(df_top: pd.DataFrame, df_bot: pd.DataFrame,
                bins: List[float], out_png: str, out_pdf: str,
                title_suffix: str, dpi: int = 300):
    """
    NaN-safe plotting using IntervalIndex keys (prevents cast errors when bins/strata are empty).
    """
    import matplotlib.pyplot as plt

    # Build IntervalIndex once and use it as the ground truth for bin order
    intervals = pd.IntervalIndex.from_breaks(bins, closed="left")
    xticklabels = [f"[{l:g},{r:g})" for l, r in zip(intervals.left, intervals.right)]
    x = np.arange(len(intervals))

    fig, axes = plt.subplots(2, 1, figsize=(6.0, 5.0), constrained_layout=False)
    ax1, ax2 = axes

    # ---------------- Panel (A): shift prevalence vs entropy ----------------
    share_map = {}
    for _, row in df_top.iterrows():
        share_map[row["_bin"]] = row["shift_share"]
    y_top = [share_map.get(I, np.nan) for I in intervals]

    ax1.bar(x, y_top)
    ax1.set_xticks(x, xticklabels, rotation=0)
    ax1.set_ylabel("Shift share (pass-1)")
    ax1.set_title(f"(A) Shifts cluster at high uncertainty — {title_suffix}")
    ax1.grid(True, axis="y", alpha=0.25)

    # ---------------- Panel (B): gated reconsideration success --------------
    bot = df_bot.copy()
    bot["N"] = pd.to_numeric(bot["N"], errors="coerce").fillna(0)

    def series_for(stratum: str):
        s = bot[bot["_stratum"] == stratum].copy()
        rate_map = {b: r for b, r in zip(s["_bin"], s["rate"])}
        se_map   = {b: r for b, r in zip(s["_bin"], s["se"])}
        N_map    = {b: (0 if (pd.isna(n)) else int(n)) for b, n in zip(s["_bin"], s["N"])}

        rates = np.array([rate_map.get(I, np.nan) for I in intervals], float)
        ses   = np.array([se_map.get(I, np.nan) for I in intervals], float)
        Ns    = np.array([N_map.get(I, 0)       for I in intervals], int)
        return rates, ses, Ns

    r1, s1, n1 = series_for("baseline ≥1/8")
    r0, s0, n0 = series_for("baseline 0/8")

    ax2.errorbar(x, r1, yerr=s1, fmt="o-", capsize=4, linewidth=2, label="baseline ≥1/8")
    ax2.errorbar(x, r0, yerr=s0, fmt="s--", capsize=4, linewidth=2, label="baseline 0/8")
    ax2.set_xticks(x, xticklabels, rotation=0)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_ylabel("P(pass-2 any-correct | injected)")
    ax2.set_title("(B) Triggering reconsideration helps only when prior ≥1/8")
    ax2.grid(True, axis="y", alpha=0.25)

    # Annotate sample sizes under each x tick: N_high/N_low
    for xi, (n_hi, n_lo) in enumerate(zip(n1, n0)):
        ax2.text(xi, -0.08, f"N={n_hi}/{n_lo}", ha="center", va="top",
                 fontsize=9, transform=ax2.get_xaxis_transform())

    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=True)
    fig.subplots_adjust(bottom=0.22, hspace=0.35)

    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

# ------------------------------- Main ---------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan_root", type=str, required=True,
                    help="Root directory containing step*/.../*.jsonl|json(.gz)")
    ap.add_argument("--split", type=str, default=None,
                    help="Only include files whose NAMES contain this substring (e.g., 'test').")
    ap.add_argument("--bins", nargs="+", type=float, default=[0,1,2,3,4],
                    help="Entropy bin edges (closed-open).")
    ap.add_argument("--min_step", type=int, default=None)
    ap.add_argument("--max_step", type=int, default=None)

    ap.add_argument("--carpark_success_op", choices=["gt","ge","eq"], default="gt")
    ap.add_argument("--carpark_soft_threshold", type=float, default=0.0)

    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--dataset_name", type=str, default="MIXED")
    ap.add_argument("--model_name", type=str, default="MODEL")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--make_plot", action="store_true")

    args = ap.parse_args()

    # Scan
    files = scan_files_step_only(
        root=args.scan_root,
        split_substr=args.split,
        skip_substrings=None,
    )
    if not files:
        sys.exit("No files found. Check --scan_root / --split.")

    # Load
    df = load_dataframe(
        files=files,
        carpark_op=args.carpark_success_op,
        carpark_thr=args.carpark_soft_threshold,
        min_step=args.min_step, max_step=args.max_step
    )
    if df.empty:
        sys.exit("No usable rows after parsing (missing entropy or p1 correctness).")

    # Summaries
    bins = list(args.bins)
    grp_top, df_bot = summarize_for_figure(df, bins=bins)

    # Save CSVs
    slug = f"{args.dataset_name}__{args.model_name}".replace(" ", "_")
    out_dir = args.out_dir or os.path.join(args.scan_root, "uncertainty_gated_effect")
    os.makedirs(out_dir, exist_ok=True)
    csv_top = os.path.join(out_dir, f"uncertainty_shift_prevalence__{slug}.csv")
    csv_bot = os.path.join(out_dir, f"uncertainty_gated_success__{slug}.csv")
    grp_top.to_csv(csv_top, index=False)
    df_bot.to_csv(csv_bot, index=False)
    print(f"[saved] {csv_top}")
    print(f"[saved] {csv_bot}")

    # Plot
    if args.make_plot:
        png_path = os.path.join(out_dir, f"uncertainty_gated_effect__{slug}.png")
        pdf_path = os.path.join(out_dir, f"uncertainty_gated_effect__{slug}.pdf")
        plot_figure(
            df_top=grp_top, df_bot=df_bot, bins=bins,
            out_png=png_path, out_pdf=pdf_path,
            title_suffix=args.model_name, dpi=args.dpi
        )
        print(f"[saved] {png_path}")
        print(f"[saved] {pdf_path}")

    # Console preview
    with pd.option_context("display.width", 140):
        print("\n[Top panel] Shift prevalence vs entropy (pass-1 sample-level):")
        print(grp_top.to_string(index=False))
        print("\n[Bottom panel] P(pass-2 any-correct | injected) by entropy bin and stratum:")
        print(df_bot.to_string(index=False))

if __name__ == "__main__":
    main()
