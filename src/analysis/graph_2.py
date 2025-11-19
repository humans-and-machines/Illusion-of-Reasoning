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

import os
import re
import json
import argparse
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

try:
    # Package imports
    from .labels import aha_gpt_for_rec
    from .utils import coerce_bool, coerce_float, get_problem_id, nat_step_from_path
except ImportError:  # pragma: no cover - script fallback
    import os as _os
    import sys as _sys

    _ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _ROOT not in _sys.path:
        _sys.path.append(_ROOT)
    from analysis.labels import aha_gpt_for_rec  # type: ignore
    from analysis.utils import coerce_bool, coerce_float, get_problem_id, nat_step_from_path  # type: ignore

# ---------- Style (Times-like) ----------
STYLE_PARAMS = {
    "axes.titlesize" : 14,
    "axes.labelsize" : 13,
    "font.size"      : 13,
    "legend.fontsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "font.family"    : "serif",
    "font.serif"     : ["Times New Roman", "Times", "Nimbus Roman No9 L", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
}
mpl.rcParams.update(STYLE_PARAMS)

# ---------- Filename step parser ----------
def scan_files(root: str, split_substr: Optional[str]) -> List[str]:
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if not fn.endswith(".jsonl"):
                continue
            if split_substr and split_substr not in fn:
                continue
            out.append(os.path.join(dp, fn))
    out.sort()
    return out

# ---------- Carpark success policy ----------
def _extract_soft_reward(rec: Dict[str, Any], p1: Dict[str, Any]) -> Optional[float]:
    return coerce_float(rec.get("soft_reward", p1.get("soft_reward")))

def _make_carpark_success_fn(op: str, thr: float):
    def _cmp(val: Any) -> Optional[int]:
        x = coerce_float(val)
        if x is None: return None
        if op == "gt": return int(x >  thr)
        if op == "ge": return int(x >= thr)
        if op == "eq": return int(x == thr)
        return int(x > thr)
    return _cmp

# ---------- Row loader ----------
def load_rows(files_by_domain: Dict[str, List[str]],
              gpt_keys: List[str],
              gpt_subset_native: bool,
              min_step: Optional[int],
              max_step: Optional[int],
              carpark_success_fn) -> pd.DataFrame:
    rows = []
    for dom, files in files_by_domain.items():
        dlow = str(dom).lower()
        for path in files:
            step_from_name = nat_step_from_path(path)
            with open(path, "r", encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln: continue
                    try:
                        rec = json.loads(ln)
                    except Exception:
                        continue
                    p1 = rec.get("pass1") or {}
                    if not isinstance(p1, dict):
                        continue

                    step = rec.get("step", step_from_name if step_from_name is not None else None)
                    if step is None:
                        continue
                    try:
                        step = int(step)
                    except Exception:
                        continue
                    if min_step is not None and step < min_step:
                        continue
                    if max_step is not None and step > max_step:
                        continue

                    # success (accuracy)
                    if dlow.startswith("carpark"):
                        success = carpark_success_fn(_extract_soft_reward(rec, p1))
                    else:
                        success = coerce_bool(p1.get("is_correct_pred"))
                    if success is None:
                        continue

                    # entropies
                    e_think  = coerce_float(p1.get("entropy_think"))
                    e_answer = coerce_float(p1.get("entropy_answer"))
                    e_joint  = coerce_float(p1.get("entropy"))
                    if e_joint is None:
                        # fall back: mean of available think/answer
                        vals = [v for v in (e_think, e_answer) if v is not None]
                        e_joint = float(np.mean(vals)) if vals else None
                    if e_think is None and e_answer is None and e_joint is None:
                        continue

                    aha = aha_gpt_for_rec(p1, rec, gpt_subset_native, gpt_keys, dom)
                    pid = get_problem_id(rec) or f"unnamed_{hash(ln)%10**9}"

                    rows.append({
                        "domain": str(dom),
                        "problem_id": pid,
                        "step": step,
                        "correct": int(success),
                        "aha": int(aha),
                        "entropy_think": e_think,
                        "entropy_answer": e_answer,
                        "entropy_joint": e_joint,
                    })
    return pd.DataFrame(rows)

# ---------- Binning and aggregation ----------
def make_bins(series: pd.Series, n_bins: int, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    s = series.dropna()
    if s.empty:
        return np.array([]), np.array([])
    if mode == "quantile":
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(s, qs))
        # ensure at least 2 distinct edges
        if len(edges) < 2:
            edges = np.array([s.min(), s.max() + 1e-9])
    else:
        lo, hi = float(s.min()), float(s.max())
        if lo == hi:
            lo, hi = lo - 1e-9, hi + 1e-9
        edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers

def aggregate_bins(df_dom: pd.DataFrame, metric_col: str, n_bins: int, mode: str,
                   min_per_bar: int) -> pd.DataFrame:
    s = df_dom[metric_col]
    edges, centers = make_bins(s, n_bins, mode)
    if edges.size == 0:
        return pd.DataFrame(columns=[
            "domain","metric","bin_idx","bin_left","bin_right",
            "n_total","n_aha","n_noaha","acc_aha","acc_noaha"
        ])

    cats = pd.cut(s, bins=edges, include_lowest=True, right=True)
    tmp = df_dom.assign(_bin=cats).dropna(subset=["_bin"])
    # group by bin and aha
    g = tmp.groupby(["_bin", "aha"], observed=True)
    acc = g["correct"].mean().rename("acc").reset_index()
    cnt = g["correct"].size().rename("n").reset_index()
    m = acc.merge(cnt, on=["_bin","aha"], how="outer")
    # pivot aha to columns
    piv_acc = m.pivot(index="_bin", columns="aha", values="acc")
    piv_cnt = m.pivot(index="_bin", columns="aha", values="n").fillna(0).astype(int)

    # filter bins that lack support in a bar
    ok = (piv_cnt.get(0, 0) >= min_per_bar) & (piv_cnt.get(1, 0) >= min_per_bar)
    piv_acc = piv_acc[ok]
    piv_cnt = piv_cnt[ok]
    kept_bins = piv_acc.index

    # build output rows
    rows = []
    for i, b in enumerate(kept_bins):
        left, right = float(b.left), float(b.right)
        n0 = int(piv_cnt.loc[b, 0]) if 0 in piv_cnt.columns else 0
        n1 = int(piv_cnt.loc[b, 1]) if 1 in piv_cnt.columns else 0
        a0 = float(piv_acc.loc[b, 0]) if 0 in piv_acc.columns else np.nan
        a1 = float(piv_acc.loc[b, 1]) if 1 in piv_acc.columns else np.nan
        rows.append({
            "metric": metric_col,
            "bin_idx": i,
            "bin_left": left,
            "bin_right": right,
            "bin_center": 0.5*(left+right),
            "n_total": n0 + n1,
            "n_noaha": n0,
            "n_aha": n1,
            "acc_noaha": a0,
            "acc_aha": a1,
        })
    return pd.DataFrame(rows)

# ---------- Plotting ----------
def plot_domain_panels(domain: str,
                       stats_by_metric: Dict[str, pd.DataFrame],
                       out_path_png: str,
                       dpi: int = 600,
                       title_prefix: Optional[str] = None):
    metrics = [("entropy_think","Think entropy"),
               ("entropy_answer","Answer entropy"),
               ("entropy_joint","Joint entropy")]
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.2), sharey=True)
    if title_prefix:
        fig.suptitle(f"{title_prefix} — {domain}", y=1.05)

    for ax, (mcol, mlabel) in zip(axes, metrics):
        stat = stats_by_metric.get(mcol)
        ax.set_title(mlabel)
        ax.set_xlabel("Entropy bins")
        ax.set_ylim(0.0, 1.0)
        if stat is None or stat.empty:
            ax.text(0.5, 0.5, "No bins with sufficient data",
                    ha="center", va="center", transform=ax.transAxes, fontsize=11)
            continue
        # x positions
        x = np.arange(len(stat))
        width = 0.4
        # bars
        ax.bar(x - width/2, stat["acc_noaha"], width=width, label="No Aha", alpha=0.85)
        ax.bar(x + width/2, stat["acc_aha"],   width=width, label="Aha",    alpha=0.85)
        # xtick labels as bin ranges (compact)
        labels = [f"[{l:.2f},{r:.2f}]" for l, r in zip(stat["bin_left"], stat["bin_right"])]
        ax.set_xticks(x, labels, rotation=45, ha="right")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
    axes[0].set_ylabel("Accuracy")

    # single legend for the row
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.savefig(out_path_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    # Domain roots
    ap.add_argument("--root_crossword", type=str, default=None)
    ap.add_argument("--root_math", type=str, default=None)
    ap.add_argument("--root_math2", type=str, default=None, help="Optional second Math model folder")
    ap.add_argument("--root_carpark", type=str, default=None)
    ap.add_argument("results_root", nargs="?", default=None,
                    help="Fallback single root if domain-specific roots are not provided.")
    ap.add_argument("--split", default=None, help="Substring filter (e.g., 'test')")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--dataset_name", default="MIXED")
    ap.add_argument("--model_name", default="Qwen2.5-1.5B")

    # GPT label policy
    ap.add_argument("--gpt_mode", choices=["canonical","broad"], default="canonical")
    ap.add_argument("--no_gpt_subset_native", action="store_true",
                    help="Disable domain-aware gate; use raw GPT shift flags")

    # Binning
    ap.add_argument("--n_bins", type=int, default=10)
    ap.add_argument("--bucket_mode", choices=["uniform","quantile"], default="uniform",
                    help="Bin edges strategy per domain/metric distribution")
    ap.add_argument("--min_per_bar", type=int, default=25,
                    help="Min examples per (bin, Aha-state) to render the bar")

    # Steps (default hard cap 1000 for safety)
    ap.add_argument("--min_step", type=int, default=None)
    ap.add_argument("--max_step", type=int, default=1000,
                    help="Upper bound on step (default 1000). Set higher to disable cap.")

    # Carpark policy
    ap.add_argument("--carpark_success_op", choices=["gt","ge","eq"], default="gt")
    ap.add_argument("--carpark_soft_threshold", type=float, default=0.0)

    # Rendering
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    # Roots
    files_by_domain: Dict[str, List[str]] = {}
    first_root = None
    if args.root_crossword:
        files_by_domain["Crossword"] = scan_files(args.root_crossword, args.split); first_root = first_root or args.root_crossword
    if args.root_math:
        files_by_domain["Math"] = scan_files(args.root_math, args.split); first_root = first_root or args.root_math
    if args.root_math2:
        files_by_domain["Math2"] = scan_files(args.root_math2, args.split); first_root = first_root or args.root_math2
    if args.root_carpark:
        files_by_domain["Carpark"] = scan_files(args.root_carpark, args.split); first_root = first_root or args.root_carpark
    if not files_by_domain:
        if not args.results_root:
            raise SystemExit("Provide --root_* folders or a fallback results_root.")
        files_by_domain["All"] = scan_files(args.results_root, args.split)
        first_root = args.results_root

    total_files = sum(len(v) for v in files_by_domain.values())
    if total_files == 0:
        raise SystemExit("No JSONL files found. Check roots/--split.")

    out_dir = args.out_dir or os.path.join(first_root, "entropy_histograms")
    os.makedirs(out_dir, exist_ok=True)

    # GPT keys & policy
    gpt_subset_native = not args.no_gpt_subset_native
    gpt_keys = (["change_way_of_thinking", "shift_in_reasoning_v1"] if args.gpt_mode == "canonical"
                else ["change_way_of_thinking", "shift_in_reasoning_v1",
                      "shift_llm", "shift_gpt", "pivot_llm", "rechecked"])

    # Carpark comparator
    carpark_success_fn = _make_carpark_success_fn(args.carpark_success_op, args.carpark_soft_threshold)

    # Load
    df = load_rows(files_by_domain, gpt_keys, gpt_subset_native,
                   min_step=args.min_step, max_step=args.max_step,
                   carpark_success_fn=carpark_success_fn)
    if df.empty:
        raise SystemExit("No rows after filtering.")

    # Build per-domain stats & figures
    csv_rows = []
    slug = f"{args.dataset_name}__{args.model_name}".replace(" ", "_")
    for domain in sorted(df["domain"].unique()):
        sub = df[df["domain"] == domain].copy()
        stats_by_metric: Dict[str, pd.DataFrame] = {}
        for mcol in ["entropy_think","entropy_answer","entropy_joint"]:
            avail = sub[mcol].dropna()
            if avail.empty:
                stats_by_metric[mcol] = pd.DataFrame()
                continue
            stat = aggregate_bins(sub, mcol, n_bins=args.n_bins, mode=args.bucket_mode,
                                min_per_bar=args.min_per_bar)

            if "domain" not in stat.columns:
                stat.insert(0, "domain", domain)
            # Only insert 'metric' if aggregate_bins didn't already add it.
            if "metric" not in stat.columns:
                stat.insert(1, "metric", mcol)
            stats_by_metric[mcol] = stat
            csv_rows.append(stat)

        # Plot
        fig_path = os.path.join(out_dir, f"{slug}__{domain}__entropy_hist_panels.png")
        plot_domain_panels(domain, stats_by_metric, fig_path, dpi=args.dpi,
                           title_prefix=f"Accuracy vs Entropy (Aha vs No-Aha)")

    # Save CSV
    if csv_rows:
        table = pd.concat(csv_rows, axis=0, ignore_index=True)
        out_csv = os.path.join(out_dir, f"entropy_hist__{slug}.csv")
        table.to_csv(out_csv, index=False)
        with pd.option_context("display.max_columns", None, "display.width", 120):
            print("\nPer-bin accuracy (head):")
            print(table.head(12).to_string(index=False))
        print(f"\nSaved CSV  -> {out_csv}")
    print(f"Saved figs -> {out_dir}/*__entropy_hist_panels.[png|pdf]")

if __name__ == "__main__":
    main()
