#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pass2_effects.py  —  Raw effect of Pass-2 by domain, split by Pass-1 solvability

What it does
------------
• Loads PASS1/PASS2 records from your JSONLs (same format as your other scripts).
• Groups at the (domain, step, problem) level, averaging over the 8 samples:
    p1_acc = mean(correct_pass1 over samples)
    p2_acc = mean(correct_pass2 over samples)
    raw_effect = p2_acc - p1_acc
• Splits problems into two buckets:
    - P1_ANY  : p1_acc > 0  (at least one of 8 correct in Pass-1)
    - P1_NONE : p1_acc = 0  (none correct in Pass-1)
• Produces a 3-panel figure (Carpark / Crossword / Math):
    bar = mean(raw_effect) per bucket, with 95% bootstrap CI; n annotated
• Writes CSVs for per-problem rows and domain summaries.

Outputs
-------
graphs/
  pass2_raw_effects_{tag}.png, .pdf
  tables/pass2_per_problem_{tag}.csv
  tables/pass2_summary_{tag}.csv

Notes
-----
• Carpark correctness uses soft reward thresholds (configurable).
• You can pool across steps per problem via --pool_across_steps (optional).
"""

import argparse, json, os, re, sys, glob
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# ---------- Matplotlib typography ----------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 13,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ---------- Helpers ----------
def truthy(x) -> bool:
    if x is True: return True
    if isinstance(x, (int, float)): return x != 0
    if isinstance(x, str): return x.strip().lower() in {"1","true","yes","y"}
    return False

STEP_PAT = re.compile(r"step[-_]?(\d{1,5})|global_step[-_]?(\d{1,5})", re.I)
def extract_step(rec: Dict[str, Any], src: str) -> int:
    if isinstance(rec.get("step"), (int, float)): return int(rec["step"])
    m = STEP_PAT.search(src)
    if m:
        for g in m.groups():
            if g: return int(g)
    return 0

# Group by *problem* (your preference), with fallbacks that are problem-like only.
FALLBACK_KEYS = ["problem_id", "question", "clue", "title", "id", "uid"]
def group_key_for(obj: dict, line_idx: int) -> str:
    v = obj.get("problem", None)
    if v is not None and not isinstance(v, (dict, list)):
        return f"problem:{str(v)}"
    for k in FALLBACK_KEYS:
        vv = obj.get(k, None)
        if vv is not None and not isinstance(vv, (dict, list)):
            return f"{k}:{str(vv)}"
    return f"__LINE__:{line_idx}"

# ----- Correctness (PASS1 / PASS2) -----
def carpark_correct(d: Dict[str, Any], op: str, thr: float) -> bool:
    def decide(x: float) -> bool:
        return (x >= thr) if op=="ge" else (x > thr) if op=="gt" else (x <= thr) if op=="le" else (x < thr)
    for k in ("soft_1", "soft_reward"):
        v = d.get(k, None)
        if isinstance(v, (int, float)): return decide(float(v))
    # optional boolean fallbacks if present
    for k in ("is_correct", "correct", "correct_exact", "is_correct_pred"):
        if k in d: return truthy(d[k])
    return False

def general_correct(d: Dict[str, Any]) -> bool:
    for k in ("is_correct", "correct", "correct_exact", "is_correct_pred"):
        if k in d: return truthy(d[k])
    return False

def pass_correct_for_domain(pass_dict: Dict[str, Any], domain: str, op: str, thr: float) -> bool:
    if not isinstance(pass_dict, dict): return False
    if domain.lower().startswith("carpark"):
        return carpark_correct(pass_dict, op, thr)
    return general_correct(pass_dict)

# ---------- IO ----------
def expand_dirs(patterns: List[str]) -> List[Path]:
    out: List[Path] = []
    for p in patterns:
        if any(ch in p for ch in "*?[]"):
            out += [Path(x) for x in glob.glob(p)]
        else:
            out.append(Path(p))
    return [p for p in out if p.exists() and p.is_dir()]

def iter_jsonl(root_dirs: List[str], split: str | None) -> Iterable[Tuple[str, str]]:
    for root in expand_dirs(root_dirs):
        for f in root.rglob("*.jsonl"):
            fp = str(f)
            if split and split not in os.path.basename(fp):
                continue
            yield (root.name, fp)  # (root_dir_name, file_path)

# ---------- Core aggregation ----------
def load_per_problem(args, domain_name: str, roots: List[str]) -> pd.DataFrame:
    """
    Return per-(problem, step) rows:
      domain, step, problem_key, n_p1, n_p2, p1_acc, p2_acc, raw_effect, p1_any_correct
    """
    # Collect sample-level tuples grouped by (step, problem)
    bucket: Dict[Tuple[int, str], Dict[str, List[int]]] = defaultdict(lambda: {"p1": [], "p2": []})

    for _, fp in iter_jsonl(roots, args.split):
        try:
            with open(fp, "r", encoding="utf-8") as fh:
                for line_idx, line in enumerate(fh):
                    line = line.strip()
                    if not line: continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    step = extract_step(rec, fp)
                    if step < args.min_step or step > args.max_step:
                        continue

                    # group key (problem)
                    gkey = group_key_for(rec, line_idx)

                    p1 = rec.get("pass1", {})
                    p2 = rec.get("pass2", {})

                    ok1 = pass_correct_for_domain(p1, domain_name, args.carpark_success_op, args.carpark_soft_threshold)
                    ok2 = pass_correct_for_domain(p2, domain_name, args.carpark_success_op, args.carpark_soft_threshold)

                    bucket[(step, gkey)]["p1"].append(int(ok1))
                    # only count pass2 if present (avoid bias if totally missing)
                    if isinstance(p2, dict) and p2:
                        bucket[(step, gkey)]["p2"].append(int(ok2))
        except Exception:
            continue

    rows = []
    for (step, gkey), vals in bucket.items():
        p1_list = vals["p1"]
        p2_list = vals["p2"]
        if len(p1_list) == 0 or len(p2_list) == 0:
            # require both passes to have at least one sample
            continue
        p1_acc = float(np.mean(p1_list))
        p2_acc = float(np.mean(p2_list))
        rows.append({
            "domain": domain_name,
            "step": int(step),
            "problem_key": gkey,
            "n_p1": int(len(p1_list)),
            "n_p2": int(len(p2_list)),
            "p1_acc": p1_acc,
            "p2_acc": p2_acc,
            "raw_effect": p2_acc - p1_acc,
            "p1_any_correct": int(p1_acc > 0.0),
        })
    return pd.DataFrame(rows)

def bootstrap_ci(x: np.ndarray, B: int = 2000, ci: float = 0.95, seed: int = 123) -> Tuple[float, float]:
    if x.size == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = []
    n = len(x)
    for _ in range(B):
        sample = x[rng.integers(0, n, size=n)]
        means.append(float(np.nanmean(sample)))
    lo = np.quantile(means, (1.0-ci)/2.0)
    hi = np.quantile(means, 1.0-(1.0-ci)/2.0)
    return (float(lo), float(hi))

# ---------- Plot ----------
def minimal_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.25)

def plot_pass2_effects(df_all: pd.DataFrame, out_base: str, dpi: int = 600, title: str | None = None):
    domains = ["Carpark", "Crossword", "Math"]
    labels = {0: "P1 NONE", 1: "P1 ≥ 1"}
    colors = {0: "#1f77b4", 1: "#d62728"}  # blue / red-ish

    fig, axes = plt.subplots(3, 1, figsize=(6.2, 8.2), sharex=True, constrained_layout=True)

    for ax, dom in zip(axes, domains):
        minimal_axes(ax)
        sub = df_all[df_all["domain"] == dom].copy()
        if sub.empty:
            ax.text(0.5, 0.5, f"No data for {dom}", ha="center", va="center")
            continue

        # Two groups
        vals0 = sub.loc[sub["p1_any_correct"] == 0, "raw_effect"].to_numpy()
        vals1 = sub.loc[sub["p1_any_correct"] == 1, "raw_effect"].to_numpy()

        means = [np.nanmean(vals0) if vals0.size else np.nan, np.nanmean(vals1) if vals1.size else np.nan]
        cis = [bootstrap_ci(vals0), bootstrap_ci(vals1)]
        ns = [vals0.size, vals1.size]

        x = np.array([0, 1], dtype=float)
        y = np.array(means, dtype=float)
        yerr = np.array([[y[i]-cis[i][0] if not np.isnan(y[i]) else 0.0 for i in range(2)],
                         [cis[i][1]-y[i] if not np.isnan(y[i]) else 0.0 for i in range(2)]])

        # bars
        for i in range(2):
            ax.bar(x[i], y[i], width=0.55, color=colors[i], alpha=0.85, label=labels[i])
            # errorbar
            ax.errorbar(x[i], y[i], yerr=np.c_[y[i]-cis[i][0], cis[i][1]-y[i]].T,
                        fmt="none", ecolor="k", elinewidth=1.2, capsize=4, capthick=1.2)

            # annotate n
            ax.text(x[i], (0 if np.isnan(y[i]) else y[i]) + 0.02, f"n={ns[i]}", ha="center", va="bottom", fontsize=11)

        ax.axhline(0.0, color="k", lw=1, alpha=0.4)
        ax.set_ylabel("Raw effect:  $\\bar p_2 - \\bar p_1$")
        ax.set_title(dom, loc="left", fontsize=14, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_xticklabels([labels[0], labels[1]])

        # Optional: jittered points of per-problem raw effects (faint)
        jitter = (np.random.rand(sub.shape[0]) - 0.5) * 0.10
        for grp in (0, 1):
            pts = sub.loc[sub["p1_any_correct"] == grp, "raw_effect"].to_numpy()
            xx = np.full_like(pts, 0.0 if grp == 0 else 1.0, dtype=float) + ((np.random.rand(pts.size)-0.5)*0.20)
            ax.scatter(xx, pts, s=8, alpha=0.25, color=colors[grp], edgecolors="none")

        # y-limits a bit adaptive
        ymin = min(-0.05, np.nanmin(sub["raw_effect"]) - 0.02) if sub["raw_effect"].size else -0.05
        ymax = max( 0.05, np.nanmax(sub["raw_effect"]) + 0.05) if sub["raw_effect"].size else 0.05
        ax.set_ylim(ymin, ymax)

    axes[-1].set_xlabel("Pass-1 solvability bucket")

    if title:
        fig.suptitle(title, y=1.02, fontsize=14, fontweight="bold")

    fig.savefig(out_base + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    print(f"[ok] wrote {out_base}.png / .pdf")

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser()
    # roots
    ap.add_argument("--roots_crossword", nargs="+", default=[], help="Paths or globs to Crossword results")
    ap.add_argument("--roots_math",      nargs="+", default=[], help="Paths or globs to Math results")
    ap.add_argument("--roots_carpark",   nargs="+", default=[], help="Paths or globs to Carpark results")
    ap.add_argument("--split",           type=str,  default="test")

    # correctness policy for Carpark
    ap.add_argument("--carpark_success_op",    type=str, default="ge", choices=["ge","gt","le","lt"])
    ap.add_argument("--carpark_soft_threshold",type=float, default=0.1)

    # step filter
    ap.add_argument("--min_step", type=int, default=0)
    ap.add_argument("--max_step", type=int, default=1000)

    # pooling
    ap.add_argument("--pool_across_steps", action="store_true",
                    help="If set, average raw_effect across steps per (domain, problem) before summarizing/plotting.")

    # output
    ap.add_argument("--outdir",      type=str, default="graphs")
    ap.add_argument("--outfile_tag", type=str, default=None)
    ap.add_argument("--dpi",         type=int, default=600)
    ap.add_argument("--title",       type=str, default="Raw Effect of Pass-2 by Pass-1 Solvability")

    return ap.parse_args()

def main():
    args = parse_args()

    out_tag = args.outfile_tag or "combined"
    outdir = Path(args.outdir)
    (outdir / "tables").mkdir(parents=True, exist_ok=True)

    # Load per-problem rows for each domain
    dfs = []
    if args.roots_carpark:
        dfs.append(load_per_problem(args, "Carpark", args.roots_carpark))
    if args.roots_crossword:
        dfs.append(load_per_problem(args, "Crossword", args.roots_crossword))
    if args.roots_math:
        dfs.append(load_per_problem(args, "Math", args.roots_math))

    if not dfs:
        print("[error] No data found. Provide --roots_*.", file=sys.stderr)
        sys.exit(2)

    df = pd.concat(dfs, ignore_index=True)

    # Optional pooling across steps: average raw_effect per (domain, problem)
    if args.pool_across_steps:
        agg = (df
               .groupby(["domain", "problem_key"], as_index=False)
               .agg(n_p1=("n_p1","sum"),
                    n_p2=("n_p2","sum"),
                    p1_acc=("p1_acc","mean"),
                    p2_acc=("p2_acc","mean"),
                    raw_effect=("raw_effect","mean"),
                    p1_any_correct=("p1_any_correct","max")))
        df_plot = agg
    else:
        df_plot = df.copy()

    # Save per-problem table
    per_problem_csv = outdir / "tables" / f"pass2_per_problem_{out_tag}.csv"
    df_plot.to_csv(per_problem_csv, index=False)
    print(f"[ok] wrote {per_problem_csv}")

    # Summaries per domain & bucket
    rows = []
    for dom in ["Carpark", "Crossword", "Math"]:
        sub = df_plot[df_plot["domain"] == dom]
        for grp in [0, 1]:
            vals = sub.loc[sub["p1_any_correct"] == grp, "raw_effect"].to_numpy()
            mean = float(np.nanmean(vals)) if vals.size else np.nan
            lo, hi = bootstrap_ci(vals)
            rows.append({"domain": dom,
                         "bucket": "P1_NONE" if grp == 0 else "P1_ANY",
                         "n_problems": int(vals.size),
                         "mean_raw_effect": mean,
                         "ci_lo": lo, "ci_hi": hi})
    summary_df = pd.DataFrame(rows)
    summary_csv = outdir / "tables" / f"pass2_summary_{out_tag}.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"[ok] wrote {summary_csv}")

    # Plot
    out_base = str(outdir / f"pass2_raw_effects_{out_tag}")
    plot_pass2_effects(df_plot, out_base, dpi=args.dpi, title=args.title)

if __name__ == "__main__":
    main()
