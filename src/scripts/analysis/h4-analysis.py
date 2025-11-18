
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extra figures for artificial recheck analysis.

Reads raw logs (same format as compare_artificial_recheck.py expects) and emits:
  • accuracy_by_step_pass1_pass2.png
  • entropy_before_after_by_step.png
  • heatmap_step_entropy_improvement.png  (P(correct after recheck | p1 wrong, artificial))
  • p1_entropy_hist.png, p2_entropy_hist.png
  • delta_correct_scatter.png  (p1_entropy vs (p2_correct - p1_correct))

Usage:
  python more_recheck_pictures.py /path/to/results_root --split test --out_dir /tmp/recheck_extra
"""

import os, re, json, argparse, math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEP_PAT = re.compile(r"step(\d+)", re.I)

def nat_step_from_path(path: str) -> Optional[int]:
    m = STEP_PAT.search(path); return int(m.group(1)) if m else None

def scan_files(root: str, split_substr: Optional[str]) -> List[str]:
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if not fn.endswith(".jsonl"): continue
            if split_substr and split_substr not in fn: continue
            out.append(os.path.join(dp, fn))
    out.sort(); return out

def coerce_bool(x) -> Optional[int]:
    if x is None: return None
    if isinstance(x, bool): return int(x)
    if isinstance(x, (int, np.integer)): return int(bool(x))
    if isinstance(x, str):
        s=x.strip().lower()
        if s in ("1","true","t","yes","y"): return 1
        if s in ("0","false","f","no","n"): return 0
    try: return int(bool(x))
    except: return None

def get_correct(d: Dict[str, Any]) -> Optional[int]:
    for k in ("is_correct_after_reconsideration", "is_correct_pred"):
        v = d.get(k); cb = coerce_bool(v)
        if cb is not None: return int(cb)
    return None

def get_entropy(d: Dict[str, Any]) -> Optional[float]:
    v = d.get("entropy")
    try: return float(v) if v is not None else None
    except: return None

def is_artificial_recheck(p2: Dict[str, Any]) -> int:
    markers = p2.get("reconsider_markers") or []
    if isinstance(markers, list) and "injected_cue" in markers: return 1
    return int(bool(coerce_bool(p2.get("has_reconsider_cue"))))

def load_pairs(files: List[str]) -> pd.DataFrame:
    rows = []
    for path in files:
        s_from = nat_step_from_path(path)
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln: continue
                try: rec = json.loads(ln)
                except: continue
                step = rec.get("step", s_from if s_from is not None else None)
                if step is None: continue
                prob = rec.get("problem") or rec.get("clue") or rec.get("row_key") or f"idx:{rec.get('dataset_index')}"
                p1 = rec.get("pass1") or {}
                p2 = rec.get("pass2") or {}
                p1c = get_correct(p1); p2c = get_correct(p2)
                if p1c is None or p2c is None: continue
                rows.append({
                    "step": int(step),
                    "problem": str(prob),
                    "sample_idx": rec.get("sample_idx"),
                    "p1_correct": int(p1c),
                    "p2_correct": int(p2c),
                    "p1_entropy": get_entropy(p1),
                    "p2_entropy": get_entropy(p2),
                    "artificial": is_artificial_recheck(p2),
                    "source_file": path,
                })
    d = pd.DataFrame(rows)
    for c in ["p1_entropy", "p2_entropy"]:
        if c in d.columns: d[c] = pd.to_numeric(d[c], errors="coerce")
    return d

def wilson_ci(k: int, n: int) -> Tuple[float,float]:
    if n<=0: return (float("nan"), float("nan"))
    z=1.96; p=k/n; z2=z*z; den=1.0+z2/n
    center=(p + z2/(2*n))/den
    half=(z*math.sqrt((p*(1-p)/n) + (z2/(4*n*n))))/den
    return (max(0.0, center-half), min(1.0, center+half))

def plot_accuracy_by_step(df: pd.DataFrame, out_png: str):
    g = df.groupby("step", as_index=False).agg(
        n=("p1_correct","size"),
        acc1=("p1_correct","mean"),
        acc2=("p2_correct","mean")
    ).sort_values("step")
    fig, ax = plt.subplots(figsize=(7.8,4.6), dpi=140)
    ax.plot(g["step"], g["acc1"], marker="o", label="PASS-1")
    ax.plot(g["step"], g["acc2"], marker="o", label="PASS-2")
    ax.set_xlabel("Training step"); ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by step (PASS-1 vs PASS-2)")
    ax.grid(True, alpha=0.3); ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)

def plot_entropy_by_step(df: pd.DataFrame, out_png: str):
    g = df.groupby("step", as_index=False).agg(
        p1=("p1_entropy","mean"),
        p2=("p2_entropy","mean")
    ).sort_values("step")
    fig, ax = plt.subplots(figsize=(7.8,4.6), dpi=140)
    ax.plot(g["step"], g["p1"], marker="o", label="PASS-1 entropy")
    ax.plot(g["step"], g["p2"], marker="o", label="PASS-2 entropy")
    ax.set_xlabel("Training step"); ax.set_ylabel("Mean entropy")
    ax.set_title("Entropy before vs after (by step)")
    ax.grid(True, alpha=0.3); ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)

def plot_heatmap_step_entropy_improve(df: pd.DataFrame, out_png: str, bins: int = 8):
    sub = df[(df["artificial"]==1) & (df["p1_correct"]==0)].copy()
    sub = sub[np.isfinite(sub["p1_entropy"])]
    if sub.empty:
        return
    # Bin entropy into quantiles
    sub["q"] = pd.qcut(sub["p1_entropy"], q=bins, duplicates="drop")
    # Map bins to indices and midpoints
    cats = sub["q"].cat.categories
    mids = []
    for c in cats:
        mids.append(0.5*(c.left + c.right))
    bin_index = {cats[i]:i for i in range(len(cats))}
    sub["bin"] = sub["q"].map(bin_index)
    g = sub.groupby(["step","bin"], as_index=False).agg(
        rate=("p2_correct","mean"),
        n=("p2_correct","size")
    )
    steps = np.sort(g["step"].unique())
    B = len(cats)
    M = np.full((len(steps), B), np.nan)
    for i, s in enumerate(steps):
        row = g[g["step"]==s]
        for _, r in row.iterrows():
            M[i, int(r["bin"])] = r["rate"]
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=150)
    im = ax.imshow(M, aspect="auto", origin="lower", interpolation="nearest")
    ax.set_yticks(np.arange(len(steps))); ax.set_yticklabels(steps)
    ax.set_xticks(np.arange(B)); ax.set_xticklabels([f"{m:.2f}" for m in mids], rotation=45, ha="right")
    ax.set_xlabel("PASS-1 entropy (bin center)"); ax.set_ylabel("Training step")
    ax.set_title("Correction rate after artificial recheck (p1 wrong)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(correct)")
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)

def plot_entropy_hists(df: pd.DataFrame, out_dir: str):
    for col, fn in [("p1_entropy","p1_entropy_hist.png"), ("p2_entropy","p2_entropy_hist.png")]:
        x = pd.to_numeric(df[col], errors="coerce").to_numpy()
        x = x[np.isfinite(x)]
        if x.size == 0: continue
        fig, ax = plt.subplots(figsize=(6.0,4.2), dpi=140)
        ax.hist(x, bins=40, alpha=0.9)
        ax.set_xlabel(col); ax.set_ylabel("count"); ax.set_title(f"{col} distribution")
        ax.grid(True, alpha=0.2)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, fn)); plt.close(fig)

def plot_delta_correct_scatter(df: pd.DataFrame, out_png: str):
    d = df.copy()
    d["delta"] = d["p2_correct"] - d["p1_correct"]
    x = pd.to_numeric(d["p1_entropy"], errors="coerce").to_numpy()
    y = pd.to_numeric(d["delta"], errors="coerce").to_numpy()
    sel = np.isfinite(x) & np.isfinite(y)
    x = x[sel]; y = y[sel]
    if x.size == 0: return
    fig, ax = plt.subplots(figsize=(6.4,4.2), dpi=140)
    ax.scatter(x, y, s=12, alpha=0.6)
    ax.set_xlabel("PASS-1 entropy"); ax.set_ylabel("Δ correctness (PASS-2 - PASS-1)")
    ax.set_title("Sample-level change vs PASS-1 entropy")
    ax.grid(True, alpha=0.2)
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_root")
    ap.add_argument("--split", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--bins", type=int, default=8)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.results_root, "artificial_recheck_extra")
    os.makedirs(out_dir, exist_ok=True)

    files = scan_files(args.results_root, args.split)
    if not files:
        raise SystemExit("No JSONL files found.")

    df = load_pairs(files)
    if df.empty:
        raise SystemExit("No comparable rows parsed.")

    plot_accuracy_by_step(df, os.path.join(out_dir, "accuracy_by_step_pass1_pass2.png"))
    plot_entropy_by_step(df, os.path.join(out_dir, "entropy_before_after_by_step.png"))
    plot_heatmap_step_entropy_improve(df, os.path.join(out_dir, "heatmap_step_entropy_improvement.png"), bins=args.bins)
    plot_entropy_hists(df, out_dir)
    plot_delta_correct_scatter(df, os.path.join(out_dir, "delta_correct_scatter.png"))

    # Bonus: write the parsed pairs for downstream use
    df.to_csv(os.path.join(out_dir, "pairs_table.csv"), index=False)
    print("Wrote figures to", out_dir)

if __name__ == "__main__":
    main()
