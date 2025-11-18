#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
graph_3.py  (PASS1 ONLY, per-metric variants; accuracy + Reasoning Shift counts)
--------------------------------------------------------------------------------
Produces the same 3-panel plot (Carpark/Crossword/Math) for EACH of:
  • Answer Entropy (pass1)
  • Think  Entropy (pass1)
  • Answer+Think   (pass1; sum or mean via --combined_mode)

For each metric, writes TWO figures:
  1) Accuracy (%) vs entropy bins (Aha vs No Aha, side-by-side bars)
  2) Reasoning Shift COUNT vs entropy bins (Aha only)

NEW: Also writes numeric tables (CSV):
  • graphs/tables/graph_3_pass1_table_<metric>_<tag>__per_domain.csv
  • graphs/tables/graph_3_pass1_table_<metric>_<tag>__overall.csv
"""

import argparse, json, os, re, sys, glob as _glob
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ---------- Global typography (Times, size 14) & nice PDFs ----------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "pdf.fonttype": 42,  # embed TrueType in PDF
    "ps.fonttype": 42,
})

AHA_KEYS_CANON = ["change_way_of_thinking", "shift_in_reasoning_v1"]
AHA_KEYS_BROAD = AHA_KEYS_CANON + ["shift_llm","shift_gpt","pivot_llm","rechecked"]

# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser()
    # Multiple roots per domain (paths or globs)
    ap.add_argument("--roots_crossword", nargs="+", default=[], help="Paths or globs to xword results")
    ap.add_argument("--roots_math",      nargs="+", default=[], help="Paths or globs to math results")
    ap.add_argument("--roots_carpark",   nargs="+", default=[], help="Paths or globs to carpark results")
    ap.add_argument("--ignore_glob",     nargs="*", default=["*compare-1shot*"], help="Glob patterns to exclude")

    # Optional: include only certain temperatures by path match 'temp-<value>'
    ap.add_argument("--only_temps", nargs="*", default=None,
                    help="Keep files whose path contains temp-<value> (e.g., 0.7)")

    ap.add_argument("--split",    type=str, default="test")
    ap.add_argument("--gpt_mode", type=str, default="canonical", choices=["canonical","broad"])

    # Carpark soft-score gating (pass1)
    ap.add_argument("--carpark_success_op",    type=str, default="ge", choices=["ge","gt","le","lt"])
    ap.add_argument("--carpark_soft_threshold",type=float, default=0.1)

    # Binning & range
    ap.add_argument("--bins",       type=int, default=10)
    ap.add_argument("--binning",    type=str, default="uniform", choices=["uniform","quantile"])
    ap.add_argument("--share_bins", type=str, default="global",   choices=["global","per_domain"])
    ap.add_argument("--entropy_min", type=float, default=None, help="Fixed min for bins (e.g., 0.0)")
    ap.add_argument("--entropy_max", type=float, default=None, help="Fixed max for bins (e.g., 4.0)")

    # Step filter
    ap.add_argument("--min_step",   type=int, default=0)
    ap.add_argument("--max_step",   type=int, default=1000)

    # Figure & output
    ap.add_argument("--outdir",      type=str, default="graphs")
    ap.add_argument("--outfile_tag", type=str, default=None)
    ap.add_argument("--dpi",         type=int, default=300)
    ap.add_argument("--width_in",    type=float, default=5.5)
    ap.add_argument("--height_in",   type=float, default=8.0)
    ap.add_argument("--title",       type=str, default=None)
    ap.add_argument("--y_pad",       type=float, default=6.0,
                    help="headroom above local ymax for Accuracy plots, in percentage points")

    # Colors
    ap.add_argument("--cmap",        type=str, default="YlGnBu",
                    help="Matplotlib colormap (e.g., YlGnBu, PuBuGn, cividis, magma).")

    # Which metrics to render & combine rule for answer+think
    ap.add_argument("--which_metrics", nargs="+",
                    choices=["answer","think","answer_plus"], default=["answer","think","answer_plus"],
                    help="Which plots to render")
    ap.add_argument("--combined_mode", type=str, default="sum", choices=["sum","mean"],
                    help="For answer_plus: sum or mean of pass1 answer & think entropies")

    return ap.parse_args()

# ──────────────────────────────────────────────────────────────────────────────
def truthy(x):
    if x is True: return True
    if isinstance(x, (int,float)): return x != 0
    if isinstance(x, str): return x.strip().lower() in {"1","true","yes","y"}
    return False

def detect_aha_pass1(rec: Dict[str,Any], mode: str) -> int:
    keys = AHA_KEYS_CANON if mode == "canonical" else AHA_KEYS_BROAD
    p1 = rec.get("pass1", {})
    if not isinstance(p1, dict): return 0
    return 1 if any(truthy(p1.get(k, False)) for k in keys) else 0

def extract_step(rec: Dict[str,Any], src_path: str) -> int:
    if isinstance(rec.get("step"), (int,float)): return int(rec["step"])
    m = re.search(r"step[-_]?(\d{1,5})", src_path) or re.search(r"global_step[-_]?(\d{1,5})", src_path)
    return int(m.group(1)) if m else 0

# ----- PASS1 entropy extractors -----
def _num(d: Dict[str,Any], names: List[str]) -> float | None:
    if not isinstance(d, dict): return None
    for n in names:
        v = d.get(n, None)
        if isinstance(v, (int,float)): return float(v)
    return None

def extract_pass1_answer_entropy(rec: Dict[str,Any]) -> float | None:
    p1 = rec.get("pass1", {})
    v = _num(p1, ["answer_entropy","entropy_answer"])
    if v is not None: return v
    tok = p1.get("answer_token_entropies") or p1.get("token_entropies")
    if isinstance(tok, list) and tok:
        try:
            vals = [float(t) for t in tok]
            if vals: return float(np.mean(vals))
        except Exception:
            pass
    return None

def extract_pass1_think_entropy(rec: Dict[str,Any]) -> float | None:
    p1 = rec.get("pass1", {})
    v = _num(p1, ["entropy_think","think_entropy"])
    if v is not None: return v
    tok = p1.get("think_token_entropies")
    if isinstance(tok, list) and tok:
        try:
            vals = [float(t) for t in tok]
            if vals: return float(np.mean(vals))
        except Exception:
            pass
    return None

def extract_pass1_answer_plus(rec: Dict[str,Any], mode: str) -> float | None:
    a = extract_pass1_answer_entropy(rec)
    t = extract_pass1_think_entropy(rec)
    if a is None and t is None: return None
    if a is None: return t
    if t is None: return a
    return (a + t) if mode == "sum" else (a + t)/2.0

# ----- correctness (PASS1) -----
def carpark_correct_pass1(rec: Dict[str,Any], op: str, thr: float) -> bool:
    p1 = rec.get("pass1", {})
    if not isinstance(p1, dict): return False
    def decide(x: float) -> bool:
        return (x >= thr) if op=="ge" else (x > thr) if op=="gt" else (x <= thr) if op=="le" else (x < thr)
    for key in ("soft_1","soft_reward"):
        v = p1.get(key, None)
        if isinstance(v, (int,float)): return decide(float(v))
    for k in ("is_correct","correct","correct_exact","is_correct_pred"):
        if k in p1: return truthy(p1[k])
    return False

def general_correct_pass1(rec: Dict[str,Any]) -> bool:
    p1 = rec.get("pass1", {})
    if not isinstance(p1, dict): return False
    for k in ("is_correct","correct","correct_exact","is_correct_pred"):
        if k in p1: return truthy(p1[k])
    return False

# ----- file discovery helpers -----
def expand_paths(paths: List[str]) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        if any(ch in p for ch in "*?[]"):
            out += [Path(x) for x in _glob.glob(p)]
        else:
            out.append(Path(p))
    return [p for p in out if p.exists() and p.is_dir()]

def ignored(path: str, ignore_globs: List[str]) -> bool:
    from fnmatch import fnmatch
    return any(fnmatch(path, pat) for pat in ignore_globs)

def temp_match(path: str, only_temps: List[str] | None) -> bool:
    if not only_temps:
        return True
    cands: List[str] = []
    for t in only_temps:
        s = str(t).strip()
        try:
            f = float(s)
            cands += [f"{f:g}", f"{f:.2f}", f"{f:.1f}"]
        except Exception:
            cands.append(s)
    return any(f"temp-{c}" in path for c in cands)

def iter_jsonl_files_many(roots: List[str], ignore_globs: List[str], only_temps: List[str] | None) -> Iterable[str]:
    for root in expand_paths(roots):
        for f in root.rglob("*.jsonl"):
            fp = str(f)
            if ignored(fp, ignore_globs):     continue
            if not temp_match(fp, only_temps): continue
            yield fp

# ----- loading per metric -----
def load_rows_from_roots_metric(
    roots: List[str], domain: str, split: str, min_step: int, max_step: int,
    gpt_mode: str, carpark_op: str, carpark_thr: float,
    ignore_globs: List[str], only_temps: List[str] | None,
    metric: str, combined_mode: str
) -> List[Dict[str,Any]]:
    data: List[Dict[str,Any]] = []
    if not roots: return data
    for fp in iter_jsonl_files_many(roots, ignore_globs, only_temps):
        try:
            with open(fp, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line: continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if split and rec.get("split") and str(rec["split"]) != split:
                        continue
                    step = extract_step(rec, fp)
                    if step < min_step or step > max_step:
                        continue

                    if metric == "answer":
                        e = extract_pass1_answer_entropy(rec)
                    elif metric == "think":
                        e = extract_pass1_think_entropy(rec)
                    else:
                        e = extract_pass1_answer_plus(rec, combined_mode)

                    if e is None:
                        continue
                    aha = detect_aha_pass1(rec, gpt_mode)
                    if domain == "Carpark":
                        corr = carpark_correct_pass1(rec, carpark_op, carpark_thr)
                    else:
                        corr = general_correct_pass1(rec)
                    data.append({"domain":domain, "entropy":float(e), "aha":int(aha), "correct":int(corr)})
        except Exception:
            continue
    return data

# ----- binning -----
def compute_edges(entropies: np.ndarray, bins: int, binning: str,
                  entropy_min: float | None, entropy_max: float | None) -> np.ndarray:
    if (entropy_min is not None) and (entropy_max is not None):
        if entropy_max <= entropy_min:
            raise ValueError("--entropy_max must be > --entropy_min")
        return np.linspace(float(entropy_min), float(entropy_max), bins+1)
    if entropies.size == 0:
        return np.array([0.0, 1.0])
    if binning == "quantile":
        qs = np.linspace(0.0, 1.0, bins+1)
        edges = np.unique(np.quantile(entropies, qs))
        if len(edges) < 3:
            e_min = float(entropies.min()) if entropy_min is None else float(entropy_min)
            e_max = float(entropies.max()) if entropy_max is None else float(entropy_max)
            if e_max <= e_min: e_max = e_min + 1e-6
            edges = np.linspace(e_min, e_max, bins+1)
    else:
        e_min = float(entropies.min()) if entropy_min is None else float(entropy_min)
        e_max = float(entropies.max()) if entropy_max is None else float(entropy_max)
        if e_max <= e_min: e_max = e_min + 1e-6
        edges = np.linspace(e_min, e_max, bins+1)
    return edges

def binned_accuracy(ent: np.ndarray, aha: np.ndarray, corr: np.ndarray,
                    edges: np.ndarray, aha_flag: int):
    ent_clip = np.clip(ent, edges[0], np.nextafter(edges[-1], -np.inf))
    idx = np.digitize(ent_clip, edges) - 1
    nb = len(edges)-1
    acc = np.full(nb, np.nan, dtype=float)
    for b in range(nb):
        m = (idx == b) & (aha == aha_flag)
        n = int(m.sum())
        if n > 0:
            acc[b] = float(corr[m].sum()) / n
    centers = 0.5*(edges[:-1] + edges[1:])
    return centers, acc

def binned_aha_counts(ent: np.ndarray, aha: np.ndarray, edges: np.ndarray):
    ent_clip = np.clip(ent, edges[0], np.nextafter(edges[-1], -np.inf))
    idx = np.digitize(ent_clip, edges) - 1
    nb = len(edges)-1
    counts = np.zeros(nb, dtype=int)
    for b in range(nb):
        m = (idx == b) & (aha == 1)
        counts[b] = int(m.sum())
    centers = 0.5*(edges[:-1] + edges[1:])
    return centers, counts

# ----- build per-bin table (NEW) -----
def compute_bin_table(metric_name: str, scope: str, domain: str,
                      ent: np.ndarray, aha: np.ndarray, corr: np.ndarray,
                      edges: np.ndarray) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per entropy bin:
      metric, scope, domain, bin_lo, bin_hi, bin_center,
      n_total, n_aha, n_noaha, pct_aha, acc_noshift, acc_shift
    """
    if ent.size == 0:
        return pd.DataFrame(columns=[
            "metric","scope","domain","bin_lo","bin_hi","bin_center",
            "n_total","n_aha","n_noaha","pct_aha","acc_noshift","acc_shift"
        ])

    ent_clip = np.clip(ent, edges[0], np.nextafter(edges[-1], -np.inf))
    idx = np.digitize(ent_clip, edges) - 1
    nb = len(edges) - 1

    rows = []
    for b in range(nb):
        mask_bin = (idx == b)
        n_total = int(mask_bin.sum())
        if n_total == 0:
            rows.append([metric_name, scope, domain, float(edges[b]), float(edges[b+1]),
                         float(0.5*(edges[b]+edges[b+1])), 0, 0, 0, np.nan, np.nan, np.nan])
            continue
        aha_bin = aha[mask_bin]
        corr_bin = corr[mask_bin]
        n_aha = int((aha_bin == 1).sum())
        n_no  = n_total - n_aha
        pct_aha = (n_aha / n_total) if n_total > 0 else np.nan

        acc_shift   = float(corr_bin[aha_bin == 1].mean()) if n_aha > 0 else np.nan
        acc_noshift = float(corr_bin[aha_bin == 0].mean()) if n_no  > 0 else np.nan

        rows.append([metric_name, scope, domain, float(edges[b]), float(edges[b+1]),
                     float(0.5*(edges[b]+edges[b+1])), n_total, n_aha, n_no, pct_aha, acc_noshift, acc_shift])

    df = pd.DataFrame(rows, columns=[
        "metric","scope","domain","bin_lo","bin_hi","bin_center",
        "n_total","n_aha","n_noaha","pct_aha","acc_noshift","acc_shift"
    ])

    # TOTAL row across all bins (pooled)
    n_total = int(len(ent))
    n_aha   = int((aha == 1).sum())
    n_no    = n_total - n_aha
    pct_aha = (n_aha / n_total) if n_total > 0 else np.nan
    acc_shift_all   = float(corr[aha == 1].mean()) if n_aha > 0 else np.nan
    acc_noshift_all = float(corr[aha == 0].mean()) if n_no  > 0 else np.nan

    total_row = pd.DataFrame([[
        metric_name, scope, domain, np.nan, np.nan, np.nan,
        n_total, n_aha, n_no, pct_aha, acc_noshift_all, acc_shift_all
    ]], columns=df.columns)
    total_row["bin_label"] = "TOTAL"
    df["bin_label"] = df.apply(lambda r: f"[{r.bin_lo:.2f},{r.bin_hi:.2f}]", axis=1)
    df = pd.concat([df, total_row], ignore_index=True)
    return df

# ----- aesthetics & saving -----
def minimal_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.25)

def save_all_formats(fig, out_base: str, dpi: int = 300):
    png_path = f"{out_base}.png"
    pdf_path = f"{out_base}.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"[ok] wrote {png_path}")
    print(f"[ok] wrote {pdf_path}")

# ----- renderers -----
def render_metric_accuracy(args, metric: str, color_noaha, color_aha):
    rows: List[Dict[str,Any]] = []
    rows += load_rows_from_roots_metric(args.roots_carpark,   "Carpark",  args.split, args.min_step, args.max_step,
                                        args.gpt_mode, args.carpark_success_op, args.carpark_soft_threshold,
                                        args.ignore_glob, args.only_temps, metric, args.combined_mode)
    rows += load_rows_from_roots_metric(args.roots_crossword, "Crossword",args.split, args.min_step, args.max_step,
                                        args.gpt_mode, args.carpark_success_op, args.carpark_soft_threshold,
                                        args.ignore_glob, args.only_temps, metric, args.combined_mode)
    rows += load_rows_from_roots_metric(args.roots_math,      "Math",     args.split, args.min_step, args.max_step,
                                        args.gpt_mode, args.carpark_success_op, args.carpark_soft_threshold,
                                        args.ignore_glob, args.only_temps, metric, args.combined_mode)
    if not rows:
        print(f"[warn] No PASS1 records with {metric} entropy found (accuracy).", file=sys.stderr)
        return False

    domains = ["Carpark","Crossword","Math"]
    per_dom = {d: {"entropy":[], "aha":[], "correct":[]} for d in domains}
    for r in rows:
        per_dom[r["domain"]]["entropy"].append(r["entropy"])
        per_dom[r["domain"]]["aha"].append(r["aha"])
        per_dom[r["domain"]]["correct"].append(r["correct"])
    for d in domains:
        for k in per_dom[d]:
            per_dom[d][k] = np.asarray(per_dom[d][k], dtype=float if k=="entropy" else int)

    # Build edges
    if args.share_bins == "global":
        all_ent = np.concatenate([per_dom[d]["entropy"] for d in domains if per_dom[d]["entropy"].size])
        global_edges = compute_edges(all_ent, args.bins, args.binning, args.entropy_min, args.entropy_max)
        edges_by_dom = {d: global_edges for d in domains}
        overall_edges = global_edges
    else:
        edges_by_dom = {d: compute_edges(per_dom[d]["entropy"], args.bins, args.binning,
                                         args.entropy_min, args.entropy_max) for d in domains}
        # For the "ALL" table we still want global edges so bins align when summing
        all_ent = np.concatenate([per_dom[d]["entropy"] for d in domains if per_dom[d]["entropy"].size])
        overall_edges = compute_edges(all_ent, args.bins, args.binning, args.entropy_min, args.entropy_max)

    x_label = {"answer":"Answer Entropy (Binned)",
               "think":"Think Entropy (Binned)",
               "answer_plus":"Answer+Think Entropy (Binned)"}[metric]
    file_tag = {"answer":"answer", "think":"think", "answer_plus":"answer_plus"}[metric]

    # -------- Tables (NEW) ----------
    tag = args.outfile_tag or "combined"
    tables_dir = Path(args.outdir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Per-domain tables
    per_domain_tables = []
    for d in domains:
        ent = per_dom[d]["entropy"]; aha = per_dom[d]["aha"]; corr = per_dom[d]["correct"]
        edges = edges_by_dom[d]
        df_d = compute_bin_table(metric, scope="domain", domain=d, ent=ent, aha=aha, corr=corr, edges=edges)
        per_domain_tables.append(df_d)
    df_per_domain = pd.concat(per_domain_tables, ignore_index=True)
    df_per_domain.to_csv(tables_dir / f"graph_3_pass1_table_{file_tag}_{tag}__per_domain.csv", index=False)

    # Overall (ALL domains pooled) using overall_edges
    ent_all = np.concatenate([per_dom[d]["entropy"] for d in domains if per_dom[d]["entropy"].size])
    aha_all = np.concatenate([per_dom[d]["aha"]     for d in domains if per_dom[d]["aha"].size])
    corr_all= np.concatenate([per_dom[d]["correct"] for d in domains if per_dom[d]["correct"].size])
    df_overall = compute_bin_table(metric, scope="overall", domain="ALL",
                                   ent=ent_all, aha=aha_all, corr=corr_all, edges=overall_edges)
    df_overall.to_csv(tables_dir / f"graph_3_pass1_table_{file_tag}_{tag}__overall.csv", index=False)

    # -------- Plots (unchanged except small refactor) ----------
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(args.width_in, args.height_in),
                             sharex=True, constrained_layout=True)

    for ax, d in zip(axes, domains):
        ent = per_dom[d]["entropy"]; aha = per_dom[d]["aha"]; corr = per_dom[d]["correct"]
        minimal_axes(ax)
        if ent.size == 0:
            ax.text(0.5, 0.5, f"No data for {d}", ha="center", va="center")
            continue
        edges = edges_by_dom[d]
        x0, acc_no  = binned_accuracy(ent, aha, corr, edges, aha_flag=0)
        x1, acc_yes = binned_accuracy(ent, aha, corr, edges, aha_flag=1)

        acc_no_pct  = np.array(acc_no,  dtype=float) * 100.0
        acc_yes_pct = np.array(acc_yes, dtype=float) * 100.0

        w = (x0[1]-x0[0])*0.80 if len(x0) > 1 else 0.5
        ax.bar(x0 - w/4, acc_no_pct,  width=w/2, label="No Aha", color=color_noaha)
        ax.bar(x0 + w/4, acc_yes_pct, width=w/2, label="Aha",    color=color_aha)

        vals = np.concatenate([acc_no_pct[~np.isnan(acc_no_pct)], acc_yes_pct[~np.isnan(acc_yes_pct)]])
        if vals.size:
            ymax = float(vals.max())
            ax.set_ylim(0, min(100.0, ymax + args.y_pad))
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(d, loc="left", fontsize=14, fontweight="bold")

    # Tick labels from Carpark edges (just to standardize)
    ref_edges = edges_by_dom["Carpark"]
    centers = 0.5*(ref_edges[:-1] + ref_edges[1:])
    tick_labels = [f"[{ref_edges[i]:.2f},{ref_edges[i+1]:.2f}]" for i in range(len(ref_edges)-1)]
    axes[-1].set_xticks(centers)
    axes[-1].set_xticklabels(tick_labels, rotation=45, ha="right")
    axes[-1].set_xlabel(x_label)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    if args.title:
        fig.suptitle(args.title, y=1.02, fontsize=14, fontweight="bold")

    out_base = os.path.join(args.outdir, f"graph_3_pass1_bins_{file_tag}_{tag}")
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    save_all_formats(fig, out_base, dpi=args.dpi)
    return True

def render_metric_counts(args, metric: str, color_aha):
    # Same rows / edges as accuracy, but Aha counts per bin
    rows: List[Dict[str,Any]] = []
    rows += load_rows_from_roots_metric(args.roots_carpark,   "Carpark",  args.split, args.min_step, args.max_step,
                                        args.gpt_mode, args.carpark_success_op, args.carpark_soft_threshold,
                                        args.ignore_glob, args.only_temps, metric, args.combined_mode)
    rows += load_rows_from_roots_metric(args.roots_crossword, "Crossword",args.split, args.min_step, args.max_step,
                                        args.gpt_mode, args.carpark_success_op, args.carpark_soft_threshold,
                                        args.ignore_glob, args.only_temps, metric, args.combined_mode)
    rows += load_rows_from_roots_metric(args.roots_math,      "Math",     args.split, args.min_step, args.max_step,
                                        args.gpt_mode, args.carpark_success_op, args.carpark_soft_threshold,
                                        args.ignore_glob, args.only_temps, metric, args.combined_mode)
    if not rows:
        print(f"[warn] No PASS1 records with {metric} entropy found (counts).", file=sys.stderr)
        return False

    domains = ["Carpark","Crossword","Math"]
    per_dom = {d: {"entropy":[], "aha":[]} for d in domains}
    for r in rows:
        per_dom[r["domain"]]["entropy"].append(r["entropy"])
        per_dom[r["domain"]]["aha"].append(r["aha"])
    for d in domains:
        per_dom[d]["entropy"] = np.asarray(per_dom[d]["entropy"], dtype=float)
        per_dom[d]["aha"]     = np.asarray(per_dom[d]["aha"], dtype=int)

    # Edges
    if args.share_bins == "global":
        all_ent = np.concatenate([per_dom[d]["entropy"] for d in domains if per_dom[d]["entropy"].size])
        global_edges = compute_edges(all_ent, args.bins, args.binning, args.entropy_min, args.entropy_max)
        edges_by_dom = {d: global_edges for d in domains}
    else:
        edges_by_dom = {d: compute_edges(per_dom[d]["entropy"], args.bins, args.binning,
                                         args.entropy_min, args.entropy_max) for d in domains}

    x_label = {"answer":"Answer Entropy (Binned)",
               "think":"Think Entropy (Binned)",
               "answer_plus":"Answer+Think Entropy (Binned)"}[metric]
    file_tag = {"answer":"answer", "think":"think", "answer_plus":"answer_plus"}[metric]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(args.width_in, args.height_in),
                             sharex=True, constrained_layout=True)

    for ax, d in zip(axes, domains):
        ent = per_dom[d]["entropy"]; aha = per_dom[d]["aha"]
        minimal_axes(ax)
        if ent.size == 0:
            ax.text(0.5, 0.5, f"No data for {d}", ha="center", va="center")
            continue

        edges = edges_by_dom[d]
        x, counts = binned_aha_counts(ent, aha, edges)

        w = (x[1]-x[0])*0.80 if len(x) > 1 else 0.5
        ax.bar(x, counts, width=w, label="Aha", color=color_aha)

        if counts.size:
            ymax = float(counts.max())
            ax.set_ylim(0, ymax*1.06 + (1.0 if ymax < 10 else 0.0))
        # Two-line y-label
        lab = ax.set_ylabel("Reasoning Shift\n(Count)")
        lab.set_multialignment("center")
        ax.set_title(d, loc="left", fontsize=14, fontweight="bold")

    ref_edges = edges_by_dom["Carpark"]
    centers = 0.5*(ref_edges[:-1] + ref_edges[1:])
    tick_labels = [f"[{ref_edges[i]:.2f},{ref_edges[i+1]:.2f}]" for i in range(len(ref_edges)-1)]
    axes[-1].set_xticks(centers)
    axes[-1].set_xticklabels(tick_labels, rotation=45, ha="right")
    axes[-1].set_xlabel(x_label)

    if args.title:
        fig.suptitle(args.title, y=1.02, fontsize=14, fontweight="bold")

    tag = args.outfile_tag or "combined"
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    out_base = os.path.join(args.outdir, f"graph_3_pass1_counts_{file_tag}_{tag}")
    save_all_formats(fig, out_base, dpi=args.dpi)
    return True

def main():
    args = parse_args()

    try:
        cmap = plt.get_cmap(args.cmap)
    except Exception:
        cmap = plt.get_cmap("YlGnBu")
    color_noaha = cmap(0.35)  # lighter
    color_aha   = cmap(0.75)  # darker

    anything = False
    for metric in args.which_metrics:
        a_ok = render_metric_accuracy(args, metric, color_noaha, color_aha)
        c_ok = render_metric_counts(args,   metric, color_aha)
        anything = anything or a_ok or c_ok

    if not anything:
        sys.exit(2)

if __name__ == "__main__":
    main()
