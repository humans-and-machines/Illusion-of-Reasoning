#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Forced Aha Effect — analysis + publication figures.

Adds:
  • Pastel1 palette for uncertainty & stepwise figures
  • Clearer labels (y: "Delta Accuracy (pp) for second pass")
  • Series names: "Per-draw (single completion)", "Per-problem mean of 8 (avg of 8 completions)"
  • New combined figure: uncertainty buckets (left) + stepwise overlay of all three metrics (right)

Figures (saved under <out_dir>/figures):
  - overall_deltas.{png,pdf}
  - stepwise_delta_sample.{png,pdf}
  - stepwise_delta_cluster_any.{png,pdf}
  - stepwise_delta_cluster_mean.{png,pdf}
  - stepwise_overlay.{png,pdf}                  <-- NEW (all three deltas in one subplot)
  - any_conversion_waterfall.{png,pdf}
  - any_headroom_scatter.{png,pdf}
  - uncertainty_buckets.{png,pdf}               <-- updated labels & Pastel1 by default
  - uncertainty_and_stepwise.{png,pdf}          <-- NEW two-subfigure figure
  - overview_deltas_and_waterfall.{png,pdf}
"""

from __future__ import annotations

import os, re, json, argparse, random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# ===================== Matplotlib (Times everywhere) =====================
import matplotlib
# Force Times / Times New Roman across all figures
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,            # embed TrueType (editable text in PDF)
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"],
    "mathtext.fontset": "stix",                        # Times-like math without usetex
    "font.size": 12,
    "axes.titlesize": 12, "axes.labelsize": 12,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 12, "figure.titlesize": 12,
})
import matplotlib.pyplot as plt

# ===================== Paper labels & palette (for overview/waterfall) =====================

METRIC_LABELS: Dict[str, str] = {
    "sample": "Per-draw accuracy",
    "cluster_mean": "Per-problem mean of 8",
    "cluster_any": "Per-problem any-correct",
}

DEFAULT_COLORS: Dict[str, str] = {
    "bar_primary":   "#2C7BB6",  # overview bars
    "bar_secondary": "#FDAE61",
    "bar_tertiary":  "#ABD9E9",
    "gain":          "#1B7837",  # Incorrect→Correct
    "loss":          "#D73027",  # Correct→Incorrect
    "stable_pos":    "#4D4D4D",  # Stayed Correct
    "stable_neg":    "#BDBDBD",  # Stayed Incorrect
}

def _parse_colors(s: Optional[str]) -> Dict[str, str]:
    if not s:
        return {}
    out: Dict[str, str] = {}
    for part in s.split(","):
        if ":" in part:
            k, v = part.split(":", 1)
            out[k.strip()] = v.strip()
    return out


# --- Colormap helpers (for Pastel1/Set2 etc.) ---
def _cmap_colors(name: str):
    """Matplotlib ≥3.7 colormap access with legacy fallback; returns list of RGB tuples."""
    try:
        cols = matplotlib.colormaps[name].colors  # works for qualitative maps like Pastel1/Set2
    except Exception:
        # legacy fallback
        from matplotlib import cm
        cols = cm.get_cmap(name).colors
    # ensure 3-tuple RGB (strip alpha if present)
    out = []
    for c in cols:
        r, g, b = c[:3]
        out.append((float(r), float(g), float(b)))
    return out

def _darken_colors(cols, factor: float = 0.8):
    """Darken a list of RGB tuples by multiplying by factor (<1 darker, >1 lighter)."""
    factor = float(factor)
    return [(max(0.0, r*factor), max(0.0, g*factor), max(0.0, b*factor)) for (r, g, b) in cols]

# ===================== small utils =====================

STEP_PAT = re.compile(r"step(\d+)", re.I)

def nat_step_from_path(path: str) -> Optional[int]:
    m = STEP_PAT.search(path)
    return int(m.group(1)) if m else None

def scan_files(root: str) -> List[str]:
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.endswith(".jsonl"):
                out.append(os.path.join(dp, fn))
    out.sort()
    return out

def coerce_bool(x) -> Optional[int]:
    if x is None: return None
    if isinstance(x, bool): return int(x)
    if isinstance(x, (int, np.integer)): return int(bool(x))
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("1","true","t","yes","y"): return 1
        if s in ("0","false","f","no","n"): return 0
    try:
        return int(bool(x))
    except Exception:
        return None

def first_nonempty(d: Dict[str, Any], keys: List[str]) -> Optional[Dict[str, Any]]:
    for k in keys:
        v = d.get(k)
        if isinstance(v, dict) and len(v) > 0:
            return v
    return None

def extract_correct_flag(pass_obj: Dict[str, Any]) -> Optional[int]:
    for k in ("is_correct_pred", "is_correct", "correct"):
        if k in pass_obj: return coerce_bool(pass_obj.get(k))
    for cont_key in ("sample", "completion"):
        v = pass_obj.get(cont_key)
        if isinstance(v, dict):
            for k in ("is_correct_pred", "is_correct", "correct"):
                if k in v: return coerce_bool(v[k])
    return None

def extract_entropy(pass_obj: Dict[str, Any], preferred: str = "entropy_answer") -> Optional[float]:
    if preferred in pass_obj and pass_obj[preferred] is not None:
        try: return float(pass_obj[preferred])
        except Exception: pass
    if "entropy" in pass_obj and pass_obj["entropy"] is not None:
        try: return float(pass_obj["entropy"])
        except Exception: pass
    return None

def extract_sample_idx(rec: Dict[str, Any], pass_obj: Dict[str, Any]) -> Optional[int]:
    for k in ("sample_idx", "sample_index", "idx", "i"):
        if k in rec:
            try: return int(rec[k])
            except Exception: pass
        if k in pass_obj:
            try: return int(pass_obj[k])
            except Exception: pass
    return None

def wilson_ci(k: int, n: int) -> Tuple[float, float]:
    if n <= 0: return (np.nan, np.nan)
    z = 1.959963984540054
    p = k / n
    denom = 1 + z*z/n
    centre = p + z*z/(2*n)
    adj = z * ((p*(1-p) + z*z/(4*n))/n) ** 0.5
    lo = (centre - adj) / denom
    hi = (centre + adj) / denom
    return (max(0.0, lo), min(1.0, hi))

def savefig(fig: plt.Figure, outbase: str):
    d = os.path.dirname(outbase)
    os.makedirs(d, exist_ok=True)
    fig.savefig(outbase + ".png", bbox_inches="tight", dpi=300)
    fig.savefig(outbase + ".pdf", bbox_inches="tight")
    plt.close(fig)

# ===================== loaders =====================

PASS1_KEYS = ["pass1", "p1", "first_pass"]
# For second-pass style results, prefer explicitly named forced/second-pass fields,
# but also fall back to multi-cue variants when present.
PASS2_KEYS = [
    "pass2_forced",
    "pass2",
    "pass2c",
    "pass2b",
    "pass2a",
    "p2",
    "forced",
    "forced_aha",
    "aha_forced",
]

def _maybe_int(x, default=-1) -> int:
    try: return int(x)
    except Exception: return default

def _common_fields(rec: Dict[str, Any], step_from_name: Optional[int]) -> Dict[str, Any]:
    dataset = rec.get("dataset")
    model   = rec.get("model")
    problem = rec.get("problem") or rec.get("question") or rec.get("row_key") or "unknown"
    step    = rec.get("step", step_from_name if step_from_name is not None else None)
    split   = rec.get("split")
    return {"dataset": dataset, "model": model, "problem": str(problem), "step": _maybe_int(step, -1), "split": split}

def load_samples_from_root(
    root: str,
    split_value: Optional[str],
    variant: str,
    entropy_field: str,
    pass2_key: Optional[str] = None,
) -> pd.DataFrame:
    files = scan_files(root)
    rows: List[Dict[str, Any]] = []
    for path in files:
        step_from_name = nat_step_from_path(path)
        with open(path, "r", encoding="utf-8") as fh:
            for ln in fh:
                if not ln.strip(): continue
                try:
                    rec = json.loads(ln)
                except Exception:
                    continue
                if split_value is not None and str(rec.get("split", "")).lower() != str(split_value).lower():
                    continue
                if variant == "pass1":
                    pass_obj = first_nonempty(rec, PASS1_KEYS)
                else:
                    if pass2_key:
                        pass_obj = rec.get(pass2_key) or {}
                    else:
                        pass_obj = first_nonempty(rec, PASS2_KEYS)
                if not pass_obj: continue
                corr = extract_correct_flag(pass_obj)
                if corr is None: continue
                meta = _common_fields(rec, step_from_name)
                sidx = extract_sample_idx(rec, pass_obj)
                row = {**meta, "sample_idx": sidx, "correct": int(corr)}
                if variant == "pass1":
                    row["entropy_p1"] = extract_entropy(pass_obj, preferred=entropy_field)
                rows.append(row)
    return pd.DataFrame(rows)

def load_samples_from_single_root_with_both(
    root: str,
    split_value: Optional[str],
    entropy_field: str,
    pass2_key: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    files = scan_files(root)
    rows1: List[Dict[str, Any]] = []
    rows2: List[Dict[str, Any]] = []
    for path in files:
        step_from_name = nat_step_from_path(path)
        with open(path, "r", encoding="utf-8") as fh:
            for ln in fh:
                if not ln.strip(): continue
                try:
                    rec = json.loads(ln)
                except Exception:
                    continue
                if split_value is not None and str(rec.get("split", "")).lower() != str(split_value).lower():
                    continue
                meta = _common_fields(rec, step_from_name)
                p1 = first_nonempty(rec, PASS1_KEYS)
                if p1:
                    c1 = extract_correct_flag(p1)
                    if c1 is not None:
                        rows1.append({**meta, "sample_idx": extract_sample_idx(rec, p1),
                                      "correct": int(c1), "entropy_p1": extract_entropy(p1, preferred=entropy_field)})
                if pass2_key:
                    p2 = rec.get(pass2_key) or {}
                else:
                    p2 = first_nonempty(rec, PASS2_KEYS)
                if p2:
                    c2 = extract_correct_flag(p2)
                    if c2 is not None:
                        rows2.append({**meta, "sample_idx": extract_sample_idx(rec, p2),
                                      "correct": int(c2)})
    return pd.DataFrame(rows1), pd.DataFrame(rows2)

# ===================== pairing helpers =====================

def _choose_merge_keys(df_left: pd.DataFrame, df_right: pd.DataFrame) -> List[str]:
    keys: List[str] = []
    must = ["problem"]
    optional = ["step", "dataset", "model", "split"]
    for k in must:
        if (k in df_left.columns) and (k in df_right.columns):
            keys.append(k)
    for k in optional:
        if (k in df_left.columns) and (k in df_right.columns):
            if df_left[k].notna().any() and df_right[k].notna().any():
                keys.append(k)
    if not keys:
        raise SystemExit("No common non-null keys to merge on. Need at least 'problem'.")
    return keys

def _fill_missing_id_cols(df: pd.DataFrame, cols: List[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = df[col].fillna("(missing)")

# ===================== pairing =====================

def pair_samples(df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    _fill_missing_id_cols(df1, ["dataset","model","split"])
    _fill_missing_id_cols(df2, ["dataset","model","split"])
    keys = _choose_merge_keys(df1, df2)
    if ("sample_idx" in df1.columns) and ("sample_idx" in df2.columns) \
       and df1["sample_idx"].notna().any() and df2["sample_idx"].notna().any():
        keys = keys + ["sample_idx"]
    left = df1.rename(columns={"correct":"correct1"})
    right = df2.rename(columns={"correct":"correct2"})
    pairs = left.merge(right, on=keys, how="inner")
    print(f"[info] Sample-level pairing on keys: {keys}  (pairs={len(pairs)})")
    return pairs, keys

def build_clusters(df: pd.DataFrame, label: str) -> pd.DataFrame:
    df = df.copy()
    keys = [k for k in ["dataset","model","split","problem","step"] if k in df.columns]
    if "problem" not in keys:
        raise SystemExit("Cluster build requires at least 'problem' in records.")
    aggs = {f"n_{label}":("correct","size"), f"k_{label}":("correct","sum")}
    if label == "p1" and "entropy_p1" in df.columns:
        aggs["entropy_p1_cluster"] = ("entropy_p1", "mean")
    g = df.groupby(keys, as_index=False).agg(**aggs)
    g[f"acc_{label}"] = g[f"k_{label}"] / g[f"n_{label}"]
    g[f"any_{label}"] = (g[f"k_{label}"] > 0).astype(int)
    return g

def pair_clusters(df1: pd.DataFrame, df2: pd.DataFrame) -> pdDataFrame:
    c1 = build_clusters(df1, "p1")
    c2 = build_clusters(df2, "p2")
    _fill_missing_id_cols(c1, ["dataset","model","split"])
    _fill_missing_id_cols(c2, ["dataset","model","split"])
    keys = _choose_merge_keys(c1, c2)
    merged = c1.merge(c2, on=keys, how="inner")
    print(f"[info] Cluster-level pairing on keys: {keys}  (clusters={len(merged)})")
    return merged

# ===================== stats =====================

def mcnemar_pvalue_table(a:int,b:int,c:int,d:int) -> float:
    try:
        from statsmodels.stats.contingency_tables import mcnemar
        res = mcnemar([[a,b],[c,d]], exact=False, correction=True)
        return float(res.pvalue)
    except Exception:
        if (b + c) == 0:
            return 1.0
        chi2 = (abs(b - c) - 1)**2 / (b + c)
        if   chi2 >= 10.83: return 0.001
        elif chi2 >= 6.63:  return 0.01
        elif chi2 >= 3.84:  return 0.05
        elif chi2 >= 2.71:  return 0.10
        else:               return 0.2

def mcnemar_from_pairs(df: pd.DataFrame, col1: str, col2: str) -> Tuple[int,int,int,int,float]:
    a = int(((df[col1]==0) & (df[col2]==0)).sum())
    b = int(((df[col1]==0) & (df[col2]==1)).sum())
    c = int(((df[col1]==1) & (df[col2]==0)).sum())
    d = int(((df[col1]==1) & (df[col2]==1)).sum())
    p = mcnemar_pvalue_table(a,b,c,d)
    return a,b,c,d,p

def paired_t_and_wilcoxon(diffs: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    t_p = w_p = None
    try:
        import scipy.stats as st
        if diffs.size >= 2 and np.all(np.isfinite(diffs)):
            t_p = float(st.ttest_rel(diffs, np.zeros_like(diffs), nan_policy="omit").pvalue)
            nz = diffs[np.abs(diffs) > 1e-12]
            if nz.size >= 1:
                try:
                    w_p = float(st.wilcoxon(nz).pvalue)
                except Exception:
                    w_p = None
    except Exception:
        pass
    return t_p, w_p

# ===================== summarizers =====================

def summarize_sample_level(pairs: pd.DataFrame) -> Dict[str, Any]:
    n = int(len(pairs))
    k1 = int(pairs["correct1"].sum()); k2 = int(pairs["correct2"].sum())
    acc1 = k1 / n if n else float("nan"); acc2 = k2 / n if n else float("nan")
    lo1, hi1 = wilson_ci(k1, n); lo2, hi2 = wilson_ci(k2, n)
    a,b,c,d,p = mcnemar_from_pairs(pairs, "correct1", "correct2")
    return {"metric":"sample","n_units":n,"acc_pass1":acc1,"acc_pass1_lo":lo1,"acc_pass1_hi":hi1,
            "acc_pass2":acc2,"acc_pass2_lo":lo2,"acc_pass2_hi":hi2,
            "delta_pp":(acc2-acc1)*100.0 if np.isfinite(acc1) and np.isfinite(acc2) else np.nan,
            "wins_pass2":b,"wins_pass1":c,"both_correct":d,"both_wrong":a,
            "p_mcnemar":p,"p_ttest":None,"p_wilcoxon":None}

def summarize_cluster_any(merged: pd.DataFrame) -> Dict[str, Any]:
    n = int(len(merged))
    k1 = int(merged["any_p1"].sum()); k2 = int(merged["any_p2"].sum())
    acc1 = k1 / n if n else float("nan"); acc2 = k2 / n if n else float("nan")
    lo1, hi1 = wilson_ci(k1, n); lo2, hi2 = wilson_ci(k2, n)
    a,b,c,d,p = mcnemar_from_pairs(merged, "any_p1", "any_p2")
    return {"metric":"cluster_any","n_units":n,"acc_pass1":acc1,"acc_pass1_lo":lo1,"acc_pass1_hi":hi1,
            "acc_pass2":acc2,"acc_pass2_lo":lo2,"acc_pass2_hi":hi2,
            "delta_pp":(acc2-acc1)*100.0 if np.isfinite(acc1) and np.isfinite(acc2) else np.nan,
            "wins_pass2":b,"wins_pass1":c,"both_correct":d,"both_wrong":a,
            "p_mcnemar":p,"p_ttest":None,"p_wilcoxon":None}

def summarize_cluster_mean(merged: pd.DataFrame) -> Dict[str, Any]:
    acc1 = float(merged["acc_p1"].mean()) if len(merged) else float("nan")
    acc2 = float(merged["acc_p2"].mean()) if len(merged) else float("nan")
    delta_pp = (acc2 - acc1) * 100.0 if np.isfinite(acc1) and np.isfinite(acc2) else np.nan
    diffs = (merged["acc_p2"] - merged["acc_p1"]).to_numpy(dtype=float)
    t_p, w_p = paired_t_and_wilcoxon(diffs)
    return {"metric":"cluster_mean","n_units":int(len(merged)),
            "acc_pass1":acc1,"acc_pass1_lo":None,"acc_pass1_hi":None,
            "acc_pass2":acc2,"acc_pass2_lo":None,"acc_pass2_hi":None,
            "delta_pp":delta_pp,"wins_pass2":None,"wins_pass1":None,
            "both_correct":None,"both_wrong":None,
            "p_mcnemar":None,"p_ttest":t_p,"p_wilcoxon":w_p}

# ===================== bootstrapping helpers =====================

def rng(seed: int):
    try: s = int(seed)
    except Exception: s = 0
    random.seed(s); np.random.seed(s)

def _bootstrap_delta(ppairs: pd.DataFrame, metric: str, n_boot: int, seed: int) -> Tuple[float,float]:
    rng(seed)
    deltas: List[float] = []
    idx = np.arange(len(ppairs))
    if metric == "sample":
        for _ in range(n_boot):
            b = ppairs.iloc[np.random.choice(idx, size=len(idx), replace=True)]
            deltas.append((b["correct2"].mean() - b["correct1"].mean()) * 100.0)
    elif metric == "cluster_any":
        for _ in range(n_boot):
            b = ppairs.iloc[np.random.choice(idx, size=len(idx), replace=True)]
            deltas.append((b["any_p2"].mean() - b["any_p1"].mean()) * 100.0)
    elif metric == "cluster_mean":
        for _ in range(n_boot):
            b = ppairs.iloc[np.random.choice(idx, size=len(idx), replace=True)]
            deltas.append((b["acc_p2"].mean() - b["acc_p1"].mean()) * 100.0)
    if not deltas: return (np.nan, np.nan)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(lo), float(hi)

def _bootstrap_delta_by_step(step_to_df: Dict[int, pd.DataFrame], metric: str, n_boot: int, seed: int) -> Dict[int, Tuple[float,float]]:
    rng(seed)
    ci: Dict[int, Tuple[float,float]] = {}
    for step, df in step_to_df.items():
        if len(df) == 0: ci[step] = (np.nan, np.nan); continue
        s = int(step)
        lo, hi = _bootstrap_delta(df, metric, n_boot, int(seed) + s)
        ci[step] = (lo, hi)
    return ci

# ===================== plotting (overview/waterfall unchanged fonts due to global rcParams) =====================

def plot_overall_deltas(out_dir: str, summary_rows: List[Dict[str, Any]],
                        pairs_df: pd.DataFrame, clusters_df: pd.DataFrame,
                        n_boot: int, seed: int, palette: Dict[str, str]):
    desired = ["sample", "cluster_mean", "cluster_any"]
    by_metric = {r["metric"]: r for r in summary_rows}
    labels, heights, ci_los, ci_his, colors = [], [], [], [], []
    bar_cols = [palette["bar_primary"], palette["bar_secondary"], palette["bar_tertiary"]]
    for i, lab in enumerate(desired):
        r = by_metric.get(lab)
        if r is None: continue
        dpp = r["delta_pp"]; labels.append(METRIC_LABELS.get(lab, lab)); heights.append(dpp)
        if lab == "sample":
            lo, hi = _bootstrap_delta(pairs_df, "sample", n_boot, seed)
        elif lab == "cluster_any":
            lo, hi = _bootstrap_delta(clusters_df, "cluster_any", n_boot, seed+1)
        else:
            lo, hi = _bootstrap_delta(clusters_df, "cluster_mean", n_boot, seed+2)
        ci_los.append(dpp - lo); ci_his.append(hi - dpp); colors.append(bar_cols[i%len(bar_cols)])
    if not labels: return
    fig = plt.figure(figsize=(7.0, 4.2)); ax = fig.add_axes([0.12,0.16,0.83,0.74])
    x = np.arange(len(labels))
    ax.bar(x, heights, color=colors, yerr=[ci_los, ci_his], capsize=4)
    ax.axhline(0, linewidth=0.8, color="#444"); ax.set_xticks(x)
    ax.set_xticklabels(labels); ax.set_ylabel("Δ accuracy (pp)"); ax.set_title("Forced Aha — Overall Δ (95% bootstrap CI)")
    for i, lab in enumerate(desired):
        r = by_metric.get(lab); 
        if r is None: continue
        p = r.get("p_mcnemar") if lab in ("sample","cluster_any") else r.get("p_ttest")
        if p is not None and np.isfinite(p):
            txt = "p≈0" if (p == 0 or p < 1e-300) else f"p={p:.1e}"
            ax.text(x[i], heights[i], txt, ha="center", va="bottom", fontsize=10)
    savefig(fig, os.path.join(out_dir, "figures", "overall_deltas"))

def _step_groups_for_boot(pairs_df: pd.DataFrame, clusters_df: pd.DataFrame):
    steps = sorted(set(pairs_df["step"].unique()).union(set(clusters_df["step"].unique())))
    sample_step = {s: pairs_df[pairs_df["step"]==s] for s in steps}
    any_step    = {s: clusters_df[clusters_df["step"]==s] for s in steps}
    mean_step   = any_step
    return steps, sample_step, any_step, mean_step

def plot_conversion_waterfall(out_dir: str, clusters_df: pd.DataFrame, palette: Dict[str, str]):
    a = int(((clusters_df["any_p1"]==0) & (clusters_df["any_p2"]==0)).sum())
    b = int(((clusters_df["any_p1"]==0) & (clusters_df["any_p2"]==1)).sum())
    c = int(((clusters_df["any_p1"]==1) & (clusters_df["any_p2"]==0)).sum())
    d = int(((clusters_df["any_p1"]==1) & (clusters_df["any_p2"]==1)).sum())
    total = a+b+c+d if (a+b+c+d) else 1
    labels = ["Incorrect → Correct","Correct → Incorrect","Stayed Correct","Stayed Incorrect"]
    vals   = [b,c,d,a]
    cols   = [palette["gain"],palette["loss"],palette["stable_pos"],palette["stable_neg"]]
    pct    = [100.0*v/total for v in vals]
    fig = plt.figure(figsize=(7.2,4.2)); ax = fig.add_axes([0.28,0.16,0.68,0.78])
    y = np.arange(len(vals)); ax.barh(y, vals, color=cols); ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("Count (problems × steps)"); ax.set_title("Any-correct conversions (overall)")
    for i,(v,p) in enumerate(zip(vals,pct)): ax.text(v, i, f" {v}  ({p:.1f}%)", va="center", ha="left")
    ax.xaxis.grid(True, linestyle=":", alpha=0.4); savefig(fig, os.path.join(out_dir,"figures","any_conversion_waterfall"))

def plot_headroom_scatter(out_dir: str, step_df: pd.DataFrame):
    sub = step_df[step_df["metric"]=="cluster_any"].copy().sort_values("step")
    x = sub["acc_pass1"].to_numpy(float); y = sub["delta_pp"].to_numpy(float)
    fig = plt.figure(figsize=(6.0,4.5)); ax = fig.add_axes([0.14,0.14,0.82,0.82])
    ax.scatter(x,y,s=30)
    if len(x)>=2 and np.isfinite(x).all() and np.isfinite(y).all():
        k,b = np.polyfit(x,y,1); xs = np.linspace(min(x),max(x),100); ax.plot(xs,k*xs+b,ls="--")
    ax.axhline(0,lw=0.8); ax.set_xlabel("Baseline any-correct (pass-1)")
    ax.set_ylabel("Δ any-correct (pp)"); ax.set_title("Headroom plot: Δ(any) vs baseline any (per step)")
    savefig(fig, os.path.join(out_dir,"figures","any_headroom_scatter"))

# ===================== NEW/UPDATED: uncertainty buckets, stepwise overlay, combined =====================

def _compute_uncertainty_bucket_data(pairs_df: pd.DataFrame, clusters_df: pd.DataFrame,
                                     n_boot: int, seed: int):
    # sample-level buckets (need entropy_p1)
    df_s = pairs_df.dropna(subset=["entropy_p1"]).copy()
    df_c = clusters_df.dropna(subset=["entropy_p1_cluster"]).copy()
    if df_s.empty or df_c.empty:
        return None
    df_s["bucket"] = pd.qcut(df_s["entropy_p1"], q=5, labels=False)
    df_c["bucket"] = pd.qcut(df_c["entropy_p1_cluster"], q=5, labels=False)

    def agg_delta_sample(g: pd.DataFrame) -> float: return (g["correct2"].mean() - g["correct1"].mean())*100.0
    def agg_delta_mean(g: pd.DataFrame)   -> float: return (g["acc_p2"].mean()   - g["acc_p1"].mean())  *100.0

    def boot(groups: List[pd.DataFrame], fn, seed0: int):
        rng(seed0); los,his,deltas=[],[],[]
        for i,g in enumerate(groups):
            if g.empty: deltas.append(np.nan); los.append(np.nan); his.append(np.nan); continue
            idx=np.arange(len(g)); boots=[]
            for b in range(n_boot):
                gg=g.iloc[np.random.choice(idx,size=len(idx),replace=True)]
                boots.append(fn(gg))
            lo,hi=np.percentile(boots,[2.5,97.5]); deltas.append(fn(g)); los.append(float(lo)); his.append(float(hi))
        return np.array(deltas), np.array(los), np.array(his)

    s_groups=[df_s[df_s["bucket"]==b] for b in range(5)]
    c_groups=[df_c[df_c["bucket"]==b] for b in range(5)]
    s_delta,s_lo,s_hi = boot(s_groups, agg_delta_sample, seed+20)
    c_delta,c_lo,c_hi = boot(c_groups, agg_delta_mean,   seed+25)
    x = np.arange(5)
    return dict(x=x, s_delta=s_delta, s_lo=s_lo, s_hi=s_hi, c_delta=c_delta, c_lo=c_lo, c_hi=c_hi)

def plot_uncertainty_buckets(out_dir: str,
                             pairs_df: pd.DataFrame,
                             clusters_df: pd.DataFrame,
                             n_boot: int,
                             seed: int,
                             palette_name: str = "Pastel1",
                             darken: float = 0.80):
    data = _compute_uncertainty_bucket_data(pairs_df, clusters_df, n_boot, seed)
    if data is None:
        print("[warn] No entropy_p1 available; skipping uncertainty_buckets.")
        return

    # --- palette (darker) ---
    cols = _darken_colors(_cmap_colors(palette_name), darken)
    c1, c2 = cols[1], cols[2]  # pick two distinct hues

    fig = plt.figure(figsize=(7.2, 4.6))
    ax = fig.add_axes([0.10, 0.15, 0.86, 0.75])

    # Per-draw (single completion)
    ax.errorbar(data["x"]-0.05, data["s_delta"],
                yerr=[data["s_delta"]-data["s_lo"], data["s_hi"]-data["s_delta"]],
                fmt="o-", capsize=4, label="Per-draw (single completion)", color=c1)

    # Per-problem mean of 8
    ax.errorbar(data["x"]+0.05, data["c_delta"],
                yerr=[data["c_delta"]-data["c_lo"], data["c_hi"]-data["c_delta"]],
                fmt="s-", capsize=4, label="Per-problem mean of 8 (avg of 8 completions)", color=c2)

    ax.axhline(0, linewidth=0.8)
    ax.set_xticks(data["x"]); ax.set_xticklabels([f"Q{b+1}" for b in range(5)])
    ax.set_xlabel("Baseline pass-1 answer entropy (quintiles)")
    ax.set_ylabel("Delta Accuracy (pp) for second pass")
    ax.set_title("Uncertainty buckets: Forced Aha gains by entropy")
    ax.legend()
    savefig(fig, os.path.join(out_dir, "figures", "uncertainty_buckets"))

def plot_stepwise_overlay(out_dir: str, step_df: pd.DataFrame,
                          pairs_df: pd.DataFrame, clusters_df: pd.DataFrame,
                          n_boot: int, seed: int,
                          palette_name: str = "Pastel1",
                          darken: float = 0.80):
    steps, sample_step, any_step, mean_step = _step_groups_for_boot(pairs_df, clusters_df)
    ci_sample = _bootstrap_delta_by_step(sample_step, "sample", n_boot, seed)
    ci_any    = _bootstrap_delta_by_step(any_step, "cluster_any", n_boot, seed+5)
    ci_mean   = _bootstrap_delta_by_step(mean_step, "cluster_mean", n_boot, seed+10)

    # --- palette (darker) ---
    cols = _darken_colors(_cmap_colors(palette_name), darken)
    c_s, c_m, c_a = cols[1], cols[2], cols[4]

    def _series(metric: str):
        sub = step_df[step_df["metric"]==metric].sort_values("step")
        x = sub["step"].to_numpy(int); y = sub["delta_pp"].to_numpy(float)
        if metric=="sample": lo,hi = zip(*[ci_sample[int(s)] for s in x])
        elif metric=="cluster_mean": lo,hi = zip(*[ci_mean[int(s)] for s in x])
        else: lo,hi = zip(*[ci_any[int(s)] for s in x])
        return x,y,np.array(lo),np.array(hi)

    x_s, y_s, lo_s, hi_s = _series("sample")
    x_m, y_m, lo_m, hi_m = _series("cluster_mean")
    x_a, y_a, lo_a, hi_a = _series("cluster_any")

    fig = plt.figure(figsize=(8.6, 4.3)); ax = fig.add_axes([0.10,0.15,0.86,0.75])

    ax.plot(x_s, y_s, marker="o", color=c_s, label="Per-draw (single completion)")
    ax.fill_between(x_s, lo_s, hi_s, color=c_s, alpha=0.22, linewidth=0)

    ax.plot(x_m, y_m, marker="s", color=c_m, label="Per-problem mean of 8 (avg of 8)")
    ax.fill_between(x_m, lo_m, hi_m, color=c_m, alpha=0.22, linewidth=0)

    ax.plot(x_a, y_a, marker="^", color=c_a, label="Per-problem any-correct")
    ax.fill_between(x_a, lo_a, hi_a, color=c_a, alpha=0.18, linewidth=0)

    ax.axhline(0, lw=0.8)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Delta Accuracy (pp) for second pass")
    ax.set_title("Stepwise Δ — overlay of per-draw / mean-of-8 / any-correct (95% bootstrap CIs)")
    ax.legend(ncol=1)
    savefig(fig, os.path.join(out_dir, "figures", "stepwise_overlay"))

def plot_uncertainty_and_stepwise(out_dir: str, step_df: pd.DataFrame,
                                  pairs_df: pd.DataFrame, clusters_df: pd.DataFrame,
                                  n_boot: int, seed: int,
                                  palette_name: str = "Pastel1",
                                  darken: float = 0.80):
    data = _compute_uncertainty_bucket_data(pairs_df, clusters_df, n_boot, seed)
    if data is None:
        print("[warn] No entropy available; skipping combined uncertainty_and_stepwise.")
        return
    steps, sample_step, any_step, mean_step = _step_groups_for_boot(pairs_df, clusters_df)
    ci_sample = _bootstrap_delta_by_step(sample_step, "sample", n_boot, seed)
    ci_any    = _bootstrap_delta_by_step(any_step, "cluster_any", n_boot, seed+5)
    ci_mean   = _bootstrap_delta_by_step(mean_step, "cluster_mean", n_boot, seed+10)

    # --- palette (darker) ---
    cols = _darken_colors(_cmap_colors(palette_name), darken)
    c1, c2, c3 = cols[1], cols[2], cols[4]

    def _series(metric: str):
        sub = step_df[step_df["metric"]==metric].sort_values("step")
        x = sub["step"].to_numpy(int); y = sub["delta_pp"].to_numpy(float)
        if metric=="sample": lo,hi = zip(*[ci_sample[int(s)] for s in x])
        elif metric=="cluster_mean": lo,hi = zip(*[ci_mean[int(s)] for s in x])
        else: lo,hi = zip(*[ci_any[int(s)] for s in x])
        return x,y,np.array(lo),np.array(hi)

    x_s,y_s,lo_s,hi_s = _series("sample")
    x_m,y_m,lo_m,hi_m = _series("cluster_mean")
    x_a,y_a,lo_a,hi_a = _series("cluster_any")

    fig = plt.figure(figsize=(12.4, 4.8))
    gs = fig.add_gridspec(1, 2, left=0.08, right=0.98, top=0.92, bottom=0.18, wspace=0.26)
    axL = fig.add_subplot(gs[0, 0]); axR = fig.add_subplot(gs[0, 1])

    # Left: uncertainty buckets
    axL.errorbar(data["x"]-0.05, data["s_delta"],
                 yerr=[data["s_delta"]-data["s_lo"], data["s_hi"]-data["s_delta"]],
                 fmt="o-", capsize=4, color=c1, label="Per-draw (single completion)")
    axL.errorbar(data["x"]+0.05, data["c_delta"],
                 yerr=[data["c_delta"]-data["c_lo"], data["c_hi"]-data["c_delta"]],
                 fmt="s-", capsize=4, color=c2, label="Per-problem mean of 8 (avg of 8 completions)")
    axL.axhline(0, lw=0.8)
    axL.set_xticks(data["x"]); axL.set_xticklabels([f"Q{b+1}" for b in range(5)])
    axL.set_xlabel("Baseline pass-1 answer entropy (quintiles)")
    axL.set_ylabel("Delta Accuracy (pp) for second pass")
    axL.set_title("Uncertainty buckets: Forced Aha gains by entropy")
    axL.legend()

    # Right: stepwise overlay
    axR.plot(x_s, y_s, marker="o", color=c1, label="Per-draw (single completion)")
    axR.fill_between(x_s, lo_s, hi_s, color=c1, alpha=0.22, linewidth=0)
    axR.plot(x_m, y_m, marker="s", color=c2, label="Per-problem mean of 8 (avg of 8)")
    axR.fill_between(x_m, lo_m, hi_m, color=c2, alpha=0.22, linewidth=0)
    axR.plot(x_a, y_a, marker="^", color=c3, label="Per-problem any-correct")
    axR.fill_between(x_a, lo_a, hi_a, color=c3, alpha=0.18, linewidth=0)
    axR.axhline(0, lw=0.8)
    axR.set_xlabel("Training step")
    axR.set_ylabel("Delta Accuracy (pp) for second pass")
    axR.set_title("Stepwise Δ — overlay (95% bootstrap CIs)")
    axR.legend()

    savefig(fig, os.path.join(out_dir, "figures", "uncertainty_and_stepwise"))

# ===================== existing overview side-by-side (still Times due to global rcParams) =====================

def plot_overview_side_by_side(out_dir: str,
                               step_df: pd.DataFrame,
                               pairs_df: pd.DataFrame,
                               clusters_df: pd.DataFrame,
                               n_boot: int,
                               seed: int,
                               palette_name: str = "Pastel1",
                               darken: float = 0.80):
    """
    Side-by-side overview with the SAME inputs as plot_uncertainty_and_stepwise:
      (left) Overall Δ bars (bootstrap CIs), no p-values shown.
      (right) Any-correct conversions waterfall (Incorrect→Correct first).
    Left bars use a darkened qualitative palette; right uses semantic colors.
    """
    # ----- Recompute overall deltas from inputs -----
    has_pairs = len(pairs_df) > 0
    s_sample = summarize_sample_level(pairs_df) if has_pairs else None
    s_any    = summarize_cluster_any(clusters_df)
    s_mean   = summarize_cluster_mean(clusters_df)

    bar_specs = []
    if s_sample is not None:
        bar_specs.append(("sample",       "Per-draw\naccuracy",      s_sample))
    bar_specs.append(("cluster_mean", "Per-problem\nmean of 8",   s_mean))
    bar_specs.append(("cluster_any",  "Per-problem\nany-correct", s_any))

    labels  = [lbl for _, lbl, _ in bar_specs]
    heights = [stat["delta_pp"] for _, _, stat in bar_specs]

    # Bootstrap CIs for bars
    ci_los, ci_his = [], []
    for idx, (key, _, _) in enumerate(bar_specs):
        if key == "sample":
            lo, hi = _bootstrap_delta(pairs_df, "sample", n_boot, seed)
        elif key == "cluster_any":
            lo, hi = _bootstrap_delta(clusters_df, "cluster_any", n_boot, seed + 1)
        else:  # cluster_mean
            lo, hi = _bootstrap_delta(clusters_df, "cluster_mean", n_boot, seed + 2)
        dpp = heights[idx]
        ci_los.append(dpp - lo)
        ci_his.append(hi - dpp)

    # ----- Palette for left bars (darkened qualitative) -----
    cols = _darken_colors(_cmap_colors(palette_name), darken)
    bar_colors = [cols[i % len(cols)] for i in (1, 2, 4)][:len(labels)]

    # ----- Waterfall counts (semantic colors on the right) -----
    a = int(((clusters_df["any_p1"] == 0) & (clusters_df["any_p2"] == 0)).sum())  # stayed incorrect
    b = int(((clusters_df["any_p1"] == 0) & (clusters_df["any_p2"] == 1)).sum())  # incorrect -> correct
    c = int(((clusters_df["any_p1"] == 1) & (clusters_df["any_p2"] == 0)).sum())  # correct -> incorrect
    d = int(((clusters_df["any_p1"] == 1) & (clusters_df["any_p2"] == 1)).sum())  # stayed correct
    total = max(a + b + c + d, 1)

    wf_vals   = [b, c, d, a]
    wf_labels = ["Incorrect →\nCorrect", "Correct →\nIncorrect", "Stayed\nCorrect", "Stayed\nIncorrect"]
    wf_colors = [DEFAULT_COLORS["gain"], DEFAULT_COLORS["loss"],
                 DEFAULT_COLORS["stable_pos"], DEFAULT_COLORS["stable_neg"]]
    wf_pct    = [100.0 * v / total for v in wf_vals]

    # ----- Layout: bring panels slightly closer (wspace=0.28) -----
    fig = plt.figure(figsize=(12.2, 4.2))
    gs = fig.add_gridspec(nrows=1, ncols=2,
                          left=0.07, right=0.99, top=0.92, bottom=0.24, wspace=0.28)
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])

    # Left: Overall deltas (no p-values)
    x = np.arange(len(labels))
    axL.bar(x, heights, color=bar_colors, yerr=[ci_los, ci_his], capsize=4)
    axL.axhline(0, linewidth=0.8, color="#444444")
    axL.set_xticks(x)
    axL.set_xticklabels(labels)
    axL.set_ylabel("Δ accuracy (pp)\nsecond pass (forced Aha)")
    axL.set_title("Forced Aha — Overall Δ (95% bootstrap CI)")

    # Right: Waterfall with fixed semantic palette
    y = np.arange(len(wf_vals))
    axR.barh(y, wf_vals, color=wf_colors)
    axR.set_yticks(y)
    axR.set_yticklabels(wf_labels)
    axR.set_xlabel("Count\n(problems × steps)")
    axR.set_title("Any-correct conversions (overall)")
    axR.xaxis.grid(True, linestyle=":", alpha=0.35)

    # annotate counts + percents
    for i, (v, p) in enumerate(zip(wf_vals, wf_pct)):
        axR.text(v + max(total * 0.004, 8), i, f"{v:,}  ({p:.1f}%)",
                 va="center", ha="left")

    savefig(fig, os.path.join(out_dir, "figures", "overview_deltas_and_waterfall"))

# ===================== main =====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root1")
    ap.add_argument("root2", nargs="?", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument(
        "--pass2_key",
        default=None,
        help=(
            "Optional specific second-pass key to analyze "
            "(e.g., 'pass2', 'pass2a', 'pass2b', 'pass2c'). "
            "Defaults to the first available in PASS2_KEYS."
        ),
    )

    ap.add_argument("--make_plots", action="store_true")
    ap.add_argument("--n_boot", type=int, default=800)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--colors", default=None, help="Overrides for overview/waterfall palette, e.g. 'gain:#009E73,loss:#D55E00'")

    ap.add_argument("--entropy_field", default="entropy_answer",
                    help='Entropy source for buckets (default: "entropy_answer"; fallback to "entropy")')
    ap.add_argument("--series_palette", default="Dark2",
                    help="Qualitative colormap for uncertainty/stepwise series (default: Pastel1)")
    ap.add_argument("--darken", type=float, default=0.80,
                    help="Darken factor for series colors (0.0–1.0, lower = darker, default 0.80)")

    args = ap.parse_args()

    # Default out_dir depends on chosen pass2 key so that multi-cue runs can coexist.
    suffix = ""
    if args.pass2_key and args.pass2_key != "pass2":
        suffix = f"_{args.pass2_key}"
    out_dir = args.out_dir or os.path.join(args.root1, f"forced_aha_effect{suffix}")
    os.makedirs(out_dir, exist_ok=True)

    if args.root2:
        df1 = load_samples_from_root(
            args.root1,
            args.split,
            variant="pass1",
            entropy_field=args.entropy_field,
        )
        df2 = load_samples_from_root(
            args.root2,
            args.split,
            variant="pass2",
            entropy_field=args.entropy_field,
            pass2_key=args.pass2_key,
        )
    else:
        df1, df2 = load_samples_from_single_root_with_both(
            args.root1,
            args.split,
            entropy_field=args.entropy_field,
            pass2_key=args.pass2_key,
        )
    if df1.empty: raise SystemExit("No sample-level pass-1 rows found (after filtering).")
    if df2.empty: raise SystemExit("No sample-level pass-2 rows found (after filtering).")

    print(f"[info] Loaded: pass1 N={len(df1)} rows; pass2 N={len(df2)} rows")

    pairs, key_cols = pair_samples(df1, df2)
    clusters = pair_clusters(df1, df2)
    if clusters.empty: raise SystemExit("No overlapping clusters between pass-1 and pass-2 after adaptive keying.")

    summary_rows: List[Dict[str, Any]] = []
    if not pairs.empty: summary_rows.append(summarize_sample_level(pairs))
    summary_rows.append(summarize_cluster_any(clusters))
    summary_rows.append(summarize_cluster_mean(clusters))

    summ_df = pd.DataFrame(summary_rows)
    summ_df.to_csv(os.path.join(out_dir, "forced_aha_summary.csv"), index=False)

    # per-step table
    step_rows: List[Dict[str, Any]] = []
    if not pairs.empty:
        for step, sub in pairs.groupby("step"):
            a,b,c,d,p = mcnemar_from_pairs(sub, "correct1", "correct2")
            n=int(len(sub)); k1=int(sub["correct1"].sum()); k2=int(sub["correct2"].sum())
            acc1=k1/n if n else np.nan; acc2=k2/n if n else np.nan
            step_rows.append({"metric":"sample","step":int(step),"n_units":n,
                              "acc_pass1":acc1,"acc_pass2":acc2,"delta_pp":(acc2-acc1)*100.0 if n else np.nan,
                              "p_mcnemar":p,"p_ttest":None,"p_wilcoxon":None,
                              "wins_pass2":b,"wins_pass1":c,"both_correct":d,"both_wrong":a})
    for step, sub in clusters.groupby("step"):
        a,b,c,d,p = mcnemar_from_pairs(sub, "any_p1", "any_p2")
        n=int(len(sub)); acc1=float(sub["any_p1"].mean()) if n else np.nan; acc2=float(sub["any_p2"].mean()) if n else np.nan
        step_rows.append({"metric":"cluster_any","step":int(step),"n_units":n,
                          "acc_pass1":acc1,"acc_pass2":acc2,"delta_pp":(acc2-acc1)*100.0 if n else np.nan,
                          "p_mcnemar":p,"p_ttest":None,"p_wilcoxon":None,
                          "wins_pass2":b,"wins_pass1":c,"both_correct":d,"both_wrong":a})
    for step, sub in clusters.groupby("step"):
        n=int(len(sub)); acc1=float(sub["acc_p1"].mean()) if n else np.nan; acc2=float(sub["acc_p2"].mean()) if n else np.nan
        diffs=(sub["acc_p2"]-sub["acc_p1"]).to_numpy(float); t_p,w_p=paired_t_and_wilcoxon(diffs)
        step_rows.append({"metric":"cluster_mean","step":int(step),"n_units":n,
                          "acc_pass1":acc1,"acc_pass2":acc2,"delta_pp":(acc2-acc1)*100.0 if n else np.nan,
                          "p_mcnemar":None,"p_ttest":t_p,"p_wilcoxon":w_p,
                          "wins_pass2":None,"wins_pass1":None,"both_correct":None,"both_wrong":None})
    step_df = pd.DataFrame(step_rows).sort_values(["metric","step"]).reset_index(drop=True)
    step_df.to_csv(os.path.join(out_dir, "forced_aha_by_step.csv"), index=False)

    # console
    def verdict_line(r: Dict[str, Any]) -> str:
        metric=r["metric"]; acc1=r["acc_pass1"]; acc2=r["acc_pass2"]; dpp=r["delta_pp"]; n=r["n_units"]
        if metric in ("sample","cluster_any"):
            p=r["p_mcnemar"]; verdict="YES" if (np.isfinite(dpp) and dpp>0 and (p is not None) and p<0.05) else ("MAYBE" if (dpp and np.isfinite(dpp) and dpp>0) else "NO")
            return f"[{METRIC_LABELS.get(metric,metric)}] N={n} acc1={acc1:.4f} acc2={acc2:.4f} Δ={dpp:+.2f}pp  McNemar p={p:.4g} -> {verdict}"
        else:
            t_p=r.get("p_ttest"); w_p=r.get("p_wilcoxon"); sig=((t_p is not None and t_p<0.05) or (w_p is not None and w_p<0.05))
            verdict="YES" if (np.isfinite(dpp) and dpp>0 and sig) else ("MAYBE" if (dpp and np.isfinite(dpp) and dpp>0) else "NO")
            return f"[{METRIC_LABELS.get(metric,metric)}] N={n} mean-acc1={acc1:.4f} mean-acc2={acc2:.4f} Δ={dpp:+.2f}pp t_p={t_p if t_p is not None else 'nan'} W={w_p if w_p is not None else 'nan'} -> {verdict}"

    print("\n=== Forced Aha Effect — Overall ===")
    for r in summary_rows: print(verdict_line(r))

    print("\nWrote:", os.path.join(out_dir, "forced_aha_summary.csv"))
    print("By-step:", os.path.join(out_dir, "forced_aha_by_step.csv"))

    if args.make_plots:
        print("[info] Generating figures...")
        palette = {**DEFAULT_COLORS, **_parse_colors(args.colors)}
        plot_overall_deltas(out_dir, summary_rows, pairs, clusters, n_boot=args.n_boot, seed=args.seed, palette=palette)
        plot_conversion_waterfall(out_dir, clusters, palette=palette)
        plot_headroom_scatter(out_dir, step_df)
        plot_uncertainty_buckets(out_dir, pairs, clusters,
                                 n_boot=args.n_boot, seed=args.seed,
                                 palette_name=args.series_palette, darken=args.darken)
        plot_stepwise_overlay(out_dir, step_df, pairs, clusters,
                              n_boot=args.n_boot, seed=args.seed,
                              palette_name=args.series_palette, darken=args.darken)
        plot_uncertainty_and_stepwise(out_dir, step_df, pairs, clusters,
                                      n_boot=args.n_boot, seed=args.seed,
                                      palette_name=args.series_palette, darken=args.darken)
        plot_overview_side_by_side(
                out_dir, step_df, pairs, clusters,
                n_boot=args.n_boot, seed=args.seed,
                palette_name=args.series_palette, darken=args.darken
            )
        print("[info] Figures saved to:", os.path.join(out_dir, "figures"))

    if not pairs.empty:
        print("Sample-level paired on keys:", ", ".join(key_cols))
    else:
        print("Sample-level pairing not available; reported only cluster metrics.")

if __name__ == "__main__":
    main()
