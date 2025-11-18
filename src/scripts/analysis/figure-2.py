#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Uncertainty → Correctness: Counts, Densities, Accuracy, and Regression (All in One)
Times (12 pt) + optional exact A4 PDFs
-------------------------------------------------------------------------------

Figures written:
  1) unc_vs_correct_4hists__<DS>__<MODEL>.png/.pdf          # 4-panel COUNT(CORRECT) histograms
  2) unc_vs_correct_overlaid__<DS>__<MODEL>.png/.pdf        # overlaid densities (CORRECT only)
  3) unc_vs_corr_incorr_by_type__<DS>__<MODEL>.png/.pdf     # 2×2 CORRECT vs INCORRECT (All, Words, LLM, Formal)
  4) unc_accuracy_by_bin__<DS>__<MODEL>.png/.pdf            # per-bin accuracy with Wilson 95% CIs
  5) acc_vs_uncertainty_regression__<DS>__<MODEL>.png/.pdf  # GLM: correct ~ C(problem)+C(step)+aha:C(perplexity_bucket)

CSVs:
  - unc_vs_correct_4hists.csv
  - unc_vs_correct_overlaid.csv
  - unc_vs_corr_incorr_by_type.csv
  - unc_accuracy_by_bin.csv
  - acc_vs_uncertainty_regression.csv

Use --a4_pdf to save exact A4 PDFs; text is Times/Times New Roman (12 pt).
"""

import os, re, json, argparse
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ----------------- Global style: Times, size 12 -----------------

def set_global_fonts(font_family: str = "Times New Roman", font_size: int = 12):
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,   # embed TrueType → preserves 12 pt
        "font.family": "serif",
        "font.serif": [font_family, "Times", "DejaVu Serif", "Computer Modern Roman"],
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "figure.titlesize": font_size,
        "mathtext.fontset": "dejavuserif",
    })

def a4_size_inches(orientation: str = "landscape") -> Tuple[float, float]:
    if str(orientation).lower().startswith("p"):
        return (8.27, 11.69)
    return (11.69, 8.27)

# ----------------- Utils -----------------

STEP_PAT = re.compile(r"step(\d+)", re.I)

def slugify(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")

def nat_step_from_path(path: str) -> Optional[int]:
    m = STEP_PAT.search(path); return int(m.group(1)) if m else None

def scan_files(root: str, split_substr: Optional[str]) -> List[str]:
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if not fn.endswith(".jsonl"): continue
            if split_substr and split_substr not in fn: continue
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
    return int(bool(x))

def lighten_hex(hex_color: str, factor: float = 0.65) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16)/255.0 for i in (0,2,4))
    r = 1 - (1-r)*factor; g = 1 - (1-g)*factor; b = 1 - (1-b)*factor
    return "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))

# ----------------- Labels & uncertainty -----------------

def _any_keys_true(p1: Dict[str, Any], rec: Dict[str, Any], keys: List[str]) -> int:
    for k in keys:
        v = p1.get(k, rec.get(k, None))
        if v is None: continue
        out = coerce_bool(v)
        if out is not None and out == 1:
            return 1
    return 0

def _aha_native(p1: Dict[str, Any]) -> int:
    aha_raw = coerce_bool(p1.get("has_reconsider_cue"))
    markers = p1.get("reconsider_markers") or []
    if isinstance(markers, list) and ("injected_cue" in markers):
        return 0
    return 0 if aha_raw is None else int(aha_raw)

def _choose_uncertainty(p1: Dict[str, Any], pref: str = "answer") -> Optional[float]:
    if pref == "answer":
        x = p1.get("entropy_answer") or p1.get("entropy") or p1.get("entropy_think")
        return float(x) if x is not None else None
    if pref == "overall":
        x = p1.get("entropy") or p1.get("entropy_answer") or p1.get("entropy_think")
        return float(x) if x is not None else None
    if pref == "think":
        x = p1.get("entropy_think") or p1.get("entropy") or p1.get("entropy_answer")
        return float(x) if x is not None else None
    return None

# ----------------- Load samples (Words & GPT) -----------------

def load_pass1_rows(files: List[str], unc_field: str, gpt_keys: List[str], gate_gpt_by_words: bool) -> pd.DataFrame:
    rows = []
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
                if not p1: continue

                problem = rec.get("problem") or rec.get("clue") or rec.get("row_key")
                if problem is None:
                    di = rec.get("dataset_index"); problem = f"idx:{di}" if di is not None else "unknown"
                step = rec.get("step", step_from_name if step_from_name is not None else None)
                if step is None: continue
                step = int(step)

                corr_raw = coerce_bool(p1.get("is_correct_pred"))
                if corr_raw is None: continue
                correct = int(corr_raw)

                unc = _choose_uncertainty(p1, unc_field)
                if unc is None: continue

                words = _aha_native(p1)
                gpt = _any_keys_true(p1, rec, gpt_keys)
                if gate_gpt_by_words:
                    gpt = int(gpt and words)

                rows.append({
                    "problem": str(problem),
                    "step": step,
                    "correct": correct,
                    "uncertainty": float(unc),
                    "aha_words": int(words),
                    "aha_gpt": int(gpt),
                })
    d = pd.DataFrame(rows)
    if d.empty:
        raise RuntimeError("No usable PASS-1 rows found (missing labels and/or uncertainty).")
    return d

# ----------------- Formal computation -----------------

def _build_problem_step_for_formal(d_samples: pd.DataFrame) -> pd.DataFrame:
    base = (
        d_samples.groupby(["step", "problem"], as_index=False)
        .agg(
            n_samples=("correct", "size"),
            freq_correct=("correct", "mean"),
            aha_any_gpt=("aha_gpt", "max"),
            aha_rate_gpt=("aha_gpt", "mean"),
        )
        .sort_values(["problem", "step"])
        .reset_index(drop=True)
    )
    pcs = (
        d_samples.groupby(["step", "problem"])
        .apply(lambda g: float(g.loc[g["aha_gpt"] == 1, "correct"].mean())
               if (g["aha_gpt"] == 1).any() else np.nan)
        .reset_index(name="p_correct_given_shift")
    )
    ps = base.merge(pcs, on=["step", "problem"], how="left")
    for c in ("n_samples", "aha_any_gpt"):
        ps[c] = ps[c].astype(int)
    ps["freq_correct"] = ps["freq_correct"].astype(float)
    ps["aha_rate_gpt"] = ps["aha_rate_gpt"].astype(float)
    ps["p_correct_given_shift"] = ps["p_correct_given_shift"].astype(float)
    return ps

def _mark_formal_pairs(ps: pd.DataFrame,
                       delta1: float, delta2: float,
                       min_prior_steps: int,
                       delta3: Optional[float]) -> pd.DataFrame:
    need = {"step","problem","freq_correct","aha_rate_gpt","aha_any_gpt","p_correct_given_shift"}
    if not need.issubset(ps.columns):
        raise ValueError(f"Formal marking missing columns: {need - set(ps.columns)}")
    flags = np.zeros(len(ps), dtype=int)
    idx = 0
    for _, sub in ps.sort_values(["problem","step"]).groupby("problem", sort=False):
        freq = sub["freq_correct"].to_numpy(float)
        rate = sub["aha_rate_gpt"].to_numpy(float)
        shift = sub["aha_any_gpt"].to_numpy(int)
        p_plus = sub["p_correct_given_shift"].to_numpy(float)
        for j in range(len(sub)):
            if j < min_prior_steps:
                flags[idx] = 0
            else:
                prior_ok = (float(np.max(freq[:j])) < delta1) and (float(np.max(rate[:j])) < delta2)
                if not (prior_ok and shift[j] == 1):
                    flags[idx] = 0
                else:
                    if delta3 is None:
                        flags[idx] = 1
                    else:
                        flags[idx] = int(np.isfinite(p_plus[j]) and ((p_plus[j] - freq[j]) > delta3))
            idx += 1
    out = ps.copy()
    out["aha_formal_pair"] = flags
    return out

def _label_formal_samples(d_samples: pd.DataFrame, ps_formal: pd.DataFrame) -> pd.DataFrame:
    formal_keys = {(str(r["problem"]), int(r["step"]))
                   for _, r in ps_formal.loc[ps_formal["aha_formal_pair"] == 1, ["problem","step"]].iterrows()}
    keys = list(zip(d_samples["problem"].astype(str), d_samples["step"].astype(int)))
    d = d_samples.copy()
    d["aha_formal"] = [int((k in formal_keys) and (g == 1)) for k, g in zip(keys, d["aha_gpt"].astype(int))]
    return d

# ----------------- Wilson CI & helpers -----------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0: return (np.nan, np.nan)
    p = k / n; z2 = z*z; den = 1 + z2/n
    center = (p + z2/(2*n)) / den
    half = (z * np.sqrt((p*(1-p)/n) + (z2/(4*n*n)))) / den
    return (max(0.0, center - half), min(1.0, center + half))

def make_edges_from_std(std_vals: np.ndarray, bins: int, xlim: Optional[Tuple[float,float]] = None) -> np.ndarray:
    if xlim is None:
        lo, hi = np.nanpercentile(std_vals, [1, 99])
    else:
        lo, hi = xlim
    return np.linspace(lo, hi, int(max(10, bins)) + 1)

def density_from_hist(values_std: np.ndarray, edges: np.ndarray, smooth_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    if values_std.size == 0:
        centers = 0.5*(edges[:-1] + edges[1:])
        return centers, np.zeros_like(centers)
    hist, _ = np.histogram(values_std, bins=edges, density=True)
    if smooth_k and smooth_k > 1:
        k = max(1, int(smooth_k))
        if k % 2 == 0: k += 1
        kernel = np.ones(k, dtype=float) / k
        hist = np.convolve(hist, kernel, mode="same")
    centers = 0.5*(edges[:-1] + edges[1:])
    return centers, hist

def compute_correct_hist(values_std: np.ndarray, correct: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.histogram(values_std, bins=edges, weights=correct.astype(float))[0].astype(int)

# ----------------- Figure 1: 4-panel count histograms -----------------

def plot_four_correct_hists(d_all: pd.DataFrame, edges: np.ndarray, out_png: str, out_pdf: str,
                            title_suffix: str, a4_pdf: bool, a4_orientation: str):
    centers = 0.5*(edges[:-1] + edges[1:])
    mu = d_all["uncertainty"].mean(); sd = d_all["uncertainty"].std(ddof=0)
    d_all = d_all.copy()
    d_all["uncertainty_std"] = (d_all["uncertainty"] - mu) / (sd + 1e-8)

    panels = [
        ("All samples", d_all),
        ('Words of "Aha!"', d_all[d_all["aha_words"] == 1]),
        ('LLM-Detected "Aha!"', d_all[d_all["aha_gpt"] == 1]),
        ('Formal "Aha!"', d_all[d_all["aha_formal"] == 1] if "aha_formal" in d_all.columns else d_all.iloc[0:0]),
    ]

    fig_size = a4_size_inches(a4_orientation) if a4_pdf else (16.5, 4.8)
    fig, axes = plt.subplots(1, 4, figsize=fig_size, dpi=150, sharey=True)
    width = (edges[1] - edges[0]) * 0.95

    for ax, (ttl, dfp) in zip(axes, panels):
        if dfp.empty:
            counts = np.zeros(len(edges)-1, dtype=int)
        else:
            counts = compute_correct_hist(dfp["uncertainty_std"].to_numpy(),
                                          dfp["correct"].to_numpy(), edges)
        ax.bar(centers, counts, width=width, color="#7f9cd8", edgecolor="none")
        ax.set_title(f"{ttl}\n{title_suffix}")
        ax.set_xlabel("uncertainty_std")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Count of CORRECT")

    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(out_png)
    if a4_pdf:
        fig.set_size_inches(*a4_size_inches(a4_orientation))
    fig.savefig(out_pdf)
    plt.close(fig)

# ----------------- Figure 2: overlaid densities (CORRECT only) -----------------

def plot_overlaid_densities(d_all: pd.DataFrame, edges: np.ndarray, out_png: str, out_pdf: str,
                            title_suffix: str, smooth_bins: int, a4_pdf: bool, a4_orientation: str) -> str:
    mu = d_all["uncertainty"].mean(); sd = d_all["uncertainty"].std(ddof=0)
    d = d_all.copy()
    d["uncertainty_std"] = (d["uncertainty"] - mu) / (sd + 1e-8)

    sel_all   = d[d["correct"] == 1]
    sel_words = sel_all[sel_all["aha_words"] == 1]
    sel_llm   = sel_all[sel_all["aha_gpt"] == 1]
    sel_form  = sel_all[sel_all["aha_formal"] == 1] if "aha_formal" in d.columns else d.iloc[0:0]

    series = [
        ("All (correct)",  sel_all["uncertainty_std"].to_numpy(),  "#666666"),
        ('Words (correct∧words)', sel_words["uncertainty_std"].to_numpy(), "#1f77b4"),
        ('LLM (correct∧gpt)',     sel_llm["uncertainty_std"].to_numpy(),   "#ff7f0e"),
        ('Formal (correct∧formal)',sel_form["uncertainty_std"].to_numpy(),  "#2ca02c"),
    ]

    curves = []
    for label, vals, color in series:
        x, y = density_from_hist(vals, edges, smooth_k=int(max(0, smooth_bins)))
        curves.append((label, x, y, color, vals.size))

    fig_size = a4_size_inches(a4_orientation) if a4_pdf else (12.0, 4.8)
    fig, ax = plt.subplots(figsize=fig_size, dpi=150)

    for label, x, y, color, n in curves:
        ax.plot(x, y, lw=2.2, label=f"{label}  (n={n})", color=color)

    ax.set_title(f"Area-normalized density of uncertainty_std (CORRECT only)\n{title_suffix}")
    ax.set_xlabel("uncertainty_std")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=False)

    fig.tight_layout()
    fig.savefig(out_png)
    if a4_pdf:
        fig.set_size_inches(*a4_size_inches(a4_orientation))
    fig.savefig(out_pdf)
    plt.close(fig)

    centers = 0.5*(edges[:-1] + edges[1:])
    out_csv = out_png.replace(".png", ".csv")
    df_out = pd.DataFrame({"bin_center": centers})
    for label, x, y, color, n in curves:
        key = re.sub(r"[^A-Za-z0-9_]+", "_", label.lower())
        df_out[key] = y
    df_out.to_csv(out_csv, index=False)
    return out_csv

# ----------------- Figure 3: 2×2 panels (CORRECT vs INCORRECT) -----------------

def plot_correct_incorrect_by_type(d_all: pd.DataFrame, edges: np.ndarray, out_png: str, out_pdf: str,
                                   title_suffix: str, smooth_bins: int, a4_pdf: bool, a4_orientation: str) -> str:
    mu = d_all["uncertainty"].mean(); sd = d_all["uncertainty"].std(ddof=0)
    d = d_all.copy()
    d["uncertainty_std"] = (d["uncertainty"] - mu) / (sd + 1e-8)

    panels = [
        ("All samples",       slice(None)),
        ('Words of "Aha!"',   d["aha_words"] == 1),
        ('LLM-Detected "Aha!"', d["aha_gpt"] == 1),
        ('Formal "Aha!"',     (d["aha_formal"] == 1) if "aha_formal" in d.columns else (d["uncertainty_std"] == 1e9)),
    ]

    fig_size = a4_size_inches(a4_orientation) if a4_pdf else (12.0, 7.6)
    fig, axes = plt.subplots(2, 2, figsize=fig_size, dpi=150, sharex=True, sharey=True)
    axes = np.array(axes).reshape(2,2)
    color_corr, color_inc = "#1f77b4", "#d62728"

    rows = []
    for ax, (label, mask) in zip(axes.flat, panels):
        dd = d if isinstance(mask, slice) else d[mask].copy()
        vals_corr = dd.loc[dd["correct"] == 1, "uncertainty_std"].to_numpy()
        vals_inc  = dd.loc[dd["correct"] == 0, "uncertainty_std"].to_numpy()
        x1, y1 = density_from_hist(vals_corr, edges, smooth_k=int(max(0, smooth_bins)))
        x0, y0 = density_from_hist(vals_inc,  edges, smooth_k=int(max(0, smooth_bins)))
        ax.plot(x1, y1, lw=2.2, color=color_corr, label=f"Correct (n={vals_corr.size})")
        ax.plot(x0, y0, lw=2.2, color=color_inc,  label=f"Incorrect (n={vals_inc.size})", ls="--")
        ax.set_title(f"{label}\n{title_suffix}")
        ax.set_xlabel("uncertainty_std"); ax.set_ylabel("Density")
        ax.grid(True, alpha=0.3)
        for xi, yi1, yi0 in zip(x1, y1, y0):
            rows.append({"panel": label, "bin_center": xi, "density_correct": yi1, "density_incorrect": yi0})

    handles = [Line2D([0],[0], color=color_corr, lw=2.2, label="Correct"),
               Line2D([0],[0], color=color_inc,  lw=2.2, ls="--", label="Incorrect")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.02))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_png)
    if a4_pdf:
        fig.set_size_inches(*a4_size_inches(a4_orientation))
    fig.savefig(out_pdf)
    plt.close(fig)

    out_csv = out_png.replace(".png", ".csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv

# ----------------- Figure 4: per-bin accuracy with Wilson CIs -----------------

def per_bin_accuracy(values_std: np.ndarray, correct: np.ndarray, edges: np.ndarray):
    n = np.histogram(values_std, bins=edges)[0]
    k = np.histogram(values_std, bins=edges, weights=correct.astype(float))[0]
    acc = np.divide(k, n, out=np.full_like(k, np.nan, dtype=float), where=n>0)
    lo, hi = np.full_like(acc, np.nan, dtype=float), np.full_like(acc, np.nan, dtype=float)
    for i, (ki, ni) in enumerate(zip(k.astype(int), n.astype(int))):
        if ni > 0:
            lo[i], hi[i] = wilson_ci(ki, ni)
    centers = 0.5*(edges[:-1] + edges[1:])
    return centers, acc, lo, hi, k.astype(int), n.astype(int)

def plot_accuracy_by_bin_overlay(d_all: pd.DataFrame, edges: np.ndarray, out_png: str, out_pdf: str,
                                 title_suffix: str, a4_pdf: bool, a4_orientation: str) -> str:
    mu = d_all["uncertainty"].mean(); sd = d_all["uncertainty"].std(ddof=0)
    d = d_all.copy()
    d["uncertainty_std"] = (d["uncertainty"] - mu) / (sd + 1e-8)

    mask_all   = np.ones(len(d), dtype=bool)
    mask_words = (d["aha_words"] == 1)
    mask_llm   = (d["aha_gpt"] == 1)
    mask_form  = (d["aha_formal"] == 1) if "aha_formal" in d.columns else np.zeros(len(d), dtype=bool)

    series = [("All", mask_all, "#666666"),
              ("Words", mask_words, "#1f77b4"),
              ("LLM", mask_llm, "#ff7f0e"),
              ("Formal", mask_form, "#2ca02c")]

    fig_size = a4_size_inches(a4_orientation) if a4_pdf else (12.5, 4.8)
    fig, ax = plt.subplots(figsize=fig_size, dpi=150)
    rows = []

    for label, m, color in series:
        vals = d.loc[m, "uncertainty_std"].to_numpy()
        corr = d.loc[m, "correct"].to_numpy().astype(int)
        x, acc, lo, hi, k, n = per_bin_accuracy(vals, corr, edges)
        ax.plot(x, acc, lw=2.0, color=color, label=f"{label}")
        ax.fill_between(x, lo, hi, color=color, alpha=0.15)
        for xi, ai, li, hi_, ki, ni in zip(x, acc, lo, hi, k, n):
            rows.append({"variant": label, "bin_center": xi,
                         "bin_left": xi-(edges[1]-edges[0])/2, "bin_right": xi+(edges[1]-edges[0])/2,
                         "n": int(ni), "k": int(ki), "acc": ai, "lo": li, "hi": hi_})

    ax.set_title(f"Per-bin accuracy with Wilson 95% CIs\n{title_suffix}")
    ax.set_xlabel("uncertainty_std"); ax.set_ylabel("Accuracy"); ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3); ax.legend(loc="best", frameon=False)

    fig.tight_layout()
    fig.savefig(out_png)
    if a4_pdf:
        fig.set_size_inches(*a4_size_inches(a4_orientation))
    fig.savefig(out_pdf)
    plt.close(fig)

    out_csv = out_png.replace(".png", ".csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv

# ----------------- Figure 5: Regression curves (GLM with interaction) -----------------

def fit_glm_interaction(d: pd.DataFrame, aha_col: str, bucket_col: str, ridge_alpha: float = 0.5):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    formula = f"correct ~ C(problem) + C(step) + {aha_col}:C({bucket_col})"
    model = smf.glm(formula, data=d, family=sm.families.Binomial())
    used = "none"
    try:
        res = model.fit(cov_type="HC1")
        if not np.isfinite(res.params).all():
            raise RuntimeError("Unstable MLE; switching to ridge.")
    except Exception:
        res = model.fit_regularized(alpha=float(ridge_alpha), L1_wt=0.0)
        used = "ridge"
    return res, model, used

def predict_formula(res, model, df_new: pd.DataFrame) -> np.ndarray:
    """Robust predict for formula models: prefers res.predict; falls back to patsy design."""
    try:
        return np.asarray(res.predict(df_new))
    except Exception:
        pass
    # Build design matrix with the original design_info
    try:
        from patsy import build_design_matrices
        design_info = getattr(getattr(model, "data", None), "design_info", None)
        X = build_design_matrices([design_info], df_new, return_type="dataframe")[0]
        linpred = np.dot(np.asarray(X), np.asarray(res.params))
        return model.family.link.inverse(linpred)
    except Exception:
        # Final fallback: assume df_new is already numeric exog
        return np.asarray(model.predict(res.params, df_new))

def predict_margins_by_bucket(res, model, base_df: pd.DataFrame,
                              aha_col: str, bucket_col: str,
                              bucket_levels: List[Any],
                              aha_val: int,
                              B: int = 500, rng_seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(rng_seed)
    means, lo, hi = [], [], []
    n = len(base_df); idx = np.arange(n)

    # Ensure bucket column in base_df is categorical with the same ordered levels
    cat = pd.Categorical(base_df[bucket_col], categories=bucket_levels, ordered=True)

    for lev in bucket_levels:
        base = base_df.copy()
        base[aha_col] = int(aha_val)
        base[bucket_col] = pd.Categorical([lev]*n, categories=bucket_levels, ordered=True)

        p = predict_formula(res, model, base)
        mu = float(np.mean(p)); means.append(mu)

        if B > 0:
            bs = np.empty(B, dtype=float)
            for b in range(B):
                take = rng.choice(idx, size=n, replace=True)
                bs[b] = float(np.mean(predict_formula(res, model, base.iloc[take])))
            lo_b, hi_b = np.nanpercentile(bs, [2.5, 97.5])
        else:
            lo_b = hi_b = np.nan
        lo.append(lo_b); hi.append(hi_b)

    return np.array(means), np.array(lo), np.array(hi)

def plot_regression_curves(df: pd.DataFrame, ppx_bucket_col: str, out_png: str, out_pdf: str,
                           dataset: str, model_name: str, B: int = 500,
                           a4_pdf: bool = False, a4_orientation: str = "landscape") -> str:
    variants = [("aha_words","Words"), ("aha_gpt","LLM"), ("aha_formal","Formal")]
    # lock the category order
    df = df.copy()
    df[ppx_bucket_col] = pd.Categorical(df[ppx_bucket_col])   # current order from qcut

    bucket_levels = list(df[ppx_bucket_col].cat.categories)
    bucket_labels = [str(x) for x in bucket_levels]

    fig_size = a4_size_inches(a4_orientation) if a4_pdf else (16.5, 4.8)
    fig, axes = plt.subplots(1, 3, figsize=fig_size, dpi=150, sharey=True)
    colors = {0:"#999999", 1:"#2f5597"}
    rows_for_csv = []

    for ax, (col, title) in zip(axes, variants):
        if col not in df.columns:
            ax.set_visible(False); continue

        d = df.copy()
        d[col] = d[col].astype(int)

        res, mdl, used = fit_glm_interaction(d, aha_col=col, bucket_col=ppx_bucket_col, ridge_alpha=0.5)

        y0, lo0, hi0 = predict_margins_by_bucket(res, mdl, d, col, ppx_bucket_col, bucket_levels, aha_val=0, B=B)
        y1, lo1, hi1 = predict_margins_by_bucket(res, mdl, d, col, ppx_bucket_col, bucket_levels, aha_val=1, B=B)

        x = np.arange(len(bucket_levels))
        ax.plot(x, y0, marker="o", color=colors[0], label=f"{title}: aha=0")
        ax.fill_between(x, lo0, hi0, color=colors[0], alpha=0.15)
        ax.plot(x, y1, marker="o", color=colors[1], label=f"{title}: aha=1")
        ax.fill_between(x, lo1, hi1, color=colors[1], alpha=0.15)

        ax.set_xticks(x); ax.set_xticklabels(bucket_labels, rotation=25, ha="right")
        ax.set_xlabel("perplexity bucket")
        ax.set_title(f"{title}\n{dataset}, {model_name}")
        ax.grid(True, alpha=0.3)

        for i, lev in enumerate(bucket_levels):
            rows_for_csv.append({"variant": title, "bucket": str(lev), "aha": 0,
                                 "mean": y0[i], "lo": lo0[i], "hi": hi0[i]})
            rows_for_csv.append({"variant": title, "bucket": str(lev), "aha": 1,
                                 "mean": y1[i], "lo": lo1[i], "hi": hi1[i]})

    axes[0].set_ylabel("Predicted accuracy")
    fig.legend([Line2D([0],[0], color=colors[0], marker="o", lw=2, label="aha=0"),
                Line2D([0],[0], color=colors[1], marker="o", lw=2, label="aha=1")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.02))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_png)
    if a4_pdf:
        fig.set_size_inches(*a4_size_inches(a4_orientation))
    fig.savefig(out_pdf)
    plt.close(fig)

    out_csv = out_png.replace(".png", ".csv")
    pd.DataFrame(rows_for_csv).to_csv(out_csv, index=False)
    return out_csv

# ----------------- Main -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_root", help="Root containing step*/.../*.jsonl")
    ap.add_argument("--split", default=None, help="Filter filenames by substring (e.g., 'test')")
    ap.add_argument("--out_dir", default=None, help="Default: <results_root>/unc_correct_all")
    ap.add_argument("--dataset_name", default="MATH-500")
    ap.add_argument("--model_name", default="Qwen2.5-1.5B")

    # Uncertainty + GPT label policy
    ap.add_argument("--unc_field", choices=["answer","overall","think"], default="answer")
    ap.add_argument("--gpt_mode", choices=["canonical","broad"], default="canonical")
    ap.add_argument("--gpt_gate_by_words", action="store_true", help="Gate GPT by Words (LLM≤Words).")

    # Formal thresholds
    ap.add_argument("--delta1", type=float, default=0.13)
    ap.add_argument("--delta2", type=float, default=0.13)
    ap.add_argument("--delta3", type=float, default=None)
    ap.add_argument("--min_prior_steps", type=int, default=2)

    # Binning / smoothing
    ap.add_argument("--hist_bins", type=int, default=10)
    ap.add_argument("--density_bins", type=int, default=10)
    ap.add_argument("--acc_bins", type=int, default=10)
    ap.add_argument("--ppx_buckets", type=int, default=8, help="Perplexity quantile buckets for regression.")
    ap.add_argument("--smooth_bins", type=int, default=5, help="Moving-average window for densities (0=off).")
    ap.add_argument("--xlim_std", nargs=2, type=float, default=None,
                    help="Explicit x-limits for uncertainty_std (e.g., --xlim_std -1.0 3.0)")

    # Fonts & A4
    ap.add_argument("--font_family", default="Times New Roman")
    ap.add_argument("--font_size", type=int, default=12)
    ap.add_argument("--a4_pdf", action="store_true", help="Save PDFs at exact A4 page size.")
    ap.add_argument("--a4_orientation", choices=["landscape","portrait"], default="landscape")

    # Bootstrap for GLM predicted CIs
    ap.add_argument("--B_ci", type=int, default=500)

    args = ap.parse_args()

    set_global_fonts(args.font_family, args.font_size)

    out_dir = args.out_dir or os.path.join(args.results_root, "unc_correct_all")
    os.makedirs(out_dir, exist_ok=True)
    slug = f"{slugify(args.dataset_name)}__{slugify(args.model_name)}"
    title_suffix = f"{args.dataset_name}, {args.model_name}"

    files = scan_files(args.results_root, args.split)
    if not files:
        raise SystemExit("No JSONL files found. Check --results_root/--split.")

    # GPT keys
    if args.gpt_mode == "canonical":
        gpt_keys = ["change_way_of_thinking", "shift_in_reasoning_v1"]
    else:
        gpt_keys = ["change_way_of_thinking","shift_in_reasoning_v1","shift_llm","shift_gpt","pivot_llm","rechecked"]

    # Load samples
    d_all = load_pass1_rows(files, args.unc_field, gpt_keys, gate_gpt_by_words=args.gpt_gate_by_words)

    # Formal labeling
    ps = _build_problem_step_for_formal(d_all)
    ps = _mark_formal_pairs(ps, delta1=float(args.delta1), delta2=float(args.delta2),
                            min_prior_steps=int(args.min_prior_steps), delta3=(None if args.delta3 is None else float(args.delta3)))
    d_all = _label_formal_samples(d_all, ps)

    # Standardize & edges
    mu = d_all["uncertainty"].mean(); sd = d_all["uncertainty"].std(ddof=0)
    d_all["uncertainty_std"] = (d_all["uncertainty"] - mu) / (sd + 1e-8)
    edges_hist = make_edges_from_std(d_all["uncertainty_std"].to_numpy(), bins=int(args.hist_bins),   xlim=args.xlim_std)
    edges_den  = make_edges_from_std(d_all["uncertainty_std"].to_numpy(), bins=int(args.density_bins), xlim=args.xlim_std)
    edges_acc  = make_edges_from_std(d_all["uncertainty_std"].to_numpy(), bins=int(args.acc_bins),     xlim=args.xlim_std)

    # ---- FIG 1: count histograms
    out_png_h = os.path.join(out_dir, f"unc_vs_correct_4hists__{slug}.png")
    out_pdf_h = os.path.join(out_dir, f"unc_vs_correct_4hists__{slug}.pdf")
    plot_four_correct_hists(d_all, edges_hist, out_png_h, out_pdf_h, title_suffix,
                            a4_pdf=bool(args.a4_pdf), a4_orientation=args.a4_orientation)

    # ---- FIG 2: overlaid densities (correct only)
    out_png_d = os.path.join(out_dir, f"unc_vs_correct_overlaid__{slug}.png")
    out_pdf_d = os.path.join(out_dir, f"unc_vs_correct_overlaid__{slug}.pdf")
    grid_csv  = plot_overlaid_densities(d_all, edges_den, out_png_d, out_pdf_d, title_suffix,
                                        smooth_bins=int(args.smooth_bins),
                                        a4_pdf=bool(args.a4_pdf), a4_orientation=args.a4_orientation)

    # ---- FIG 3: 2×2 correct vs incorrect by type
    out_png_ci = os.path.join(out_dir, f"unc_vs_corr_incorr_by_type__{slug}.png")
    out_pdf_ci = os.path.join(out_dir, f"unc_vs_corr_incorr_by_type__{slug}.pdf")
    grid_ci_csv = plot_correct_incorrect_by_type(d_all, edges_den, out_png_ci, out_pdf_ci, title_suffix,
                                                 smooth_bins=int(args.smooth_bins),
                                                 a4_pdf=bool(args.a4_pdf), a4_orientation=args.a4_orientation)

    # ---- FIG 4: per-bin accuracy with Wilson CIs (overlay)
    out_png_acc = os.path.join(out_dir, f"unc_accuracy_by_bin__{slug}.png")
    out_pdf_acc = os.path.join(out_dir, f"unc_accuracy_by_bin__{slug}.pdf")
    acc_csv = plot_accuracy_by_bin_overlay(d_all, edges_acc, out_png_acc, out_pdf_acc,
                                           title_suffix, a4_pdf=bool(args.a4_pdf),
                                           a4_orientation=args.a4_orientation)

    # ---- FIG 5: regression curves (your model) by category
    # Build perplexity and quantile buckets
    d_all["perplexity"] = np.exp(d_all["uncertainty"].astype(float))
    d_all["perplexity_bucket"] = pd.qcut(d_all["perplexity"],
                                         q=int(max(3, args.ppx_buckets)),
                                         duplicates="drop").astype("category")

    out_png_reg = os.path.join(out_dir, f"acc_vs_uncertainty_regression__{slug}.png")
    out_pdf_reg = os.path.join(out_dir, f"acc_vs_uncertainty_regression__{slug}.pdf")
    reg_csv = plot_regression_curves(d_all, "perplexity_bucket", out_png_reg, out_pdf_reg,
                                     dataset=args.dataset_name, model_name=args.model_name,
                                     B=int(args.B_ci), a4_pdf=bool(args.a4_pdf),
                                     a4_orientation=args.a4_orientation)

    print("WROTE:")
    print("  (1) Panels (counts):", out_png_h, "|", out_pdf_h)
    print("  (2) Overlaid density:", out_png_d, "|", out_pdf_d, " CSV:", grid_csv)
    print("  (3) Correct vs Incorrect (2×2):", out_png_ci, "|", out_pdf_ci, " CSV:", grid_ci_csv)
    print("  (4) Accuracy by bin:", out_png_acc, "|", out_pdf_acc, " CSV:", acc_csv)
    print("  (5) Regression curves:", out_png_reg, "|", out_pdf_reg, " CSV:", reg_csv)

if __name__ == "__main__":
    main()
