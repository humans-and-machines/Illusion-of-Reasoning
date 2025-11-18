#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
H2: Are Aha! Moments Important During Different Stages of Training?
-------------------------------------------------------------------

Per-step model (for stepwise GLMs):
    correct ~ C(problem) + aha + uncertainty_std [+ aha:uncertainty_std]

Where:
  - correct         : pass1.is_correct_pred (0/1)
  - aha             : source via --aha_source (gpt|native)
  - uncertainty     : entropy; choose --unc_field {answer|overall|think}
  - uncertainty_std : z-score across the full dataset (global scaling)

This version adds:
  • One-figure uncertainty-buckets view for ALL THREE Aha definitions
    (Words, LLM-Detected, Formal) with Wilson 95% CIs.
  • A 100-bin histogram of uncertainty with Aha COUNTS overlaid (Words/GPT/Formal).
  • Formal Aha support with δ1, δ2, optional δ3 gain-at-shift and min_prior_steps.
  • Usual step-wise GLMs (ridge fallback), AME CIs, and pooled step-FE model.

Outputs (default under <results_root>/h2_analysis):
  - h2_pass1_samples.csv
  - h2_step_regression.csv
  - h2_ame_grid.csv
  - h2_balance_by_step.csv
  - h2_fdr_summary.txt
  - h2_interaction_summary.txt (if --interaction)
  - h2_pooled_aha_by_step.csv

  - h2_diag_panel.png
  - aha_coef_vs_step.png
  - aha_ame_vs_step.png
  - aha_ame_with_ci.png
  - uncertainty_coef_vs_step.png
  - naive_delta_vs_step.png
  - aha_ame_grid.png
  - h2_pooled_aha_by_step.png

  # NEW:
  - h2_aha_vs_uncertainty_buckets__<dataset>__<model>.png/.pdf
  - h2_aha_vs_uncertainty_buckets.csv
  - h2_uncertainty_hist_100bins__<dataset>__<model>.png/.pdf
  - h2_uncertainty_hist_100bins.csv
"""

import os
import re
import json
import argparse
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ----------------------------- file scanning -----------------------------
STEP_PAT = re.compile(r"step(\d+)", re.I)
TEMP_PATS = [re.compile(r"temp(?:erature)?[_-]?([0-9]*\.?[0-9]+)", re.I)]

def _get_param(res, name: str, default=np.nan) -> float:
    """Robustly fetch a coefficient by name from statsmodels results, even for fit_regularized."""
    try:
        p = getattr(res, "params", None)
        if p is None:
            return default
        if isinstance(p, pd.Series):
            return float(p.get(name, default))
        names = getattr(getattr(res, "model", None), "exog_names", None)
        if names is not None:
            try:
                idx = names.index(name)
                return float(p[idx])
            except Exception:
                return default
    except Exception:
        pass
    return default

def nat_step_from_path(path: str) -> Optional[int]:
    m = STEP_PAT.search(path)
    return int(m.group(1)) if m else None

def maybe_temp_from_path(path: str) -> Optional[float]:
    for pat in TEMP_PATS:
        m = pat.search(path)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return None

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

# ----------------------------- helpers -----------------------------

def coerce_bool(x) -> Optional[int]:
    if x is None: return None
    if isinstance(x, bool): return int(x)
    if isinstance(x, (int, np.integer)): return int(bool(x))
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("1","true","t","yes","y"): return 1
        if s in ("0","false","f","no","n"): return 0
    return int(bool(x))

def _get_aha_gpt(p1: Dict[str, Any], rec: Dict[str, Any]) -> Optional[int]:
    """Prefer GPT/LLM-labeled shift flags if present (various aliases)."""
    candidates = [
        ("p1", "shift_in_reasoning_v1"), ("p1", "shift_llm"),
        ("p1", "shift_gpt"), ("p1", "pivot_llm"),
        ("p1", "rechecked"), ("root", "rechecked"),
        ("p1", "change_way_of_thinking"), ("root", "change_way_of_thinking"),
    ]
    for loc, key in candidates:
        v = p1.get(key) if loc == "p1" else rec.get(key)
        if v is None: continue
        out = coerce_bool(v)
        if out is not None: return int(out)
    return None

def _get_aha_native(p1: Dict[str, Any]) -> Optional[int]:
    """Native reconsider cue, ignoring any injected cue on PASS-1."""
    aha_raw = coerce_bool(p1.get("has_reconsider_cue"))
    markers = p1.get("reconsider_markers") or []
    if isinstance(markers, list) and ("injected_cue" in markers):
        return 0
    return 0 if aha_raw is None else int(aha_raw)

def _choose_uncertainty(p1: Dict[str, Any], pref: str = "answer") -> Optional[float]:
    """Pick uncertainty (entropy) per preference with sensible fallbacks."""
    if pref == "answer":
        x = p1.get("entropy_answer")
        if x is None: x = p1.get("entropy")
        if x is None: x = p1.get("entropy_think")
        return float(x) if x is not None else None
    if pref == "overall":
        x = p1.get("entropy")
        if x is None: x = p1.get("entropy_answer")
        if x is None: x = p1.get("entropy_think")
        return float(x) if x is not None else None
    if pref == "think":
        x = p1.get("entropy_think")
        if x is None: x = p1.get("entropy")
        if x is None: x = p1.get("entropy_answer")
        return float(x) if x is not None else None
    return None

# ----------------------------- loader -----------------------------

def load_pass1_rows(files: List[str], unc_field: str, aha_source: str) -> pd.DataFrame:
    rows = []
    for path in files:
        step_from_name = nat_step_from_path(path)
        temp_from_path = maybe_temp_from_path(path)
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

                # IDs
                problem = rec.get("problem") or rec.get("clue") or rec.get("row_key")
                if problem is None:
                    di = rec.get("dataset_index")
                    problem = f"idx:{di}" if di is not None else "unknown"
                step = rec.get("step", step_from_name if step_from_name is not None else None)
                if step is None: continue

                # Outcome
                corr_raw = coerce_bool(p1.get("is_correct_pred"))
                if corr_raw is None: continue
                correct = int(corr_raw)

                # Aha
                aha_gpt = _get_aha_gpt(p1, rec)
                aha_native = _get_aha_native(p1)
                if aha_source == "gpt":
                    aha = aha_gpt if aha_gpt is not None else aha_native
                else:
                    aha = aha_native if aha_native is not None else aha_gpt
                if aha is None:  # no label at all -> drop
                    continue
                aha = int(aha)

                # Uncertainty
                unc = _choose_uncertainty(p1, unc_field)
                if unc is None:  # can't control for uncertainty -> drop
                    continue

                # Temperature (if present)
                t = (rec.get("temperature") or p1.get("temperature") or
                     rec.get("config", {}).get("temperature") or temp_from_path)

                rows.append({
                    "problem": str(problem),
                    "step": int(step),
                    "sample_idx": rec.get("sample_idx"),
                    "correct": correct,
                    "aha": aha,
                    "uncertainty": float(unc),
                    "temperature": None if t is None else float(t),
                    "source_file": path,
                })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No usable PASS-1 rows found (missing aha and/or uncertainty).")
    return df

# ----------------------------- modeling -----------------------------

def _fit_glm_force_ridge(d, formula: str, l2: float):
    """Fit Binomial GLM with ridge (L2) directly to avoid MLE IRLS overflows."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    model = smf.glm(formula, data=d, family=sm.families.Binomial())
    res = model.fit_regularized(alpha=float(l2), L1_wt=0.0)
    return res, model, "ridge"

def _fit_glm_with_ridge_if_needed(d: pd.DataFrame, formula: str, l2: float):
    """Try MLE; if unstable, fall back to ridge."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    model = smf.glm(formula, data=d, family=sm.families.Binomial())
    used = "none"
    try:
        res = model.fit(cov_type="HC1")
        if not np.isfinite(res.params).all() or ("aha" in res.params.index and abs(res.params["aha"]) > 10):
            raise RuntimeError("Unstable MLE; switching to ridge.")
    except Exception:
        res = model.fit_regularized(alpha=float(l2), L1_wt=0.0)
        used = "ridge"
    return res, model, used

def _predict_from_formula(res, model, df_new):
    """Predict P(correct=1) for df_new using the fit-time design_info when needed."""
    try:
        return np.asarray(res.predict(df_new))
    except Exception:
        pass
    try:
        from patsy import build_design_matrices
        design_info = getattr(getattr(model, "data", None), "design_info", None)
        X = build_design_matrices([design_info], df_new, return_type="dataframe")[0]
        linpred = np.dot(np.asarray(X), np.asarray(res.params))
        return model.family.link.inverse(linpred)
    except Exception:
        return np.asarray(model.predict(res.params, df_new))

def fit_stepwise_glms(df: pd.DataFrame,
                      out_dir: str,
                      interaction: bool = False,
                      penalty: str = "ridge",
                      l2: float = 1.0,
                      bootstrap_ame: int = 200,
                      ame_grid: int = 9,
                      fdr_alpha: float = 0.05) -> pd.DataFrame:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import multipletests

    steps = sorted(df["step"].unique().tolist())
    rows, ame_grid_rows, bal_rows = [], [], []

    ame_grid = int(max(3, ame_grid))
    u_grid = np.linspace(-2.0, 2.0, ame_grid)

    for s in steps:
        d = df[df["step"] == s].copy()
        if d.empty:
            continue

        # Balance stats
        g0 = d[d["aha"] == 0]; g1 = d[d["aha"] == 1]
        bal_rows.append({
            "step": s,
            "n": len(d),
            "n_aha0": int(len(g0)),
            "n_aha1": int(len(g1)),
            "mean_unc_aha0": float(g0["uncertainty"].mean()) if len(g0) else np.nan,
            "mean_unc_aha1": float(g1["uncertainty"].mean()) if len(g1) else np.nan,
            "aha_ratio": float(d["aha"].mean()),
        })

        if d["aha"].nunique() < 2:
            naive_delta = (g1["correct"].mean() - g0["correct"].mean()) if (len(g0) and len(g1)) else np.nan
            rows.append({
                "step": s, "n": len(d),
                "penalty": "n/a",
                "aha_coef": np.nan, "aha_se": np.nan, "aha_z": np.nan, "aha_p": np.nan,
                "aha_ame": np.nan, "aha_ame_lo": np.nan, "aha_ame_hi": np.nan,
                "inter_coef": np.nan, "inter_se": np.nan, "inter_z": np.nan, "inter_p": np.nan,
                "unc_coef": np.nan, "unc_se": np.nan, "unc_z": np.nan, "unc_p": np.nan,
                "acc": d["correct"].mean(), "aha_ratio": d["aha"].mean(),
                "mean_uncertainty": d["uncertainty"].mean(),
                "naive_delta": naive_delta,
            })
            continue

        # Formula
        formula = "correct ~ C(problem) + aha + uncertainty_std"
        if interaction:
            formula += " + aha:uncertainty_std"

        # Fit
        if penalty in ("ridge", "firth"):
            res, model, used = _fit_glm_force_ridge(d, formula, l2)
        elif penalty == "none":
            res, model, used = _fit_glm_with_ridge_if_needed(d, formula, 0.0)
            used = "none" if used == "none" else "ridge"
        else:
            res, model, used = _fit_glm_force_ridge(d, formula, l2)

        # Coefs + SE/z/p (may be NaN under ridge)
        b_aha = _get_param(res, "aha", np.nan)
        b_unc = _get_param(res, "uncertainty_std", np.nan)
        b_int = np.nan
        for key in ("aha:uncertainty_std", "uncertainty_std:aha"):
            val = _get_param(res, key, np.nan)
            if np.isfinite(val):
                b_int = val
                break
        try:
            se_aha = float(getattr(res, "bse", pd.Series()).get("aha", np.nan))
            se_unc = float(getattr(res, "bse", pd.Series()).get("uncertainty_std", np.nan))
            se_int = np.nan
            for key in ("aha:uncertainty_std", "uncertainty_std:aha"):
                se_int_try = getattr(res, "bse", pd.Series()).get(key, np.nan)
                if np.isfinite(se_int_try):
                    se_int = float(se_int_try); break
            z_aha = b_aha / se_aha if np.isfinite(se_aha) and se_aha else np.nan
            z_unc = b_unc / se_unc if np.isfinite(se_unc) and se_unc else np.nan
            z_int = b_int / se_int if np.isfinite(se_int) and se_int else np.nan
            p_aha = float(getattr(res, "pvalues", pd.Series()).get("aha", np.nan))
            p_unc = float(getattr(res, "pvalues", pd.Series()).get("uncertainty_std", np.nan))
            p_int = np.nan
            for key in ("aha:uncertainty_std", "uncertainty_std:aha"):
                pv = getattr(res, "pvalues", pd.Series()).get(key, np.nan)
                if np.isfinite(pv): p_int = float(pv); break
        except Exception:
            se_aha = se_unc = se_int = z_aha = z_unc = z_int = p_aha = p_unc = p_int = np.nan

        # AME(aha) at mean uncertainty
        base = d.copy()
        ubar = float(base["uncertainty_std"].mean())
        base["uncertainty_std"] = ubar
        d1 = base.copy(); d1["aha"] = 1
        d0 = base.copy(); d0["aha"] = 0
        p1 = _predict_from_formula(res, model, d1)
        p0 = _predict_from_formula(res, model, d0)
        ame = float(np.mean(p1 - p0))

        # Bootstrap CI for AME
        ame_lo = ame_hi = np.nan
        if int(bootstrap_ame) > 0 and len(d) > 10:
            rng = np.random.default_rng(0)
            B = int(bootstrap_ame)
            bs = np.empty(B, dtype=float)
            idx = np.arange(len(d))
            for b in range(B):
                take = rng.choice(idx, size=len(d), replace=True)
                dd = d.iloc[take].copy()
                r2, m2, _ = _fit_glm_force_ridge(dd, formula, l2)
                bb = dd.copy()
                ubar2 = float(bb["uncertainty_std"].mean()); bb["uncertainty_std"] = ubar2
                d1b = bb.copy(); d0b = bb.copy()
                d1b["aha"] = 1; d0b["aha"] = 0
                p1b = _predict_from_formula(r2, m2, d1b)
                p0b = _predict_from_formula(r2, m2, d0b)
                bs[b] = float(np.mean(p1b - p0b))
            ame_lo, ame_hi = np.nanpercentile(bs, [2.5, 97.5])

        naive_delta = d.loc[d["aha"] == 1, "correct"].mean() - d.loc[d["aha"] == 0, "correct"].mean()

        rows.append({
            "step": s, "n": len(d), "penalty": used,
            "aha_coef": float(b_aha), "aha_se": float(se_aha), "aha_z": float(z_aha), "aha_p": float(p_aha),
            "aha_ame": float(ame), "aha_ame_lo": float(ame_lo), "aha_ame_hi": float(ame_hi),
            "inter_coef": float(b_int), "inter_se": float(se_int), "inter_z": float(z_int), "inter_p": float(p_int),
            "unc_coef": float(b_unc), "unc_se": float(se_unc), "unc_z": float(z_unc), "unc_p": float(p_unc),
            "acc": d["correct"].mean(), "aha_ratio": d["aha"].mean(),
            "mean_uncertainty": d["uncertainty"].mean(),
            "naive_delta": float(naive_delta),
        })

    out = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    out.to_csv(os.path.join(out_dir, "h2_step_regression.csv"), index=False)
    return out

# ----------------------------- plotting utils -----------------------------

def _lineplot(x, y, xlabel, ylabel, title, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=140)
    ax.plot(x, y, marker="o")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, alpha=0.3); fig.tight_layout(); fig.savefig(path); plt.close(fig)

def plot_diag_panel(df: pd.DataFrame, out_dir: str):
    """(1) Uncertainty vs Aha, (2) Uncertainty vs step, (3) Aha vs step."""
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), dpi=140)

    # (1) Uncertainty vs Aha
    u0 = df.loc[df["aha"] == 0, "uncertainty_std"].values
    u1 = df.loc[df["aha"] == 1, "uncertainty_std"].values
    try:
        axes[0].boxplot([u0, u1], tick_labels=["aha=0", "aha=1"], showfliers=False)
    except TypeError:
        axes[0].boxplot([u0, u1], labels=["aha=0", "aha=1"], showfliers=False)
    axes[0].set_title("Uncertainty vs Aha"); axes[0].set_ylabel("uncertainty_std")

    # (2) Mean uncertainty vs step
    m = df.groupby("step", as_index=False).agg(mu=("uncertainty_std", "mean"))
    axes[1].plot(m["step"], m["mu"], marker="o")
    axes[1].set_title("Uncertainty vs step"); axes[1].set_xlabel("Training step"); axes[1].set_ylabel("mean uncertainty_std")
    axes[1].grid(True, alpha=0.3)

    # (3) Aha ratio vs step
    ar = df.groupby("step", as_index=False).agg(r=("aha", "mean"))
    axes[2].plot(ar["step"], ar["r"], marker="o")
    axes[2].set_title("Aha vs step"); axes[2].set_xlabel("Training step"); axes[2].set_ylabel("P(aha=1)")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "h2_diag_panel.png")); plt.close(fig)

def plot_ame_with_ci(reg: pd.DataFrame, out_dir: str):
    need = {"aha_ame", "aha_ame_lo", "aha_ame_hi"}
    if not need.issubset(reg.columns): return
    reg = reg.dropna(subset=["aha_ame"])  
    if reg.empty: return
    fig, ax = plt.subplots(figsize=(7.8, 4.6), dpi=140)
    ax.plot(reg["step"], reg["aha_ame"], marker="o", label="AME(aha)")
    if reg[["aha_ame_lo", "aha_ame_hi"]].notna().all().all():
        ax.fill_between(reg["step"], reg["aha_ame_lo"], reg["aha_ame_hi"], alpha=0.2, label="95% CI")
    ax.set_xlabel("Training step"); ax.set_ylabel("AME(aha)"); ax.set_title("Aha AME with bootstrap CI")
    ax.grid(True, alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "aha_ame_with_ci.png")); plt.close(fig)

# ----------------------------- NEW: Buckets + Histogram -----------------------------

def wilson_ci(k: int, n: int, z: float = 1.96):
    if n <= 0: return (np.nan, np.nan)
    p = k / n; z2 = z*z; den = 1 + z2/n
    center = (p + z2/(2*n)) / den
    half = (z * np.sqrt((p*(1-p)/n) + (z2/(4*n*n)))) / den
    return (max(0.0, center - half), min(1.0, center + half))

def _aha_gpt_eff(p1, rec, gate_by_words=False):
    gpt = _get_aha_gpt(p1, rec); words = _get_aha_native(p1)
    if gpt is None: gpt = 0
    if words is None: words = 0
    return int((gpt and words) if gate_by_words else gpt), int(words)

def _load_all_defs_for_buckets(files: List[str], unc_field: str, gate_gpt_by_words: bool):
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

                # IDs
                problem = rec.get("problem") or rec.get("clue") or rec.get("row_key")
                if problem is None:
                    di = rec.get("dataset_index"); problem = f"idx:{di}" if di is not None else "unknown"
                step = rec.get("step", step_from_name if step_from_name is not None else None)
                if step is None: continue
                step = int(step)

                # Outcome
                corr_raw = coerce_bool(p1.get("is_correct_pred"))
                if corr_raw is None: continue
                correct = int(corr_raw)

                # Uncertainty
                unc = _choose_uncertainty(p1, unc_field)
                if unc is None: continue

                # Aha labels
                gpt_eff, words = _aha_gpt_eff(p1, rec, gate_by_words=gate_gpt_by_words)

                rows.append({
                    "problem": str(problem),
                    "step": step,
                    "correct": correct,
                    "uncertainty": float(unc),
                    "aha_gpt": int(gpt_eff),
                    "aha_words": int(words),
                })
    d = pd.DataFrame(rows)
    if d.empty: raise RuntimeError("No rows for buckets/hist figure.")
    return d

def _build_problem_step_for_formal(d_samples: pd.DataFrame) -> pd.DataFrame:
    """
    Per-(step, problem) aggregates needed for Formal:
      - n_samples: #samples at (step, problem)
      - freq_correct: P(correct | q, step)
      - aha_any_gpt / aha_rate_gpt
      - p_correct_given_shift: P(correct | q, step, shift=1) if any shift; else NaN
    """
    # Base aggregates
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

    # Robust P(correct | shift=1): force a Series and name it explicitly
    pcs = (
        d_samples.groupby(["step", "problem"])      # as_index=True -> apply returns a Series
        .apply(lambda g: float(g.loc[g["aha_gpt"] == 1, "correct"].mean())
                        if (g["aha_gpt"] == 1).any() else np.nan)
        .reset_index(name="p_correct_given_shift")
    )

    ps = base.merge(pcs, on=["step", "problem"], how="left")

    # Types
    for c in ("n_samples", "aha_any_gpt"):
        ps[c] = ps[c].astype(int)
    ps["freq_correct"] = ps["freq_correct"].astype(float)
    ps["aha_rate_gpt"] = ps["aha_rate_gpt"].astype(float)
    ps["p_correct_given_shift"] = ps["p_correct_given_shift"].astype(float)

    return ps

def _mark_formal(ps: pd.DataFrame, delta1: float, delta2: float, min_prior_steps: int, delta3: Optional[float]):
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
                if not (prior_ok and shift[j]==1): 
                    flags[idx] = 0
                else:
                    if delta3 is None:
                        flags[idx] = 1
                    else:
                        flags[idx] = int(np.isfinite(p_plus[j]) and ((p_plus[j]-freq[j]) > delta3))
            idx += 1
    out = ps.copy(); out["aha_formal_pair"] = flags; return out

def _label_formal_samples(d_samples: pd.DataFrame, ps_formal: pd.DataFrame):
    """Sample-level Formal=1 if (problem,step) is formal_pair AND sample has GPT shift (=1)."""
    formal_keys = {(str(r["problem"]), int(r["step"])) for _, r in ps_formal.loc[ps_formal["aha_formal_pair"]==1, ["problem","step"]].iterrows()}
    keys = list(zip(d_samples["problem"].astype(str), d_samples["step"].astype(int)))
    d = d_samples.copy()
    d["aha_formal"] = [int((k in formal_keys) and (g==1)) for k, g in zip(keys, d["aha_gpt"].astype(int))]
    return d

def _make_uncertainty_buckets(d_samples: pd.DataFrame, n_buckets: int):
    d = d_samples.copy()
    mu = d["uncertainty"].mean(); sd = d["uncertainty"].std(ddof=0)
    d["uncertainty_std"] = (d["uncertainty"] - mu) / (sd + 1e-8)
    d["unc_bucket"] = pd.qcut(d["uncertainty_std"], q=int(max(3, n_buckets)), duplicates="drop")
    d["bucket_id"] = d["unc_bucket"].cat.codes
    d["bucket_label"] = d["unc_bucket"].astype(str)
    return d

def _aggregate_buckets(d: pd.DataFrame, label_col: str) -> pd.DataFrame:
    g = (d.groupby(["bucket_id","bucket_label"], as_index=False)
           .agg(n=("uncertainty", "size"), k_aha=(label_col,"sum")))
    g["aha_ratio"] = g["k_aha"] / g["n"]
    lo, hi = [], []
    for _, r in g.iterrows():
        l, h = wilson_ci(int(r["k_aha"]), int(r["n"]))
        lo.append(l); hi.append(h)
    g["lo"] = lo; g["hi"] = hi
    return g.sort_values("bucket_id").reset_index(drop=True)

def plot_uncertainty_buckets_three(d_words: pd.DataFrame,
                                   d_gpt: pd.DataFrame,
                                   d_formal: pd.DataFrame,
                                   out_path: str,
                                   title_suffix: str = ""):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), dpi=150, sharey=True)
    items = [
        ("Words of \"Aha!\"", _aggregate_buckets(d_words, "aha_words")),
        ("LLM-Detected \"Aha!\"", _aggregate_buckets(d_gpt, "aha_gpt")),
        ("Formal \"Aha!\"", _aggregate_buckets(d_formal, "aha_formal")),
    ]
    for ax, (ttl, tab) in zip(axes, items):
        ax.plot(tab["bucket_id"], tab["aha_ratio"], marker="o", label="Aha ratio")
        ax.fill_between(tab["bucket_id"], tab["lo"], tab["hi"], alpha=0.2, label="95% CI")
        ax.set_xticks(tab["bucket_id"])
        ax.set_xticklabels(tab["bucket_label"], rotation=25, ha="right")
        ax.set_xlabel("uncertainty_std quantile bucket")
        ax.set_ylabel("Aha ratio")
        ax.set_title(f"{ttl}\n{title_suffix}")
        ax.grid(True, alpha=0.35)
    line = Line2D([0],[0], color="C0", marker="o", label="Aha ratio")
    band = Patch(facecolor="C0", alpha=0.2, label="95% CI")
    fig.legend(handles=[line, band], loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=[0,0.08,1,1])
    fig.savefig(out_path); fig.savefig(out_path.replace(".png",".pdf")); plt.close(fig)

def make_all3_uncertainty_buckets_figure(files: List[str],
                                         out_dir: str,
                                         dataset: str,
                                         model: str,
                                         unc_field: str,
                                         n_buckets: int,
                                         gate_gpt_by_words: bool,
                                         delta1: float, delta2: float,
                                         min_prior_steps: int,
                                         delta3: Optional[float]):
    # 1) Load all samples with Words + GPT
    d_all = _load_all_defs_for_buckets(files, unc_field, gate_gpt_by_words)
    # 2) Formal at (step,problem) then label samples
    ps = _build_problem_step_for_formal(d_all)
    ps = _mark_formal(ps, delta1=delta1, delta2=delta2, min_prior_steps=min_prior_steps, delta3=delta3)
    d_all = _label_formal_samples(d_all, ps)
    # 3) Bucketize (quantiles)
    d_all = _make_uncertainty_buckets(d_all, n_buckets=n_buckets)
    # Views
    d_words  = d_all.copy()
    d_gpt    = d_all.copy()
    d_formal = d_all.copy()
    # Plot & CSV
    slug = f"{dataset.replace(' ','_')}__{model.replace(' ','_')}"
    out_png = os.path.join(out_dir, f"h2_aha_vs_uncertainty_buckets__{slug}.png")
    plot_uncertainty_buckets_three(d_words, d_gpt, d_formal, out_png, title_suffix=f"{dataset}, {model}")

    rows = []
    for name, tab in [("words", _aggregate_buckets(d_words, "aha_words")),
                      ("gpt",   _aggregate_buckets(d_gpt, "aha_gpt")),
                      ("formal",_aggregate_buckets(d_formal, "aha_formal"))]:
        t = tab.copy(); t["series"] = name
        rows.append(t[["bucket_id","bucket_label","n","k_aha","aha_ratio","lo","hi","series"]])
    out_csv = os.path.join(out_dir, "h2_aha_vs_uncertainty_buckets.csv")
    pd.concat(rows, ignore_index=True).to_csv(out_csv, index=False)
    return out_png, out_csv, d_all, ps

def plot_uncertainty_hist_100bins(d_all: pd.DataFrame,
                                  ps: pd.DataFrame,
                                  out_dir: str,
                                  dataset: str,
                                  model: str,
                                  bins: int = 100):
    """
    Histogram of uncertainty_std with total counts and Aha COUNTS per bin
    (Words, GPT, Formal). Writes a PNG/PDF and a CSV with bin stats.
    """
    # Ensure we have std
    mu = d_all["uncertainty"].mean(); sd = d_all["uncertainty"].std(ddof=0)
    d_all = d_all.copy()
    d_all["uncertainty_std"] = (d_all["uncertainty"] - mu) / (sd + 1e-8)

    # Bin edges & centers
    hist_range = (d_all["uncertainty_std"].min(), d_all["uncertainty_std"].max())
    edges = np.linspace(hist_range[0], hist_range[1], int(max(10, bins))+1)
    centers = 0.5*(edges[:-1] + edges[1:])

    # Total histogram
    n_total, _ = np.histogram(d_all["uncertainty_std"].values, bins=edges)

    # Counts for each Aha def
    n_words, _  = np.histogram(d_all.loc[d_all["aha_words"]==1,  "uncertainty_std"].values, bins=edges)
    n_gpt, _    = np.histogram(d_all.loc[d_all["aha_gpt"]==1,    "uncertainty_std"].values, bins=edges)

    # Formal sample-level already in d_all (from _label_formal_samples)
    if "aha_formal" not in d_all.columns:
        # derive from ps if missing (shouldn't happen here)
        formal_keys = {(str(r["problem"]), int(r["step"])) for _, r in ps.loc[ps["aha_formal_pair"]==1, ["problem","step"]].iterrows()}
        keys = list(zip(d_all["problem"].astype(str), d_all["step"].astype(int)))
        d_all["aha_formal"] = [int((k in formal_keys) and (g==1)) for k, g in zip(keys, d_all["aha_gpt"].astype(int))]
    n_formal, _ = np.histogram(d_all.loc[d_all["aha_formal"]==1, "uncertainty_std"].values, bins=edges)

    # CSV
    slug = f"{dataset.replace(' ','_')}__{model.replace(' ','_')}"
    out_csv = os.path.join(out_dir, "h2_uncertainty_hist_100bins.csv")
    pd.DataFrame({
        "bin_left": edges[:-1], "bin_right": edges[1:], "bin_center": centers,
        "n_total": n_total.astype(int),
        "n_words": n_words.astype(int),
        "n_gpt": n_gpt.astype(int),
        "n_formal": n_formal.astype(int),
    }).to_csv(out_csv, index=False)

    # Plot
    fig, ax1 = plt.subplots(figsize=(12.0, 4.8), dpi=150)
    width = (edges[1]-edges[0])
    ax1.bar(centers, n_total, width=width*0.95, color="#CCCCCC", edgecolor="none", label="Total (bin count)")

    ax2 = ax1.twinx()
    ax2.plot(centers, n_words,  marker="o", lw=1.8, label='Words Aha (count)')
    ax2.plot(centers, n_gpt,    marker="o", lw=1.8, label='LLM Aha (count)')
    ax2.plot(centers, n_formal, marker="o", lw=1.8, label='Formal Aha (count)')

    ax1.set_xlabel("uncertainty_std"); ax1.set_ylabel("Total samples per bin")
    ax2.set_ylabel("Aha count per bin")
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"Uncertainty histogram (bins={bins}) with Aha counts\n{dataset}, {model}")

    # Legend combined
    bars_proxy = Patch(facecolor="#CCCCCC", label="Total (bin count)")
    lines = [Line2D([0],[0], color=ax2.lines[i].get_color(), lw=1.8, marker="o", label=ax2.lines[i].get_label()) for i in range(3)]
    leg = ax1.legend(handles=[bars_proxy]+lines, loc="upper left", frameon=False)
    fig.tight_layout()
    out_png = os.path.join(out_dir, f"h2_uncertainty_hist_100bins__{slug}.png")
    fig.savefig(out_png); fig.savefig(out_png.replace(".png",".pdf"))
    plt.close(fig)
    return out_png, out_csv

# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_root", help="Root containing step*/.../*.jsonl")
    ap.add_argument("--split", default=None, help="Filter filenames by substring, e.g. 'test'")
    ap.add_argument("--out_dir", default=None, help="Output dir (default: <results_root>/h2_analysis)")
    ap.add_argument("--min_step", type=int, default=None)
    ap.add_argument("--max_step", type=int, default=None)
    ap.add_argument("--unc_field", choices=["answer","overall","think"], default="answer",
                    help="Which entropy field to use as uncertainty (default: answer entropy).")
    ap.add_argument("--aha_source", choices=["gpt","native"], default="gpt",
                    help="Prefer GPT-labeled shift (default) or native reconsider cue.")
    ap.add_argument("--interaction", action="store_true", help="Include aha×uncertainty_std interaction.")
    ap.add_argument("--compare_native", action="store_true", help="Also fit and plot using native aha labels.")
    ap.add_argument("--penalty", choices=["none","ridge","firth"], default="ridge",
                    help='Penalty for step-wise GLMs; "firth" currently falls back to ridge.')
    ap.add_argument("--l2", type=float, default=1.0, help="L2 strength for ridge when used.")
    ap.add_argument("--bootstrap_ame", type=int, default=200, help="Bootstrap reps for AME CIs (per step).")
    ap.add_argument("--ame_grid", type=int, default=9, help="Number of u grid points in [-2,2] for AME(u).")
    ap.add_argument("--fdr_alpha", type=float, default=0.05, help="BH/FDR alpha for step-wise aha p-values.")

    # NEW: buckets + histogram + Formal thresholds
    ap.add_argument("--unc_buckets", type=int, default=6,
                    help="Number of quantile buckets in the all-3 Aha ratio figure (default: 6).")
    ap.add_argument("--hist_bins", type=int, default=100,
                    help="Number of bins for uncertainty histogram with Aha counts (default: 100).")
    ap.add_argument("--delta1", type=float, default=0.13, help="Formal δ1 prior-failure threshold.")
    ap.add_argument("--delta2", type=float, default=0.13, help="Formal δ2 prior-stability threshold.")
    ap.add_argument("--delta3", type=float, default=None,
                    help="Optional Formal gain-at-shift; if set, require P(correct|shift)-P(correct) > δ3.")
    ap.add_argument("--min_prior_steps", type=int, default=2,
                    help="Formal: require at least this many prior steps.")
    ap.add_argument("--gpt_gate_by_words", action="store_true",
                    help="Gate GPT shifts by Words cue (so LLM<=Words).")

    # Optional dataset/model labels for filenames
    ap.add_argument("--dataset_name", default="MATH-500")
    ap.add_argument("--model_name", default="Qwen2.5-1.5B")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.results_root, "h2_analysis")
    os.makedirs(out_dir, exist_ok=True)

    files = scan_files(args.results_root, args.split)
    if not files:
        raise SystemExit("No JSONL files found. Check path or --split.")

    # -------- Primary run for stepwise GLMs (selected aha_source) --------
    df = load_pass1_rows(files, args.unc_field, args.aha_source)
    if args.min_step is not None: df = df[df["step"] >= args.min_step]
    if args.max_step is not None: df = df[df["step"] <= args.max_step]
    if df.empty: raise SystemExit("No rows left after step filtering.")

    # Global standardization
    mu = df["uncertainty"].mean(); sd = df["uncertainty"].std(ddof=0)
    df["uncertainty_std"] = (df["uncertainty"] - mu) / (sd + 1e-8)

    # Save samples
    samples_csv = os.path.join(out_dir, "h2_pass1_samples.csv"); df.to_csv(samples_csv, index=False)

    # Per-step GLMs
    reg = fit_stepwise_glms(df, out_dir,
                            interaction=args.interaction,
                            penalty=args.penalty,
                            l2=args.l2,
                            bootstrap_ame=args.bootstrap_ame,
                            ame_grid=args.ame_grid,
                            fdr_alpha=args.fdr_alpha)

    # Diagnostics & standard plots
    plot_diag_panel(df, out_dir)
    if not reg.empty:
        _lineplot(reg["step"], reg["aha_coef"], "Training step", "β(aha)", "Aha coefficient vs. step",
                  os.path.join(out_dir, "aha_coef_vs_step.png"))
        _lineplot(reg["step"], reg["aha_ame"], "Training step", "AME(aha)", "Aha average marginal effect vs. step",
                  os.path.join(out_dir, "aha_ame_vs_step.png"))
        _lineplot(reg["step"], reg["unc_coef"], "Training step", "β(uncertainty_std)",
                  "Uncertainty coefficient vs. step",
                  os.path.join(out_dir, "uncertainty_coef_vs_step.png"))
        _lineplot(reg["step"], reg["naive_delta"], "Training step",
                  "Δ accuracy (aha=1 − aha=0)", "Naïve Δaccuracy vs. step",
                  os.path.join(out_dir, "naive_delta_vs_step.png"))
        plot_ame_with_ci(reg, out_dir)

    # -------- NEW: All-3 Aha uncertainty buckets (quantiles) --------
    buckets_png, buckets_csv, d_all, ps_formal = make_all3_uncertainty_buckets_figure(
        files=files,
        out_dir=out_dir,
        dataset=args.dataset_name,
        model=args.model_name,
        unc_field=args.unc_field,
        n_buckets=int(args.unc_buckets),
        gate_gpt_by_words=bool(args.gpt_gate_by_words),
        delta1=float(args.delta1), delta2=float(args.delta2),
        min_prior_steps=int(args.min_prior_steps),
        delta3=(None if args.delta3 is None else float(args.delta3)),
    )
    print("Buckets figure:", buckets_png)
    print("Buckets CSV   :", buckets_csv)

    # -------- NEW: 100-bin uncertainty histogram with Aha COUNTS --------
    hist_png, hist_csv = plot_uncertainty_hist_100bins(
        d_all=d_all, ps=ps_formal, out_dir=out_dir,
        dataset=args.dataset_name, model=args.model_name,
        bins=int(args.hist_bins)
    )
    print("Histogram (100 bins):", hist_png)
    print("Histogram CSV       :", hist_csv)

    # Pooled model (ridge-regularized step FE with aha×step)
    try:
        import statsmodels.api as sm; import statsmodels.formula.api as smf
        formula = "correct ~ C(problem) + C(step) + aha + uncertainty_std + aha:C(step)"
        model = smf.glm(formula, data=df, family=sm.families.Binomial())
        res = model.fit_regularized(alpha=max(0.1, args.l2), L1_wt=0.0)
        base = _get_param(res, "aha", 0.0); rows = []
        for s in sorted(df["step"].unique()):
            t1 = f"aha:C(step)[T.{s}]"; t2 = f"C(step)[T.{s}]:aha"
            delta = _get_param(res, t1, np.nan);  delta = _get_param(res, t2, 0.0) if not np.isfinite(delta) else delta
            effect = float(base) + (float(delta) if np.isfinite(delta) else 0.0)
            rows.append({"step": int(s), "aha_effect": effect})
        pooled = pd.DataFrame(rows).sort_values("step")
        pooled.to_csv(os.path.join(out_dir, "h2_pooled_aha_by_step.csv"), index=False)
        fig, ax = plt.subplots(figsize=(7.8, 4.6), dpi=140)
        ax.plot(pooled["step"], pooled["aha_effect"], marker="o")
        ax.set_xlabel("Training step"); ax.set_ylabel("Pooled effect of aha (log-odds)")
        ax.set_title("Pooled GLM (step FE): per-step aha effect (ridge)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "h2_pooled_aha_by_step.png")); plt.close(fig)
    except Exception:
        print("Pooled model skipped (statsmodels not available).")

    # Console recap
    print(f"Wrote samples CSV: {samples_csv}")
    print(f"Wrote step regression CSV: {os.path.join(out_dir, 'h2_step_regression.csv')}")
    print("Plots written:")
    print("  h2_diag_panel.png")
    print("  aha_coef_vs_step.png, aha_ame_vs_step.png, uncertainty_coef_vs_step.png, naive_delta_vs_step.png")
    print("  aha_ame_with_ci.png, aha_ame_grid.png (if grid used elsewhere)")
    print("  h2_pooled_aha_by_step.png")
    print("  h2_aha_vs_uncertainty_buckets__*.png/.pdf, h2_uncertainty_hist_100bins__*.png/.pdf")
    print("CSVs:")
    print("  h2_balance_by_step.csv, h2_ame_grid.csv, h2_fdr_summary.txt, h2_pooled_aha_by_step.csv")
    print("  h2_aha_vs_uncertainty_buckets.csv, h2_uncertainty_hist_100bins.csv")

if __name__ == "__main__":
    main()
