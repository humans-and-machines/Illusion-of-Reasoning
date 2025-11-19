#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
H3: Can Aha!/Second-Pass Help When the Model is Uncertain?

We test whether the *second pass* (phase=1) improves accuracy more when the
model is uncertain, using PASS-1 uncertainty as the moderator.

Inputs
------
- Root directory containing step*/.../*.jsonl produced by your two-pass runners.
  Each JSON line has keys:
    problem/clue/row_key, step, sample_idx,
    pass1{...}, pass2{...} with fields:
      is_correct_pred (bool), entropy / entropy_answer / entropy_think (floats)

What this script builds
-----------------------
1) Pair-level (wide) table: one row per (problem, step, sample_idx)
   with PASS-1/2 correctness and PASS-1 uncertainty.

2) Long table: duplicates each pair into two rows:
      phase=0 → PASS-1, phase=1 → PASS-2
   Includes: correct (0/1), uncertainty (from PASS-1), bucket (by quantiles)

3) Pooled GLM (Binomial, robust HC1 SE):
      correct ~ C(problem) + C(step) + phase + uncertainty_std + phase:uncertainty_std
   Effect of "phase" tells us if second-pass helps on average; the interaction
   tells us if that help grows with uncertainty.

4) Bucket GLM (heterogeneous effect across uncertainty buckets):
      correct ~ C(problem) + C(step) + phase + C(bucket) + phase:C(bucket)
   Also computes per-bucket AME of toggling phase 0→1.

5) Plots:
   - Accuracy by uncertainty bucket for phase 0 vs 1
   - Per-bucket AME (phase toggle)

CLI
---
python h3-analysis.py /path/to/results \
  --split test \
  --uncertainty_field entropy_answer \
  --num_buckets 4

Notes
-----
- Uses PASS-1 uncertainty for both phases (keeps moderator fixed within pair).
- Robust to missing fields; pairs without both passes are dropped from modeling.
"""

import os
import re
import json
import argparse
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------- utils -----------------------------

STEP_PAT = re.compile(r"step(\d+)", re.I)

def nat_step_from_path(path: str) -> Optional[int]:
    m = STEP_PAT.search(path)
    return int(m.group(1)) if m else None

def scan_files(root: str, split_substr: Optional[str]) -> List[str]:
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if not fn.endswith(".jsonl"):
                continue
            if split_substr and split_substr not in fn:
                continue
            out.append(os.path.join(dp, fn))
    out.sort(key=lambda p: (nat_step_from_path(p) or 0, p))
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

# -------------------------- data loading -------------------------

def load_pairs(files: List[str],
               uncertainty_field: str = "entropy") -> pd.DataFrame:
    """
    Build a PAIRS (wide) DataFrame with columns:
      problem, step, sample_idx,
      correct_p1, correct_p2,
      unc1, unc2 (unc2 kept for reference, not used in models),
      source_file
    Only keep pairs that have both pass1 and pass2 correctness.
    """
    rows = []
    for path in files:
        step_from_name = nat_step_from_path(path)
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rec = json.loads(ln)
                except Exception:
                    continue

                # Identify the "problem" robustly across runners
                prob = rec.get("problem") or rec.get("clue") or rec.get("row_key")
                if prob is None:
                    di = rec.get("dataset_index")
                    prob = f"idx:{di}" if di is not None else "unknown"

                step = rec.get("step", step_from_name if step_from_name is not None else None)
                if step is None:
                    continue

                p1 = rec.get("pass1") or {}
                p2 = rec.get("pass2") or {}
                if not p1 or not p2:
                    # Need both passes for H3 comparisons
                    continue

                c1 = coerce_bool(p1.get("is_correct_pred"))
                c2 = coerce_bool(p2.get("is_correct_pred"))
                if c1 is None or c2 is None:
                    continue

                # Uncertainty (we'll use PASS-1 as moderator)
                u1 = p1.get(uncertainty_field)
                u2 = p2.get(uncertainty_field)

                rows.append(dict(
                    problem=str(prob),
                    step=int(step),
                    sample_idx=rec.get("sample_idx", None),
                    correct_p1=int(c1),
                    correct_p2=int(c2),
                    unc1=None if u1 is None else float(u1),
                    unc2=None if u2 is None else float(u2),
                    source_file=path
                ))

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No pairs found with both PASS-1 and PASS-2. "
                           "Check --split, paths, or that pass2 exists in logs.")
    return df

def pairs_to_long(df_pairs: pd.DataFrame,
                  num_buckets: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert pairs→long:
      phase=0 (pass1), phase=1 (pass2)
    Uncertainty is taken from PASS-1 (unc1) for both rows.

    Adds:
      - uncertainty_std (z-score)
      - bucket (0..num_buckets-1) by quantiles on unc1
    """
    # Drop rows without unc1 (for modeling); keep a copy for descriptive CSV.
    df_pairs_model = df_pairs.dropna(subset=["unc1"]).copy()

    long_rows = []
    for _, r in df_pairs_model.iterrows():
        pair_id = f"{r['problem']}||{int(r['step'])}||{r.get('sample_idx', 'NA')}"
        for p in (0, 1):
            long_rows.append(dict(
                problem=r["problem"],
                step=int(r["step"]),
                pair_id=pair_id,
                phase=p,  # 0=P1, 1=P2
                correct=int(r["correct_p2"] if p == 1 else r["correct_p1"]),
                uncertainty=float(r["unc1"]),
            ))
    long_df = pd.DataFrame(long_rows)

    # Standardize uncertainty
    mu = long_df["uncertainty"].mean()
    sd = long_df["uncertainty"].std(ddof=0)
    long_df["uncertainty_std"] = (long_df["uncertainty"] - mu) / (sd + 1e-8)

    # Buckets based on PASS-1 uncertainty distribution (shared across phases)
    # Compute on unique pairs to avoid double-counting
    uq = df_pairs_model[["unc1"]].copy()
    uq = uq.rename(columns={"unc1": "unc"})
    try:
        uq["bucket"] = pd.qcut(uq["unc"], q=num_buckets, labels=False, duplicates="drop")
    except ValueError:
        # Not enough unique values; fall back to single bucket
        uq["bucket"] = 0
    # Map back
    edges = sorted(uq["bucket"].dropna().unique().tolist())
    if not edges:
        long_df["bucket"] = 0
    else:
        # Build a simple rank-based bucketer against the original unc1 values
        # to keep consistent with qcut above:
        # Create a mapping by merging on unc
        df_pairs_model = df_pairs_model.merge(uq[["unc", "bucket"]].drop_duplicates(),
                                              left_on="unc1", right_on="unc", how="left")
        bucket_map = dict(zip(df_pairs_model["pair_id"] if "pair_id" in df_pairs_model else
                              (df_pairs_model["problem"].astype(str) + "||" +
                               df_pairs_model["step"].astype(str) + "||" +
                               df_pairs_model["sample_idx"].astype(str)),
                              df_pairs_model["bucket"]))
        # annotate long_df
        # Make the same pair_id key we used above
        if "pair_id" not in df_pairs_model.columns:
            df_pairs_model["pair_id"] = (df_pairs_model["problem"].astype(str) + "||" +
                                         df_pairs_model["step"].astype(str) + "||" +
                                         df_pairs_model["sample_idx"].astype(str))
        long_df["bucket"] = long_df["pair_id"].map(bucket_map).fillna(0).astype(int)

    # Ensure categorical types for fixed effects
    long_df["problem"] = long_df["problem"].astype(str)
    long_df["step"] = long_df["step"].astype(int)

    return df_pairs_model, long_df

# --------------------------- modeling -----------------------------

def fit_pooled_glm(df: pd.DataFrame, out_txt: str) -> Dict[str, float]:
    """
    GLM (Binomial, logit link, robust HC1 SE):
      correct ~ C(problem) + C(step) + phase + uncertainty_std + phase:uncertainty_std
    Returns key stats for phase and the interaction; writes full summary.
    """
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except Exception as e:
        raise RuntimeError("statsmodels is required. pip install statsmodels") from e

    model = smf.glm(
        "correct ~ C(problem) + C(step) + phase + uncertainty_std + phase:uncertainty_std",
        data=df,
        family=sm.families.Binomial()
    )
    res = model.fit(cov_type="HC1")

    with open(out_txt, "w", encoding="utf-8") as fh:
        fh.write(res.summary().as_text())
        fh.write("\n")

    params, bse, pvals = res.params, res.bse, res.pvalues

    stats = dict(
        b_phase=float(params.get("phase", np.nan)),
        se_phase=float(bse.get("phase", np.nan)),
        p_phase=float(pvals.get("phase", np.nan)),
        b_unc=float(params.get("uncertainty_std", np.nan)),
        se_unc=float(bse.get("uncertainty_std", np.nan)),
        p_unc=float(pvals.get("uncertainty_std", np.nan)),
        b_phase_x_unc=float(params.get("phase:uncertainty_std", np.nan)),
        se_phase_x_unc=float(bse.get("phase:uncertainty_std", np.nan)),
        p_phase_x_unc=float(pvals.get("phase:uncertainty_std", np.nan)),
    )

    # Average Marginal Effect (AME) of toggling phase 0→1 (overall)
    df1 = df.copy(); df1["phase"] = 1
    df0 = df.copy(); df0["phase"] = 0
    p1 = res.predict(df1)
    p0 = res.predict(df0)
    stats["ame_phase"] = float(np.mean(p1 - p0))

    with open(out_txt, "a", encoding="utf-8") as fh:
        fh.write(f"\nAverage Marginal Effect (phase 0→1): {stats['ame_phase']:.4f}\n")

    return stats

def fit_bucket_glm(df: pd.DataFrame, out_txt: str, num_buckets: int) -> pd.DataFrame:
    """
    Heterogeneous effect by uncertainty bucket:
      correct ~ C(problem) + C(step) + phase + C(bucket) + phase:C(bucket)

    Returns a DataFrame with per-bucket log-odds effect and AME estimates.
    """
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except Exception as e:
        raise RuntimeError("statsmodels is required. pip install statsmodels") from e

    model = smf.glm(
        "correct ~ C(problem) + C(step) + phase + C(bucket) + phase:C(bucket)",
        data=df,
        family=sm.families.Binomial()
    )
    res = model.fit(cov_type="HC1")

    with open(out_txt, "w", encoding="utf-8") as fh:
        fh.write(res.summary().as_text())
        fh.write("\n")

    # Log-odds base effect for phase
    base_phase = float(res.params.get("phase", np.nan))

    # Build per-bucket log-odds deltas (phase toggle)
    buckets = sorted(df["bucket"].unique().tolist())
    recs = []
    for b in buckets:
        term = f"phase:C(bucket)[T.{b}]"
        interaction = float(res.params.get(term, 0.0))
        beta = base_phase + interaction  # log-odds change for bucket b

        # AME in bucket b: predict with phase toggled
        df_b = df[df["bucket"] == b].copy()
        if df_b.empty:
            ame = np.nan
        else:
            df1 = df_b.copy(); df1["phase"] = 1
            df0 = df_b.copy(); df0["phase"] = 0
            ame = float(np.mean(res.predict(df1) - res.predict(df0)))

        p_beta = res.pvalues.get("phase", np.nan)
        if term in res.pvalues:
            # Wald p for the *total* effect isn't directly given; we report the base p
            # and the interaction p separately for transparency.
            p_inter = float(res.pvalues[term])
        else:
            p_inter = np.nan

        recs.append(dict(
            bucket=int(b),
            log_odds_phase=beta,
            p_interaction=p_inter,
            ame_phase=ame
        ))

    out = pd.DataFrame(recs).sort_values("bucket").reset_index(drop=True)
    return out

# -------------------------- visualization ------------------------

def plot_acc_by_bucket(long_df: pd.DataFrame, out_png: str):
    """
    Accuracy by bucket for phase 0 vs 1.
    """
    agg = (long_df
           .groupby(["bucket", "phase"], as_index=False)
           .agg(acc=("correct","mean"), n=("correct","size")))
    buckets = sorted(agg["bucket"].unique())

    fig, ax = plt.subplots(figsize=(7.5,4.5), dpi=140)
    for ph in (0,1):
        sub = agg[agg["phase"] == ph]
        ax.plot(sub["bucket"], sub["acc"], marker="o", label=f"phase={ph}")
    ax.set_xticks(buckets)
    ax.set_xlabel("Uncertainty bucket (by PASS-1)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by uncertainty bucket (phase 0 vs 1)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

def plot_ame_by_bucket(bucket_df: pd.DataFrame, out_png: str):
    """
    Per-bucket AME of phase toggle (probability units).
    """
    fig, ax = plt.subplots(figsize=(7.5,4.5), dpi=140)
    ax.plot(bucket_df["bucket"], bucket_df["ame_phase"], marker="o")
    ax.axhline(0.0, ls="--", lw=1)
    ax.set_xticks(bucket_df["bucket"].tolist())
    ax.set_xlabel("Uncertainty bucket (by PASS-1)")
    ax.set_ylabel("Δ Accuracy from phase 0→1 (AME)")
    ax.set_title("Phase effect by uncertainty bucket (AME)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

# ----------------------------- main ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_root", help="Root with step*/.../*.jsonl")
    ap.add_argument("--split", default=None, help="Substring to filter filenames (e.g. 'test')")
    ap.add_argument("--out_dir", default=None, help="Output dir (default: <root>/h3_analysis)")
    ap.add_argument("--uncertainty_field", default="entropy",
                    choices=["entropy", "entropy_answer", "entropy_think"],
                    help="Which field to use as PASS-1 uncertainty moderator")
    ap.add_argument("--num_buckets", type=int, default=4, help="Quantile buckets for uncertainty")
    ap.add_argument("--min_step", type=int, default=None, help="Optional min step")
    ap.add_argument("--max_step", type=int, default=None, help="Optional max step")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.results_root, "h3_analysis")
    os.makedirs(out_dir, exist_ok=True)

    files = scan_files(args.results_root, args.split)
    if not files:
        raise SystemExit("No JSONL files found. Check the path or --split.")

    # Build pairs
    pairs = load_pairs(files, uncertainty_field=args.uncertainty_field)

    # Optional step filtering
    if args.min_step is not None:
        pairs = pairs[pairs["step"] >= args.min_step]
    if args.max_step is not None:
        pairs = pairs[pairs["step"] <= args.max_step]

    if pairs.empty:
        raise SystemExit("No pairs after filtering.")

    # Persist the raw/wide pairs
    pairs_csv = os.path.join(out_dir, "h3_pairs.csv")
    pairs.to_csv(pairs_csv, index=False)

    # Long format with standardized uncertainty + buckets
    pairs_model, long_df = pairs_to_long(pairs, num_buckets=args.num_buckets)
    long_csv = os.path.join(out_dir, "h3_long.csv")
    long_df.to_csv(long_csv, index=False)

    # Pooled GLM
    pooled_txt = os.path.join(out_dir, "pooled_glm.txt")
    pooled_stats = fit_pooled_glm(long_df, pooled_txt)

    # Bucket GLM
    bucket_txt = os.path.join(out_dir, "bucket_glm.txt")
    bucket_df = fit_bucket_glm(long_df, bucket_txt, num_buckets=args.num_buckets)
    bucket_csv = os.path.join(out_dir, "bucket_effects.csv")
    bucket_df.to_csv(bucket_csv, index=False)

    # Plots
    acc_png = os.path.join(out_dir, "acc_by_bucket.png")
    plot_acc_by_bucket(long_df, acc_png)

    ame_png = os.path.join(out_dir, "ame_by_bucket.png")
    plot_ame_by_bucket(bucket_df, ame_png)

    # Console recap
    print(f"Wrote PAIRS CSV: {pairs_csv}")
    print(f"Wrote LONG CSV:  {long_csv}")
    print(f"Wrote pooled GLM summary:  {pooled_txt}")
    print(f"Wrote bucket GLM summary:  {bucket_txt}")
    print(f"Wrote bucket effects CSV:  {bucket_csv}")
    print(f"Wrote plots: {acc_png} and {ame_png}\n")

    print("Key pooled effects (log-odds, robust SE):")
    print(f"  β_phase           = {pooled_stats['b_phase']:.4f} "
          f"(se={pooled_stats['se_phase']:.4f}, p={pooled_stats['p_phase']:.3g})")
    print(f"  β_uncertainty     = {pooled_stats['b_unc']:.4f} "
          f"(se={pooled_stats['se_unc']:.4f}, p={pooled_stats['p_unc']:.3g})")
    print(f"  β_phase×uncertainty = {pooled_stats['b_phase_x_unc']:.4f} "
          f"(se={pooled_stats['se_phase_x_unc']:.4f}, p={pooled_stats['p_phase_x_unc']:.3g})")
    print(f"  AME(phase 0→1)    = {pooled_stats['ame_phase']:.4f}")

if __name__ == "__main__":
    main()
