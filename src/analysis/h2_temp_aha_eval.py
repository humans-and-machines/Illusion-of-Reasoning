#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
H2: Temperature x Aha! analysis (+ stages of training, + three Aha definitions)
-------------------------------------------------------------------------------
Evaluate whether temperature (default vs low-temp=0.3) and Aha markers affect accuracy
across training stages, using only overlapping (problem, step[, sample_idx]) between runs.

Aha definitions:
  • words  : pass1.has_reconsider_cue (excl. injected)
  • gpt    : canonical LLM shift (optionally gated by words)
  • formal : problem–step level: prior failure & prior shift-stability & shift now,
             computed within each temperature run, then AND-gated with sample-level GPT Aha

CLI:
  --aha {words|gpt|gpt_broad|formal|none|all}
  --gate_gpt_by_words (for gpt/gpt_broad)
  --formal_delta1 0.13  --formal_delta2 0.13  --formal_min_prior_steps 2

Stage bucketing:
  --stage_mode {quantile|fixed}
  --stage_quantiles 0.33 0.66
  --stage_bounds lo hi

Outputs (per Aha mode; under <out_dir>/<aha_mode>/ when --aha all):
  - h2_glm_summary.txt
  - h2_glm_coefficients.csv
  - h2_group_accuracy.csv
  - h2_group_accuracy_by_step.csv
  - h2_stage_group_accuracy.csv
  - h2_stage_aha_counts.csv
  - h2_stage_glm_summary.txt
  - h2_stage_glm_coefficients.csv
  - h2_stage_info.json
  - h2_overlap_summary.json (kept once at top-level when single mode; per-mode when --aha all)
"""

import os, re, json, argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------- file scanning & misc ----------

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
                pass  # record-level split filtering happens in loader
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

# ---------- Aha helpers ----------

def _aha_words(p1: Dict[str, Any]) -> int:
    v = coerce_bool(p1.get("has_reconsider_cue"))
    markers = p1.get("reconsider_markers") or []
    if isinstance(markers, list) and ("injected_cue" in markers):
        return 0  # exclude injected/forced cue
    return 0 if v is None else int(v)

def _aha_gpt_canonical(p1: Dict[str, Any], rec: Dict[str, Any]) -> int:
    for k in ("change_way_of_thinking", "shift_in_reasoning_v1"):
        v = p1.get(k, rec.get(k, None))
        if v is not None and coerce_bool(v) == 1:
            return 1
    return 0

def _aha_gpt_broad(p1: Dict[str, Any], rec: Dict[str, Any]) -> int:
    keys = ["change_way_of_thinking","shift_in_reasoning_v1","shift_llm","shift_gpt","pivot_llm","rechecked"]
    for k in keys:
        v = p1.get(k, rec.get(k, None))
        if v is not None and coerce_bool(v) == 1:
            return 1
    return 0

# ---------- load a root into per-sample rows ----------

def load_samples(root: str,
                 split_filter: Optional[str],
                 aha_mode: str = "gpt",
                 gate_gpt_by_words: bool = True) -> pd.DataFrame:
    """
    Returns per-sample rows with columns:
      dataset, model, split, problem, step, sample_idx, correct, aha_words, aha_gpt, aha
    """
    files = scan_files(root, split_filter)
    rows = []
    for path in files:
        step_from_name = nat_step_from_path(path)
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                if not ln.strip(): continue
                try:
                    rec = json.loads(ln)
                except Exception:
                    continue

                if split_filter is not None:
                    if str(rec.get("split", "")).lower() != str(split_filter).lower():
                        continue

                p1 = rec.get("pass1") or rec.get("p1") or rec.get("first_pass") or {}
                if not p1:
                    continue

                step = rec.get("step", step_from_name if step_from_name is not None else None)
                if step is None:
                    continue

                prob = rec.get("problem") or rec.get("question") or rec.get("row_key")
                if prob is None:
                    di = rec.get("dataset_index")
                    prob = f"idx:{di}" if di is not None else "unknown"

                correct = coerce_bool(p1.get("is_correct_pred") if "is_correct_pred" in p1 else p1.get("is_correct", p1.get("correct")))
                if correct is None:
                    continue

                # Aha features
                words = _aha_words(p1)
                if aha_mode == "gpt":
                    gpt_raw = _aha_gpt_canonical(p1, rec)
                elif aha_mode == "gpt_broad":
                    gpt_raw = _aha_gpt_broad(p1, rec)
                else:
                    gpt_raw = 0
                gpt = int(gpt_raw and words) if (aha_mode.startswith("gpt") and gate_gpt_by_words) else int(gpt_raw)

                aha = words if aha_mode == "words" else (gpt if aha_mode.startswith("gpt") else 0)

                rows.append({
                    "dataset": rec.get("dataset"),
                    "model":   rec.get("model"),
                    "split":   rec.get("split"),
                    "problem": str(prob),
                    "step":    int(step),
                    "sample_idx": rec.get("sample_idx", rec.get("sample_index", rec.get("idx", rec.get("i", None)))),
                    "correct": int(correct),
                    "aha_words": int(words),
                    "aha_gpt":   int(gpt),
                    "aha": int(aha),  # generic slot for selected aha
                })
    return pd.DataFrame(rows)

# ---------- overlap + merge helpers ----------

def restrict_to_overlap(dfA: pd.DataFrame, dfB: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Intersect on keys. Always require (problem, step); include sample_idx if present on both.
    Returns filtered (dfA, dfB, keys_used).
    """
    keys = ["problem","step"]
    use_idx = ("sample_idx" in dfA.columns) and ("sample_idx" in dfB.columns) \
              and dfA["sample_idx"].notna().any() and dfB["sample_idx"].notna().any()
    if use_idx:
        keys.append("sample_idx")

    keyA = dfA[keys].drop_duplicates()
    keyB = dfB[keys].drop_duplicates()
    common = keyA.merge(keyB, on=keys, how="inner")
    if common.empty:
        raise SystemExit("No overlapping (problem, step[, sample_idx]) between runs.")

    mergedA = common.merge(dfA, on=keys, how="inner").sort_values(keys).reset_index(drop=True)
    mergedB = common.merge(dfB, on=keys, how="inner").sort_values(keys).reset_index(drop=True)
    return mergedA, mergedB, keys

# ---------- stage bucketing ----------

def assign_stage(df: pd.DataFrame,
                 mode: str = "quantile",
                 quantiles: Tuple[float,float] = (0.33, 0.66),
                 bounds: Optional[Tuple[int,int]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    d = df.copy()
    steps = np.sort(d["step"].unique())
    info: Dict[str, Any] = {"mode": mode}

    if mode == "fixed":
        if not bounds or len(bounds) != 2:
            raise SystemExit("--stage_mode fixed requires --stage_bounds lo hi")
        lo, hi = int(bounds[0]), int(bounds[1])
        info.update({"bounds": [lo, hi]})
    else:
        q1, q2 = quantiles
        lo = int(np.quantile(steps, q1))
        hi = int(np.quantile(steps, q2))
        info.update({"quantiles": [float(q1), float(q2)], "cutpoints": [lo, hi]})

    def _lab(s):
        if s <= lo: return "early"
        if s <= hi: return "mid"
        return "late"

    d["stage"] = d["step"].apply(_lab)
    return d, info

# ---------- FORMAL Aha (problem–step) ----------

def build_problem_step(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per (problem, step, run_label) aggregate from sample-level:
      n_samples, freq_correct, aha_any_gpt, aha_rate_gpt
    """
    cols_needed = ["problem","step","run_label","correct","aha_gpt"]
    if not set(cols_needed).issubset(df.columns):
        raise ValueError("build_problem_step: missing columns")
    g = (df.groupby(["run_label","problem","step"], as_index=False)
           .agg(n_samples=("correct","size"),
                freq_correct=("correct","mean"),
                aha_any_gpt=("aha_gpt","max"),
                aha_rate_gpt=("aha_gpt","mean")))
    g[["n_samples","aha_any_gpt"]] = g[["n_samples","aha_any_gpt"]].astype(int)
    return g.sort_values(["run_label","problem","step"]).reset_index(drop=True)

def mark_formal_by_run(ps: pd.DataFrame,
                       delta1: float,
                       delta2: float,
                       min_prior_steps: int) -> pd.DataFrame:
    """
    For each run_label & problem, mark formal at step j if:
      max past freq_correct < delta1  AND  max past aha_rate_gpt < delta2  AND  aha_any_gpt[j] == 1
    Adds column 'aha_formal_ps' (0/1).
    """
    need = {"run_label","problem","step","freq_correct","aha_rate_gpt","aha_any_gpt"}
    if not need.issubset(ps.columns): raise ValueError("mark_formal_by_run: missing cols")
    ps = ps.sort_values(["run_label","problem","step"]).copy()
    flags = np.zeros(len(ps), dtype=int); idx = 0
    for (run, prob), sub in ps.groupby(["run_label","problem"], sort=False):
        sub = sub.sort_values("step")
        freq = sub["freq_correct"].to_numpy(float)
        rate = sub["aha_rate_gpt"].to_numpy(float)
        shift = sub["aha_any_gpt"].to_numpy(int)
        for j in range(len(sub)):
            if j < min_prior_steps:
                flags[idx] = 0
            else:
                prior_ok = (float(np.max(freq[:j])) < delta1) and (float(np.max(rate[:j])) < delta2)
                flags[idx] = int(prior_ok and (shift[j] == 1))
            idx += 1
    ps["aha_formal_ps"] = flags.astype(int)
    return ps

def attach_formal_sample_level(df: pd.DataFrame,
                               delta1: float,
                               delta2: float,
                               min_prior_steps: int) -> pd.DataFrame:
    """
    Compute formal at (run_label, problem, step) level; merge to samples and AND-gate with sample-level GPT Aha.
    Returns df with new columns: aha_formal_ps, aha_formal.
    """
    ps = build_problem_step(df)
    ps = mark_formal_by_run(ps, delta1=delta1, delta2=delta2, min_prior_steps=min_prior_steps)
    d = df.merge(ps[["run_label","problem","step","aha_formal_ps"]],
                 on=["run_label","problem","step"], how="left").fillna({"aha_formal_ps":0})
    d["aha_formal_ps"] = d["aha_formal_ps"].astype(int)
    # sample-level formal = problem-step formal AND sample's GPT Aha
    d["aha_formal"] = (d["aha_formal_ps"] & d["aha_gpt"]).astype(int)
    return d

# ---------- modeling ----------

def fit_glm_binomial(df: pd.DataFrame,
                     aha_col: Optional[str],
                     cluster_by: str = "problem",
                     out_txt: Optional[str] = None) -> Tuple[Dict[str, Any], pd.DataFrame]:
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except Exception as e:
        raise RuntimeError("statsmodels is required (pip install statsmodels)") from e

    d = df.copy()
    d["step_std"] = (d["step"] - d["step"].mean()) / (d["step"].std(ddof=0) + 1e-8)

    f_terms = ["C(problem)", "step_std", "temp_low"]
    if aha_col:
        f_terms += [aha_col, f"temp_low:{aha_col}"]
    formula = "correct ~ " + " + ".join(f_terms)

    model = smf.glm(formula, data=d, family=sm.families.Binomial())
    if cluster_by == "problem":
        groups = pd.Categorical(d["problem"]).codes
        cov_type, cov_kwds = "cluster", {"groups": groups, "use_correction": True, "df_correction": True}
    else:
        cov_type, cov_kwds = "HC1", {}

    try:
        res = model.fit(cov_type=cov_type, cov_kwds=cov_kwds)
    except TypeError:
        res = model.fit(cov_type=cov_type)

    coef_df = pd.DataFrame({
        "term": res.params.index,
        "coef": res.params.values,
        "se":   res.bse.values,
        "z":    (res.params.values / np.where(res.bse.values==0, np.nan, res.bse.values)),
        "p":    res.pvalues.values,
    })

    if out_txt:
        with open(out_txt, "w", encoding="utf-8") as fh:
            fh.write(res.summary().as_text())
            fh.write(f"\n\nFormula: {formula}\nCovariance: {cov_type} (clustered by {cluster_by})\n")

    def _get(name):
        return float(res.params.get(name, np.nan)), float(res.pvalues.get(name, np.nan))

    out = {"formula": formula, "N": int(len(d))}
    out["coef_temp_low"], out["p_temp_low"] = _get("temp_low")
    if aha_col:
        out["coef_aha"], out["p_aha"] = _get(aha_col)
        out["coef_interaction"], out["p_interaction"] = _get(f"temp_low:{aha_col}")
    return out, coef_df

def fit_glm_stage_interaction(df: pd.DataFrame,
                              aha_col: Optional[str],
                              cluster_by: str,
                              out_txt: Optional[str]) -> Tuple[Dict[str, Any], pd.DataFrame]:
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except Exception as e:
        raise RuntimeError("statsmodels is required") from e

    d = df.copy()
    d["step_std"] = (d["step"] - d["step"].mean()) / (d["step"].std(ddof=0) + 1e-8)
    base_terms = ["C(problem)", "step_std", "temp_low", "C(stage)"]
    if aha_col:
        base_terms += [aha_col, f"temp_low:{aha_col}"]
    base_terms += ["temp_low:C(stage)"]
    if aha_col:
        base_terms += [f"{aha_col}:C(stage)"]

    formula = "correct ~ " + " + ".join(base_terms)
    model = smf.glm(formula, data=d, family=sm.families.Binomial())
    if cluster_by == "problem":
        groups = pd.Categorical(d["problem"]).codes
        cov_type, cov_kwds = "cluster", {"groups": groups, "use_correction": True, "df_correction": True}
    else:
        cov_type, cov_kwds = "HC1", {}

    try:
        res = model.fit(cov_type=cov_type, cov_kwds=cov_kwds)
    except TypeError:
        res = model.fit(cov_type=cov_type)

    coef_df = pd.DataFrame({
        "term": res.params.index,
        "coef": res.params.values,
        "se":   res.bse.values,
        "z":    (res.params.values / np.where(res.bse.values==0, np.nan, res.bse.values)),
        "p":    res.pvalues.values,
    })

    if out_txt:
        with open(out_txt, "a", encoding="utf-8") as fh:
            fh.write("\n\n=== Stage-interaction GLM ===\n")
            fh.write(res.summary().as_text())
            fh.write(f"\n\nFormula: {formula}\nCovariance: {cov_type} (clustered by {cluster_by})\n")

    def _maybe_get(name):
        return (float(res.params.get(name, np.nan)),
                float(res.pvalues.get(name, np.nan)))

    out = {"formula": formula, "N": int(len(d))}
    out["coef_temp_low"], out["p_temp_low"] = _maybe_get("temp_low")
    if aha_col:
        out["coef_aha"], out["p_aha"] = _maybe_get(aha_col)
        out["coef_temp_low_x_aha"], out["p_temp_low_x_aha"] = _maybe_get(f"temp_low:{aha_col}")
    return out, coef_df

# ---------- group accuracy tables ----------

def compute_group_acc(df: pd.DataFrame, aha_col: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    grp_cols = ["temp_low"] + ([aha_col] if aha_col else [])
    overall = (df.groupby(grp_cols, as_index=False)
                 .agg(n=("correct","size"), k=("correct","sum")))
    overall["accuracy"] = overall["k"] / overall["n"]
    by_step = (df.groupby(["step"] + grp_cols, as_index=False)
                 .agg(n=("correct","size"), k=("correct","sum")))
    by_step["accuracy"] = by_step["k"] / by_step["n"]
    return overall, by_step

def compute_stage_group_acc(df: pd.DataFrame, aha_col: Optional[str]) -> pd.DataFrame:
    grp_cols = ["stage", "temp_low"] + ([aha_col] if aha_col else [])
    acc = (df.groupby(grp_cols, as_index=False)
             .agg(n=("correct","size"), k=("correct","sum")))
    acc["accuracy"] = acc["k"] / acc["n"]
    return acc

def compute_stage_aha_counts(df: pd.DataFrame, aha_col: Optional[str]) -> pd.DataFrame:
    if aha_col is None: return pd.DataFrame()
    sub = df[df[aha_col] == 1]
    counts = (sub.groupby(["stage","temp_low"], as_index=False)
                .agg(n_aha=("correct","size")))
    return counts

# ---------- Orchestrate one Aha mode ----------

def evaluate_for_aha_mode(combined_in: pd.DataFrame,
                          aha_mode_name: str,
                          out_dir: str,
                          cluster_by: str,
                          formal_cfg: Dict[str, Any]) -> None:
    """
    Run baseline + stage-interaction + stage-specific GLMs and write group accuracies
    for a given Aha mode: 'words' | 'gpt' | 'gpt_broad' | 'formal' | 'none'
    """
    os.makedirs(out_dir, exist_ok=True)
    df = combined_in.copy()

    # Select aha column name
    if aha_mode_name == "words":
        aha_col = "aha_words"
    elif aha_mode_name in ("gpt","gpt_broad"):
        aha_col = "aha"
    elif aha_mode_name == "formal":
        # formal uses problem-step flags; compute by run and AND with sample GPT
        df = attach_formal_sample_level(df,
                                        delta1=float(formal_cfg["delta1"]),
                                        delta2=float(formal_cfg["delta2"]),
                                        min_prior_steps=int(formal_cfg["min_prior_steps"]))
        aha_col = "aha_formal"
    else:
        aha_col = None  # 'none'

    # -------- Baseline GLM --------
    summ_base, coef_base = fit_glm_binomial(df, aha_col=aha_col, cluster_by=cluster_by,
                                            out_txt=os.path.join(out_dir, "h2_glm_summary.txt"))
    coef_base.to_csv(os.path.join(out_dir, "h2_glm_coefficients.csv"), index=False)

    # -------- Stage-interaction GLM --------
    summ_int, coef_int = fit_glm_stage_interaction(df, aha_col=aha_col, cluster_by=cluster_by,
                                                   out_txt=os.path.join(out_dir, "h2_stage_glm_summary.txt"))
    coef_int["which"] = "interaction"
    coef_int.to_csv(os.path.join(out_dir, "h2_stage_glm_coefficients.csv"), index=False)

    # -------- Stage-specific GLMs --------
    with open(os.path.join(out_dir, "h2_stage_glm_summary.txt"), "a", encoding="utf-8") as fh:
        fh.write("\n\n=== Stage-specific GLMs ===\n")
    stage_coefs = []
    for stage in ["early","mid","late"]:
        sub = df[df["stage"] == stage]
        if sub.empty: 
            continue
        summ_s, coef_s = fit_glm_binomial(sub, aha_col=aha_col, cluster_by=cluster_by, out_txt=None)
        coef_s["which"] = f"stage_{stage}"
        stage_coefs.append(coef_s)
        with open(os.path.join(out_dir, "h2_stage_glm_summary.txt"), "a", encoding="utf-8") as fh:
            fh.write(f"\n[Stage: {stage}] N={summ_s['N']}\nFormula: {summ_s['formula']}\n")
            fh.write(f"  temp_low coef={summ_s.get('coef_temp_low', np.nan):+.4f}  p={summ_s.get('p_temp_low', np.nan):.3g}\n")
            if aha_col:
                fh.write(f"  {aha_col} coef={summ_s.get('coef_aha', np.nan):+.4f}  p={summ_s.get('p_aha', np.nan):.3g}\n")
                fh.write(f"  interaction coef={summ_s.get('coef_interaction', np.nan):+.4f}  p={summ_s.get('p_interaction', np.nan):.3g}\n")
    if stage_coefs:
        pd.concat(stage_coefs, ignore_index=True).to_csv(
            os.path.join(out_dir, "h2_stage_glm_coefficients.csv"), mode="a", index=False)

    # -------- Group accuracies & counts --------
    overall_acc, step_acc = compute_group_acc(df, aha_col=aha_col)
    overall_acc.to_csv(os.path.join(out_dir, "h2_group_accuracy.csv"), index=False)
    step_acc.to_csv(os.path.join(out_dir, "h2_group_accuracy_by_step.csv"), index=False)

    stage_acc = compute_stage_group_acc(df, aha_col=aha_col)
    stage_acc.to_csv(os.path.join(out_dir, "h2_stage_group_accuracy.csv"), index=False)

    stage_aha = compute_stage_aha_counts(df, aha_col=aha_col)
    if not stage_aha.empty:
        stage_aha.to_csv(os.path.join(out_dir, "h2_stage_aha_counts.csv"), index=False)

    # One mode’s overlap/stage info snapshot
    info = {
        "aha_mode": aha_mode_name,
        "glm_headline": summ_base,
        "stage_interaction_headline": summ_int,
    }
    with open(os.path.join(out_dir, "h2_stage_info.json"), "w") as fh:
        json.dump(info, fh, indent=2)

    # Console snippet
    print(f"\n[{aha_mode_name.upper()}] Baseline GLM: {summ_base['formula']}  N={summ_base['N']}")
    print(f"  temp_low coef={summ_base.get('coef_temp_low', np.nan):+.4f}  p={summ_base.get('p_temp_low', np.nan):.3g}")
    if aha_col:
        print(f"  {aha_col} coef={summ_base.get('coef_aha', np.nan):+.4f}  p={summ_base.get('p_aha', np.nan):.3g}")
        print(f"  temp_low:{aha_col} coef={summ_base.get('coef_interaction', np.nan):+.4f}  p={summ_base.get('p_interaction', np.nan):.3g}")
    print("  Wrote ->", out_dir)

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root_high", help="Results root for baseline temp (e.g., GRPO-1.5B).")
    ap.add_argument("root_low",  help="Results root for low-temp run (e.g., GRPO-1.5B-low-temp).")
    ap.add_argument("--split", default=None, help="Record-level split filter (e.g., 'test').")
    ap.add_argument("--out_dir", default=None, help="Output directory (default: <root_high>/h2_temp_aha).")

    # Aha selection
    ap.add_argument("--aha", choices=["words","gpt","gpt_broad","formal","none","all"], default="gpt",
                    help="Aha feature(s) to evaluate.")
    ap.add_argument("--gate_gpt_by_words", action="store_true",
                    help="Require GPT Aha to also have Words cue (aha_gpt ⊆ aha_words).")

    # Formal thresholds
    ap.add_argument("--formal_delta1", type=float, default=0.13,
                    help="Formal prior failure threshold on freq_correct (default 0.13).")
    ap.add_argument("--formal_delta2", type=float, default=0.13,
                    help="Formal prior shift-stability threshold on aha_rate_gpt (default 0.13).")
    ap.add_argument("--formal_min_prior_steps", type=int, default=2,
                    help="Formal minimum prior steps before eligibility (default 2).")

    # Stage bucketing
    ap.add_argument("--stage_mode", choices=["quantile","fixed"], default="quantile",
                    help="Stage definition: tertiles (quantile) or fixed cutpoints.")
    ap.add_argument("--stage_quantiles", nargs=2, type=float, default=(0.33, 0.66),
                    help="Quantiles for early/mid/late when stage_mode=quantile.")
    ap.add_argument("--stage_bounds", nargs=2, type=int, default=None,
                    help="Fixed cutpoints (lo hi) when stage_mode=fixed.")

    # GLM cov type
    ap.add_argument("--cluster_by", choices=["problem","none"], default="problem",
                    help="Covariance type for GLM (default: cluster by problem).")

    args = ap.parse_args()
    out_root = args.out_dir or os.path.join(args.root_high, "h2_temp_aha")
    os.makedirs(out_root, exist_ok=True)

    # Load both runs using gpt (to populate aha_gpt/words regardless of top-level mode)
    df_hi = load_samples(args.root_high, args.split, aha_mode="gpt", gate_gpt_by_words=args.gate_gpt_by_words)
    df_lo = load_samples(args.root_low,  args.split, aha_mode="gpt", gate_gpt_by_words=args.gate_gpt_by_words)
    if df_hi.empty or df_lo.empty:
        raise SystemExit("One directory has no usable sample-level rows. Check --split and file structure.")

    # Restrict to overlap
    dA, dB, keys = restrict_to_overlap(df_hi, df_lo)

    # Annotate temperature labels
    dA["temp_low"] = 0; dA["run_label"] = "high"
    dB["temp_low"] = 1; dB["run_label"] = "low"

    # Combine
    cols = ["dataset","model","split","problem","step","sample_idx","correct",
            "aha_words","aha_gpt","aha","temp_low","run_label"]
    combined = pd.concat([dA[cols], dB[cols]], ignore_index=True)
    combined["correct"] = combined["correct"].astype(int)
    combined["temp_low"] = combined["temp_low"].astype(int)

    # Stage assignment (based on combined steps)
    if args.stage_mode == "fixed":
        combined, stage_info = assign_stage(combined, mode="fixed",
                                            bounds=tuple(args.stage_bounds) if args.stage_bounds else None)
    else:
        q1, q2 = tuple(args.stage_quantiles)
        combined, stage_info = assign_stage(combined, mode="quantile", quantiles=(q1, q2))

    # Shared top-level overlap & stage info (write once)
    info_header = {
        "keys_used": keys,
        "n_overlap_units_per_run": int(len(dA)),
        "n_combined_rows": int(len(combined)),
        "root_high": args.root_high,
        "root_low":  args.root_low,
        "stage_info": stage_info
    }
    with open(os.path.join(out_root, "h2_stage_info.json"), "w") as fh:
        json.dump(info_header, fh, indent=2)

    # Formal config
    formal_cfg = {
        "delta1": args.formal_delta1,
        "delta2": args.formal_delta2,
        "min_prior_steps": args.formal_min_prior_steps,
    }

    # Determine which modes to run
    modes = ["words","gpt","formal"] if args.aha == "all" else [args.aha]
    # Map legacy 'gpt_broad' (if requested) into this list
    if args.aha == "gpt_broad":
        modes = ["gpt_broad"]
    if args.aha == "none":
        modes = ["none"]

    for mode in modes:
        out_dir = os.path.join(out_root, mode) if len(modes) > 1 else out_root
        # For GPT broad, we need aha_gpt computed under broad; recompute a temp df for that mode
        df_mode = combined.copy()
        if mode == "gpt_broad":
            # recompute aha_gpt for broad keys using original raw records would be ideal;
            # here we approximate by reloading with aha_mode='gpt_broad' and re-merging overlap keys
            df_hi_b = load_samples(args.root_high, args.split, aha_mode="gpt_broad",
                                   gate_gpt_by_words=args.gate_gpt_by_words)
            df_lo_b = load_samples(args.root_low,  args.split, aha_mode="gpt_broad",
                                   gate_gpt_by_words=args.gate_gpt_by_words)
            dA_b, dB_b, _ = restrict_to_overlap(df_hi_b, df_lo_b)
            dA_b["temp_low"]=0; dA_b["run_label"]="high"
            dB_b["temp_low"]=1; dB_b["run_label"]="low"
            cols_b = ["problem","step","sample_idx","aha_words","aha_gpt"]
            df_mode = (df_mode.drop(columns=["aha_words","aha_gpt"], errors="ignore")
                               .merge(pd.concat([dA_b[cols_b], dB_b[cols_b]]),
                                      on=["problem","step","sample_idx"], how="left"))

        evaluate_for_aha_mode(df_mode, mode, out_dir,
                              cluster_by=args.cluster_by,
                              formal_cfg=formal_cfg)

    # Console header
    print("\n=== H2 Temperature x Aha! (+ stages; overlap only) ===")
    print(f"Keys used for overlap: {keys}")
    print(f"N units per run after overlap: {len(dA)}")
    print(f"N combined rows (both temps): {len(combined)}")
    if stage_info.get("mode") == "fixed":
        print(f"Stage cuts (fixed): early <= {stage_info['bounds'][0]} < mid <= {stage_info['bounds'][1]} < late")
    else:
        cp = stage_info.get("cutpoints", [])
        q  = stage_info.get("quantiles", [])
        print(f"Stage cuts (quantiles {q}): early <= {cp[0]} < mid <= {cp[1]} < late")
    print("Output root:", out_root)

if __name__ == "__main__":
    main()
