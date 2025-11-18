#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
H3: Can Aha! Moments Help When the Model is Uncertain?
+ Re-asking (pass2) at QUESTION & PROMPT levels, with Aha variants
+ Wilson CIs, step-wise & bucket-wise plots/tables
---------------------------------------------------------------

Uncertainty measure
-------------------
Default is --measure entropy (nats). Buckets are built from entropy values.
(If you set --measure perplexity, the loader converts entropy->perplexity via exp(entropy).)
We keep the column name "perplexity_bucket" for backward compatibility with your H2/H3 formula.

What this script does
---------------------
1) Robustly loads your JSONL (pass1 & pass2), extracts entropy/ppx, Aha labels:
   - Words (excludes injected/forced cues)
   - GPT-4o canonical (optionally gated by Words)
   - Formal (problem-step, gated by GPT)
2) H3 GLMs per Aha variant:
   correct ~ C(problem) + step_std + aha * C(perplexity_bucket)
3) Re-asking analysis (pass2 vs pass1), with **Wilson 95% CIs**:
   • QUESTION level (per problem): fraction solved ≥1 time (overall, by step, by bucket)
   • PROMPT level (per pair): accuracy (overall, by step, by bucket)
   • PROMPT-level Δ histogram (with/without forced-insight split)
4) Answers written to **h3_answers.txt**:
   (A) Given ≥1 correct in pass1, does pass2 increase overall accuracy?
   (B) Given 0 correct in pass1, does pass2 achieve ≥1 correct?
   Each answered overall, step-wise, and bucket-wise.

Outputs
-------
CSV:
  - h3_glm_bucket_margins.csv
  - h3_bucket_group_accuracy.csv
  - h3_pass2_prompt_level.csv                 # per-pair aligned rows
  - h3_pass2_question_level.csv               # per-problem aligned rows
  - h3_pass2_conditioned_summary.csv          # overall A & B answers (macro & micro)
  - h3_pass2_conditioned_by_forcing.csv       # (optional) split by forced insight
  - h3_prompt_level_overall.csv               # k/n with Wilson CIs
  - h3_prompt_level_by_step.csv               # step-wise k/n with CIs
  - h3_prompt_level_by_bucket.csv             # bucket-wise k/n with CIs
  - h3_question_level_overall.csv             # k/n (any-correct) with Wilson CIs
  - h3_question_level_by_step.csv             # step-wise any-correct with CIs
  - h3_question_level_by_bucket.csv           # bucket-wise any-correct with CIs
  - (optional) h3_pass2_prompt_level_by_aha.csv
  - (optional) h3_pass2_question_level_by_aha.csv

Plots (if --make_plots):
  - h3_plot_question_overall.png              # bars + Wilson CIs
  - h3_plot_question_by_step.png              # lines + CIs by step
  - h3_plot_question_by_bucket.png            # points/bars + CIs by bucket
  - h3_plot_prompt_overall.png                # bars + CIs
  - h3_plot_prompt_by_step.png                # lines + CIs
  - h3_plot_prompt_by_bucket.png              # points/bars + CIs
  - h3_plot_prompt_level_delta.png            # Δ histogram (± forced)

PDF (if --make_pdf):
  - h3_summary.pdf                            # GLM bucket margins summary page

Text:
  - h3_answers.txt                            # answers to (A) and (B), overall/step/bucket

"""

import os, re, json, argparse, math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

# ---------------- util ----------------

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
        s = x.strip().lower()
        if s in ("1","true","t","yes","y"): return 1
        if s in ("0","false","f","no","n"): return 0
    return int(bool(x))

# ---------------- Aha labels ----------------

def _aha_words(p: Dict[str, Any]) -> int:
    v = coerce_bool(p.get("has_reconsider_cue"))
    markers = p.get("reconsider_markers") or []
    if isinstance(markers, list) and ("injected_cue" in markers or "forced_insight" in markers):
        return 0
    return 0 if v is None else int(v)

def _aha_gpt_canonical(p: Dict[str, Any], rec: Dict[str, Any]) -> int:
    for k in ("change_way_of_thinking", "shift_in_reasoning_v1"):
        v = p.get(k, rec.get(k, None))
        if v is not None and coerce_bool(v) == 1:
            return 1
    return 0

def _aha_gpt_broad(p: Dict[str, Any], rec: Dict[str, Any]) -> int:
    keys = ["change_way_of_thinking","shift_in_reasoning_v1","shift_llm","shift_gpt","pivot_llm","rechecked"]
    for k in keys:
        v = p.get(k, rec.get(k, None))
        if v is not None and coerce_bool(v) == 1:
            return 1
    return 0

# ---------------- Uncertainty / Entropy & Perplexity ----------------

def _first_num(d: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for k in keys:
        v = d.get(k)
        if isinstance(v, (int, float, np.floating)): return float(v)
        if isinstance(v, str):
            try: return float(v)
            except: pass
    return None

def _any_key_contains(d: Dict[str, Any], must: List[str], any_of: List[str]) -> Optional[float]:
    for k, v in d.items():
        ks = k.lower()
        if all(s in ks for s in must) and any(s in ks for s in any_of):
            if isinstance(v, (int, float, np.floating)): return float(v)
            if isinstance(v, str):
                try: return float(v)
                except: pass
    return None

def extract_uncertainty_or_ppx(pass_dict: Dict[str, Any],
                               unc_field: str,
                               measure: str) -> Optional[float]:
    if pass_dict is None: return None
    d = {str(k): v for k, v in pass_dict.items()}

    if unc_field == "answer":
        ppx = (_any_key_contains(d, ["perplexity"], ["answer"]) or
               _first_num(d, ["answer_perplexity","perplexity_answer"]))
        ent_n = (_first_num(d, ["answer_entropy_nats","entropy_answer_nats"]) or
                 _any_key_contains(d, ["entropy","nats"], ["answer"]))
        ent = (_first_num(d, ["entropy_answer","answer_entropy"]) or
               _any_key_contains(d, ["entropy"], ["answer"]))
    elif unc_field == "think":
        ppx = _any_key_contains(d, ["perplexity"], ["think"])
        ent_n = _first_num(d, ["think_entropy_nats"])
        ent = (_first_num(d, ["entropy_think","think_entropy"]) or
               _any_key_contains(d, ["entropy"], ["think"]))
    else:
        ppx = (_first_num(d, ["perplexity"]) or
               _any_key_contains(d, ["perplexity"], [""]))
        ent_n = _first_num(d, ["overall_entropy_nats","entropy_nats"])
        ent = _first_num(d, ["overall_entropy","entropy"])

    if measure == "perplexity":
        if ppx is not None: return float(ppx)
        if ent_n is not None: return float(np.exp(ent_n))
        if ent is not None: return float(np.exp(ent))
        ent_any = _any_key_contains(d, ["entropy"], [""])
        if ent_any is not None: return float(np.exp(ent_any))
        return None
    else:  # "entropy"
        if ent_n is not None: return float(ent_n)
        if ent is not None: return float(ent)
        if ppx is not None and ppx > 0: return float(np.log(ppx))
        ppx_any = _any_key_contains(d, ["perplexity"], [""])
        if ppx_any is not None and ppx_any > 0: return float(np.log(ppx_any))
        return None

# ---------------- prompt key & forced-insight ----------------

def _extract_prompt_key(pass_dict: Dict[str, Any], rec: Dict[str, Any]) -> str:
    for k in ("prompt_id","prompt_key","prompt_variant","template_id","prompt_name","prompt_version"):
        v = (pass_dict or {}).get(k, rec.get(k, None))
        if v not in (None, ""): return str(v)
    txt = (pass_dict or {}).get("prompt") or rec.get("prompt")
    if isinstance(txt, str) and txt:
        return f"textlen{len(txt)}"
    return "unknown"

def _detect_forced_insight(p: Dict[str, Any], rec: Dict[str, Any]) -> int:
    markers = set((p.get("reconsider_markers") or []) + (rec.get("reconsider_markers") or []))
    pv = (p.get("prompt_variant") or rec.get("prompt_variant") or "")
    s = str(pv).lower()
    forced = (
        ("forced_insight" in markers) or
        ("injected_cue" in markers) or
        ("force" in s and "insight" in s) or
        ("reconsider" in s and "force" in s)
    )
    return int(bool(forced))

# ---------------- loader (pass1 + pass2) ----------------

def load_rows(files: List[str],
              gpt_mode: str = "canonical",
              gate_gpt_by_words: bool = True,
              unc_field: str = "answer",
              measure: str = "entropy") -> pd.DataFrame:
    rows = []
    pair_id = 0
    gpt_fn = _aha_gpt_canonical if gpt_mode == "canonical" else _aha_gpt_broad

    for path in files:
        step_from_name = nat_step_from_path(path)
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                if not ln.strip(): continue
                try:
                    rec = json.loads(ln)
                except Exception:
                    continue

                step = rec.get("step", step_from_name if step_from_name is not None else None)
                if step is None: continue

                prob = rec.get("problem") or rec.get("clue") or rec.get("row_key")
                if prob is None:
                    di = rec.get("dataset_index")
                    prob = f"idx:{di}" if di is not None else f"unk:{pair_id}"

                p1 = rec.get("pass1") or {}
                c1 = coerce_bool(p1.get("is_correct_pred"))
                if c1 is None: continue
                words1 = _aha_words(p1)
                gpt1_raw = gpt_fn(p1, rec)
                gpt1 = int(gpt1_raw and words1) if gate_gpt_by_words else int(gpt1_raw)
                unc1 = extract_uncertainty_or_ppx(p1, unc_field, measure)
                prompt1 = _extract_prompt_key(p1, rec)

                rows.append({
                    "pair_id": pair_id, "pass_id": 1,
                    "problem": str(prob), "step": int(step),
                    "prompt_key": prompt1,
                    "correct": int(c1),
                    "aha_words": int(words1), "aha_gpt": int(gpt1),
                    "unc": None if unc1 is None else float(unc1),
                })

                p2 = rec.get("pass2") or None
                if p2 is not None:
                    c2 = coerce_bool(p2.get("is_correct_pred"))
                    if c2 is not None:
                        words2 = _aha_words(p2)
                        gpt2_raw = gpt_fn(p2, rec)
                        gpt2 = int(gpt2_raw and words2) if gate_gpt_by_words else int(gpt2_raw)
                        unc2 = extract_uncertainty_or_ppx(p2, unc_field, measure)
                        prompt2 = _extract_prompt_key(p2, rec)
                        rows.append({
                            "pair_id": pair_id, "pass_id": 2,
                            "problem": str(prob), "step": int(step),
                            "prompt_key": prompt2,
                            "correct": int(c2),
                            "aha_words": int(words2), "aha_gpt": int(gpt2),
                            "unc": None if unc2 is None else float(unc2),
                            "forced_insight": int(_detect_forced_insight(p2, rec)),
                        })
                pair_id += 1

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No usable rows (pass1/pass2) found.")
    return df

# ---------------- Formal Aha (problem-step; from pass1) ----------------

def build_problem_step(df_pass1: pd.DataFrame) -> pd.DataFrame:
    grp = df_pass1.groupby(["step","problem"], as_index=False)
    ps = grp.agg(
        n_samples=("correct","size"),
        freq_correct=("correct","mean"),
        aha_any_gpt=("aha_gpt","max"),
        aha_rate_gpt=("aha_gpt","mean"),
    )
    for c in ("n_samples","aha_any_gpt"): ps[c] = ps[c].astype(int)
    ps["freq_correct"] = ps["freq_correct"].astype(float)
    ps["aha_rate_gpt"] = ps["aha_rate_gpt"].astype(float)
    return ps.sort_values(["problem","step"]).reset_index(drop=True)

def mark_formal(ps: pd.DataFrame, delta1: float, delta2: float, min_prior_steps: int) -> pd.DataFrame:
    need = {"step","problem","freq_correct","aha_rate_gpt","aha_any_gpt"}
    if not need.issubset(ps.columns): raise ValueError("mark_formal: missing cols")
    ps = ps.sort_values(["problem","step"]).copy()
    flags = np.zeros(len(ps), dtype=int); idx = 0
    for _, sub in ps.groupby("problem", sort=False):
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
    ps["aha_formal_ps"] = flags
    return ps

# ---------------- Bucketing (entropy-based by default) ----------------

def add_perplexity_buckets(df_pass1: pd.DataFrame, n_buckets: int, method: str,
                           custom_edges: Optional[List[float]] = None) -> pd.DataFrame:
    d = df_pass1.copy()
    if "unc" not in d.columns or d["unc"].dropna().empty:
        raise SystemExit(
            "No pass1 rows with uncertainty/entropy. "
            "Ensure your records carry entropy fields (e.g., entropy, entropy_answer, entropy_think)."
        )
    d = d[~d["unc"].isna()].copy()
    if method == "fixed":
        if not custom_edges:
            raise ValueError("fixed bucketing requires --bucket_edges like 0.2,0.5,1,1.5,2,3")
        edges = sorted(set(custom_edges))
        d["perplexity_bucket"] = pd.cut(d["unc"], bins=edges, include_lowest=True)
    else:
        d["perplexity_bucket"] = pd.qcut(d["unc"], q=n_buckets, duplicates="drop")
    d["perplexity_bucket"] = d["perplexity_bucket"].astype(str)
    return d

# ---------------- Wilson CI ----------------

def wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    z = 1.959963984540054 if abs(alpha - 0.05) < 1e-12 else float(np.abs(np.sqrt(2)*np.erfcinv(alpha)))
    phat = k / n
    denom = 1 + z*z/n
    center = (phat + z*z/(2*n)) / denom
    half = (z * math.sqrt(phat*(1-phat)/n + z*z/(4*n*n))) / denom
    lo, hi = center - half, center + half
    return max(0.0, lo), min(1.0, hi)

# ---------------- GLM with interaction ----------------

def _cov_spec(df: pd.DataFrame, cluster_by: str):
    if cluster_by == "problem":
        groups = pd.Categorical(df["problem"]).codes
        return "cluster", {"groups": groups, "use_correction": True, "df_correction": True}
    return "HC1", None

def fit_glm_bucket_interaction(df: pd.DataFrame, aha_col: str,
                               strict_interaction_only: bool,
                               cluster_by: str,
                               out_txt: str) -> Tuple[Dict[str, Any], "object"]:
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except Exception as e:
        raise RuntimeError("statsmodels is required (pip install statsmodels)") from e

    d = df.copy()
    d["step_std"] = (d["step"] - d["step"].mean()) / (d["step"].std(ddof=0) + 1e-8)
    d = d[~d["perplexity_bucket"].isna()].copy()

    if strict_interaction_only:
        formula = f"correct ~ C(problem) + step_std + {aha_col}:C(perplexity_bucket)"
    else:
        formula = f"correct ~ C(problem) + step_std + {aha_col} + C(perplexity_bucket) + {aha_col}:C(perplexity_bucket)"

    model = smf.glm(formula, data=d, family=sm.families.Binomial())
    cov_type, cov_kwds = _cov_spec(d, cluster_by)
    try:
        res = model.fit(cov_type=cov_type, cov_kwds=(cov_kwds or {}))
    except TypeError:
        minimal_kw = {"groups": cov_kwds.get("groups")} if cov_kwds and "groups" in cov_kwds else {}
        res = model.fit(cov_type=cov_type, cov_kwds=minimal_kw)

    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as fh:
        fh.write(res.summary().as_text())
        fh.write(f"\nCovariance: {cov_type}")
        if cov_kwds and "groups" in cov_kwds: fh.write(" (clustered by problem)")

    buckets = sorted(d["perplexity_bucket"].unique().tolist())
    rows = []
    for b in buckets:
        d1 = d.copy(); d0 = d.copy()
        d1[aha_col] = 1; d0[aha_col] = 0
        d1["perplexity_bucket"] = b; d0["perplexity_bucket"] = b
        ame_b = float(np.mean(res.predict(d1) - res.predict(d0)))
        sub = d[d["perplexity_bucket"] == b]
        rows.append({"bucket": b, "N": int(len(sub)), "share_aha": float(sub[aha_col].mean()), "AME_bucket": ame_b})

    acc = float(d["correct"].mean())
    return ({"N": int(len(d)), "acc_overall": acc, "bucket_rows": rows}, res)

def bucket_group_accuracy(df: pd.DataFrame, aha_col: str) -> pd.DataFrame:
    g = (df.groupby(["perplexity_bucket", aha_col], as_index=False)
           .agg(n=("correct","size"), k=("correct","sum")))
    g["accuracy"] = g["k"] / g["n"]
    g = g.rename(columns={aha_col: "aha"})
    return g[["perplexity_bucket","aha","n","k","accuracy"]].copy()

# ---------------- Re-asking core tables ----------------

def compute_reasking_tables(df_all: pd.DataFrame, pass1_bucket_col: str = "perplexity_bucket"
                            ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d1 = df_all[df_all["pass_id"] == 1].copy()
    d2 = df_all[df_all["pass_id"] == 2].copy()
    if d2.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    m = d1.merge(d2, on=["pair_id","problem","step"], suffixes=("1","2"))
    if m.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if pass1_bucket_col in d1.columns:
        m = m.merge(d1[["pair_id", pass1_bucket_col]].drop_duplicates(), on="pair_id", how="left")

    pairs_df = m[[
        "pair_id","problem","step","prompt_key1","prompt_key2",
        "correct1","correct2","forced_insight2"
    ]].rename(columns={
        "prompt_key1":"prompt_key_pass1", "prompt_key2":"prompt_key_pass2",
        "forced_insight2":"forced_insight"
    })
    pairs_df["delta"] = pairs_df["correct2"] - pairs_df["correct1"]
    if pass1_bucket_col in m.columns:
        pairs_df[pass1_bucket_col] = m[pass1_bucket_col]

    g = (m.groupby("problem", as_index=False)
           .agg(n_pairs=("pair_id","size"),
                any_pass1=("correct1","max"),
                any_pass2=("correct2","max"),
                mean_pass1=("correct1","mean"),
                mean_pass2=("correct2","mean")))
    g["delta_any"] = g["any_pass2"] - g["any_pass1"]
    g["delta_mean"] = g["mean_pass2"] - g["mean_pass1"]
    probs_df = g

    a_problems = probs_df[probs_df["any_pass1"] == 1]["problem"]
    mm = m[m["problem"].isin(a_problems)]
    A = {
        "condition": "pass1_any_correct==1",
        "n_problems": int(len(a_problems)),
        "macro_mean_acc_pass1": float(probs_df.loc[probs_df["any_pass1"]==1, "mean_pass1"].mean()) if len(a_problems) else np.nan,
        "macro_mean_acc_pass2": float(probs_df.loc[probs_df["any_pass1"]==1, "mean_pass2"].mean()) if len(a_problems) else np.nan,
        "macro_delta_mean": float((probs_df.loc[probs_df["any_pass1"]==1, "delta_mean"]).mean()) if len(a_problems) else np.nan,
        "micro_acc_pass1": float(mm["correct1"].mean()) if not mm.empty else np.nan,
        "micro_acc_pass2": float(mm["correct2"].mean()) if not mm.empty else np.nan,
        "micro_delta": float(mm["correct2"].mean() - mm["correct1"].mean()) if not mm.empty else np.nan,
        "share_keep_any_correct_in_pass2": float(probs_df.loc[probs_df["any_pass1"]==1, "any_pass2"].mean()) if len(a_problems) else np.nan,
    }

    b_problems = probs_df[probs_df["any_pass1"] == 0]["problem"]
    mm2 = m[m["problem"].isin(b_problems)]
    B = {
        "condition": "pass1_any_correct==0",
        "n_problems": int(len(b_problems)),
        "share_any_pass2": float(probs_df.loc[probs_df["any_pass1"]==0, "any_pass2"].mean()) if len(b_problems) else np.nan,
        "macro_mean_acc_pass1": float(probs_df.loc[probs_df["any_pass1"]==0, "mean_pass1"].mean()) if len(b_problems) else np.nan,
        "macro_mean_acc_pass2": float(probs_df.loc[probs_df["any_pass1"]==0, "mean_pass2"].mean()) if len(b_problems) else np.nan,
        "macro_delta_mean": float((probs_df.loc[probs_df["any_pass1"]==0, "delta_mean"]).mean()) if len(b_problems) else np.nan,
        "micro_acc_pass1": float(mm2["correct1"].mean()) if not mm2.empty else np.nan,
        "micro_acc_pass2": float(mm2["correct2"].mean()) if not mm2.empty else np.nan,
        "micro_delta": float(mm2["correct2"].mean() - mm2["correct1"].mean()) if not mm2.empty else np.nan,
    }

    cond_df = pd.DataFrame([A, B])

    cond_forced_rows = []
    if "forced_insight2" in m.columns:
        for forced in [0, 1]:
            mf = m[m["forced_insight2"] == forced]
            if mf.empty: continue
            gf = (mf.groupby("problem", as_index=False)
                    .agg(n_pairs=("pair_id","size"),
                         any_pass1=("correct1","max"),
                         any_pass2=("correct2","max"),
                         mean_pass1=("correct1","mean"),
                         mean_pass2=("correct2","mean")))
            gf["delta_mean"] = gf["mean_pass2"] - gf["mean_pass1"]
            a_prob = gf[gf["any_pass1"] == 1]
            b_prob = gf[gf["any_pass1"] == 0]
            cond_forced_rows.append({
                "forced_insight": forced,
                "condition": "pass1_any_correct==1",
                "n_problems": int(len(a_prob)),
                "macro_mean_acc_pass1": float(a_prob["mean_pass1"].mean()) if len(a_prob) else np.nan,
                "macro_mean_acc_pass2": float(a_prob["mean_pass2"].mean()) if len(a_prob) else np.nan,
                "macro_delta_mean": float(a_prob["delta_mean"].mean()) if len(a_prob) else np.nan,
            })
            cond_forced_rows.append({
                "forced_insight": forced,
                "condition": "pass1_any_correct==0",
                "n_problems": int(len(b_prob)),
                "share_any_pass2": float(b_prob["any_pass2"].mean()) if len(b_prob) else np.nan,
                "macro_mean_acc_pass1": float(b_prob["mean_pass1"].mean()) if len(b_prob) else np.nan,
                "macro_mean_acc_pass2": float(b_prob["mean_pass2"].mean()) if len(b_prob) else np.nan,
                "macro_delta_mean": float(b_prob["delta_mean"].mean()) if len(b_prob) else np.nan,
            })
    cond_forced_df = pd.DataFrame(cond_forced_rows) if cond_forced_rows else pd.DataFrame()

    return pairs_df, probs_df, cond_df, cond_forced_df

# ---------------- Aggregations with Wilson CIs ----------------

def _ensure_groupcol(df: pd.DataFrame, group_keys: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    if group_keys:
        return df.copy(), group_keys
    d = df.copy(); d["__all__"] = "all"; return d, ["__all__"]

def question_level_any_with_ci(m_pairs: pd.DataFrame, group_keys: List[str]) -> pd.DataFrame:
    # per PROBLEM within group, then proportion of problems with any-correct
    d, gk = _ensure_groupcol(m_pairs, group_keys)
    perprob = (d.groupby(gk + ["problem"], as_index=False)
                 .agg(any_pass1=("correct1","max"),
                      any_pass2=("correct2","max")))
    out_rows = []
    for keys, sub in perprob.groupby(gk):
        if not isinstance(keys, tuple): keys = (keys,)
        n = int(len(sub))
        k1 = int(sub["any_pass1"].sum()); p1 = k1 / n if n else float("nan")
        k2 = int(sub["any_pass2"].sum()); p2 = k2 / n if n else float("nan")
        lo1, hi1 = wilson_ci(k1, n); lo2, hi2 = wilson_ci(k2, n)
        row = {k:v for k, v in zip(gk, keys)}
        row.update(dict(n_problems=n,
                        any_pass1=p1, any_pass1_lo=lo1, any_pass1_hi=hi1,
                        any_pass2=p2, any_pass2_lo=lo2, any_pass2_hi=hi2,
                        delta=p2 - p1 if n else float("nan")))
        out_rows.append(row)
    return pd.DataFrame(out_rows).sort_values(gk).reset_index(drop=True)

def prompt_level_acc_with_ci(m_pairs: pd.DataFrame, group_keys: List[str]) -> pd.DataFrame:
    # per pair (prompt) Bernoulli; compute k/n with Wilson CI
    d, gk = _ensure_groupcol(m_pairs, group_keys)
    out_rows = []
    for keys, sub in d.groupby(gk):
        if not isinstance(keys, tuple): keys = (keys,)
        n = int(len(sub))
        k1 = int(sub["correct1"].sum()); p1 = k1 / n if n else float("nan")
        k2 = int(sub["correct2"].sum()); p2 = k2 / n if n else float("nan")
        lo1, hi1 = wilson_ci(k1, n); lo2, hi2 = wilson_ci(k2, n)
        row = {k:v for k, v in zip(gk, keys)}
        row.update(dict(n_pairs=n,
                        acc_pass1=p1, acc_pass1_lo=lo1, acc_pass1_hi=hi1,
                        acc_pass2=p2, acc_pass2_lo=lo2, acc_pass2_hi=hi2,
                        delta=p2 - p1 if n else float("nan")))
        out_rows.append(row)
    return pd.DataFrame(out_rows).sort_values(gk).reset_index(drop=True)

# ---------------- Plots (with error bars) ----------------

def _plot_bars_with_ci(ax, labels, p1, p1_lo, p1_hi, p2, p2_lo, p2_hi, title, ylabel):
    import numpy as np

    # Coerce to arrays
    p1     = np.asarray(p1, dtype=float)
    p1_lo  = np.asarray(p1_lo, dtype=float)
    p1_hi  = np.asarray(p1_hi, dtype=float)
    p2     = np.asarray(p2, dtype=float)
    p2_lo  = np.asarray(p2_lo, dtype=float)
    p2_hi  = np.asarray(p2_hi, dtype=float)

    # Handle NaNs safely
    def _nan0(a): return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    p1, p1_lo, p1_hi = _nan0(p1), _nan0(p1_lo), _nan0(p1_hi)
    p2, p2_lo, p2_hi = _nan0(p2), _nan0(p2_lo), _nan0(p2_hi)

    # Build asymmetric error bars (2 × N): [lower; upper], nonnegative
    yerr1 = np.vstack([np.maximum(p1 - p1_lo, 0.0), np.maximum(p1_hi - p1, 0.0)])
    yerr2 = np.vstack([np.maximum(p2 - p2_lo, 0.0), np.maximum(p2_hi - p2, 0.0)])

    x = np.arange(len(labels))
    w = 0.35

    ax.bar(x - w/2, p1, w, label="Pass1", yerr=yerr1, capsize=3)
    ax.bar(x + w/2, p2, w, label="Pass2", yerr=yerr2, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

def plot_question_overall_ci(q_overall: pd.DataFrame, out_png: str, also_pdf: bool):
    import matplotlib.pyplot as plt
    if q_overall.empty: return
    row = q_overall.iloc[0]
    labels = ["Any-correct"]
    p1 = [row["any_pass1"]]; p2 = [row["any_pass2"]]
    p1_lo=[row["any_pass1_lo"]]; p1_hi=[row["any_pass1_hi"]]
    p2_lo=[row["any_pass2_lo"]]; p2_hi=[row["any_pass2_hi"]]
    fig = plt.figure(); ax = fig.add_subplot(111)
    _plot_bars_with_ci(ax, labels, p1, p1_lo, p1_hi, p2, p2_lo, p2_hi,
                       "Question-level re-asking (overall)", "Fraction of problems solved ≥1 time")
    fig.tight_layout(); fig.savefig(out_png, bbox_inches="tight", dpi=200)
    if also_pdf: fig.savefig(out_png.replace(".png",".pdf"), bbox_inches="tight")
    plt.close(fig)

def plot_question_by_step_ci(q_step: pd.DataFrame, out_png: str, also_pdf: bool):
    import matplotlib.pyplot as plt
    if q_step.empty or "step" not in q_step.columns: return
    fig = plt.figure(); ax = fig.add_subplot(111)
    steps = q_step["step"].tolist()
    ax.errorbar(steps, q_step["any_pass1"], yerr=[q_step["any_pass1"]-q_step["any_pass1_lo"], q_step["any_pass1_hi"]-q_step["any_pass1"]],
                fmt="-o", label="Pass1")
    ax.errorbar(steps, q_step["any_pass2"], yerr=[q_step["any_pass2"]-q_step["any_pass2_lo"], q_step["any_pass2_hi"]-q_step["any_pass2"]],
                fmt="-o", label="Pass2")
    ax.set_xlabel("Step"); ax.set_ylabel("Fraction solved ≥1 time")
    ax.set_title("Question-level re-asking by step"); ax.legend()
    fig.tight_layout(); fig.savefig(out_png, bbox_inches="tight", dpi=200)
    if also_pdf: fig.savefig(out_png.replace(".png",".pdf"), bbox_inches="tight")
    plt.close(fig)

def plot_question_by_bucket_ci(q_bucket: pd.DataFrame, out_png: str, also_pdf: bool):
    import matplotlib.pyplot as plt
    if q_bucket.empty or "perplexity_bucket" not in q_bucket.columns: return
    # Order buckets as they appear
    labels = q_bucket["perplexity_bucket"].tolist()
    p1 = q_bucket["any_pass1"].tolist(); p2 = q_bucket["any_pass2"].tolist()
    p1_lo=q_bucket["any_pass1_lo"].tolist(); p1_hi=q_bucket["any_pass1_hi"].tolist()
    p2_lo=q_bucket["any_pass2_lo"].tolist(); p2_hi=q_bucket["any_pass2_hi"].tolist()
    fig = plt.figure(); ax = fig.add_subplot(111)
    _plot_bars_with_ci(ax, labels, p1, p1_lo, p1_hi, p2, p2_lo, p2_hi,
                       "Question-level re-asking by (entropy) bucket", "Fraction solved ≥1 time")
    fig.tight_layout(); fig.savefig(out_png, bbox_inches="tight", dpi=200)
    if also_pdf: fig.savefig(out_png.replace(".png",".pdf"), bbox_inches="tight")
    plt.close(fig)

def plot_prompt_overall_ci(p_overall: pd.DataFrame, out_png: str, also_pdf: bool):
    import matplotlib.pyplot as plt
    if p_overall.empty: return
    row = p_overall.iloc[0]
    labels = ["Prompt accuracy"]
    p1 = [row["acc_pass1"]]; p2 = [row["acc_pass2"]]
    p1_lo=[row["acc_pass1_lo"]]; p1_hi=[row["acc_pass1_hi"]]
    p2_lo=[row["acc_pass2_lo"]]; p2_hi=[row["acc_pass2_hi"]]
    fig = plt.figure(); ax = fig.add_subplot(111)
    _plot_bars_with_ci(ax, labels, p1, p1_lo, p1_hi, p2, p2_lo, p2_hi,
                       "Prompt-level re-asking (overall)", "Accuracy over prompts")
    fig.tight_layout(); fig.savefig(out_png, bbox_inches="tight", dpi=200)
    if also_pdf: fig.savefig(out_png.replace(".png",".pdf"), bbox_inches="tight")
    plt.close(fig)

def plot_prompt_by_step_ci(p_step: pd.DataFrame, out_png: str, also_pdf: bool):
    import matplotlib.pyplot as plt
    if p_step.empty or "step" not in p_step.columns: return
    fig = plt.figure(); ax = fig.add_subplot(111)
    steps = p_step["step"].tolist()
    ax.errorbar(steps, p_step["acc_pass1"],
                yerr=[p_step["acc_pass1"]-p_step["acc_pass1_lo"], p_step["acc_pass1_hi"]-p_step["acc_pass1"]],
                fmt="-o", label="Pass1")
    ax.errorbar(steps, p_step["acc_pass2"],
                yerr=[p_step["acc_pass2"]-p_step["acc_pass2_lo"], p_step["acc_pass2_hi"]-p_step["acc_pass2"]],
                fmt="-o", label="Pass2")
    ax.set_xlabel("Step"); ax.set_ylabel("Accuracy over prompts")
    ax.set_title("Prompt-level re-asking by step"); ax.legend()
    fig.tight_layout(); fig.savefig(out_png, bbox_inches="tight", dpi=200)
    if also_pdf: fig.savefig(out_png.replace(".png",".pdf"), bbox_inches="tight")
    plt.close(fig)

def plot_prompt_by_bucket_ci(p_bucket: pd.DataFrame, out_png: str, also_pdf: bool):
    import matplotlib.pyplot as plt
    if p_bucket.empty or "perplexity_bucket" not in p_bucket.columns: return
    labels = p_bucket["perplexity_bucket"].tolist()
    p1 = p_bucket["acc_pass1"].tolist(); p2 = p_bucket["acc_pass2"].tolist()
    p1_lo=p_bucket["acc_pass1_lo"].tolist(); p1_hi=p_bucket["acc_pass1_hi"].tolist()
    p2_lo=p_bucket["acc_pass2_lo"].tolist(); p2_hi=p_bucket["acc_pass2_hi"].tolist()
    fig = plt.figure(); ax = fig.add_subplot(111)
    _plot_bars_with_ci(ax, labels, p1, p1_lo, p1_hi, p2, p2_lo, p2_hi,
                       "Prompt-level re-asking by (entropy) bucket", "Accuracy over prompts")
    fig.tight_layout(); fig.savefig(out_png, bbox_inches="tight", dpi=200)
    if also_pdf: fig.savefig(out_png.replace(".png",".pdf"), bbox_inches="tight")
    plt.close(fig)

def plot_prompt_level_deltas(pairs_df: pd.DataFrame, out_png: str, by_forced: bool = True, also_pdf: bool = False):
    import matplotlib.pyplot as plt
    if pairs_df.empty: return
    if by_forced and "forced_insight" in pairs_df.columns:
        cats = []
        for forced, sub in pairs_df.groupby("forced_insight"):
            vals, counts = np.unique(sub["delta"], return_counts=True)
            mapping = {int(v): int(c) for v,c in zip(vals, counts)}
            cats.append((forced, [mapping.get(-1,0), mapping.get(0,0), mapping.get(1,0)]))
        labels = ["Δ=-1", "Δ=0", "Δ=+1"]
        x = np.arange(len(labels))
        fig = plt.figure(); ax = fig.add_subplot(111)
        width = 0.35
        for i, (forced, counts) in enumerate(cats):
            ax.bar(x + i*width, counts, width, label=f"forced={forced}")
        ax.set_xticks(x + width*(len(cats)-1)/2)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Count of prompts")
        ax.set_title("Prompt-level re-asking Δ (correct2 - correct1)")
        ax.legend()
        fig.tight_layout(); fig.savefig(out_png, bbox_inches="tight", dpi=200)
        if also_pdf: fig.savefig(out_png.replace(".png",".pdf"), bbox_inches="tight")
        plt.close(fig)
    else:
        vals, counts = np.unique(pairs_df["delta"], return_counts=True)
        mapping = {int(v): int(c) for v,c in zip(vals, counts)}
        labels = ["Δ=-1", "Δ=0", "Δ=+1"]
        heights = [mapping.get(-1,0), mapping.get(0,0), mapping.get(1,0)]
        fig = plt.figure(); ax = fig.add_subplot(111)
        ax.bar(labels, heights)
        ax.set_ylabel("Count of prompts")
        ax.set_title("Prompt-level re-asking Δ (correct2 - correct1)")
        fig.tight_layout(); fig.savefig(out_png, bbox_inches="tight", dpi=200)
        if also_pdf: fig.savefig(out_png.replace(".png",".pdf"), bbox_inches="tight")
        plt.close(fig)

# ---------------- A4 summary (GLM margins) ----------------

def write_a4_summary_pdf(margins_df: pd.DataFrame,
                         out_pdf: str,
                         dataset: str,
                         model: str,
                         font_family: str = "Times New Roman",
                         font_size: int = 12):
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": [font_family, "Times", "DejaVu Serif", "Computer Modern Roman"],
        "font.size": font_size,
        "axes.titlesize": font_size, "axes.labelsize": font_size,
        "xtick.labelsize": font_size, "ytick.labelsize": font_size,
        "legend.fontsize": font_size, "figure.titlesize": font_size,
    })
    A4 = (8.27, 11.69)
    fig = plt.figure(figsize=A4, dpi=300)
    ax = fig.add_axes([0.08, 0.08, 0.84, 0.84]); ax.axis("off")
    y = 0.98
    ax.text(0.0, y, f"H3 Summary — {dataset}, {model}", ha="left", va="top", weight="bold"); y -= 0.04
    ax.text(0.0, y, "Bucket-wise Aha Margins (Δ Acc):", ha="left", va="top", weight="bold"); y -= 0.02
    hdr  = ["Variant","Bucket","N","Share Aha","Δ Acc"]
    x = [0.00, 0.28, 0.56, 0.72, 0.86]
    for xi, h in zip(x, hdr): ax.text(xi, y, h, ha="left", va="top", weight="bold")
    y -= 0.012; ax.plot([0.00, 0.98],[y,y], color="black", lw=0.5); y -= 0.01
    for _, r in margins_df.head(20).iterrows():
        ax.text(x[0], y, str(r["variant"]).title(), ha="left", va="top")
        ax.text(x[1], y, r["perplexity_bucket"], ha="left", va="top")
        ax.text(x[2], y, f"{int(r['N'])}", ha="left", va="top")
        ax.text(x[3], y, f"{r['share_aha']:.3f}", ha="left", va="top")
        ax.text(x[4], y, f"{r['AME_bucket']:+.3f}", ha="left", va="top"); y -= 0.028
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.2); plt.close(fig)

# ---------------- Writing the answers.txt ----------------

def write_answers_txt(path: str,
                      cond_df: pd.DataFrame,
                      q_overall: pd.DataFrame,
                      q_by_step: pd.DataFrame,
                      q_by_bucket: pd.DataFrame,
                      p_overall: pd.DataFrame,
                      p_by_step: pd.DataFrame,
                      p_by_bucket: pd.DataFrame):
    lines = []
    lines.append("H3 Answers\n==========\n")
    # Overall conditioned
    if not cond_df.empty:
        a = cond_df[cond_df["condition"]=="pass1_any_correct==1"]
        b = cond_df[cond_df["condition"]=="pass1_any_correct==0"]
        if not a.empty:
            a = a.iloc[0]
            lines += [
                "A) Given ≥1 correct in pass1, does pass2 increase overall accuracy?",
                f"  Problems: {int(a['n_problems'])}",
                f"  Macro mean acc: pass1={a['macro_mean_acc_pass1']:.4f}, pass2={a['macro_mean_acc_pass2']:.4f}, Δ={a['macro_delta_mean']:+.4f}",
                f"  Micro mean acc: pass1={a['micro_acc_pass1']:.4f}, pass2={a['micro_acc_pass2']:.4f}, Δ={a['micro_delta']:+.4f}",
                f"  Reliability (stay any-correct in pass2): {a['share_keep_any_correct_in_pass2']:.4f}",
                "",
            ]
        if not b.empty:
            b = b.iloc[0]
            lines += [
                "B) Given wrong every time in pass1, does pass2 get it correct at least once?",
                f"  Problems: {int(b['n_problems'])}",
                f"  Share any-correct in pass2: {b['share_any_pass2']:.4f}",
                f"  Macro mean acc: pass1={b['macro_mean_acc_pass1']:.4f}, pass2={b['macro_mean_acc_pass2']:.4f}, Δ={b['macro_delta_mean']:+.4f}",
                f"  Micro mean acc: pass1={b['micro_acc_pass1']:.4f}, pass2={b['micro_acc_pass2']:.4f}, Δ={b['micro_delta']:+.4f}",
                "",
            ]
    # Overall CI recaps
    def _fmt_overall_block(title, df, name1, name2):
        if df.empty: return []
        r = df.iloc[0]
        return [
            title,
            f"  Pass1 {name1}: {r[name1]:.4f}  (95% CI: {r[name1+'_lo']:.4f}, {r[name1+'_hi']:.4f})",
            f"  Pass2 {name2}: {r[name2]:.4f}  (95% CI: {r[name2+'_lo']:.4f}, {r[name2+'_hi']:.4f})",
            f"  Δ = {r['delta']:+.4f}",
            ""
        ]
    lines += _fmt_overall_block("Question-level (any-correct, overall)",
                                q_overall, "any_pass1", "any_pass2")
    lines += _fmt_overall_block("Prompt-level (accuracy, overall)",
                                p_overall, "acc_pass1", "acc_pass2")

    # Step-wise and bucket-wise short recaps
    def _fmt_table(df, keys, metric, lo, hi, label):
        L = []
        for _, r in df.iterrows():
            keytxt = ", ".join(f"{k}={r[k]}" for k in keys)
            L.append(f"{label} [{keytxt}]  pass1={r[metric]:.4f} (CI {r[lo]:.4f},{r[hi]:.4f})")
        return L

    if not q_by_step.empty:
        lines += ["Question-level by step (any-correct):"]
        lines += _fmt_table(q_by_step, ["step"], "any_pass1", "any_pass1_lo", "any_pass1_hi", "  P1") + \
                 _fmt_table(q_by_step, ["step"], "any_pass2", "any_pass2_lo", "any_pass2_hi", "  P2") + [""]
    if not q_by_bucket.empty:
        lines += ["Question-level by (entropy) bucket (any-correct):"]
        lines += _fmt_table(q_by_bucket, ["perplexity_bucket"], "any_pass1", "any_pass1_lo", "any_pass1_hi", "  P1") + \
                 _fmt_table(q_by_bucket, ["perplexity_bucket"], "any_pass2", "any_pass2_lo", "any_pass2_hi", "  P2") + [""]

    if not p_by_step.empty:
        lines += ["Prompt-level by step (accuracy):"]
        lines += _fmt_table(p_by_step, ["step"], "acc_pass1", "acc_pass1_lo", "acc_pass1_hi", "  P1") + \
                 _fmt_table(p_by_step, ["step"], "acc_pass2", "acc_pass2_lo", "acc_pass2_hi", "  P2") + [""]
    if not p_by_bucket.empty:
        lines += ["Prompt-level by (entropy) bucket (accuracy):"]
        lines += _fmt_table(p_by_bucket, ["perplexity_bucket"], "acc_pass1", "acc_pass1_lo", "acc_pass1_hi", "  P1") + \
                 _fmt_table(p_by_bucket, ["perplexity_bucket"], "acc_pass2", "acc_pass2_lo", "acc_pass2_hi", "  P2") + [""]

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_root")
    ap.add_argument("--split", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--dataset_name", default="MATH-500")
    ap.add_argument("--model_name", default="Qwen2.5-1.5B")

    # Uncertainty / bucketing
    ap.add_argument("--unc_field", choices=["answer","overall","think"], default="answer")
    ap.add_argument("--measure", choices=["perplexity","entropy"], default="entropy")
    ap.add_argument("--n_buckets", type=int, default=5)
    ap.add_argument("--bucket_method", choices=["quantile","fixed"], default="quantile")
    ap.add_argument("--bucket_edges", default=None,
                    help="Comma-separated edges for fixed bins (entropy values, e.g., 0.2,0.5,1,1.5,2,3)")

    # GPT label policy
    ap.add_argument("--gpt_mode", choices=["canonical","broad"], default="canonical")
    ap.add_argument("--no_gate_gpt_by_words", action="store_true",
                    help="If set, GPT shifts are NOT restricted to samples with Words-cue.")

    # Formal thresholds
    ap.add_argument("--delta1", type=float, default=0.13)
    ap.add_argument("--delta2", type=float, default=0.13)
    ap.add_argument("--min_prior_steps", type=int, default=2)

    # GLM options
    ap.add_argument("--cluster_by", choices=["problem","none"], default="problem")
    ap.add_argument("--strict_interaction_only", action="store_true",
                    help="Fit correct ~ C(problem) + step_std + aha:C(perplexity_bucket).")

    # Re-asking split by Aha
    ap.add_argument("--split_reasking_by_aha", action="store_true",
                    help="Additionally write re-asking summaries split by Words/GPT/Formal Aha.")

    # Output / plots
    ap.add_argument("--tex_path", default=None)
    ap.add_argument("--make_pdf", action="store_true")
    ap.add_argument("--make_plots", action="store_true",
                    help="Emit PNG plots for question- and prompt-level re-asking effects (overall/step/bucket).")
    ap.add_argument("--also_pdf_plots", action="store_true")
    ap.add_argument("--font_family", default="Times New Roman")
    ap.add_argument("--font_size", type=int, default=12)

    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.results_root, "h3_uncertainty_buckets")
    os.makedirs(out_dir, exist_ok=True)
    files = scan_files(args.results_root, args.split)
    if not files: raise SystemExit("No JSONL files found.")

    # Load rows (pass1+pass2)
    df = load_rows(
        files,
        gpt_mode=args.gpt_mode,
        gate_gpt_by_words=not args.no_gate_gpt_by_words,
        unc_field=args.unc_field,
        measure=args.measure,
    )
    if df["pass_id"].min() > 1 or df["pass_id"].max() < 1:
        raise SystemExit("No pass1 rows present; cannot proceed.")

    # Pass1 slice for buckets & Formal flags
    d1 = df[df["pass_id"] == 1].copy()

    # Formal flags and gating
    ps = build_problem_step(d1)
    ps = mark_formal(ps, delta1=args.delta1, delta2=args.delta2, min_prior_steps=args.min_prior_steps)
    d1 = d1.merge(ps[["step","problem","aha_formal_ps"]], on=["step","problem"], how="left").fillna({"aha_formal_ps":0})
    d1["aha_formal"] = (d1["aha_formal_ps"].astype(int) & d1["aha_gpt"].astype(int)).astype(int)

    # Bucketing (entropy-based by default)
    custom_edges = None
    if args.bucket_method == "fixed" and args.bucket_edges:
        try:
            custom_edges = [float(x.strip()) for x in args.bucket_edges.split(",") if x.strip()]
        except Exception:
            raise SystemExit("Failed to parse --bucket_edges.")
    d1 = add_perplexity_buckets(d1, n_buckets=int(args.n_buckets),
                                method=args.bucket_method, custom_edges=custom_edges)

    # GLMs per Aha variant
    variants = [("aha_words","words"), ("aha_gpt","gpt"), ("aha_formal","formal")]
    all_margin_rows = []
    os.makedirs(os.path.join(out_dir, "h3_glm_fit_summaries"), exist_ok=True)
    for aha_col, tag in variants:
        sub = d1[~d1[aha_col].isna()].copy()
        if sub.empty: continue
        out_txt = os.path.join(out_dir, "h3_glm_fit_summaries", f"logit_pass1_correct_on_{aha_col}_by_bucket.txt")
        model_info, _ = fit_glm_bucket_interaction(
            df=sub,
            aha_col=aha_col,
            strict_interaction_only=args.strict_interaction_only,
            cluster_by=args.cluster_by,
            out_txt=out_txt,
        )
        for r in model_info["bucket_rows"]:
            all_margin_rows.append({
                "dataset": args.dataset_name,
                "model": args.model_name,
                "variant": tag,
                "perplexity_bucket": r["bucket"],
                "N": r["N"],
                "share_aha": r["share_aha"],
                "AME_bucket": r["AME_bucket"],
                "glm_summary_path": out_txt,
            })
        acc_df = bucket_group_accuracy(sub, aha_col=aha_col)
        acc_df.insert(0, "variant", tag)
        acc_df.to_csv(os.path.join(out_dir, f"h3_bucket_group_accuracy__{tag}.csv"), index=False)

    margins_df = pd.DataFrame(all_margin_rows).sort_values(["variant","perplexity_bucket"]).reset_index(drop=True)
    margins_df.to_csv(os.path.join(out_dir, "h3_glm_bucket_margins.csv"), index=False)

    # Concatenate group accuracies
    acc_parts = []
    for _, tag in variants:
        p = os.path.join(out_dir, f"h3_bucket_group_accuracy__{tag}.csv")
        if os.path.exists(p): acc_parts.append(pd.read_csv(p))
    if acc_parts:
        pd.concat(acc_parts, ignore_index=True).to_csv(os.path.join(out_dir, "h3_bucket_group_accuracy.csv"), index=False)

    # ---------- Re-asking analyses ----------
    d1_buckets = d1[["pair_id","perplexity_bucket","aha_words","aha_gpt","aha_formal"]].drop_duplicates()
    df2 = df.merge(d1_buckets, on="pair_id", how="left")

    pairs_df, probs_df, cond_df, cond_forced_df = compute_reasking_tables(df2, pass1_bucket_col="perplexity_bucket")

    # Raw aligned tables
    if not pairs_df.empty:
        pairs_df.to_csv(os.path.join(out_dir, "h3_pass2_prompt_level.csv"), index=False)
    if not probs_df.empty:
        probs_df.to_csv(os.path.join(out_dir, "h3_pass2_question_level.csv"), index=False)
    if not cond_df.empty:
        cond_df.to_csv(os.path.join(out_dir, "h3_pass2_conditioned_summary.csv"), index=False)
    if not cond_forced_df.empty:
        cond_forced_df.to_csv(os.path.join(out_dir, "h3_pass2_conditioned_by_forcing.csv"), index=False)

    # Aggregations with Wilson CIs
    if not pairs_df.empty:
        # PROMPT-LEVEL (overall/step/bucket)
        p_overall = prompt_level_acc_with_ci(pairs_df, [])
        p_overall.to_csv(os.path.join(out_dir, "h3_prompt_level_overall.csv"), index=False)
        p_by_step = prompt_level_acc_with_ci(pairs_df, ["step"])
        p_by_step.to_csv(os.path.join(out_dir, "h3_prompt_level_by_step.csv"), index=False)
        if "perplexity_bucket" in pairs_df.columns:
            p_by_bucket = prompt_level_acc_with_ci(pairs_df, ["perplexity_bucket"])
            p_by_bucket.to_csv(os.path.join(out_dir, "h3_prompt_level_by_bucket.csv"), index=False)
        else:
            p_by_bucket = pd.DataFrame()
    else:
        p_overall = p_by_step = p_by_bucket = pd.DataFrame()

    if not pairs_df.empty:
        # QUESTION-LEVEL (derive any-correct from pairs_df via per-problem)
        q_overall = question_level_any_with_ci(pairs_df, [])
        q_overall.to_csv(os.path.join(out_dir, "h3_question_level_overall.csv"), index=False)
        q_by_step = question_level_any_with_ci(pairs_df, ["step"])
        q_by_step.to_csv(os.path.join(out_dir, "h3_question_level_by_step.csv"), index=False)
        if "perplexity_bucket" in pairs_df.columns:
            q_by_bucket = question_level_any_with_ci(pairs_df, ["perplexity_bucket"])
            q_by_bucket.to_csv(os.path.join(out_dir, "h3_question_level_by_bucket.csv"), index=False)
        else:
            q_by_bucket = pd.DataFrame()
    else:
        q_overall = q_by_step = q_by_bucket = pd.DataFrame()

    # Optional: re-asking summaries split by Aha
    if args.split_reasking_by_aha and (not pairs_df.empty) and (not probs_df.empty):
        prompt_by_aha, problem_by_aha = split_reasking_by_aha(pairs_df, probs_df, d1)
        if not prompt_by_aha.empty:
            prompt_by_aha.to_csv(os.path.join(out_dir, "h3_pass2_prompt_level_by_aha.csv"), index=False)
        if not problem_by_aha.empty:
            problem_by_aha.to_csv(os.path.join(out_dir, "h3_pass2_question_level_by_aha.csv"), index=False)

    # Plots
    if args.make_plots:
        if not q_overall.empty:
            plot_question_overall_ci(q_overall, os.path.join(out_dir, "h3_plot_question_overall.png"), args.also_pdf_plots)
        if not q_by_step.empty:
            plot_question_by_step_ci(q_by_step, os.path.join(out_dir, "h3_plot_question_by_step.png"), args.also_pdf_plots)
        if not q_by_bucket.empty:
            plot_question_by_bucket_ci(q_by_bucket, os.path.join(out_dir, "h3_plot_question_by_bucket.png"), args.also_pdf_plots)

        if not p_overall.empty:
            plot_prompt_overall_ci(p_overall, os.path.join(out_dir, "h3_plot_prompt_overall.png"), args.also_pdf_plots)
        if not p_by_step.empty:
            plot_prompt_by_step_ci(p_by_step, os.path.join(out_dir, "h3_plot_prompt_by_step.png"), args.also_pdf_plots)
        if not p_by_bucket.empty:
            plot_prompt_by_bucket_ci(p_by_bucket, os.path.join(out_dir, "h3_plot_prompt_by_bucket.png"), args.also_pdf_plots)

        if not pairs_df.empty:
            plot_prompt_level_deltas(pairs_df, os.path.join(out_dir, "h3_plot_prompt_level_delta.png"), by_forced=True, also_pdf=args.also_pdf_plots)

    # A4 GLM summary
    if args.make_pdf:
        pdf_path = os.path.join(out_dir, "h3_summary.pdf")
        write_a4_summary_pdf(
            margins_df=margins_df,
            out_pdf=pdf_path,
            dataset=args.dataset_name, model=args.model_name,
            font_family=args.font_family, font_size=int(args.font_size),
        )
        print("A4 summary PDF:", pdf_path)

    # Write answers.txt
    answers_path = os.path.join(out_dir, "h3_answers.txt")
    write_answers_txt(answers_path, cond_df, q_overall, q_by_step, q_by_bucket,
                      p_overall, p_by_step, p_by_bucket)
    print("Answers written to:", answers_path)

    # File pointers
    print("\nWROTE:")
    for fn in [
        "h3_glm_bucket_margins.csv",
        "h3_bucket_group_accuracy.csv",
        "h3_pass2_prompt_level.csv",
        "h3_pass2_question_level.csv",
        "h3_pass2_conditioned_summary.csv",
        "h3_pass2_conditioned_by_forcing.csv",
        "h3_prompt_level_overall.csv",
        "h3_prompt_level_by_step.csv",
        "h3_prompt_level_by_bucket.csv",
        "h3_question_level_overall.csv",
        "h3_question_level_by_step.csv",
        "h3_question_level_by_bucket.csv",
        "h3_plot_question_overall.png",
        "h3_plot_question_by_step.png",
        "h3_plot_question_by_bucket.png",
        "h3_plot_prompt_overall.png",
        "h3_plot_prompt_by_step.png",
        "h3_plot_prompt_by_bucket.png",
        "h3_plot_prompt_level_delta.png",
        "h3_summary.pdf",
        "h3_answers.txt",
    ]:
        pth = os.path.join(out_dir, fn)
        if os.path.exists(pth):
            print(" ", pth)

if __name__ == "__main__":
    main()
