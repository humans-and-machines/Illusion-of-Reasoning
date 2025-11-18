#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Uncertainty buckets: Aha vs No-Aha (accuracy, odds, effects)
Qwen-7B & Llama-8B only; all temperatures & steps

Adds (kept from 1.5B script):
- Stratified CMH test (domain×entropy-bin strata) on P(correct | shift)
- GLM LRT "ANOVA" with stratum fixed effects
- Domain-wise two-proportion tests and pooled test
- Tidy CSV + concise console summary

NEW for 7B/8B:
- Flexible discovery: finds directories whose names include
  (qwen & 7b) or (llama & 8b), plus domain (math/xword/carpark) and temp tokens.
- Per-model global regressions:
    (a) Training stage:   correct ~ C(problem_id) + step_std + shift
    (b) Temperature:      correct ~ C(problem_id) + C(temp) + shift
  -> Produces N, shift share, p(correct|S=1), Δpp, AME, p-value.

Multi-metric runner:
- Single metric: use --metric entropy_answer|entropy|entropy_think
- Multiple: use --metrics entropy_answer entropy entropy_think answer_plus_think
- Default (no --metric/--metrics): runs all four.

Binning:
- fixed edges via --fixed_bins "0,0.5,1,2,inf"
- or equal-count bins via --equal_n_bins --bins K (global/domain scope)

Only steps <= 1000 are loaded by default (hard cap).
"""

import os, re, json, argparse, warnings
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import chi2

# ---------------------------
# Model/domain name handling
# ---------------------------

MODEL_LABELS = {
    "qwen7b":  "Qwen2.5-7B",
    "llama8b": "Llama3.1-8B",
}

DOMAIN_NORMALIZE = {
    "xword": "Crossword",
    "crossword": "Crossword",
    "math": "Math",
    "carpark": "Carpark",
    "rush": "Carpark",
    "rushhour": "Carpark",
}

TEMP_RE = re.compile(r"temp-([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)

def _detect_model_key(name_lower: str) -> Optional[str]:
    if "llama" in name_lower and "8b" in name_lower:
        return "llama8b"
    if "qwen" in name_lower and "7b" in name_lower:
        return "qwen7b"
    # Fallback for dirs like "GRPO-7B-math-temp-0.7"
    if "7b" in name_lower and "llama" not in name_lower:
        return "qwen7b"
    return None

def _detect_domain(name_lower: str) -> Optional[str]:
    for tok, dom in DOMAIN_NORMALIZE.items():
        if tok in name_lower:
            return dom
    return None

def _detect_temp(name_lower: str, low_alias: float) -> Optional[float]:
    m = TEMP_RE.search(name_lower)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    if "low-temp" in name_lower or "low_temp" in name_lower:
        return float(low_alias)
    return None

def discover_roots_7b8b(scan_root: str,
                        temps: List[float],
                        low_alias: float,
                        want_models: List[str],
                        want_domains: List[str],
                        verbose: bool) -> Dict[str, Dict[float, Dict[str, str]]]:
    """
    Returns: mapping_by_model[model_key][T][domain] = path
    Prefers the longest path string if duplicates exist.
    """
    temps_set = {float(x) for x in temps}
    want_models = [m.lower() for m in want_models]
    want_domains = set(want_domains)

    mapping: Dict[str, Dict[float, Dict[str, str]]] = {}

    for entry in os.listdir(scan_root):
        full = os.path.join(scan_root, entry)
        if not os.path.isdir(full):
            continue
        low = entry.lower()

        model_key = _detect_model_key(low)
        if model_key is None or model_key not in want_models:
            continue

        dom = _detect_domain(low)
        if dom is None or dom not in want_domains:
            continue

        t = _detect_temp(low, low_alias)
        if t is None or float(t) not in temps_set:
            continue

        mapping.setdefault(model_key, {})
        mapping[model_key].setdefault(float(t), {})
        prev = mapping[model_key][float(t)].get(dom)
        if prev is None or len(full) > len(prev):
            mapping[model_key][float(t)][dom] = full

    if verbose:
        print("[info] discovered roots (7B/8B):")
        for mk in mapping:
            print(f"  [{MODEL_LABELS.get(mk, mk)}]")
            for T in sorted(mapping[mk]):
                print(f"    T={T}:")
                for d, p in mapping[mk][T].items():
                    print(f"      {d:9s} -> {p}")
    return mapping

# ==========
# Utilities
# ==========

def coerce_bool(x) -> Optional[int]:
    if x is None: return None
    if isinstance(x,bool): return int(x)
    if isinstance(x,(int,np.integer)): return int(bool(x))
    if isinstance(x,str):
        s=x.strip().lower()
        if s in ("1","true","t","yes","y"): return 1
        if s in ("0","false","f","no","n"): return 0
    try: return int(bool(x))
    except Exception: return None

def coerce_float(x) -> Optional[float]:
    if x is None: return None
    try: return float(x)
    except Exception: return None

def get_pid(rec: Dict[str,Any]) -> Optional[str]:
    for k in ("problem_id","example_id","id","question","clue","title"):
        v = rec.get(k)
        if v is not None and not isinstance(v,(list,dict)): return str(v)
    si = rec.get("sample_idx")
    return f"sample_{si}" if si is not None else None

# =====================
# Aha (LLM) gating
# =====================

def aha_gate(p1: Dict[str,Any], domain: str, mode: str) -> int:
    keys = (["change_way_of_thinking","shift_in_reasoning_v1"]
            if mode=="canonical" else
            ["change_way_of_thinking","shift_in_reasoning_v1","shift_llm","shift_gpt","pivot_llm","rechecked"])
    has_recon = coerce_bool(p1.get("has_reconsider_cue")) == 1
    rec_marks = p1.get("reconsider_markers") or []
    injected  = ("injected_cue" in rec_marks)
    recon_ok  = has_recon and not injected
    pre = p1.get("_shift_prefilter_markers") or []
    judge = p1.get("shift_markers_v1") or []
    gate = (recon_ok or bool(pre) or (bool(judge) if domain.lower()=="crossword" else False))
    raw = 0
    for k in keys:
        cb = coerce_bool(p1.get(k))
        if cb == 1: raw = 1; break
    return int(raw and gate)

# ==========================
# Load pass-1 rows (all T)
# ==========================

def load_rows(dir_path: str, split: str, domain: str,
              entropy_source: str, allow_fallback: bool,
              carpark_op: str, carpark_thr: float,
              gpt_mode: str, verbose: bool,
              combined_mode: str = "sum",
              min_step: Optional[int] = None,
              max_step: Optional[int] = None,
              temp: Optional[float] = None,
              model_label: Optional[str] = None) -> pd.DataFrame:
    files = scan_step_jsonls(dir_path, split, verbose)
    recs=[]
    for fp in files:
        # infer step from path tokens like "step250"
        step=None
        for tok in os.path.normpath(fp).split(os.sep):
            if tok.lower().startswith("step"):
                digits=re.sub("[^0-9]","",tok)
                if digits:
                    try: step=int(digits)
                    except Exception: step=None

        # skip unknown step or outside bounds
        if step is None:
            continue
        if min_step is not None and step < min_step:
            continue
        if max_step is not None and step > max_step:
            continue

        with open(fp,"r",encoding="utf-8") as f:
            for ln in f:
                s=ln.strip()
                if not s: continue
                try: rec=json.loads(s)
                except Exception: continue
                p1=rec.get("pass1") or {}
                if not isinstance(p1,dict): p1={}
                pid=get_pid(rec)
                if pid is None: continue
                pid=f"{domain}::{pid}"

                # entropy (supports 'answer_plus_think')
                def ent_from(p):
                    if   entropy_source=="entropy_answer":
                        e=coerce_float(p.get("entropy_answer"))
                    elif entropy_source=="entropy":
                        e=coerce_float(p.get("entropy"))
                    elif entropy_source=="entropy_think":
                        e=coerce_float(p.get("entropy_think"))
                    elif entropy_source=="answer_plus_think":
                        ea=coerce_float(p.get("entropy_answer"))
                        et=coerce_float(p.get("entropy_think"))
                        if ea is not None and et is not None:
                            e=(ea+et) if combined_mode=="sum" else (ea+et)/2.0
                        else:
                            e=None
                            if allow_fallback:
                                e=coerce_float(p.get("entropy"))
                    else:
                        e=None
                    if e is None and allow_fallback:
                        for alt in ("entropy_answer","entropy","entropy_think"):
                            altv=coerce_float(p.get(alt))
                            if altv is not None:
                                e=altv
                                break
                    return e

                ent=ent_from(p1)
                if ent is None: continue

                # correctness
                if domain=="Carpark":
                    sr=coerce_float(p1.get("soft_reward"))
                    if sr is None: continue
                    if   carpark_op=="gt":  corr=int(sr> carpark_thr)
                    elif carpark_op=="ge":  corr=int(sr>=carpark_thr)
                    elif carpark_op=="eq":  corr=int(sr==carpark_thr)
                    else:                   corr=int(sr>=carpark_thr)
                else:
                    cb=coerce_bool(p1.get("is_correct_pred"))
                    if cb is None: continue
                    corr=int(cb)

                shift=aha_gate(p1, domain, gpt_mode)

                recs.append({
                    "model": model_label or "UNKNOWN",
                    "domain":domain,
                    "temp": temp,
                    "step":int(step),
                    "problem_id":pid,
                    "ent":ent,
                    "correct":corr,
                    "shift":shift
                })
    df=pd.DataFrame(recs)
    if verbose: print(f"[info] loaded rows for {model_label}/{domain} (steps", end="")
    if verbose: print(f" ≤ {max_step}" if max_step is not None else "", end="")
    if verbose: print(f"): {len(df)}")
    return df

def scan_step_jsonls(dir_path: str, split: str, verbose: bool) -> List[str]:
    files, alljs = [], []
    for dp,_,fns in os.walk(dir_path):
        if not os.path.basename(dp).lower().startswith("step"):
            continue
        for fn in fns:
            if not fn.lower().endswith(".jsonl"): continue
            full = os.path.join(dp, fn)
            alljs.append(full)
            if split:
                if split.lower() in fn.lower(): files.append(full)
            else:
                files.append(full)
    if split and not files and verbose:
        print(f"[warn] no files matched split='{split}' in {dir_path}; using ALL ({len(alljs)})")
        files = alljs
    files.sort()
    if verbose: print(f"[info] {dir_path}: {len(files)} JSONLs")
    return files

# =================
# Binning helpers
# =================

def _parse_fixed_edges(s: str) -> np.ndarray:
    toks = [t.strip() for t in s.split(",") if t.strip()]
    vals = []
    for t in toks:
        if t.lower() in {"inf","+inf","infinity"}:
            vals.append(np.inf)
        else:
            vals.append(float(t))
    edges = np.asarray(vals, dtype=float)
    if edges.size < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError(f"fixed_bins must be strictly increasing with ≥2 edges, got: {edges}")
    return edges

def build_edges(df: pd.DataFrame, bins:int, method:str, scope:str)->Dict[str,np.ndarray]:
    edges_by_dom={}
    if scope=="global":
        x=df["ent"].to_numpy()
        e = np.linspace(np.nanmin(x), np.nanmax(x), bins+1) if method=="uniform" \
            else np.quantile(x, np.linspace(0,1,bins+1))
        for d in df["domain"].unique(): edges_by_dom[d]=e
    else:
        for d,sub in df.groupby("domain", sort=False):
            x=sub["ent"].to_numpy()
            e = np.linspace(np.nanmin(x), np.nanmax(x), bins+1) if method=="uniform" \
                else np.quantile(x, np.linspace(0,1,bins+1))
            edges_by_dom[d]=e
    return edges_by_dom

def assign_bins(df: pd.DataFrame, edges_by_dom: Dict[str,np.ndarray]):
    df=df.copy(); df["bin"]=-1
    centers_by_dom={}
    for d,e in edges_by_dom.items():
        centers_by_dom[d]=0.5*(e[:-1]+e[1:])
        m=(df["domain"]==d)
        df.loc[m,"bin"]=np.digitize(df.loc[m,"ent"], e, right=True)-1
        df.loc[m,"bin"]=df.loc[m,"bin"].clip(lower=0, upper=len(e)-2)
    return df, centers_by_dom

def assign_bins_fixed(df: pd.DataFrame,
                      edges_by_dom: Dict[str, np.ndarray],
                      last_center_offset: float = 0.5):
    df = df.copy()
    df["bin"] = -1
    centers_by_dom = {}
    for d, e in edges_by_dom.items():
        m = (df["domain"] == d)
        x = df.loc[m, "ent"].to_numpy()
        idx = np.digitize(x, e, right=False) - 1
        idx = np.clip(idx, 0, len(e) - 2)
        df.loc[m, "bin"] = idx

        centers = 0.5 * (e[:-1] + e[1:])
        if np.isinf(e[-1]):
            centers[-1] = e[-2] + float(last_center_offset)
        centers_by_dom[d] = centers
    return df, centers_by_dom

def _assign_equal_bins_1d(n: int, B: int) -> np.ndarray:
    idx = np.arange(n)
    parts = np.array_split(idx, B)
    out = np.empty(n, dtype=int)
    for b, part in enumerate(parts):
        out[part] = b
    return out

def assign_bins_equal_count(df: pd.DataFrame, bins: int, scope: str,
                            tie_break: str = "stable", seed: int = 0) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["bin"] = -1

    def _sort_keys(frame: pd.DataFrame) -> pd.DataFrame:
        if tie_break == "random":
            jitter = rng.uniform(-1e-12, 1e-12, size=len(frame))
            return frame.assign(_j=jitter).sort_values(["ent", "_j", "domain", "problem_id", "step"], kind="mergesort").drop(columns="_j")
        else:
            return frame.sort_values(["ent", "domain", "problem_id", "step"], kind="mergesort")

    centers_by_dom: Dict[str, np.ndarray] = {}

    if scope == "global":
        srt = _sort_keys(df)
        bvec = _assign_equal_bins_1d(len(srt), bins)
        df.loc[srt.index, "bin"] = bvec
        for dom, sub in df.groupby("domain", sort=False):
            med = sub.groupby("bin", sort=False)["ent"].median()
            centers = med.reindex(range(bins)).to_numpy()
            if np.any(pd.isna(centers)):
                centers = np.where(pd.isna(centers), np.nanmedian(df["ent"].to_numpy()), centers)
            centers_by_dom[dom] = centers
    else:
        for dom, sub in df.groupby("domain", sort=False):
            srt = _sort_keys(sub)
            bvec = _assign_equal_bins_1d(len(srt), bins)
            df.loc[srt.index, "bin"] = bvec
            centers = pd.Series(bvec, index=srt.index).to_frame("bin").join(sub["ent"]).groupby("bin")["ent"].median()
            centers_by_dom[dom] = centers.reindex(range(bins)).fillna(np.nanmedian(sub["ent"])).to_numpy()

    return df, centers_by_dom

# =========================
# Stats, GLM, CI helpers
# =========================

def wilson_ci(k:int,n:int,z:float=1.96):
    if n<=0: return (np.nan,np.nan)
    p=k/n; denom=1+z*z/n
    ctr=(p+z*z/(2*n))/denom
    half=(z/denom)*np.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return (max(0,ctr-half), min(1,ctr+half))

def newcombe_diff_ci(k1,n1,k0,n0):
    if n1==0 or n0==0: return (np.nan,np.nan,np.nan)
    l1,u1=wilson_ci(k1,n1); l0,u0=wilson_ci(k0,n0)
    return ((k1/n1 - k0/n0)*100.0, (l1-u0)*100.0, (u1-l0)*100.0)

def glm_ame_bucket(sub: pd.DataFrame, n_boot:int=300, seed:int=0):
    if sub["correct"].nunique()<2 or sub["shift"].nunique()<2:
        return (np.nan,np.nan,np.nan,np.nan)
    g = sub.groupby("shift")["correct"]
    if g.nunique().min()==1:
        n1=int((sub["shift"]==1).sum()); n0=int((sub["shift"]==0).sum())
        k1=int(sub.loc[sub["shift"]==1,"correct"].sum()) if n1>0 else 0
        k0=int(sub.loc[sub["shift"]==0,"correct"].sum()) if n0>0 else 0
        diff_pp,lo_pp,hi_pp=newcombe_diff_ci(k1,n1,k0,n0)
        return (diff_pp/100.0, lo_pp/100.0, hi_pp/100.0, np.nan)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model=smf.glm("correct ~ C(problem_id) + shift", data=sub, family=sm.families.Binomial())
            res=model.fit(cov_type="cluster", cov_kwds={"groups": sub["problem_id"]}, maxiter=200)
    except (PerfectSeparationError, Exception):
        n1=int((sub["shift"]==1).sum()); n0=int((sub["shift"]==0).sum())
        k1=int(sub.loc[sub["shift"]==1,"correct"].sum()) if n1>0 else 0
        k0=int(sub.loc[sub["shift"]==0,"correct"].sum()) if n0>0 else 0
        diff_pp,lo_pp,hi_pp=newcombe_diff_ci(k1,n1,k0,n0)
        return (diff_pp/100.0, lo_pp/100.0, hi_pp/100.0, np.nan)

    X=res.model.exog; names=list(res.model.exog_names)
    j = names.index("shift") if "shift" in names else max(i for i,n in enumerate(names) if "shift" in n)
    inv=lambda z: 1.0/(1.0+np.exp(-z))
    b=res.params.to_numpy(); x1=X.copy(); x1[:,j]=1.0; x0=X.copy(); x0[:,j]=0.0
    ame=float(np.mean(inv(x1@b)-inv(x0@b)))
    cov=res.cov_params().to_numpy(); rng=np.random.default_rng(seed)
    draws=rng.multivariate_normal(mean=b, cov=cov, size=n_boot)
    ames=[float(np.mean(inv(x1@beta)-inv(x0@beta))) for beta in draws]
    lo,hi=np.percentile(ames,[2.5,97.5])
    p=float(res.pvalues.get("shift",np.nan))
    return (ame,lo,hi,p)

# ---------- NEW: Global regressions (per model) ----------

def _ame_from_fitted_glm(res, df: pd.DataFrame, shift_var: str = "shift", n_boot: int = 300, seed: int = 0):
    X = res.model.exog
    names = list(res.model.exog_names)
    j = names.index(shift_var) if shift_var in names else max(i for i, n in enumerate(names) if shift_var in n)
    inv = lambda z: 1.0 / (1.0 + np.exp(-z))
    b = res.params.to_numpy()
    x1 = X.copy(); x1[:, j] = 1.0
    x0 = X.copy(); x0[:, j] = 0.0
    ame = float(np.mean(inv(x1 @ b) - inv(x0 @ b)))
    cov = res.cov_params().to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(mean=b, cov=cov, size=n_boot)
    ames = [float(np.mean(inv(x1 @ beta) - inv(x0 @ beta))) for beta in draws]
    lo, hi = np.percentile(ames, [2.5, 97.5])
    p = float(res.pvalues.get(shift_var, np.nan))
    return ame, lo, hi, p

def run_model_regressions(df: pd.DataFrame, out_dir: str, model_label: str, slug_base: str) -> pd.DataFrame:
    """
    Runs:
      (a) correct ~ C(problem_id) + step_std + shift
      (b) correct ~ C(problem_id) + C(temp_cat) + shift
    Returns tidy df; also writes CSV.
    """
    if df.empty: 
        return pd.DataFrame()

    # Prep
    df = df.copy()
    df["step_std"] = (df["step"] - df["step"].mean()) / max(1e-9, df["step"].std(ddof=0))
    df["temp_cat"] = df["temp"].astype(str)

    # Aggregates
    n_total = int(len(df))
    n1 = int((df["shift"] == 1).sum())
    n0 = n_total - n1
    k1 = int(df.loc[df["shift"] == 1, "correct"].sum()) if n1 > 0 else 0
    k0 = int(df.loc[df["shift"] == 0, "correct"].sum()) if n0 > 0 else 0
    share_shift = (n1 / n_total) if n_total else np.nan
    acc_shift = (k1 / n1) if n1 else np.nan
    delta_pp = (k1 / n1 - k0 / n0) * 100.0 if (n1 and n0) else np.nan

    rows = []

    # (a) Training stage
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = smf.glm("correct ~ C(problem_id) + step_std + shift", data=df, family=sm.families.Binomial())
            r = m.fit(cov_type="cluster", cov_kwds={"groups": df["problem_id"]}, maxiter=200)
        ame, lo, hi, p = _ame_from_fitted_glm(r, df)
        rows.append(dict(kind="training_stage", model=model_label, N=n_total,
                         share_shift=share_shift, acc_shift=acc_shift,
                         delta_pp=delta_pp, AME=ame, p_value=p))
    except Exception:
        rows.append(dict(kind="training_stage", model=model_label, N=n_total,
                         share_shift=share_shift, acc_shift=acc_shift,
                         delta_pp=delta_pp, AME=np.nan, p_value=np.nan))

    # (b) Temperature controls
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = smf.glm("correct ~ C(problem_id) + C(temp_cat) + shift", data=df, family=sm.families.Binomial())
            r = m.fit(cov_type="cluster", cov_kwds={"groups": df["problem_id"]}, maxiter=200)
        ame, lo, hi, p = _ame_from_fitted_glm(r, df)
        rows.append(dict(kind="temperature", model=model_label, N=n_total,
                         share_shift=share_shift, acc_shift=acc_shift,
                         delta_pp=delta_pp, AME=ame, p_value=p))
    except Exception:
        rows.append(dict(kind="temperature", model=model_label, N=n_total,
                         share_shift=share_shift, acc_shift=acc_shift,
                         delta_pp=delta_pp, AME=np.nan, p_value=np.nan))

    out = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, f"global_regressions__{slug_base}__{model_label.replace(' ','_')}.csv")
    out.to_csv(out_csv, index=False)
    print("[global-reg] saved:", out_csv)
    return out

# ---------- ANOVA/CMH + domain z-tests ----------

def run_anova_and_cmh(df_binned: pd.DataFrame, out_dir: str, slug: str) -> pd.DataFrame:
    rows = []
    tables = []
    kept_strata = []
    for (dom, b), sub in df_binned.groupby(["domain", "bin"], sort=False):
        n1 = int((sub["shift"] == 1).sum())
        n0 = int((sub["shift"] == 0).sum())
        if n1 == 0 or n0 == 0:
            continue
        k1 = int(sub.loc[sub["shift"] == 1, "correct"].sum())
        k0 = int(sub.loc[sub["shift"] == 0, "correct"].sum())
        t = np.array([[k1, n1 - k1],
                      [k0, n0 - k0]], dtype=int)
        tables.append(t)
        kept_strata.append((dom, int(b)))

    cmh_res = None
    if tables:
        st = StratifiedTable(tables)
        cmh = st.test_null_odds()
        cmh_stat = float(cmh.statistic); cmh_p = float(cmh.pvalue)
        or_pooled = float(st.oddsratio_pooled)
        or_lo, or_hi = [float(x) for x in st.oddsratio_pooled_confint()]
        cmh_res = dict(test="CMH", statistic=cmh_stat, pvalue=cmh_p,
                       pooled_or=or_pooled, pooled_or_lo=or_lo, pooled_or_hi=or_hi,
                       n_strata=len(tables))
        rows.append({"test":"CMH", "domain":"ALL", "strata":len(tables),
                     "stat":cmh_stat, "p":cmh_p,
                     "effect":"pooled OR", "est":or_pooled, "lo":or_lo, "hi":or_hi})

    # Keep only strata with both S=0 and S=1
    keep_idx = pd.Series(False, index=df_binned.index)
    if kept_strata:
        for (dom, b) in kept_strata:
            keep_idx |= ((df_binned["domain"] == dom) & (df_binned["bin"] == b))
    sub_all = df_binned.loc[keep_idx].copy()
    sub_all["stratum"] = sub_all["domain"].astype(str) + ":" + sub_all["bin"].astype(str)

    glm_res = None
    if not sub_all.empty and sub_all["shift"].nunique() == 2 and sub_all["correct"].nunique() == 2:
        # (Existing) LRT for shift controlling for strata
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m0 = smf.glm("correct ~ C(stratum)", data=sub_all, family=sm.families.Binomial()).fit()
            m1 = smf.glm("correct ~ C(stratum) + shift", data=sub_all, family=sm.families.Binomial()).fit()
        lr_stat = 2.0 * (m1.llf - m0.llf)
        lr_df = (len(m1.params) - len(m0.params))
        lr_p = float(chi2.sf(lr_stat, lr_df))
        if "shift" in m1.params.index:
            b = float(m1.params["shift"]); se = float(m1.bse["shift"])
            lo, hi = b - 1.96*se, b + 1.96*se
            or_hat, or_lo, or_hi = np.exp(b), np.exp(lo), np.exp(hi)
        else:
            or_hat = or_lo = or_hi = np.nan
        glm_res = dict(test="GLM_LRT", statistic=float(lr_stat), df=lr_df, pvalue=lr_p,
                       shift_or=or_hat, shift_or_lo=or_lo, shift_or_hi=or_hi)
        rows.append({"test":"GLM_LRT", "domain":"ALL", "strata":len(kept_strata),
                     "stat":float(lr_stat), "p":lr_p,
                     "effect":"shift OR (Wald CI)", "est":or_hat, "lo":or_lo, "hi":or_hi})

        # ===== NEW: "True ANOVA" — does adding strata improve fit over shift-only? =====
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Reduced: shift only (categorical so it's the same coding as you requested)
                mR = smf.glm("correct ~ C(shift)", data=sub_all, family=sm.families.Binomial()).fit()
                # Full: add stratum FE, still keep shift as categorical (per your spec)
                mF = smf.glm("correct ~ C(stratum) + C(shift)", data=sub_all, family=sm.families.Binomial()).fit()
            lr_stat2 = 2.0 * (mF.llf - mR.llf)
            lr_df2 = (len(mF.params) - len(mR.params))
            lr_p2 = float(chi2.sf(lr_stat2, lr_df2))
            rows.append({
                "test": "ANOVA_strata_vs_shift",
                "domain": "ALL",
                "strata": len(kept_strata),
                "stat": float(lr_stat2),
                "p": lr_p2,
                "effect": f"add C(stratum) vs C(shift) (Δdf={lr_df2})",
                "est": np.nan,
                "lo": np.nan,
                "hi": np.nan
            })
        except Exception:
            rows.append({
                "test": "ANOVA_strata_vs_shift",
                "domain": "ALL",
                "strata": len(kept_strata),
                "stat": np.nan,
                "p": np.nan,
                "effect": "add C(stratum) vs C(shift)",
                "est": np.nan, "lo": np.nan, "hi": np.nan
            })

    # Domain-wise two-proportion z-tests (unchanged)
    for dom, sub in df_binned.groupby("domain", sort=False):
        n1 = int((sub["shift"] == 1).sum())
        n0 = int((sub["shift"] == 0).sum())
        if n1 == 0 or n0 == 0:
            rows.append({"test":"Z_2prop", "domain":dom, "strata":sub["bin"].nunique(),
                         "stat":np.nan, "p":np.nan,
                         "effect":"p1 - p0 (pp)", "est":np.nan, "lo":np.nan, "hi":np.nan})
            continue
        k1 = int(sub.loc[sub["shift"] == 1, "correct"].sum())
        k0 = int(sub.loc[sub["shift"] == 0, "correct"].sum())
        z_stat, p_val = proportions_ztest([k1, k0], [n1, n0], alternative='two-sided', prop_var=False)
        diff_pp, lo_pp, hi_pp = newcombe_diff_ci(k1, n1, k0, n0)
        rows.append({"test":"Z_2prop", "domain":dom, "strata":sub["bin"].nunique(),
                     "stat":float(z_stat), "p":float(p_val),
                     "effect":"p1 - p0 (pp)", "est":diff_pp, "lo":lo_pp, "hi":hi_pp})

    # Pooled two-proportion z-test (unchanged)
    pooled = df_binned.copy()
    n1 = int((pooled["shift"] == 1).sum()); n0 = int((pooled["shift"] == 0).sum())
    if n1 > 0 and n0 > 0:
        k1 = int(pooled.loc[pooled["shift"] == 1, "correct"].sum())
        k0 = int(pooled.loc[pooled["shift"] == 0, "correct"].sum())
        z_stat, p_val = proportions_ztest([k1, k0], [n1, n0], alternative='two-sided', prop_var=False)
        diff_pp, lo_pp, hi_pp = newcombe_diff_ci(k1, n1, k0, n0)
        rows.append({"test":"Z_2prop", "domain":"ALL", "strata":df_binned["bin"].nunique()*df_binned["domain"].nunique(),
                     "stat":float(z_stat), "p":float(p_val),
                     "effect":"p1 - p0 (pp)", "est":diff_pp, "lo":lo_pp, "hi":hi_pp})

    # NOTE: Keep the original fixed column order for downstream scripts
    out = pd.DataFrame(rows, columns=["test","domain","strata","stat","p","effect","est","lo","hi"])
    out_path = os.path.join(out_dir, f"anova_cmh_summary__{slug}.csv")
    out.to_csv(out_path, index=False)
    print("\n[ANOVA/CMH] saved:", out_path)
    if cmh_res is not None:
        print(f"  CMH: chi2={cmh_res['statistic']:.3f}, p={cmh_res['pvalue']:.3g}, pooled OR={cmh_res['pooled_or']:.3f} "
              f"[{cmh_res['pooled_or_lo']:.3f}, {cmh_res['pooled_or_hi']:.3f}]  (n_strata={cmh_res['n_strata']})")
    if glm_res is not None:
        print(f"  GLM LRT: chi2={glm_res['statistic']:.3f} (df={glm_res['df']}), p={glm_res['pvalue']:.3g}; "
              f"shift OR={glm_res['shift_or']:.3f} [{glm_res['shift_or_lo']:.3f}, {glm_res['shift_or_hi']:.3f}]")
    # Print the NEW true-ANOVA line if present
    anova_row = out[(out["test"]=="ANOVA_strata_vs_shift") & (out["domain"]=="ALL")]
    if not anova_row.empty:
        r = anova_row.iloc[0]
        # Δdf is embedded in the 'effect' string
        print(f"  ANOVA (stratum FE vs shift-only): chi2={r['stat']:.3f}, p={r['p']:.3g}; {r['effect']}")
    return out


def print_anova_quick(anova_df: pd.DataFrame):
    if anova_df.empty:
        print("[ANOVA/CMH] No strata with both S=0 and S=1; skipping.")
        return
    print("\n====== Quick ANOVA/CMH Summary ======")
    cmh = anova_df[(anova_df["test"]=="CMH") & (anova_df["domain"]=="ALL")]
    if not cmh.empty:
        r = cmh.iloc[0]
        print(f"CMH (pooled across strata): chi2={r['stat']:.3f}, p={r['p']:.3g}; {r['effect']}={r['est']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]")

    glm = anova_df[(anova_df["test"]=="GLM_LRT") & (anova_df["domain"]=="ALL")]
    if not glm.empty:
        r = glm.iloc[0]
        print(f"GLM LRT (with stratum FE): chi2={r['stat']:.3f}, p={r['p']:.3g}; {r['effect']}={r['est']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]")

    # NEW quick line for the true ANOVA
    anova = anova_df[(anova_df["test"]=="ANOVA_strata_vs_shift") & (anova_df["domain"]=="ALL")]
    if not anova.empty:
        r = anova.iloc[0]
        print(f"ANOVA (stratum FE vs shift-only): chi2={r['stat']:.3f}, p={r['p']:.3g}; {r['effect']}")

    for dom in ["Crossword","Math","Carpark","ALL"]:
        rows = anova_df[(anova_df["test"]=="Z_2prop") & (anova_df["domain"]==dom)]
        if not rows.empty:
            r = rows.iloc[0]
            print(f"{dom:9s}  Δpp = {r['est']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}],  z={r['stat']:.3f}, p={r['p']:.3g}")
    print("=====================================\n")
    # inside run_anova_and_cmh after fitting mF
    print("N used in ANOVA:", int(mF.nobs))           # statsmodels view
    print("N check:", int(sub_all.shape[0]))          # dataframe view


# =================
# Plotting helpers
# =================

def _safe_yerr(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.vstack([np.nan_to_num(np.maximum(0,low)), np.nan_to_num(np.maximum(0,high))])

# =========
#   Runner
# =========

def run_for_model(metric: str, args, roots_for_model, model_key: str, model_label: str,
                  combined_mode: str, min_step: Optional[int], max_step: Optional[int]):

    frames=[]
    for T in sorted(roots_for_model):
        for dom in args.domains:
            if dom not in roots_for_model[T]:
                continue
            df=load_rows(roots_for_model[T][dom], args.split, dom,
                         entropy_source=metric, allow_fallback=args.allow_metric_fallback,
                         carpark_op=args.carpark_success_op, carpark_thr=args.carpark_soft_threshold,
                         gpt_mode=args.gpt_mode, verbose=args.verbose,
                         combined_mode=combined_mode,
                         min_step=min_step, max_step=max_step,
                         temp=T, model_label=model_label)
            if not df.empty: frames.append(df)
    if not frames:
        print(f"[warn] no rows for {model_label} / metric={metric}. Skipping.")
        return

    df=pd.concat(frames, ignore_index=True)

    # Build bins
    if args.equal_n_bins:
        df_binned, centers_by_dom = assign_bins_equal_count(
            df, bins=args.bins, scope=args.bin_scope,
            tie_break=args.tie_break, seed=args.random_seed
        )
        edges_global = np.arange(args.bins + 1, dtype=float)
    else:
        if args.fixed_bins:
            fixed_edges = _parse_fixed_edges(args.fixed_bins)
            edges_by_dom = {d: fixed_edges for d in df["domain"].unique()}
            df_binned, centers_by_dom = assign_bins_fixed(
                df, edges_by_dom, last_center_offset=args.last_bin_center_offset
            )
            edges_global = fixed_edges
        else:
            edges_by_dom=build_edges(df, args.bins, args.binning, args.bin_scope)
            df_binned, centers_by_dom=assign_bins(df, edges_by_dom)
            if args.bin_scope=="global":
                edges_global=list(edges_by_dom.values())[0]
            else:
                x=df["ent"].to_numpy()
                edges_global = (np.linspace(np.nanmin(x), np.nanmax(x), args.bins+1)
                                if args.binning=="uniform" else
                                np.quantile(x, np.linspace(0,1,args.bins+1)))

    # Aha vs No-Aha accuracy (+ odds) per bucket
    aha_rows=[]
    for (dom,b), sub in df_binned.groupby(["domain","bin"], sort=False):
        for grp,mask in [("no", sub["shift"]==0), ("yes", sub["shift"]==1)]:
            ss=sub[mask]; n=int(len(ss)); k=int(ss["correct"].sum())
            if n>0:
                acc=k/n; lo,hi=wilson_ci(k,n)
                def p2odds(p): p=float(np.clip(p,1e-9,1-1e-9)); return p/(1-p)
                odds=(k+0.5)/((n-k)+0.5); odds_lo=p2odds(lo); odds_hi=p2odds(hi)
            else:
                acc=lo=hi=odds=odds_lo=odds_hi=np.nan
            aha_rows.append({"model":model_label,"domain":dom,"bin":int(b),"grp":grp,"n":n,"k":k,
                             "acc":acc,"lo":lo,"hi":hi,
                             "odds":odds,"odds_lo":odds_lo,"odds_hi":odds_hi})
    aha_df=pd.DataFrame(aha_rows)

    # Effects per bucket
    eff_rows=[]
    for (dom,b), sub in df_binned.groupby(["domain","bin"], sort=False):
        n=int(len(sub)); n1=int((sub["shift"]==1).sum()); n0=n-n1
        k1=int(sub.loc[sub["shift"]==1,"correct"].sum()) if n1>0 else 0
        k0=int(sub.loc[sub["shift"]==0,"correct"].sum()) if n0>0 else 0
        raw_pp, raw_lo, raw_hi=newcombe_diff_ci(k1,n1,k0,n0)
        ame, lo, hi, p=glm_ame_bucket(sub[["problem_id","correct","shift"]], n_boot=args.n_boot)
        eff_rows.append({"model":model_label,"domain":dom,"bin":int(b),"n":n,"n_shift":n1,"n_noshift":n0,
                         "raw_pp":raw_pp,"raw_lo":raw_lo,"raw_hi":raw_hi,
                         "ame_pp":(ame*100.0 if np.isfinite(ame) else np.nan),
                         "ame_lo_pp":(lo*100.0 if np.isfinite(lo) else np.nan),
                         "ame_hi_pp":(hi*100.0 if np.isfinite(hi) else np.nan),
                         "p_shift":p})
    eff_df=pd.DataFrame(eff_rows)

    # Combined (centers from edges_global)
    def aggregate_combined(aha_df: pd.DataFrame, edges_global: np.ndarray)->Tuple[pd.DataFrame,np.ndarray]:
        bins_idx=np.arange(len(edges_global)-1)
        rows=[]
        for grp in ["no","yes"]:
            g=aha_df[aha_df["grp"]==grp].groupby("bin", as_index=False).agg({"n":"sum","k":"sum"})
            g=g.set_index("bin").reindex(bins_idx).fillna(0)
            n=g["n"].to_numpy(dtype=float); k=g["k"].to_numpy(dtype=float)
            acc=np.divide(k,n, out=np.full_like(n,np.nan), where=n>0)
            lo=np.empty_like(acc); hi=np.empty_like(acc)
            for i,(ki,ni) in enumerate(zip(k,n)):
                if ni>0: 
                    l,h=wilson_ci(int(ki), int(ni))
                else:
                    l=h=np.nan
                lo[i]=l; hi[i]=h
            tmp=pd.DataFrame({"bin":bins_idx,"grp":grp,"n":n,"k":k,"acc":acc,"lo":lo,"hi":hi})
            rows.append(tmp.reset_index(drop=True))
        comb=pd.concat(rows, ignore_index=True)
        centers=0.5*(edges_global[:-1]+edges_global[1:])
        return comb, centers

    aha_combined, centers_combined = aggregate_combined(aha_df, edges_global)

    # Save + plot
    out_dir=args.out_dir or os.path.join(args.scan_root,"temperature_effects")
    os.makedirs(out_dir, exist_ok=True)
    slug_metric = metric if metric != "answer_plus_think" else f"answerPlusThink_{combined_mode}"
    bound_slug = f"stepMax{max_step}" if max_step is not None else "stepAll"
    bin_slug = ("equalN" if args.equal_n_bins else
                ("fixedbins" if args.fixed_bins else f"{args.binning}_{args.bin_scope}_{args.bins}bins"))
    slug=f"{model_label.replace(' ','_')}_allT_{slug_metric}_{bin_slug}_{bound_slug}"

    aha_csv      = os.path.join(out_dir, f"aha_acc_buckets__{slug}.csv")
    odds_csv     = os.path.join(out_dir, f"aha_odds_buckets__{slug}.csv")
    eff_csv      = os.path.join(out_dir, f"effects_buckets__{slug}.csv")
    aha_comb_csv = os.path.join(out_dir, f"aha_acc_buckets__{slug}_COMBINED.csv")
    aha_df.to_csv(aha_csv, index=False); aha_df.to_csv(odds_csv, index=False)
    eff_df.to_csv(eff_csv, index=False); aha_combined.to_csv(aha_comb_csv, index=False)

    # --- Plots (reuse your existing functions; omitted here for brevity) ---
    # plot_aha_hist(aha_df, centers_by_dom, None, os.path.join(out_dir, f"aha_acc_buckets__{slug}.png"))
    # plot_aha_ratio_hist(aha_df, centers_by_dom, None, os.path.join(out_dir, f"aha_odds_buckets__{slug}.png"))
    # plot_aha_hist_combined(aha_combined, 0.5*(edges_global[:-1]+edges_global[1:]),
    #                        os.path.join(out_dir, f"aha_acc_buckets__{slug}_COMBINED.png"))
    # plot_effects(eff_df, centers_by_dom, os.path.join(out_dir, f"effects_buckets__{slug}.png"))

    # --- ANOVA/CMH ---
    if not args.no_anova:
        anova_df = run_anova_and_cmh(df_binned=df_binned, out_dir=out_dir, slug=slug)
        print_anova_quick(anova_df)

    # --- Global regressions (per model) ---
    reg_df = run_model_regressions(df, out_dir=out_dir, model_label=model_label, slug_base=slug)

    print(f"\n[{model_label} | metric={metric}, steps ≤ {max_step if max_step is not None else '∞'}] Saved:")
    print("  Aha vs No-Aha accuracy (per-domain) CSV :", aha_csv)
    print("  Aha vs No-Aha odds     (per-domain) CSV :", odds_csv)
    print("  Aha vs No-Aha accuracy (COMBINED)  CSV :", aha_comb_csv)
    print("  Effects per bucket CSV               :", eff_csv)
    print("  Global regressions CSV               :", os.path.join(args.out_dir or os.path.join(args.scan_root,'temperature_effects'),
                                                                  f'global_regressions__{slug}.csv').replace('//','/'))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--scan_root", type=str, default="results")
    ap.add_argument("--temps", nargs="+", type=float, default=[0.0,0.05,0.3,0.7])
    ap.add_argument("--low_alias", type=float, default=0.3)
    ap.add_argument("--split", type=str, default="test")

    # Models/domains
    ap.add_argument("--models", nargs="+", default=["qwen7b","llama8b"],
                    choices=["qwen7b","llama8b"], help="Which models to include.")
    ap.add_argument("--domains", nargs="+", default=["Math"],
                    choices=["Math","Crossword","Carpark"], help="Domains to include (default Math).")

    # Metrics
    ap.add_argument("--metric", choices=["entropy_answer","entropy","entropy_think","answer_plus_think"],
                    default=None, help="Single metric to run (overridden by --metrics).")
    ap.add_argument("--metrics", nargs="+",
                    choices=["entropy_answer","entropy","entropy_think","answer_plus_think"],
                    help="Run these metrics in one go; overrides --metric.")
    ap.add_argument("--combined_mode", choices=["sum","mean"], default="sum",
                    help="For answer_plus_think: sum or mean of (answer, think).")

    # Step bounds (max is hard-capped to 1000)
    ap.add_argument("--min_step", type=int, default=None, help="Lower bound on step (inclusive).")
    ap.add_argument("--max_step", type=int, default=1000, help="Upper bound on step (inclusive; hard cap = 1000).")

    # Fixed bins override
    ap.add_argument("--fixed_bins", type=str, default=None,
                    help="Comma-separated entropy edges, e.g. '0,0.5,1,2,inf'. Overrides --bins/--binning/--bin_scope.")
    ap.add_argument("--last_bin_center_offset", type=float, default=0.5,
                    help="Center offset used to place the open-ended last bin (if top edge is inf).")

    # Equal-count bins (rank-based)
    ap.add_argument("--equal_n_bins", action="store_true",
                    help="Force equal-count bins (e.g., quartiles if --bins 4). Ignores --fixed_bins and numeric quantile edges.")
    ap.add_argument("--tie_break", choices=["stable","random"], default="stable",
                    help="How to break ties when many entropies are identical.")
    ap.add_argument("--random_seed", type=int, default=0,
                    help="Seed used when --tie_break=random")

    ap.add_argument("--allow_metric_fallback", action="store_true")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--binning", choices=["uniform","quantile"], default="uniform")
    ap.add_argument("--bin_scope", choices=["global","domain"], default="global")
    ap.add_argument("--carpark_success_op", choices=["gt","ge","eq"], default="ge")
    ap.add_argument("--carpark_soft_threshold", type=float, default=0.1)
    ap.add_argument("--gpt_mode", choices=["canonical","broad"], default="canonical")
    ap.add_argument("--n_boot", type=int, default=300)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no_anova", action="store_true", help="Skip CMH/GLM and two-proportion tests.")

    args=ap.parse_args()

    HARD_MAX_STEP = 1000
    eff_max = HARD_MAX_STEP if args.max_step is None else min(args.max_step, HARD_MAX_STEP)
    if args.max_step is None or args.max_step > HARD_MAX_STEP:
        print(f"[info] Capping max_step to {eff_max} (hard cap = {HARD_MAX_STEP}).")

    mapping = discover_roots_7b8b(args.scan_root, args.temps, args.low_alias,
                                  want_models=args.models, want_domains=args.domains,
                                  verbose=args.verbose)
    if not mapping:
        raise SystemExit("[error] no 7B/8B temp dirs found under scan_root.")

    metrics_to_run = args.metrics if args.metrics else ([args.metric] if args.metric else
                     ["entropy_answer","entropy","entropy_think","answer_plus_think"])

    for mk in args.models:
        model_label = MODEL_LABELS.get(mk, mk)
        roots_for_model = mapping.get(mk, {})
        if not roots_for_model:
            print(f"[warn] No runs found for {model_label}.")
            continue
        for metric in metrics_to_run:
            run_for_model(metric, args, roots_for_model, mk, model_label,
                          args.combined_mode, args.min_step, eff_max)

if __name__ == "__main__":
    main()