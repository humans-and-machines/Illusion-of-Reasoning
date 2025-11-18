#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Uncertainty buckets: Aha vs No-Aha (accuracy, odds, effects)
Qwen-1.5B; all temperatures & steps

Multi-metric runner:
- Single metric: use --metric entropy_answer|entropy|entropy_think
- Multiple: use --metrics entropy_answer entropy entropy_think answer_plus_think
- Default (no --metric/--metrics): runs all four:
    entropy_answer, entropy, entropy_think, answer_plus_think
- For 'answer_plus_think', control combine via --combined_mode sum|mean (default: sum)

STEP FILTERING
--------------
Only steps <= 1000 are loaded by default.
Use --min_step to set a lower bound. --max_step is capped at 1000.

CUSTOM BINS
-----------
Use --fixed_bins "0,0.5,1,2,inf" to set edges for bins [0,0.5), [0.5,1), [1,2), [2,∞).
Half-open on the right (except the last, open-ended bin). For plotting, the last
open-ended bin is placed at a finite x center via --last_bin_center_offset.

EQUAL-COUNT BINS (QUARTILES, QUINTILES, ...)
--------------------------------------------
Set --equal_n_bins with --bins K to force rank-based, equal-count bins (e.g., quartiles for K=4).
Works with --bin_scope global (equal N overall) or --bin_scope domain (equal N within each domain).
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

# =========================
# Discovery (folders/files)
# =========================

DIR_RE = re.compile(
    r"^GRPO-1\.5B-(?P<dom>xword|math|carpark)-(?:temp-(?P<t>low|[0-9]+(?:\.[0-9]+)?)|low-temp)$",
    re.IGNORECASE,
)

def _normT(tok: Optional[str], low_alias: float) -> Optional[float]:
    if tok is None: return None
    s = tok.lower()
    if s == "low": return float(low_alias)
    try: return float(s)
    except Exception: return None

def discover_roots(scan_root: str, temps: List[float], low_alias: float, verbose: bool):
    temps_set = {float(x) for x in temps}
    mapping: Dict[float, Dict[str, str]] = {}
    for entry in os.listdir(scan_root):
        full = os.path.join(scan_root, entry)
        if not os.path.isdir(full): continue
        low = entry.lower()
        if "compare-1shot" in low or "1shot" in low: continue
        if "7b" in low or "8b" in low: continue
        m = DIR_RE.match(entry)
        if not m: continue
        dom = {"xword":"Crossword","math":"Math","carpark":"Carpark"}[m.group("dom").lower()]
        t = _normT(m.group("t"), low_alias)
        if t is None and "low-temp" in low: t = float(low_alias)
        if t is None or float(t) not in temps_set: continue
        mapping.setdefault(float(t), {})
        prev = mapping[float(t)].get(dom)
        if prev is None or len(full) > len(prev):
            mapping[float(t)][dom] = full
    if verbose:
        print("[info] discovered roots:")
        for T in sorted(mapping):
            print(f"  T={T}:")
            for d,p in mapping[T].items(): print(f"    {d:9s} -> {p}")
    return mapping

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
              max_step: Optional[int] = None) -> pd.DataFrame:
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
                    "domain":domain,
                    "step":int(step),
                    "problem_id":pid,
                    "ent":ent,
                    "correct":corr,
                    "shift":shift
                })
    df=pd.DataFrame(recs)
    if verbose: print(f"[info] loaded rows for {domain} (steps", end="")
    if verbose: print(f" ≤ {max_step}" if max_step is not None else "", end="")
    if verbose: print(f"): {len(df)}")
    return df

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
    """
    Half-open bins [left,right), last bin open-ended if right==inf.
    For plotting, place the last bin center at (last_finite_edge + last_center_offset).
    """
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

# ---------- NEW: Equal-count binning (rank-based) ----------

def _assign_equal_bins_1d(n: int, B: int) -> np.ndarray:
    """
    Return an array of length n with values in {0,...,B-1} such that
    bins are as equal in size as possible (sizes differ by at most 1).
    """
    idx = np.arange(n)
    parts = np.array_split(idx, B)  # nearly-equal splits
    out = np.empty(n, dtype=int)
    for b, part in enumerate(parts):
        out[part] = b
    return out

def assign_bins_equal_count(df: pd.DataFrame, bins: int, scope: str,
                            tie_break: str = "stable", seed: int = 0) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Equal-count binning by rank (e.g., quartiles if bins=4).
    scope='global' => equal N overall; scope='domain' => equal N within each domain.
    Returns (df_with_bin, centers_by_dom) where centers are per-domain bin medians of entropy.
    """
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
        # per-domain centers = median entropy within each (domain,bin)
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
    """
    AME in probability units for: correct ~ C(problem_id) + shift.
    Robust to perfect separation: fallback to raw diff (prob units) + Newcombe CI.
    """
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

# =================
# Plotting helpers
# =================

def _safe_yerr(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.vstack([np.nan_to_num(np.maximum(0,low)), np.nan_to_num(np.maximum(0,high))])

def plot_aha_hist(aha_df: pd.DataFrame,
                  centers_by_dom: Dict[str, np.ndarray],
                  edges_by_dom: Optional[Dict[str, np.ndarray]],
                  out_png: str):
    colors = {"no": "#1f77b4", "yes": "#ff7f0e"}
    domains = ["Crossword", "Math", "Carpark"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharey=True)

    for ax, dom in zip(axes, domains):
        sub = aha_df[aha_df["domain"] == dom].copy()
        ax.set_title(dom)
        ax.set_xlabel("Uncertainty bin")
        ax.grid(alpha=0.2, linestyle="--", axis="y")
        if sub.empty:
            continue

        centers = centers_by_dom.get(dom, None)
        e = edges_by_dom.get(dom, None) if isinstance(edges_by_dom, dict) else None

        # Choose x positions and bar width
        if centers is not None and e is not None and len(e) >= 2:
            # numeric scale (has edges) – use entropy-like x positions
            pos = centers
            width = (e[1] - e[0]) * 0.38
            xticks = pos
            xticklabels = [f"{c:.2f}" for c in centers]
        else:
            # equal-N fallback – plot by bin index (Q1..QK)
            B = int(sub["bin"].max()) + 1 if "bin" in sub.columns else len(centers) if centers is not None else 4
            pos = np.arange(B)
            width = 0.35
            xticks = pos
            xticklabels = [f"Q{i+1}" for i in range(B)]

        bins_idx = np.arange(len(xticks))
        # Aggregate to bins for each group
        s_no = (sub[sub["grp"] == "no"]
                .groupby("bin", as_index=True)[["acc", "lo", "hi"]]
                .mean().reindex(bins_idx))
        s_yes = (sub[sub["grp"] == "yes"]
                 .groupby("bin", as_index=True)[["acc", "lo", "hi"]]
                 .mean().reindex(bins_idx))

        # x offsets for the two bars
        x_no = pos - width * 0.55
        x_yes = pos + width * 0.55

        for xpos, s, grp, lab in [(x_no, s_no, "no", "No Aha"),
                                  (x_yes, s_yes, "yes", "With Aha")]:
            y = (s["acc"] * 100.0).to_numpy()
            lo = (s["lo"] * 100.0).to_numpy()
            hi = (s["hi"] * 100.0).to_numpy()
            err = np.vstack([np.nan_to_num(np.maximum(0, y - lo)),
                             np.nan_to_num(np.maximum(0, hi - y))])
            ax.bar(xpos, np.nan_to_num(y), width=width, color=colors[grp], alpha=0.85,
                   edgecolor="none", label=lab)
            ax.errorbar(xpos, np.nan_to_num(y), yerr=err, fmt="none", ecolor="black",
                        elinewidth=1, capsize=2, alpha=0.7)

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)

    axes[0].set_ylabel("Accuracy (%)")
    # legend once, if any handles
    handles, labels = axes[0].get_legend_handles_labels()
    if labels:
        axes[0].legend(handles[:2], labels[:2], frameon=False, loc="upper left")
    fig.suptitle("Accuracy by Uncertainty Bucket — Aha vs No-Aha (steps ≤ 1000)", y=1.03, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_aha_ratio_hist(aha_df: pd.DataFrame,
                        centers_by_dom: Dict[str, np.ndarray],
                        edges_by_dom: Optional[Dict[str, np.ndarray]],
                        out_png: str):
    colors = {"no": "#1f77b4", "yes": "#ff7f0e"}
    domains = ["Crossword", "Math", "Carpark"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharey=False)

    for ax, dom in zip(axes, domains):
        sub = aha_df[aha_df["domain"] == dom].copy()
        ax.set_title(dom)
        ax.set_xlabel("Uncertainty bin")
        ax.grid(alpha=0.2, linestyle="--", axis="y")
        if sub.empty:
            continue

        centers = centers_by_dom.get(dom, None)
        e = edges_by_dom.get(dom, None) if isinstance(edges_by_dom, dict) else None

        if centers is not None and e is not None and len(e) >= 2:
            pos = centers
            width = (e[1] - e[0]) * 0.38
            xticks = pos
            xticklabels = [f"{c:.2f}" for c in centers]
        else:
            B = int(sub["bin"].max()) + 1 if "bin" in sub.columns else len(centers) if centers is not None else 4
            pos = np.arange(B)
            width = 0.35
            xticks = pos
            xticklabels = [f"Q{i+1}" for i in range(B)]

        bins_idx = np.arange(len(xticks))
        s_no = (sub[sub["grp"] == "no"]
                .groupby("bin", as_index=True)[["odds", "odds_lo", "odds_hi"]]
                .mean().reindex(bins_idx))
        s_yes = (sub[sub["grp"] == "yes"]
                 .groupby("bin", as_index=True)[["odds", "odds_lo", "odds_hi"]]
                 .mean().reindex(bins_idx))

        x_no = pos - width * 0.55
        x_yes = pos + width * 0.55

        for xpos, s, grp, lab in [(x_no, s_no, "no", "No Aha"),
                                  (x_yes, s_yes, "yes", "With Aha")]:
            y = s["odds"].to_numpy()
            lo = s["odds_lo"].to_numpy()
            hi = s["odds_hi"].to_numpy()
            err = np.vstack([np.nan_to_num(np.maximum(0, y - lo)),
                             np.nan_to_num(np.maximum(0, hi - y))])
            ax.bar(xpos, np.nan_to_num(y), width=width, color=colors[grp], alpha=0.85,
                   edgecolor="none", label=lab)
            ax.errorbar(xpos, np.nan_to_num(y), yerr=err, fmt="none", ecolor="black",
                        elinewidth=1, capsize=2, alpha=0.7)

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_yscale("log")
        ax.set_ylabel("Accuracy odds (correct / incorrect)")

    handles, labels = axes[0].get_legend_handles_labels()
    if labels:
        axes[0].legend(handles[:2], labels[:2], frameon=False, loc="upper left")
    fig.suptitle("Accuracy odds by Uncertainty Bucket — Aha vs No-Aha (steps ≤ 1000)", y=1.03, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_aha_hist_combined(aha_df: pd.DataFrame, centers: np.ndarray, out_png: str):
    colors={"no":"#1f77b4","yes":"#ff7f0e"}
    fig,ax=plt.subplots(figsize=(10.0,4.2))
    width=(centers[1]-centers[0])*0.38 if len(centers)>1 else 0.15
    x_no = centers - width*0.55; x_yes = centers + width*0.55
    bins_idx=np.arange(len(centers))
    s_no = aha_df[aha_df["grp"]=="no"].set_index("bin").reindex(bins_idx)
    s_yes= aha_df[aha_df["grp"]=="yes"].set_index("bin").reindex(bins_idx)
    for xpos,s,grp,lab in [(x_no,s_no,"no","No Aha"),(x_yes,s_yes,"yes","With Aha")]:
        y=(s["acc"]*100.0).to_numpy(); lo=(s["lo"]*100.0).to_numpy(); hi=(s["hi"]*100.0).to_numpy()
        err=_safe_yerr(y-lo, hi-y)
        ax.bar(xpos, np.nan_to_num(y), width=width, color=colors[grp], alpha=0.85,
               edgecolor="none", label=lab)
        ax.errorbar(xpos, np.nan_to_num(y), yerr=err, fmt="none", ecolor="black",
                    elinewidth=1, capsize=2, alpha=0.7)
    ax.set_xticks(centers); ax.set_xticklabels([f"{c:.2f}" for c in centers])
    ax.set_xlabel("Mean token entropy (nats)"); ax.set_ylabel("Accuracy (%)")
    ax.grid(alpha=0.2, linestyle="--", axis="y")
    ax.legend(frameon=False, loc="upper right")
    fig.suptitle("Accuracy by Uncertainty Bucket — Aha vs No-Aha (Combined; steps ≤ 1000)", y=1.03, fontsize=12)
    fig.tight_layout(); fig.savefig(out_png, dpi=200, bbox_inches="tight"); fig.savefig(out_png.replace(".png",".pdf"), bbox_inches="tight"); plt.close(fig)

def plot_effects(eff_df: pd.DataFrame, centers_by_dom: Dict[str,np.ndarray], out_png: str):
    colors={"Crossword":"#1f77b4","Math":"#2ca02c","Carpark":"#d62728"}
    fig,ax=plt.subplots(figsize=(8.6,4.4))
    for dom,sub in eff_df.groupby("domain", sort=False):
        sub=sub.sort_values("bin")
        x=centers_by_dom[dom][sub["bin"].to_numpy()]
        # raw
        raw_low = np.nan_to_num(sub["raw_pp"].to_numpy()-sub["raw_lo"].to_numpy())
        raw_high= np.nan_to_num(sub["raw_hi"].to_numpy()-sub["raw_pp"].to_numpy())
        ax.errorbar(x, sub["raw_pp"], yerr=_safe_yerr(raw_low, raw_high),
                    fmt="o", mfc="none", mec=colors[dom], ecolor=colors[dom],
                    color=colors[dom], capsize=2, alpha=0.9, label=f"{dom} (raw)")
        ax.plot(x, sub["raw_pp"], "-", color=colors[dom], lw=2, alpha=0.85)
        # AME
        ame_low = np.nan_to_num(sub["ame_pp"].to_numpy()-sub["ame_lo_pp"].to_numpy())
        ame_high= np.nan_to_num(sub["ame_hi_pp"].to_numpy()-sub["ame_pp"].to_numpy())
        ax.errorbar(x, sub["ame_pp"], yerr=_safe_yerr(ame_low, ame_high),
                    fmt="s", mfc=colors[dom], mec=colors[dom], ecolor=colors[dom],
                    capsize=2, alpha=0.9, label=f"{dom} (AME)")
        ax.plot(x, sub["ame_pp"], "--", color=colors[dom], lw=2, alpha=0.85)
    ax.axhline(0, color="k", lw=1, alpha=0.35)
    ax.set_xlabel("Mean token entropy (nats)")
    ax.set_ylabel("Effect of Aha on accuracy (pp)")
    ax.set_title("Effect vs Uncertainty — Δpp and AME per bucket (steps ≤ 1000)")
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout(); fig.savefig(out_png, dpi=200, bbox_inches="tight"); fig.savefig(out_png.replace(".png",".pdf"), bbox_inches="tight"); plt.close(fig)

# =================
# Combined helpers
# =================

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
            if ni>0: l,h=wilson_ci(int(ki), int(ni))
            else:    l=h=np.nan
        # ... (rest unchanged)
        tmp=pd.DataFrame({"bin":bins_idx,"grp":grp,"n":n,"k":k,"acc":acc,"lo":lo,"hi":hi})
        rows.append(tmp.reset_index(drop=True))
    comb=pd.concat(rows, ignore_index=True)
    centers=0.5*(edges_global[:-1]+edges_global[1:])
    return comb, centers

# =========
#   Main
# =========

def run_for_metric(metric: str, args, roots, combined_mode: str, min_step: Optional[int], max_step: Optional[int]):
    # Load pass-1 rows across temps/steps
    frames=[]
    for T in sorted(roots):
        for dom in ("Crossword","Math","Carpark"):
            if dom not in roots[T]: continue
            df=load_rows(roots[T][dom], args.split, dom,
                         entropy_source=metric, allow_fallback=args.allow_metric_fallback,
                         carpark_op=args.carpark_success_op, carpark_thr=args.carpark_soft_threshold,
                         gpt_mode=args.gpt_mode, verbose=args.verbose,
                         combined_mode=combined_mode,
                         min_step=min_step, max_step=max_step)
            if not df.empty: frames.append(df)
    if not frames:
        raise SystemExit(f"[error] no rows loaded for metric '{metric}' within step bounds.")
    df=pd.concat(frames, ignore_index=True)

    # Build edges & assign bins
    if args.equal_n_bins:
        # Force equal-count bins; ignore fixed/numeric edges
        df_binned, centers_by_dom = assign_bins_equal_count(
            df, bins=args.bins, scope=args.bin_scope,
            tie_break=args.tie_break, seed=args.random_seed
        )
        edges_global = np.arange(args.bins + 1, dtype=float)  # dummy, for combined plotting
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

    # Aha vs No-Aha accuracy (+ odds)
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
            aha_rows.append({"domain":dom,"bin":int(b),"grp":grp,"n":n,"k":k,
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
        eff_rows.append({"domain":dom,"bin":int(b),"n":n,"n_shift":n1,"n_noshift":n0,
                         "raw_pp":raw_pp,"raw_lo":raw_lo,"raw_hi":raw_hi,
                         "ame_pp":(ame*100.0 if np.isfinite(ame) else np.nan),
                         "ame_lo_pp":(lo*100.0 if np.isfinite(lo) else np.nan),
                         "ame_hi_pp":(hi*100.0 if np.isfinite(hi) else np.nan),
                         "p_shift":p})
    eff_df=pd.DataFrame(eff_rows)

    # Combined (centers come from edges_global)
    aha_combined, centers_combined = aggregate_combined(aha_df, edges_global)

    # Save + plot
    out_dir=args.out_dir or os.path.join(args.scan_root,"temperature_effects")
    os.makedirs(out_dir, exist_ok=True)
    slug_metric = metric if metric != "answer_plus_think" else f"answerPlusThink_{combined_mode}"
    bound_slug = f"stepMax{max_step}" if max_step is not None else "stepAll"
    bin_slug = ("equalN" if args.equal_n_bins else
                ("fixedbins" if args.fixed_bins else f"{args.binning}_{args.bin_scope}_{args.bins}bins"))
    slug=f"Qwen1p5B_allT_{slug_metric}_{bin_slug}_{bound_slug}"

    aha_csv      = os.path.join(out_dir, f"aha_acc_buckets__{slug}.csv")
    odds_csv     = os.path.join(out_dir, f"aha_odds_buckets__{slug}.csv")
    eff_csv      = os.path.join(out_dir, f"effects_buckets__{slug}.csv")
    aha_comb_csv = os.path.join(out_dir, f"aha_acc_buckets__{slug}_COMBINED.csv")
    aha_df.to_csv(aha_csv, index=False); aha_df.to_csv(odds_csv, index=False)
    eff_df.to_csv(eff_csv, index=False); aha_combined.to_csv(aha_comb_csv, index=False)

    figA = os.path.join(out_dir, f"aha_acc_buckets__{slug}.png")
    plot_aha_hist(aha_df, centers_by_dom, None, figA)

    figA_ratio = os.path.join(out_dir, f"aha_odds_buckets__{slug}.png")
    plot_aha_ratio_hist(aha_df, centers_by_dom, None, figA_ratio)
    figA_comb = os.path.join(out_dir, f"aha_acc_buckets__{slug}_COMBINED.png")
    plot_aha_hist_combined(aha_combined, 0.5*(edges_global[:-1]+edges_global[1:]), figA_comb)

    figB = os.path.join(out_dir, f"effects_buckets__{slug}.png")
    plot_aha_hist_path = figA  # just to avoid linter warnings about unused vars
    plot_effects(eff_df, centers_by_dom, figB)

    print(f"\n[metric={metric}, steps ≤ {max_step if max_step is not None else '∞'}] Saved:")
    print("  Aha vs No-Aha accuracy (per-domain) CSV :", aha_csv)
    print("  Aha vs No-Aha accuracy (per-domain) FIG :", figA, " (+ PDF)")
    print("  Aha vs No-Aha odds     (per-domain) CSV :", odds_csv)
    print("  Aha vs No-Aha odds     (per-domain) FIG :", figA_ratio, " (+ PDF)")
    print("  Aha vs No-Aha accuracy (COMBINED)  CSV :", aha_comb_csv)
    print("  Aha vs No-Aha accuracy (COMBINED)  FIG :", figA_comb, " (+ PDF)")
    print("  Effects per bucket CSV               :", eff_csv)
    print("  Effects per bucket FIG               :", figB, " (+ PDF)")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--scan_root", type=str, default="results")
    ap.add_argument("--temps", nargs="+", type=float, default=[0.0,0.05,0.3,0.7])
    ap.add_argument("--low_alias", type=float, default=0.3)
    ap.add_argument("--split", type=str, default="test")

    # Single-metric (back-compat) OR multi-metric
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
    args=ap.parse_args()

    # Enforce hard cap at 1000
    HARD_MAX_STEP = 1000
    eff_max = HARD_MAX_STEP if args.max_step is None else min(args.max_step, HARD_MAX_STEP)
    if args.max_step is None or args.max_step > HARD_MAX_STEP:
        print(f"[info] Capping max_step to {eff_max} (hard cap = {HARD_MAX_STEP}).")

    roots=discover_roots(args.scan_root, args.temps, args.low_alias, args.verbose)
    if not roots: raise SystemExit("[error] no 1.5B temp dirs found.")

    # Resolve which metrics to run
    if args.metrics:
        metrics_to_run = args.metrics
    elif args.metric:
        metrics_to_run = [args.metric]
    else:
        metrics_to_run = ["entropy_answer","entropy","entropy_think","answer_plus_think"]

    for metric in metrics_to_run:
        run_for_metric(metric, args, roots, args.combined_mode, args.min_step, eff_max)

if __name__ == "__main__":
    main()
