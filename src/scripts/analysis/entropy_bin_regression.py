#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
entropy_bin_regression.py
─────────────────────────
Builds a per-sample dataset with:
  domain, problem, step, sample, shift_at_1, correct_at_1, entropy_at_1,
  entropy_bin_at_1, entropy_bin_label, correct_at_2

Per-domain regressions (run twice per domain):
  correct_at_2 ~ correct_at_1 + C(entropy_bin_label) + C(problem)
(GLM Binomial logit). Cluster-robust SEs by problem; fallback to HC1 if needed.

Binning options:
  • --fixed_bins "0,0.75,1.25,2,inf"   (takes precedence)
  • --binning {uniform,quantile} with --bins K
  • --equal_n_bins                     (rank-based equal-count bins; exact quartiles when --bins 4)
      • --bin_scope {global,domain}    (equal N overall vs within each domain)
      • --tie_break {stable,random}    (stable = deterministic; random = tiny jitter)
      • --random_seed 42               (for random tie-break)

Entropy modes at pass-1 (controls entropy_at_1):
  • sum       : entropy_think + entropy_answer (preferred when parts exist; else fallback to 'entropy')
  • think     : entropy_think (fallback to 'entropy')
  • answer    : entropy_answer (fallback to 'entropy')
  • combined  : 'entropy' field as-is (fallback to avg(parts) if helpful)

Outputs (per domain & per entropy mode):
  rows__<slug>__<domain>__<mode>.csv
  model_{none|false}__<slug>__<domain>__<mode>.txt
  bin_contrasts__{none|false}__<slug>__<domain>__<mode>.csv
  bin_contrasts__<slug>__<domain>__<mode>.{png,pdf}
"""

import os, re, json, gzip, argparse, sys
from typing import Optional, List, Dict, Any, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
except Exception as e:
    sys.exit("statsmodels is required: pip install statsmodels\n" + str(e))

# ------------------------------- Path & step helpers -------------------------------

STEP_DIR_PAT = re.compile(r"(?:^|/)(step[-_]?\d{1,5})(?:/|$)", re.I)
STEP_FILE_PAT = re.compile(r"^(?:step|global[_-]?step|checkpoint)[-_]?\d{1,5}", re.I)
STEP_PATS = [
    re.compile(r"step\s*[-_]?(\d+)", re.I),
    re.compile(r"global[_-]?step\s*[-_]?(\d+)", re.I),
    re.compile(r"checkpoint\s*[-_]?(\d+)", re.I),
    re.compile(r"/(\d{2,5})(?=/|$)")
]
SKIP_DIR_DEFAULT = {"compare-1shot", "1shot", "hf_cache", "__pycache__"}

def nat_step_from_path(path: str) -> Optional[int]:
    s = str(path)
    for pat in STEP_PATS:
        m = pat.search(s)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
    return None

def scan_files(root: str, split_substr: Optional[str], skip_substrings: set) -> List[str]:
    out = []
    for dp, _, fns in os.walk(root):
        dp_norm = dp.replace("\\", "/")
        if any(s.lower() in dp_norm.lower() for s in skip_substrings):
            continue
        dir_has_step = STEP_DIR_PAT.search(dp_norm) is not None
        for fn in fns:
            low = fn.lower()
            if not (low.endswith(".jsonl") or low.endswith(".jsonl.gz") or low.endswith(".json")):
                continue
            if split_substr and split_substr not in fn:
                continue
            file_has_step = STEP_FILE_PAT.search(fn) is not None
            if not (dir_has_step or file_has_step):
                continue
            out.append(os.path.join(dp, fn))
    out.sort()
    return out

def iter_records_from_file(path: str) -> Iterable[Dict[str, Any]]:
    opener = gzip.open if path.endswith(".jsonl.gz") else open
    mode = "rt"
    try:
        with opener(path, mode, encoding="utf-8") as f:
            if path.endswith(".json"):
                txt = f.read().strip()
                if not txt:
                    return
                try:
                    obj = json.loads(txt)
                    if isinstance(obj, list):
                        for r in obj:
                            if isinstance(r, dict): yield r
                    elif isinstance(obj, dict):
                        yield obj
                except Exception:
                    for line in txt.splitlines():
                        s = line.strip()
                        if not s: continue
                        try:
                            r = json.loads(s)
                            if isinstance(r, dict): yield r
                        except Exception:
                            continue
            else:
                for line in f:
                    s = line.strip()
                    if not s: continue
                    try:
                        r = json.loads(s)
                        if isinstance(r, dict): yield r
                    except Exception:
                        continue
    except Exception:
        return

# ------------------------------- Coercion utils -------------------------------

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

def coerce_float(x) -> Optional[float]:
    if x is None: return None
    try: return float(x)
    except Exception: return None

def both_get(p1: Dict[str, Any], rec: Dict[str, Any], key: str, default=None):
    v = p1.get(key, None)
    return v if v is not None else rec.get(key, default)

# ------------------------------- Correctness helpers -------------------------------

CORRECT_KEYS = {
    "is_correct","is_correct_pred","correct","pred_correct","y_is_correct",
    "exact_match","em","acc","pass1_is_correct","pass1_correct",
    "answer_correct","label_correct"
}

def _first_str(*xs) -> Optional[str]:
    for x in xs:
        if isinstance(x, str) and x.strip():
            return x.strip()
    return None

def _find_in_obj(obj, keys: set) -> Optional[int]:
    q = [obj]
    while q:
        cur = q.pop(0)
        if isinstance(cur, dict):
            for k, v in cur.items():
                kl = str(k).lower()
                if any(cand in kl for cand in keys):
                    cb = coerce_bool(v)
                    if cb is not None:
                        return int(cb)
                if kl in {"acc","accuracy","score"}:
                    fv = coerce_float(v)
                    if fv is not None:
                        if fv in (0.0, 1.0): return int(fv)
                        if 0.0 <= fv <= 1.0: return int(fv >= 0.5)
            q.extend(cur.values())
        elif isinstance(cur, list):
            q.extend(cur)
    return None

def canon_equal(pred_canon: Optional[str], gold: Any) -> Optional[int]:
    if pred_canon is None or gold is None:
        return None
    if isinstance(gold, (list, tuple, set)):
        gset = set(str(g).strip() for g in gold if isinstance(g, str))
        return int(pred_canon in gset) if gset else None
    if isinstance(gold, str):
        return int(pred_canon.strip() == gold.strip())
    return None

def extract_correct(obj_like: Dict[str, Any], rec: Dict[str, Any]) -> Optional[int]:
    cb = _find_in_obj(obj_like, CORRECT_KEYS)
    if cb is not None: return cb
    pred_canon = _first_str(
        obj_like.get("pred_answer_canon"), rec.get("pred_answer_canon"),
        obj_like.get("final_answer_canon"), rec.get("final_answer_canon")
    )
    gold_canon = rec.get("gold_answer_canon_set") or rec.get("gold_answer_canon")
    ce = canon_equal(pred_canon, gold_canon)
    if ce is not None: return ce
    pred_raw = _first_str(
        obj_like.get("pred_answer"), rec.get("pred_answer"),
        obj_like.get("final_answer"), rec.get("final_answer"),
        obj_like.get("prediction"), rec.get("prediction")
    )
    gold_raw = _first_str(
        rec.get("gold_answer"), rec.get("answer"),
        rec.get("target"), rec.get("label")
    )
    if pred_raw is not None and gold_raw is not None:
        return int(pred_raw.strip() == gold_raw.strip())
    return None

# Carpark soft reward -> success
def carpark_success_from_soft_reward(rec: Dict[str, Any], p: Dict[str, Any], op: str, thr: float) -> Optional[int]:
    def cmp(val: Any) -> Optional[int]:
        x = coerce_float(val)
        if x is None: return None
        if op == "gt": return int(x > thr)
        if op == "ge": return int(x >= thr)
        if op == "eq": return int(x == thr)
        return int(x > thr)
    sr = rec.get("soft_reward", p.get("soft_reward"))
    return cmp(sr)

# ------------------------------- Domain / record parsing -------------------------------

def domain_from_path(path: str) -> str:
    p = path.lower()
    if ("xword" in p) or ("crossword" in p):
        return "Crossword"
    if ("carpark" in p) or ("rush" in p) or ("parking" in p):
        return "Carpark"
    return "Math"

def get_problem(rec: Dict[str, Any]) -> Optional[str]:
    # Include plain "problem" and fall back to sample_idx if needed.
    for k in ("problem_id", "example_id", "id", "uid", "question", "clue", "title", "problem"):
        v = rec.get(k)
        if v is not None and not isinstance(v, (dict, list)):
            return f"{k}:{v}"
    v = rec.get("sample_idx")
    return f"sample_{v}" if v is not None else None

def get_sample(rec: Dict[str, Any]) -> Optional[str]:
    v = rec.get("sample_idx")
    if v is None:
        return None
    try:
        return f"s{int(v)}"
    except Exception:
        return f"s{str(v)}"

def step_from_rec_or_path(rec: Dict[str, Any], path: str) -> int:
    step = rec.get("step") or rec.get("global_step") or rec.get("training_step")
    if step is None:
        step = nat_step_from_path(path)
    try:
        return int(step) if step is not None else 0
    except Exception:
        return 0

# ------------------------------- Entropy at pass-1 -------------------------------

def entropy_from_pass1(p1: Dict[str, Any], mode: str = "sum") -> Optional[float]:
    """
    Compute entropy_at_1 from pass-1:
      - sum      : entropy_think + entropy_answer (parts first; else fallback to 'entropy')
      - think    : entropy_think (fallback to 'entropy')
      - answer   : entropy_answer (fallback to 'entropy')
      - combined : 'entropy' as-is (fallback to avg(parts) if both available)
    """
    et = coerce_float(p1.get("entropy_think"))
    ea = coerce_float(p1.get("entropy_answer"))
    e  = coerce_float(p1.get("entropy"))

    if mode == "sum":
        parts = [v for v in (et, ea) if v is not None]
        if parts:
            return float(sum(parts))
        return e
    elif mode == "think":
        return et if et is not None else e
    elif mode == "answer":
        return ea if ea is not None else e
    elif mode == "combined":
        if e is not None:
            return e
        if et is not None and ea is not None:
            return 0.5 * (et + ea)
        return et if et is not None else ea
    else:
        return e

# shift_at_1 (canonical by default)
def compute_shift_at_1(p1: Dict[str, Any], rec: Dict[str, Any],
                       gpt_mode: str = "canonical") -> Optional[int]:
    keys = ["change_way_of_thinking", "shift_in_reasoning_v1"]
    if gpt_mode == "broad":
        keys += ["shift_llm", "shift_gpt", "pivot_llm", "rechecked"]
    seen = []
    for k in keys:
        v = both_get(p1, rec, k, None)
        if v is not None:
            cb = coerce_bool(v)
            if cb is not None:
                seen.append(cb)
    if not seen:
        return None
    return 1 if any(seen) else 0

# ------------------------------- Build rows (no binning yet) -------------------------------

def build_rows(files: List[str],
               carpark_op: str, carpark_thr: float,
               gpt_mode: str,
               min_step: Optional[int], max_step: Optional[int],
               entropy_mode: str) -> pd.DataFrame:
    rows = []
    for path in files:
        dom = domain_from_path(path)
        for rec in iter_records_from_file(path):
            p1 = rec.get("pass1") or {}
            p2 = rec.get("pass2") or {}

            step = step_from_rec_or_path(rec, path)
            if min_step is not None and step < min_step: continue
            if max_step is not None and step > max_step: continue

            problem = get_problem(rec)
            sample = get_sample(rec)

            # pass-1 correctness (fallback to Carpark soft_reward if needed)
            c1 = extract_correct(p1, rec)
            if c1 is None:
                c1 = carpark_success_from_soft_reward(rec, p1, carpark_op, carpark_thr)

            # pass-2 correctness (structured + Carpark fallback)
            c2 = None
            if isinstance(p2, dict) and p2:
                c2 = extract_correct(p2, rec)
                if c2 is None and dom == "Carpark":
                    c2 = carpark_success_from_soft_reward(rec, p2, carpark_op, carpark_thr)
            # top-level fallback used by Xword/Math
            if c2 is None:
                c2 = coerce_bool(rec.get("is_correct_after_reconsideration"))

            # pass-1 shift & entropies
            shift1 = compute_shift_at_1(p1, rec, gpt_mode=gpt_mode)
            ent = entropy_from_pass1(p1, mode=entropy_mode)
            ent_think = coerce_float(p1.get("entropy_think"))
            ent_answer = coerce_float(p1.get("entropy_answer"))

            rows.append({
                "domain": dom,
                "problem": problem,
                "step": step,
                "sample": sample,
                "shift_at_1": shift1,         # None / 0 / 1
                "correct_at_1": c1,           # 0/1 or None
                "entropy_at_1": ent,          # per entropy_mode
                "entropy_think_at_1": ent_think,
                "entropy_answer_at_1": ent_answer,
                "correct_at_2": c2            # 0/1 or None (target)
            })

    df = pd.DataFrame(rows)
    # keep rows needed for modeling
    df = df[ df["domain"].notna()
           & df["problem"].notna()
           & df["correct_at_1"].notna()
           & df["correct_at_2"].notna()
           & df["entropy_at_1"].notna()
           ].copy()
    df["correct_at_1"] = df["correct_at_1"].astype(int)
    df["correct_at_2"] = df["correct_at_2"].astype(int)
    return df

# ------------------------------- Binning --------------------------------------

def parse_fixed_bins(s: Optional[str]) -> Optional[List[float]]:
    if not s: return None
    out = []
    for tok in s.split(","):
        tok = tok.strip().lower()
        if tok in ("inf","+inf"): out.append(float("inf")); continue
        if tok == "-inf": out.append(float("-inf")); continue
        out.append(float(tok))
    if len(out) < 2:
        raise SystemExit("--fixed_bins must provide at least two edges (e.g., 0,1,2,inf).")
    return out

def compute_edges(values: np.ndarray, binning: str, bins: int,
                  fixed: Optional[List[float]]) -> List[float]:
    if fixed is not None:
        return fixed
    if binning == "uniform":
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise SystemExit("Non-finite min/max for uniform bins.")
        if hi <= lo:
            hi = lo + 1e-9
        return list(np.linspace(lo, hi, bins+1))
    if binning == "quantile":
        qs = np.linspace(0.0, 1.0, bins+1)
        edges = list(np.quantile(values, qs))
        eps = 1e-9
        for i in range(1, len(edges)):
            if edges[i] <= edges[i-1]:
                edges[i] = edges[i-1] + eps
        return edges
    raise SystemExit(f"Unknown --binning '{binning}'.")

def label_interval(I):
    if pd.isna(I):
        return np.nan
    if isinstance(I, str):
        return I
    try:
        return f"[{I.left:g},{I.right:g})"
    except AttributeError:
        return np.nan

def apply_binning_cut(df: pd.DataFrame, edges: List[float], scope: str = "global") -> pd.DataFrame:
    df = df.copy()
    if scope == "domain":
        parts = []
        for _, sub in df.groupby("domain"):
            parts.append(pd.cut(sub["entropy_at_1"], bins=edges, right=False, include_lowest=True))
        df["entropy_bin_at_1"] = pd.concat(parts).sort_index()
    else:
        df["entropy_bin_at_1"] = pd.cut(df["entropy_at_1"], bins=edges, right=False, include_lowest=True)
    cats = df["entropy_bin_at_1"].cat.categories
    labels = [label_interval(I) for I in cats]
    df["entropy_bin_label"] = df["entropy_bin_at_1"].cat.rename_categories(labels)
    df["entropy_bin_label"] = df["entropy_bin_label"].astype(
        pd.api.types.CategoricalDtype(categories=labels, ordered=True)
    )
    return df

# -------- Equal-N (rank-based) binning with tie-breaking (global or per-domain) --------

def _rank_based_bins(x: pd.Series, bins: int, tie_break: str, seed: int) -> np.ndarray:
    s = x.copy()
    mask = s.notna()
    n = int(mask.sum())
    if n == 0:
        return np.full(len(s), -1, dtype=int)
    if tie_break == "random":
        rng = np.random.RandomState(seed)
        scale = max(1.0, float(np.nanstd(s.values)))
        jitter = rng.uniform(-1e-9*scale, 1e-9*scale, size=n)
        vals = s[mask].values.astype(float) + jitter
        order = np.argsort(vals, kind="mergesort")
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(1, n+1, dtype=float)
    else:  # stable
        ranks = s[mask].rank(method="first").values
    idx = np.floor((ranks - 1) * bins / n).astype(int)
    idx[idx < 0] = 0
    idx[idx >= bins] = bins - 1
    out = np.full(len(s), -1, dtype=int)
    out[mask.values] = idx
    return out

def apply_equal_n_binning(df: pd.DataFrame, bins: int,
                          scope: str = "global",
                          tie_break: str = "stable",
                          seed: int = 42) -> pd.DataFrame:
    df = df.copy()
    def _apply(sub: pd.DataFrame) -> pd.DataFrame:
        idx = _rank_based_bins(sub["entropy_at_1"], bins=bins, tie_break=tie_break, seed=seed)
        sub = sub.copy()
        sub["_bin_id"] = idx
        labels = []
        for b in range(bins):
            v = sub.loc[sub["_bin_id"] == b, "entropy_at_1"]
            if v.empty:
                labels.append(f"Q{b+1} [∅]")
            else:
                lo, hi = float(np.min(v)), float(np.max(v))
                labels.append(f"Q{b+1} [{lo:g},{hi:g})")
        cat = pd.api.types.CategoricalDtype(categories=labels, ordered=True)
        sub["entropy_bin_label"] = pd.Categorical(
            [labels[b] if b >= 0 else np.nan for b in sub["_bin_id"]], dtype=cat
        )
        id_cat = pd.api.types.CategoricalDtype(categories=list(range(bins)), ordered=True)
        sub["entropy_bin_at_1"] = pd.Categorical(
            [b if b >= 0 else np.nan for b in sub["_bin_id"]], dtype=id_cat
        )
        sub.drop(columns=["_bin_id"], inplace=True)
        return sub
    if scope == "domain":
        parts = []
        for _, g in df.groupby("domain"):
            parts.append(_apply(g))
        df = pd.concat(parts, axis=0).sort_index()
    else:
        df = _apply(df)
    return df

# ------------------------------- Modeling helpers -------------------------------

def prune_subset(sub: pd.DataFrame, min_rows_per_problem: int = 2) -> pd.DataFrame:
    if sub.empty:
        return sub
    g = sub.groupby("problem")["correct_at_2"].agg(n="size", nunq="nunique").reset_index()
    keep = g[(g["n"] >= min_rows_per_problem) & (g["nunq"] > 1)]["problem"]
    dropped = len(g) - len(keep)
    if dropped > 0:
        print(f"[prune] Dropping {dropped} problem(s) with <{min_rows_per_problem} rows or no outcome variation.")
    sub = sub[sub["problem"].isin(keep)].copy()
    if "entropy_bin_label" in sub.columns and hasattr(sub["entropy_bin_label"], "cat"):
        sub["entropy_bin_label"] = sub["entropy_bin_label"].cat.remove_unused_categories()
    return sub

def fit_clustered_glm(df: pd.DataFrame, formula: str, cluster_col: str):
    model = smf.glm(formula=formula, data=df, family=sm.families.Binomial())
    try:
        res = model.fit(cov_type="cluster", cov_kwds={"groups": df[cluster_col]})
        if not np.isfinite(res.bse).all():
            print("[warn] Cluster SEs contain non-finite values; falling back to HC1 robust.")
            res = model.fit(cov_type="HC1")
        cov_used = res.cov_type
    except Exception as e:
        print(f"[warn] Cluster sandwich failed ({e}). Falling back to HC1 robust.")
        res = model.fit(cov_type="HC1")
        cov_used = "HC1"
    return res, res.summary().as_text(), cov_used

def compute_bin_ame(result, df_model: pd.DataFrame,
                    bin_col: str, baseline_label: str) -> pd.DataFrame:
    from pandas.api.types import CategoricalDtype as CDT
    if not isinstance(df_model[bin_col].dtype, CDT):
        df_model = df_model.copy()
        df_model[bin_col] = df_model[bin_col].astype("category")
    cats = df_model[bin_col].cat.categories
    cat_dtype = pd.api.types.CategoricalDtype(categories=cats, ordered=True)

    base = df_model.copy()
    base[bin_col] = pd.Categorical([baseline_label] * len(base), dtype=cat_dtype)
    p_base = result.predict(base)

    out_rows = []
    for bl in cats:
        cur = df_model.copy()
        cur[bin_col] = pd.Categorical([bl] * len(cur), dtype=cat_dtype)
        p_cur = result.predict(cur)
        ame = float(np.mean(p_cur - p_base))
        out_rows.append({"bin": bl, "ame": ame, "n_rows": int(len(cur))})
    return pd.DataFrame(out_rows)

# ------------------------------- Plotting -------------------------------

def plot_bin_contrasts(ame_none: pd.DataFrame,
                       ame_false: pd.DataFrame,
                       out_png: str, out_pdf: str,
                       title: str, dpi: int = 300):
    bins = list(ame_none["bin"]) if not ame_none.empty else list(ame_false["bin"])
    x = np.arange(len(bins))
    width = 0.38

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    if not ame_none.empty:
        ax.bar(x - width/2, ame_none["ame"].values, width, label="shift_at_1 is None")
    if not ame_false.empty:
        ax.bar(x + width/2, ame_false["ame"].values, width, label="shift_at_1 == 0")

    ax.set_xticks(x, bins, rotation=0)
    ax.set_ylabel("AME (Δ pp vs baseline bin)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(0.0, linewidth=1, color="black")

    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

# ------------------------------- Main ---------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan_root", type=str, required=True)
    ap.add_argument("--split", type=str, default=None)
    ap.add_argument("--min_step", type=int, default=None)
    ap.add_argument("--max_step", type=int, default=None)

    # Path filters (e.g., GRPO-1.5B & temp-0.7)
    ap.add_argument("--path_include", type=str, default=None,
                    help="Comma-separated substrings; all must appear in PATH (e.g., 'GRPO-1.5B,temp-0.7').")
    ap.add_argument("--path_exclude", type=str, default=None,
                    help="Comma-separated substrings; any match excludes the file.")

    # Binning
    ap.add_argument("--bin_scope", choices=["global","domain"], default="global")
    ap.add_argument("--binning", choices=["fixed","uniform","quantile"], default="uniform",
                    help="Ignored if --fixed_bins or --equal_n_bins is set.")
    ap.add_argument("--bins", type=int, default=4, help="Number of bins for uniform/quantile/equal_n.")
    ap.add_argument("--fixed_bins", type=str, default=None,
                    help="Comma-separated edges, e.g., '0,0.75,1.25,2,inf' (overrides --binning).")

    # Equal-N rank binning
    ap.add_argument("--equal_n_bins", action="store_true",
                    help="Force rank-based equal-count bins (exact quartiles when --bins 4).")
    ap.add_argument("--tie_break", choices=["stable","random"], default="stable",
                    help="Tie handling for equal-N bins: stable (deterministic) or random (tiny jitter).")
    ap.add_argument("--random_seed", type=int, default=42,
                    help="Random seed for --tie_break random.")

    # Carpark & shift
    ap.add_argument("--carpark_success_op", choices=["gt","ge","eq"], default="gt")
    ap.add_argument("--carpark_soft_threshold", type=float, default=0.0)
    ap.add_argument("--gpt_mode", choices=["canonical","broad"], default="canonical")

    # Domains
    ap.add_argument("--domains", type=str, default="Crossword,Math,Carpark")

    # Entropy modes
    ap.add_argument("--entropy_mode", choices=["sum","think","answer","combined"], default=None,
                    help="Run a single entropy mode.")
    ap.add_argument("--entropy_modes", nargs="+", choices=["sum","think","answer","combined"], default=None,
                    help="Run multiple entropy modes in one go (e.g., --entropy_modes sum think answer).")

    # Pruning
    ap.add_argument("--min_rows_per_problem", type=int, default=2,
                    help="Drop problems with fewer rows than this (prevents separation).")

    # Output / misc
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--dataset_name", type=str, default="MIXED")
    ap.add_argument("--model_name", type=str, default="MODEL")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--make_plot", action="store_true")
    ap.add_argument("--debug", action="store_true")

    args = ap.parse_args()

    # Which entropy modes to run
    if args.entropy_modes:
        modes = args.entropy_modes
    elif args.entropy_mode:
        modes = [args.entropy_mode]
    else:
        modes = ["sum", "think", "answer"]  # default: run all three

    mode_tag = {"sum":"eSUM", "think":"eTHINK", "answer":"eANSWER", "combined":"eCOMB"}

    # Scan and filter files
    skip_set = set(s.lower() for s in SKIP_DIR_DEFAULT)
    files_all = scan_files(args.scan_root, args.split, skip_set)
    if not files_all:
        sys.exit("No files found. Check --scan_root / --split.")

    def _parse_list(s):
        return [t.strip() for t in s.split(",") if t.strip()] if s else []
    inc = _parse_list(args.path_include)
    exc = _parse_list(args.path_exclude)

    files = []
    for f in files_all:
        f_low = f.lower()
        if inc and not all(i.lower() in f_low for i in inc):
            continue
        if exc and any(e.lower() in f_low for e in exc):
            continue
        files.append(f)
    if not files:
        sys.exit("No files left after path filters. Adjust --path_include/--path_exclude.")

    keep_domains = set([d.strip() for d in (args.domains or "").split(",") if d.strip()])

    # Base output root
    out_root_base = args.out_dir or os.path.join(args.scan_root, "entropy_bin_reg")
    os.makedirs(out_root_base, exist_ok=True)

    for emode in modes:
        print(f"\n=== Running entropy mode: {emode} ===")
        # Build rows for this entropy mode
        df_rows = build_rows(
            files=files,
            carpark_op=args.carpark_success_op,
            carpark_thr=args.carpark_soft_threshold,
            gpt_mode=args.gpt_mode,
            min_step=args.min_step,
            max_step=args.max_step,
            entropy_mode=emode
        )
        if args.debug:
            print(f"[info] built rows: n={len(df_rows)}")

        # Domain filter
        df_rows = df_rows[df_rows["domain"].isin(keep_domains)] if keep_domains else df_rows
        if df_rows.empty:
            print("[warn] No rows after --domains filter; skipping this mode.")
            continue

        # Choose binning path
        fixed = parse_fixed_bins(args.fixed_bins)
        if fixed is not None:
            edges = fixed
            df_rows = apply_binning_cut(df_rows, edges, scope=args.bin_scope)
            bins_info = f"fixed edges: {edges}"
        elif args.equal_n_bins:
            df_rows = apply_equal_n_binning(
                df_rows, bins=args.bins,
                scope=args.bin_scope,
                tie_break=args.tie_break,
                seed=args.random_seed
            )
            bins_info = f"equal_n_bins scope={args.bin_scope}, bins={args.bins}, tie_break={args.tie_break}, seed={args.random_seed}"
        else:
            if args.binning == "fixed":
                sys.exit("Provide --fixed_bins when --binning fixed; or use --equal_n_bins.")
            edges = compute_edges(df_rows["entropy_at_1"].to_numpy(), args.binning, args.bins, None)
            df_rows = apply_binning_cut(df_rows, edges, scope=args.bin_scope)
            bins_info = f"{args.binning} edges: {edges}"

        # Output dir + slug with mode tag
        slug = f"{args.dataset_name}__{args.model_name}".replace(" ", "_")
        slug_mode = f"{slug}__{mode_tag[emode]}"
        out_root = os.path.join(out_root_base, mode_tag[emode])
        os.makedirs(out_root, exist_ok=True)

        # Per-domain loop
        from pandas.api.types import CategoricalDtype as CDT
        for dom, df_dom in df_rows.groupby("domain", sort=False):
            out_dir = os.path.join(out_root, dom.replace(" ", "_"))
            os.makedirs(out_dir, exist_ok=True)

            # Ensure categorical (equal_n_bins + scope=domain can upcast to object)
            if not isinstance(df_dom["entropy_bin_label"].dtype, CDT):
                cats_dom = df_dom["entropy_bin_label"].dropna().drop_duplicates().tolist()
                df_dom["entropy_bin_label"] = pd.Categorical(
                    df_dom["entropy_bin_label"], categories=cats_dom, ordered=True
                )

            rows_csv = os.path.join(out_dir, f"rows__{slug_mode}__{dom}.csv")
            df_dom.to_csv(rows_csv, index=False)
            print(f"[{dom}:{emode}] rows: {len(df_dom):d}  -> {rows_csv}")

            # Two subsets
            sub_none  = df_dom[df_dom["shift_at_1"].isna()].copy()   # missing shift label
            sub_false = df_dom[df_dom["shift_at_1"] == 0].copy()     # labeled no-shift
            sub_true  = df_dom[df_dom["shift_at_1"] == 1].copy()     # labeled shift

            # Optional debug to see counts by label
            if args.debug:
                print(f"[{dom}:{emode}] shift_at_1 counts (NaN/0/1):")
                print(df_dom["shift_at_1"].value_counts(dropna=False).to_string())

            # Ensure categorical ordering, then prune + model
            for tag, sub in (("none", sub_none), ("false", sub_false), ("true", sub_true)):

                if sub.empty:
                    print(f"[{dom}:{emode}:{tag}] subset empty; skipping model.")
                    continue

                cats = df_dom["entropy_bin_label"].cat.categories
                sub["entropy_bin_label"] = pd.Categorical(sub["entropy_bin_label"], categories=cats, ordered=True)

                n_before = len(sub)
                sub = prune_subset(sub, min_rows_per_problem=args.min_rows_per_problem)
                n_after = len(sub)
                print(f"[{dom}:{emode}:{tag}] kept {n_after}/{n_before} rows after pruning")

                sub["entropy_bin_label"] = sub["entropy_bin_label"].cat.remove_unused_categories()

                if sub.empty or sub["problem"].nunique() < 2:
                    print(f"[{dom}:{emode}:{tag}] Not enough variation after pruning; skipping model.")
                    if tag == "none":   sub_none  = sub
                    else:               sub_false = sub
                    continue

                formula = "correct_at_2 ~ correct_at_1 + C(entropy_bin_label) + C(problem)"
                res, summ, cov_used = fit_clustered_glm(sub, formula, cluster_col="problem")
                txt_path = os.path.join(out_dir, f"model_{tag}__{slug_mode}__{dom}.txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(f"# cov_type: {cov_used}\n")
                    f.write(summ + "\n")
                print(f"[saved] {txt_path}")

                baseline = sub["entropy_bin_label"].cat.categories[0]
                ame = compute_bin_ame(res, sub, "entropy_bin_label", baseline)
                ame_csv = os.path.join(out_dir, f"bin_contrasts__{tag}__{slug_mode}__{dom}.csv")
                ame.to_csv(ame_csv, index=False)
                print(f"[saved] {ame_csv}")

                if tag == "none":   sub_none  = sub
                else:               sub_false = sub

            # Plot per-domain contrasts if any
            def _load(path):
                return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["bin","ame","n_rows"])
            ame_none  = _load(os.path.join(out_dir, f"bin_contrasts__none__{slug_mode}__{dom}.csv"))
            ame_false = _load(os.path.join(out_dir, f"bin_contrasts__false__{slug_mode}__{dom}.csv"))
            if args.make_plot and (not ame_none.empty or not ame_false.empty):
                png = os.path.join(out_dir, f"bin_contrasts__{slug_mode}__{dom}.png")
                pdf = os.path.join(out_dir, f"bin_contrasts__{slug_mode}__{dom}.pdf")
                title = f"{dom} — Entropy-bin contrasts (Δ pp vs baseline) — {args.model_name} [{emode}]"
                plot_bin_contrasts(ame_none, ame_false, png, pdf, title, args.dpi)
                print(f"[saved] {png}\n[saved] {pdf}")

        if args.debug:
            with pd.option_context("display.width", 160):
                print("\n[Rows head]")
                print(df_rows.head(6).to_string(index=False))
                print("\nBins used:", bins_info)

if __name__ == "__main__":
    main()
