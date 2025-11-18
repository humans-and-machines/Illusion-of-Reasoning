#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
temperature_effects.py  (parallel; updated to support Math2/7B & joint plotting)
-------------------------------------------------------------------------------
Temperature-focused shift table (raw effect + AME) with fast parallel loading.

New in this version
-------------------
• Supports a second Math series (e.g., Qwen-7B) via --math2_tpl/--label_math2
• Per-temperature raw-effect aggregation includes Math2
• Plot shows Math and Math2 as separate series (others are skipped if absent)
• Backward compatible with older usage (Crossword/Math/Carpark only)
• Robust tick handling: x-axis uses the exact temperatures you passed

Parallelism
-----------
  --workers 40          number of processes/threads (default: 40)
  --parallel process    process (default) or thread pool
  --chunksize N         batching for executor.map

Everything else is as before (step caps, robust correctness, per-temp CSV+plot).
"""

import os, re, json, argparse, sys, gzip
from typing import Optional, List, Dict, Any, Tuple, Callable, Iterable
import concurrent.futures as cf

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families.links import Logit

# Matplotlib + global styling
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L"],
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "pdf.fonttype": 42,   # embed TrueType
    "ps.fonttype": 42,
})

# ---------- patterns ----------

STEP_DIR_PAT = re.compile(r"(?:^|/)(step[-_]?\d{1,5})(?:/|$)", re.I)
STEP_PATS = [
    re.compile(r"step\s*[-_]?(\d+)", re.I),
    re.compile(r"global[_-]?step\s*[-_]?(\d+)", re.I),
    re.compile(r"checkpoint\s*[-_]?(\d+)", re.I),
    re.compile(r"/(\d{2,5})(?=/|$)")
]
TEMP_PATS = [
    re.compile(r"(?:^|[-_])temp[-_](?P<t>low|[0-9]+(?:\.[0-9]+)?)$", re.I),
    re.compile(r"(?:^|[-_])(?P<t>low|[0-9]+(?:\.[0-9]+)?)[-_]temp$", re.I),
    re.compile(r"(?:^|[-_])low[-_]temp$", re.I),
]

SKIP_DIR_DEFAULT = {"compare-1shot", "1shot", "hf_cache"}

# ---------- helpers ----------

def parse_temp_from_dir(dirname: str, low_alias: float) -> Optional[float]:
    d = dirname.lower()
    for pat in TEMP_PATS:
        m = pat.search(d)
        if not m:
            continue
        tok = m.groupdict().get("t", "low").lower()
        if tok == "low":
            return float(low_alias)
        try:
            return float(tok)
        except Exception:
            return None
    return None

def nat_step_from_path(path: str) -> Optional[int]:
    s = str(path)
    for pat in STEP_PATS:
        m = pat.search(s)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return None

def coerce_bool(x) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, np.integer)):
        return int(bool(x))
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("1","true","t","yes","y"): return 1
        if s in ("0","false","f","no","n"):  return 0
    try:
        return int(bool(x))
    except Exception:
        return None

def coerce_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None

def get_problem_id(rec: Dict[str, Any]) -> Optional[str]:
    for k in ("problem_id","example_id","id","question","problem","clue","title","uid"):
        v = rec.get(k)
        if v is not None and not isinstance(v, (dict, list)):
            return str(v)
    v = rec.get("sample_idx")
    return None if v is None else f"sample_{v}"

# ---------- shift gating ----------

def _any_keys_true(p1: Dict[str, Any], rec: Dict[str, Any], keys: List[str]) -> int:
    for k in keys:
        v = p1.get(k, rec.get(k, None))
        if v is None:
            continue
        out = coerce_bool(v)
        if out == 1:
            return 1
    return 0

def _cue_gate_for_llm(p1: Dict[str, Any], domain: Optional[str]) -> int:
    has_reconsider = coerce_bool(p1.get("has_reconsider_cue")) == 1
    rec_marks = p1.get("reconsider_markers") or []
    injected = ("injected_cue" in rec_marks)
    reconsider_ok = has_reconsider and not injected
    prefilter = p1.get("_shift_prefilter_markers") or []
    judge     = p1.get("shift_markers_v1") or []
    if str(domain).lower() == "crossword":
        return int(reconsider_ok or bool(prefilter) or bool(judge))
    else:
        return int(reconsider_ok or bool(prefilter))

def aha_gpt_for_rec(p1: Dict[str, Any], rec: Dict[str, Any],
                    gpt_subset_native: bool, gpt_keys: List[str], domain: Optional[str]) -> int:
    gpt_raw = _any_keys_true(p1, rec, gpt_keys)
    if not gpt_subset_native:
        return int(gpt_raw)
    gate = _cue_gate_for_llm(p1, domain)
    return int(gpt_raw & gate)

# ---------- correctness extraction (robust) ----------

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

def _extract_correct(p1: Dict[str, Any], rec: Dict[str, Any]) -> Optional[int]:
    for src in (p1, rec):
        cb = _find_in_obj(src, CORRECT_KEYS)
        if cb is not None:
            return cb
    pred_canon = _first_str(rec.get("pred_answer_canon"), p1.get("pred_answer_canon"))
    gold_canon = _first_str(rec.get("gold_answer_canon"), p1.get("gold_answer_canon"))
    if pred_canon is not None and gold_canon is not None:
        return int(pred_canon == gold_canon)
    pred_raw = _first_str(rec.get("pred_answer"), p1.get("pred_answer"),
                          rec.get("final_answer"), p1.get("final_answer"),
                          rec.get("pred"), p1.get("pred"),
                          rec.get("prediction"), p1.get("prediction"))
    gold_raw = _first_str(rec.get("gold_answer"), p1.get("gold_answer"),
                          rec.get("gold"), p1.get("gold"),
                          rec.get("answer"), p1.get("answer"),
                          rec.get("target"), p1.get("target"),
                          rec.get("label"), p1.get("label"))
    if pred_raw is not None and gold_raw is not None:
        return int(pred_raw == gold_raw)
    return None

# ---------- carpark success ----------

def _make_carpark_success_fn(op: str, thr: float) -> Callable[[Any], Optional[int]]:
    def _cmp(val: Any) -> Optional[int]:
        x = coerce_float(val)
        if x is None:
            return None
        if op == "gt": return int(x >  thr)
        if op == "ge": return int(x >= thr)
        if op == "eq": return int(x == thr)
        return int(x > thr)
    return _cmp

def _extract_soft_reward(rec: Dict[str, Any], p1: Dict[str, Any]) -> Optional[float]:
    return coerce_float(rec.get("soft_reward", p1.get("soft_reward")))

# ---------- file scanning & reading ----------

STEP_FILE_PAT = re.compile(r"^(?:step|global[_-]?step|checkpoint)[-_]?\d{1,5}", re.I)

def scan_files_step_only(root: str, split_substr: Optional[str], skip_substrings: set) -> List[str]:
    out = []
    for dp, _, fns in os.walk(root):
        dp_norm = dp.replace("\\", "/").lower()
        if any(s in dp_norm for s in skip_substrings):
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
    if path.endswith(".jsonl.gz"):
        opener = gzip.open; mode = "rt"
    else:
        opener = open; mode = "r"
    try:
        with opener(path, mode, encoding="utf-8") as f:
            if path.endswith(".json"):
                text = f.read().strip()
                if not text:
                    return
                try:
                    obj = json.loads(text)
                    if isinstance(obj, list):
                        for rec in obj:
                            if isinstance(rec, dict):
                                yield rec
                    elif isinstance(obj, dict):
                        yield obj
                except Exception:
                    for line in text.splitlines():
                        s = line.strip()
                        if not s:
                            continue
                        try:
                            rec = json.loads(s)
                            if isinstance(rec, dict): yield rec
                        except Exception:
                            continue
            else:
                for line in f:
                    s = line.strip()
                    if not s: continue
                    try:
                        rec = json.loads(s)
                        if isinstance(rec, dict): yield rec
                    except Exception:
                        continue
    except Exception:
        return

# ---------- stats ----------

def cond_counts(df: pd.DataFrame) -> Tuple[int, float, float, float]:
    N = int(len(df))
    n1 = int((df["shift"] == 1).sum())
    n0 = N - n1
    k1 = int(df.loc[df["shift"] == 1, "correct"].sum()) if n1 > 0 else 0
    k0 = int(df.loc[df["shift"] == 0, "correct"].sum()) if n0 > 0 else 0
    p1 = (k1 / n1) if n1 > 0 else np.nan
    p0 = (k0 / n0) if n0 > 0 else np.nan
    share = (n1 / N) if N > 0 else np.nan
    return N, share, p1, p0

def ame_glm_temp(sub_allT: pd.DataFrame) -> Tuple[float, float]:
    sub = sub_allT.copy()
    t = sub["temp"].to_numpy(dtype=float)
    sub["temp_std"] = (t - t.mean()) / (t.std(ddof=0) + 1e-12)
    sub["shift"] = sub["shift"].astype(float)
    sub["correct"] = sub["correct"].astype(float)

    model = smf.glm(
        formula="correct ~ C(problem_id) + temp_std + shift",
        data=sub,
        family=sm.families.Binomial(link=Logit()),
    )
    try:
        res = model.fit(cov_type="cluster", cov_kwds={"groups": sub["problem_id"]}, maxiter=200)
    except Exception:
        res = model.fit()
        res = res.get_robustcov_results(cov_type="cluster", groups=sub["problem_id"])

    X = res.model.exog
    names = list(res.model.exog_names)
    try:
        j_shift = names.index("shift")
    except ValueError:
        j_shift = max(i for i, nm in enumerate(names) if "shift" in nm)

    def inv_logit(z): return 1.0/(1.0+np.exp(-z))
    b = res.params.to_numpy()
    x1 = X.copy(); x1[:, j_shift] = 1.0
    x0 = X.copy(); x0[:, j_shift] = 0.0
    ame = float(np.mean(inv_logit(x1 @ b) - inv_logit(x0 @ b)))
    p_shift = float(res.pvalues.get("shift", np.nan))
    return ame, p_shift

# ---------- discovery ----------
def discover_dirs(scan_root: str, temps: List[float], low_alias: float, skip_substrings: set) -> Dict[float, Dict[str, str]]:
    temps_set = set(float(t) for t in temps)
    mapping: Dict[float, Dict[str, str]] = {}
    for root, dirs, _ in os.walk(scan_root):
        for d in dirs:
            d_lower = d.lower()
            if any(s in d_lower for s in skip_substrings):
                continue
            T = parse_temp_from_dir(d, low_alias)
            if T is None or float(T) not in temps_set:
                continue

            if   "xword"   in d_lower: dom = "Crossword"
            elif "carpark" in d_lower: dom = "Carpark"
            elif "math"    in d_lower: dom = "Math2" if "7b" in d_lower else "Math"
            elif ("low-temp" in d_lower or "low_temp" in d_lower) and "1.5b" in d_lower:
                dom = "Math"
            else:
                continue

            full = os.path.join(root, d)
            mapping.setdefault(float(T), {})
            prev = mapping[float(T)].get(dom)

            def _rank_math(p: str) -> tuple:
                b = os.path.basename(p).lower()
                score = 0
                if "1.5b" in b: score += 300
                if "qwen" in b: score += 200
                if "llama" in b: score += 100
                if "low-temp" in b or "low_temp" in b: score += 10
                return (score, len(p))

            if dom == "Math":
                if (prev is None) or (_rank_math(full) > _rank_math(prev)):
                    mapping[float(T)][dom] = full
            else:
                if (prev is None) or (len(full) > len(prev)):
                    mapping[float(T)][dom] = full

    if mapping:
        print("[info] discovered roots by temperature:")
        for T in sorted(mapping):
            items = ", ".join(f"{k}→{v}" for k, v in mapping[T].items())
            print(f"  T={T}: {items}")
    return mapping

# ---------- PARALLEL WORKER ----------

def _process_file_worker(task: Tuple[str, str, List[str], bool, Optional[int], Optional[int], str, float, float]) -> List[Dict[str, Any]]:
    """
    One file -> list of rows.
    task = (dom, path, gpt_keys, gpt_subset_native, min_step, max_step, carpark_op, carpark_thr, temp_value)
    """
    dom, path, gpt_keys, gpt_subset_native, min_step, max_step, carpark_op, carpark_thr, temp_value = task
    rows: List[Dict[str, Any]] = []
    carpark_success_fn = _make_carpark_success_fn(carpark_op, carpark_thr)
    dom_lower = dom.lower()
    step_from_name = nat_step_from_path(path)

    for rec in iter_records_from_file(path):
        p1 = rec.get("pass1") or {}
        if not isinstance(p1, dict):
            p1 = {}

        step = (rec.get("step") or rec.get("global_step") or
                rec.get("training_step") or step_from_name)
        try:
            step = int(step) if step is not None else 0
        except Exception:
            step = 0

        if min_step is not None and step < min_step:
            continue
        if max_step is not None and step > max_step:
            continue

        if dom_lower.startswith("carpark"):
            sr = _extract_soft_reward(rec, p1)
            ok = carpark_success_fn(sr)
            if ok is None:
                continue
            correct = int(ok)
        else:
            corr = _extract_correct(p1, rec)
            if corr is None:
                continue
            correct = int(corr)

        pid = get_problem_id(rec)
        if pid is None:
            continue

        shift = aha_gpt_for_rec(p1, rec, gpt_subset_native, gpt_keys, dom)

        rows.append({
            "domain": str(dom),
            "problem_id": f"{dom}::{pid}",
            "step": int(step),
            "temp": float(temp_value),
            "correct": int(correct),
            "shift": int(shift),
        })

    return rows

def load_rows_parallel(files_by_domain: Dict[str, List[str]],
                       gpt_keys: List[str], gpt_subset_native: bool,
                       min_step: Optional[int], max_step: Optional[int],
                       carpark_op: str, carpark_thr: float,
                       temp_value: float, workers: int, parallel: str, chunksize: int) -> pd.DataFrame:
    tasks: List[Tuple[str, str, List[str], bool, Optional[int], Optional[int], str, float, float]] = []
    for dom, files in files_by_domain.items():
        for path in files:
            tasks.append((dom, path, gpt_keys, gpt_subset_native, min_step, max_step, carpark_op, carpark_thr, temp_value))
    if not tasks:
        return pd.DataFrame(columns=["domain","problem_id","step","temp","correct","shift"])

    Exec = cf.ProcessPoolExecutor if parallel == "process" else cf.ThreadPoolExecutor
    rows_all: List[Dict[str, Any]] = []
    csize = max(1, chunksize if chunksize > 0 else len(tasks)//(workers*4) or 1)
    with Exec(max_workers=workers) as ex:
        for res in ex.map(_process_file_worker, tasks, chunksize=csize):
            if res:
                rows_all.extend(res)
    return pd.DataFrame(rows_all)

# ---------- NEW: per-temperature raw-effect computation & plotting ----------

def per_temp_delta(df_T_dom: pd.DataFrame) -> Tuple[float, float, int, int]:
    N, _, p1, p0 = cond_counts(df_T_dom)
    n1 = int((df_T_dom["shift"] == 1).sum())
    n0 = int((df_T_dom["shift"] == 0).sum())
    if not (np.isfinite(p1) and np.isfinite(p0)) or n1 == 0 or n0 == 0:
        return (np.nan, np.nan, n1, n0)
    delta = (p1 - p0) * 100.0
    se = 100.0 * float(np.sqrt((p1*(1-p1))/n1 + (p0*(1-p0))/n0))
    return (delta, se, n1, n0)

def make_plot(pertemp_df: pd.DataFrame, out_png: str, out_pdf: Optional[str],
              title: str, x_temps_sorted: List[float],
              label_map: Dict[str, str], dpi: int = 300):
    fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)

    # Series order; empty ones are skipped automatically
    series = [
        ("Math",      "C2", "o"),
        ("Math2",     "C4", "s"),  # second math series (e.g., Qwen-7B)
        ("Crossword", "C0", "o"),
        ("Carpark",   "C3", "o"),
    ]
    for dom_key, color, marker in series:
        sub = pertemp_df[pertemp_df["domain_key"] == dom_key].copy()
        if sub.empty:
            continue
        sub = sub.set_index("temp").reindex(sorted(set(x_temps_sorted))).reset_index()
        ax.errorbar(
            sub["temp"], sub["delta_pp"], yerr=sub["se_pp"],
            fmt=marker+"-", capsize=4, elinewidth=1, linewidth=2,
            label=label_map.get(dom_key, dom_key), color=color
        )

    # Helpful x ticks: exactly the temperatures you passed; pad edges
    xt = sorted(set(x_temps_sorted))
    ax.set_xticks(xt)
    ax.set_xticklabels([("{:.2f}".format(t)).rstrip("0").rstrip(".") for t in xt])
    if len(xt) >= 2:
        pad = 0.05 * (max(xt) - min(xt))
    else:
        pad = 0.05
    ax.set_xlim(min(xt) - pad, max(xt) + pad)

    ax.axhline(0.0, linewidth=1, linestyle="--", color="0.4")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Raw effect of shift on accuracy (pp)")
    ax.set_title(title)
    ax.legend(frameon=True, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    ax.grid(True, axis="y", alpha=0.25)

    fig.subplots_adjust(bottom=0.28)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    if out_pdf:
        fig.savefig(out_pdf, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temps", nargs="+", type=float, required=True)
    ap.add_argument("--scan_root", type=str, default=None)
    ap.add_argument("--crossword_tpl", type=str, default=None)
    ap.add_argument("--math_tpl",      type=str, default=None)   # e.g., Llama-8B
    ap.add_argument("--math2_tpl",     type=str, default=None)   # e.g., Qwen-7B
    ap.add_argument("--carpark_tpl",   type=str, default=None)

    ap.add_argument("--label_crossword", type=str, default="Crossword")
    ap.add_argument("--label_math",      type=str, default="Llama-8B-Math")
    ap.add_argument("--label_math2",     type=str, default="Qwen-7B-Math")
    ap.add_argument("--label_carpark",   type=str, default="Carpark")
    ap.add_argument("--include_math2", action="store_true", default=True)

    ap.add_argument("--split", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--dataset_name", default="MIXED")
    ap.add_argument("--model_name",   default="7B_vs_8B")

    ap.add_argument("--gpt_mode", choices=["canonical","broad"], default="canonical")
    ap.add_argument("--no_gpt_subset_native", action="store_true")
    ap.add_argument("--min_step", type=int, default=None)
    ap.add_argument("--max_step", type=int, default=None)

    ap.add_argument("--carpark_success_op", choices=["gt","ge","eq"], default="gt")
    ap.add_argument("--carpark_soft_threshold", type=float, default=0.0)

    ap.add_argument("--low_alias", type=float, default=0.3)
    ap.add_argument("--skip_substr", nargs="*", default=["compare-1shot", "1shot", "hf_cache"])

    # plotting knobs
    ap.add_argument("--make_plot", action="store_true")
    ap.add_argument("--plot_title", type=str, default=None)
    ap.add_argument("--dpi", type=int, default=300)

    # PARALLEL knobs
    ap.add_argument("--workers", type=int, default=40, help="Parallel workers (processes or threads).")
    ap.add_argument("--parallel", choices=["process","thread"], default="process", help="Process pool (default) or thread pool.")
    ap.add_argument("--chunksize", type=int, default=8, help="Task chunk size for executor.map.")

    args = ap.parse_args()

    # output dir
    if args.out_dir:
        out_dir = args.out_dir
    else:
        guess = args.scan_root or (args.crossword_tpl or args.math_tpl or args.math2_tpl or args.carpark_tpl or ".")
        out_dir = os.path.join(guess if isinstance(guess, str) else ".", "temperature_effects")
    os.makedirs(out_dir, exist_ok=True)

    # GPT keys
    gpt_subset_native = not args.no_gpt_subset_native
    gpt_keys = (["change_way_of_thinking","shift_in_reasoning_v1"]
                if args.gpt_mode == "canonical"
                else ["change_way_of_thinking","shift_in_reasoning_v1","shift_llm","shift_gpt","pivot_llm","rechecked"])

    # Step cap (hard cap 1000)
    HARD_MAX = 1000
    max_step_eff = HARD_MAX if args.max_step is None else min(args.max_step, HARD_MAX)
    if args.max_step is None or args.max_step > HARD_MAX:
        print(f"[info] Capping max_step to {max_step_eff} (hard cap = {HARD_MAX}).")

    carpark_op  = args.carpark_success_op
    carpark_thr = args.carpark_soft_threshold

    skip_set = set(s.lower() for s in args.skip_substr) | SKIP_DIR_DEFAULT

    # Discover roots
    if args.scan_root:
        roots_by_temp = discover_dirs(args.scan_root, args.temps, args.low_alias, skip_set)
    else:
        roots_by_temp = {}
        for T in args.temps:
            dmap: Dict[str, str] = {}
            if args.crossword_tpl:
                p = args.crossword_tpl.format(T=T);  dmap["Crossword"] = p if os.path.isdir(p) else None
            if args.math_tpl:
                p = args.math_tpl.format(T=T);       dmap["Math"]      = p if os.path.isdir(p) else None
            if args.math2_tpl:
                p = args.math2_tpl.format(T=T);      dmap["Math2"]     = p if os.path.isdir(p) else None
            if args.carpark_tpl:
                p = args.carpark_tpl.format(T=T);    dmap["Carpark"]   = p if os.path.isdir(p) else None
            roots_by_temp[float(T)] = {k:v for k,v in dmap.items() if v}

    if not roots_by_temp:
        sys.exit("No usable folders discovered. Check --scan_root or templates + temps.")

    domain_frames: Dict[str, List[pd.DataFrame]] = {"Crossword": [], "Math": [], "Math2": [], "Carpark": []}
    pertemp_rows: List[Dict[str, Any]] = []
    x_temps_sorted = sorted(roots_by_temp.keys())

    for T, dmap in roots_by_temp.items():
        files_by_domain: Dict[str, List[str]] = {}
        discover_keys = ["Crossword","Math","Math2","Carpark"] if args.include_math2 else ["Crossword","Math","Carpark"]
        for dom in discover_keys:
            path = dmap.get(dom)
            if not path:
                continue
            files = scan_files_step_only(path, args.split, skip_set)
            if files:
                files_by_domain[dom] = files

        if not files_by_domain:
            continue

        # PARALLEL load
        df_T = load_rows_parallel(
            files_by_domain=files_by_domain,
            gpt_keys=gpt_keys,
            gpt_subset_native=gpt_subset_native,
            min_step=args.min_step,
            max_step=max_step_eff,
            carpark_op=carpark_op,
            carpark_thr=carpark_thr,
            temp_value=T,
            workers=args.workers,
            parallel=args.parallel,
            chunksize=args.chunksize,
        )

        if df_T.empty:
            print(f"[warn] T={T}: no rows loaded after parsing/gating.")
            for dom_dbg, flist in files_by_domain.items():
                print(f"    files[{dom_dbg}]={len(flist)} under {dmap.get(dom_dbg)}")
            continue

        parts = []
        for dom_show in ["Crossword","Math","Math2","Carpark"]:
            n_dom = int((df_T["domain"] == dom_show).sum())
            parts.append(f"{dom_show}={n_dom}")
        print(f"[info] loaded rows @ T={T}: " + ", ".join(parts))

        for dom in ["Crossword","Math","Math2","Carpark"]:
            sub = df_T[df_T["domain"] == dom]
            if not sub.empty:
                domain_frames[dom].append(sub)

        # per-temp raw effects (include Math2)
        for dom in ["Crossword","Math","Math2","Carpark"]:
            sub = df_T[df_T["domain"] == dom]
            if sub.empty:
                continue
            delta_pp, se_pp, n1, n0 = per_temp_delta(sub)
            pertemp_rows.append({
                "temp": float(T),
                "domain_key": dom,
                "domain": dom,
                "delta_pp": delta_pp,
                "se_pp": se_pp,
                "n_shift": n1,
                "n_noshift": n0,
            })

    label_map = {
        "Crossword": args.label_crossword,
        "Math": args.label_math,
        "Math2": args.label_math2,
        "Carpark": args.label_carpark,
    }

    # Domain-level aggregates with AME per domain
    rows = []
    for dom in ["Crossword","Math","Math2","Carpark"]:
        if not domain_frames[dom]:
            continue
        df = pd.concat(domain_frames[dom], ignore_index=True)
        N, share, p1, p0 = cond_counts(df)
        delta_pp = (p1 - p0) * 100.0 if (np.isfinite(p1) and np.isfinite(p0)) else np.nan
        ame, p_shift = ame_glm_temp(df)
        rows.append({
            "domain_key": dom,
            "domain": label_map[dom],
            "N": N,
            "share_shift": share,
            "acc_shift": p1,
            "delta_pp": delta_pp,
            "AME": ame,
            "p": p_shift,
        })

    if not rows:
        sys.exit("No per-domain aggregates available.")

    tab = pd.DataFrame(rows, columns=["domain","N","share_shift","acc_shift","delta_pp","AME","p"])

    slug = f"{args.dataset_name}__{args.model_name}".replace(" ", "_")
    out_dir = args.out_dir or os.path.join(args.scan_root or ".", "temperature_effects")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"temperature_shift_table__{slug}.csv")
    tab.to_csv(out_csv, index=False)

    print("[info] Aggregates by domain:")
    with pd.option_context("display.max_columns", None, "display.width", 140):
        disp = tab.copy()
        disp["share_shift"] = disp["share_shift"].map(lambda v: f"{v:.4f}" if pd.notna(v) else "--")
        disp["acc_shift"]   = disp["acc_shift"].map(  lambda v: f"{v:.4f}" if pd.notna(v) else "--")
        disp["delta_pp"]    = disp["delta_pp"].map(   lambda v: f"{v:+.2f}" if pd.notna(v) else "--")
        disp["AME"]         = disp["AME"].map(        lambda v: f"{v:.4f}" if pd.notna(v) else "--")
        disp["p"]           = disp["p"].map(          lambda v: f"{v:.3g}" if pd.notna(v) else "--")
        print(disp.to_string(index=False))
    print(f"\nSaved CSV -> {out_csv}")

    if pertemp_rows:
        pertemp = pd.DataFrame(pertemp_rows)
        pertemp["domain"] = pertemp["domain_key"].map(label_map).fillna(pertemp["domain_key"])
        pertemp_csv = os.path.join(out_dir, f"temperature_shift_raw_effects__{slug}.csv")
        pertemp.sort_values(["domain_key","temp"]).to_csv(pertemp_csv, index=False)
        print(f"Saved per-temperature raw effects CSV -> {pertemp_csv}")

        if args.make_plot:
            plot_title = args.plot_title or f"Raw effect vs temperature — {args.model_name}"
            png_path = os.path.join(out_dir, f"temperature_shift_plot__{slug}.png")
            pdf_path = os.path.join(out_dir, f"temperature_shift_plot__{slug}.pdf")
            make_plot(
                pertemp_df=pertemp,
                out_png=png_path,
                out_pdf=pdf_path,
                title=plot_title,
                x_temps_sorted=x_temps_sorted,
                label_map=label_map,
                dpi=args.dpi,
            )
            print(f"Saved plot PNG -> {png_path}")
            print(f"Saved plot PDF -> {pdf_path}")

if __name__ == "__main__":
    main()
