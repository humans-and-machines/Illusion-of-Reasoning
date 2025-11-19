#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build Table: Shift frequency and conditional accuracy (per domain + overall),
and print a LaTeX-ready one-sentence summary with test results.

Domains supported:
- Crossword
- Math (labelled as Qwen1.5B-Math)
- Math2 (labelled as Qwen7B-Math)
- Carpark (success via soft_reward OP threshold)

Outputs
-------
1) CSV  : <out_dir>/shift_accuracy__<dataset>__<model>.csv
          columns = [domain, n_total, n_shift, shift_rate,
                      acc_shift, acc_noshift, delta_pp]
          plus a final "Overall" row.
2) TEX  : <out_dir>/shift_accuracy_summary__<dataset>__<model>.tex
          Contains a one-sentence LaTeX paragraph with the numbers filled in.
3) Console: prints the same LaTeX sentence and both p-values
   (GLM FE+cluster Wald test; two-proportion z-test).
"""

import os
import re
import json
import argparse
from typing import Optional, List, Dict, Any, Callable

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from math import erf, sqrt

# ---------- File / step parsing ----------

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
    out.sort()
    return out

# ---------- Small utils ----------

def coerce_bool(x) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, np.integer)):
        return int(bool(x))
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("1", "true", "t", "yes", "y"): return 1
        if s in ("0", "false", "f", "no", "n"): return 0
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

# ---------- Domain-aware LLM shift gating ----------

def _any_keys_true(p1: Dict[str, Any], rec: Dict[str, Any], keys: List[str]) -> int:
    for k in keys:
        v = p1.get(k, rec.get(k, None))
        if v is None:
            continue
        out = coerce_bool(v)
        if out is not None and out == 1:
            return 1
    return 0

def _cue_gate_for_llm(p1: Dict[str, Any], domain: Optional[str]) -> int:
    has_reconsider = coerce_bool(p1.get("has_reconsider_cue")) == 1
    rec_marks = p1.get("reconsider_markers") or []
    injected = ("injected_cue" in rec_marks)
    reconsider_ok = has_reconsider and not injected
    prefilter_cues = p1.get("_shift_prefilter_markers") or []
    judge_cues     = p1.get("shift_markers_v1") or []
    if str(domain).lower() == "crossword":
        return int(reconsider_ok or bool(prefilter_cues) or bool(judge_cues))
    else:
        return int(reconsider_ok or bool(prefilter_cues))

def aha_gpt_for_rec(p1: Dict[str, Any], rec: Dict[str, Any],
                    gpt_subset_native: bool, gpt_keys: List[str], domain: Optional[str]) -> int:
    gpt_raw = _any_keys_true(p1, rec, gpt_keys)
    if not gpt_subset_native:
        return int(gpt_raw)
    gate = _cue_gate_for_llm(p1, domain)
    return int(gpt_raw & gate)

# ---------- Carpark success helpers ----------

def _extract_soft_reward(rec: Dict[str, Any], p1: Dict[str, Any]) -> Optional[float]:
    # prefer top-level; fallback to pass1
    return coerce_float(rec.get("soft_reward", p1.get("soft_reward")))

def _make_carpark_success_fn(op: str, thr: float) -> Callable[[Any], Optional[int]]:
    def _cmp(val: Any) -> Optional[int]:
        x = coerce_float(val)
        if x is None: return None
        if op == "gt": return int(x >  thr)
        if op == "ge": return int(x >= thr)
        if op == "eq": return int(x == thr)
        return int(x > thr)  # safe default
    return _cmp

# ---------- Load rows ----------

def load_rows(files_by_domain: Dict[str, List[str]],
              gpt_keys: List[str],
              gpt_subset_native: bool,
              min_step: Optional[int],
              max_step: Optional[int],
              carpark_success_fn: Callable[[Any], Optional[int]]) -> pd.DataFrame:
    rows = []
    for dom, files in files_by_domain.items():
        dom_lower = str(dom).lower()
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
                    p1 = rec.get("pass1") or {}
                    if not isinstance(p1, dict):
                        continue

                    step = rec.get("step", step_from_name if step_from_name is not None else None)
                    if step is None:
                        continue
                    try:
                        step = int(step)
                    except Exception:
                        continue
                    if min_step is not None and step < min_step: continue
                    if max_step is not None and step > max_step: continue

                    # success
                    if dom_lower.startswith("carpark"):
                        sr = _extract_soft_reward(rec, p1)
                        success = carpark_success_fn(sr)
                    else:
                        success = coerce_bool(p1.get("is_correct_pred"))
                    if success is None:
                        continue

                    pid = get_problem_id(rec)
                    if pid is None:
                        continue
                    # prefix problem with domain to avoid collisions across domains
                    pid_full = f"{dom}::{pid}"

                    shift = aha_gpt_for_rec(p1, rec, gpt_subset_native, gpt_keys, dom)

                    rows.append({
                        "domain": str(dom),
                        "problem_id": pid_full,
                        "step": int(step),
                        "correct": int(success),
                        "shift": int(shift),
                    })
    return pd.DataFrame(rows)

# ---------- Aggregation ----------

def cond_counts(df: pd.DataFrame) -> Dict[str, Any]:
    n_total = int(len(df))
    n_shift = int((df["shift"] == 1).sum())
    n_noshift = n_total - n_shift
    k_shift = int(df.loc[df["shift"] == 1, "correct"].sum()) if n_shift>0 else 0
    k_noshift = int(df.loc[df["shift"] == 0, "correct"].sum()) if n_noshift>0 else 0
    p_shift = (k_shift / n_shift) if n_shift>0 else np.nan
    p_noshift = (k_noshift / n_noshift) if n_noshift>0 else np.nan
    shift_rate = (n_shift / n_total) if n_total>0 else np.nan
    return dict(n_total=n_total, n_shift=n_shift, n_noshift=n_noshift,
                k_shift=k_shift, k_noshift=k_noshift,
                p_shift=p_shift, p_noshift=p_noshift, shift_rate=shift_rate)

def two_prop_z_test(k1, n1, k0, n0) -> float:
    """Two-sample z test for proportions (two-sided). Returns p-value."""
    if n1==0 or n0==0: return float("nan")
    p1 = k1/n1; p0 = k0/n0
    p_pool = (k1 + k0) / (n1 + n0)
    se = sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n0))
    if se == 0: return float("nan")
    z = (p1 - p0) / se
    # normal CDF via erf
    def norm_cdf(x): return 0.5 * (1.0 + erf(x / sqrt(2.0)))
    pval = 2.0 * (1.0 - norm_cdf(abs(z)))
    return float(pval)

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_crossword", type=str, default=None)
    ap.add_argument("--root_math", type=str, default=None)   # Qwen1.5B-Math
    ap.add_argument("--root_math2", type=str, default=None)  # Qwen7B-Math
    ap.add_argument("--root_carpark", type=str, default=None)

    ap.add_argument("--label_math", type=str, default="Qwen1.5B-Math")
    ap.add_argument("--label_math2", type=str, default="Qwen7B-Math")

    ap.add_argument("results_root", nargs="?", default=None,
                    help="Fallback single root if domain-specific roots are not provided.")
    ap.add_argument("--split", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--dataset_name", default="MIXED")
    ap.add_argument("--model_name", default="MIXED_MODELS")

    # GPT label policy
    ap.add_argument("--gpt_mode", choices=["canonical","broad"], default="canonical")
    ap.add_argument("--no_gpt_subset_native", action="store_true")

    # Step filters (hard cap at 1000)
    ap.add_argument("--min_step", type=int, default=None)
    ap.add_argument("--max_step", type=int, default=None)

    # Carpark success policy
    ap.add_argument("--carpark_success_op", choices=["gt","ge","eq"], default="gt")
    ap.add_argument("--carpark_soft_threshold", type=float, default=0.0)

    args = ap.parse_args()

    # Map display labels
    label_map = {
        "Crossword": "Crossword",
        "Math": args.label_math.strip() or "Qwen1.5B-Math",
        "Math2": args.label_math2.strip() or "Qwen7B-Math",
        "Carpark": "Carpark",
    }

    files_by_domain: Dict[str, List[str]] = {}
    first_root = None
    if args.root_crossword:
        files_by_domain["Crossword"] = scan_files(args.root_crossword, args.split)
        first_root = first_root or args.root_crossword
    if args.root_math:
        files_by_domain["Math"] = scan_files(args.root_math, args.split)
        first_root = first_root or args.root_math
    if args.root_math2:
        files_by_domain["Math2"] = scan_files(args.root_math2, args.split)
        first_root = first_root or args.root_math2
    if args.root_carpark:
        files_by_domain["Carpark"] = scan_files(args.root_carpark, args.split)
        first_root = first_root or args.root_carpark
    if not files_by_domain:
        if not args.results_root:
            raise SystemExit("Provide --root_* folders or a fallback results_root.")
        files_by_domain["All"] = scan_files(args.results_root, args.split)
        first_root = args.results_root

    total_files = sum(len(v) for v in files_by_domain.values())
    if total_files == 0:
        raise SystemExit("No JSONL files found. Check roots/--split.")

    out_dir = args.out_dir or os.path.join(first_root, "tables_shift_accuracy")
    os.makedirs(out_dir, exist_ok=True)

    # GPT keys
    gpt_subset_native = not args.no_gpt_subset_native
    gpt_keys = (["change_way_of_thinking","shift_in_reasoning_v1"] if args.gpt_mode == "canonical"
                else ["change_way_of_thinking","shift_in_reasoning_v1","shift_llm","shift_gpt","pivot_llm","rechecked"])

    # Hard cap ≤ 1000
    HARD_MAX_STEP = 1000
    max_step_eff = HARD_MAX_STEP if args.max_step is None else min(args.max_step, HARD_MAX_STEP)
    if args.max_step is None or args.max_step > HARD_MAX_STEP:
        print(f"[info] Capping max_step to {max_step_eff} (hard cap = {HARD_MAX_STEP}).")

    carpark_success_fn = _make_carpark_success_fn(args.carpark_success_op, args.carpark_soft_threshold)

    # Load all rows
    df = load_rows(files_by_domain, gpt_keys, gpt_subset_native,
                   min_step=args.min_step, max_step=max_step_eff,
                   carpark_success_fn=carpark_success_fn)
    if df.empty:
        raise SystemExit("No rows after filtering.")

    # Per-domain table
    rows = []
    for dom_key in sorted(df["domain"].unique(), key=str):
        sub = df[df["domain"] == dom_key]
        cc = cond_counts(sub)
        rows.append({
            "domain": label_map.get(dom_key, dom_key),
            "n_total": cc["n_total"],
            "n_shift": cc["n_shift"],
            "shift_rate": cc["shift_rate"],
            "acc_shift": cc["p_shift"],
            "acc_noshift": cc["p_noshift"],
            "delta_pp": (cc["p_shift"] - cc["p_noshift"]) * 100.0 if np.isfinite(cc["p_shift"]) and np.isfinite(cc["p_noshift"]) else np.nan,
        })

    # Overall (pooled) row
    cc_all = cond_counts(df)
    rows.append({
        "domain": "Overall",
        "n_total": cc_all["n_total"],
        "n_shift": cc_all["n_shift"],
        "shift_rate": cc_all["shift_rate"],
        "acc_shift": cc_all["p_shift"],
        "acc_noshift": cc_all["p_noshift"],
        "delta_pp": (cc_all["p_shift"] - cc_all["p_noshift"]) * 100.0 if np.isfinite(cc_all["p_shift"]) and np.isfinite(cc_all["p_noshift"]) else np.nan,
    })

    table = pd.DataFrame(rows, columns=["domain","n_total","n_shift","shift_rate","acc_shift","acc_noshift","delta_pp"])

    # --- Tests (overall) ---
    # 1) GLM with FE by problem + cluster-robust SEs (preferred)
    glm = smf.glm(formula="correct ~ C(problem_id) + shift",
                  data=df, family=sm.families.Binomial(link=sm.families.links.logit()))
    try:
        res = glm.fit(cov_type="cluster", cov_kwds={"groups": df["problem_id"]}, maxiter=200)
    except Exception:
        res = glm.fit()
        res = res.get_robustcov_results(cov_type="cluster", groups=df["problem_id"])
    p_glm = float(res.pvalues.get("shift", np.nan))

    # 2) Two-proportion z-test
    p_z = two_prop_z_test(cc_all["k_shift"], cc_all["n_shift"], cc_all["k_noshift"], cc_all["n_noshift"])

    # Save CSV
    slug = f"{args.dataset_name}__{args.model_name}".replace(" ", "_")
    out_csv = os.path.join(out_dir, f"shift_accuracy__{slug}.csv")
    table.to_csv(out_csv, index=False)

    # Build LaTeX one-liner
    sr_pct = cc_all["shift_rate"] * 100.0 if np.isfinite(cc_all["shift_rate"]) else float("nan")
    a1 = cc_all["p_shift"] * 100.0 if np.isfinite(cc_all["p_shift"]) else float("nan")
    a0 = cc_all["p_noshift"] * 100.0 if np.isfinite(cc_all["p_noshift"]) else float("nan")
    # Convert p to "p<0.05" style
    def p_to_tex(p):
        if not np.isfinite(p): return r"$p$=--"
        if p < 1e-4: return r"$p<10^{-4}$"
        if p < 0.001: return r"$p<0.001$"
        if p < 0.01: return r"$p<0.01$"
        if p < 0.05: return r"$p<0.05$"
        return rf"$p={p:.3f}$"

    sentence = (
        r"Table~\ref{tab:shift-accuracy} reports the frequency of detected shifts ($S_{i,j}$) "
        r"and conditional accuracy with and without a shift. "
        rf"Across domains and models, shifts are rare ($\approx {sr_pct:.1f}\%$ of samples overall) "
        rf"and, when they occur, accuracy is lower than in non-shifted traces "
        rf"({a1:.1f}\% vs.\ {a0:.1f}\% correct; "
        r"test: logistic GLM with problem fixed effects and cluster-robust SEs, "
        rf"{p_to_tex(p_glm)})."
    )

    out_tex = os.path.join(out_dir, f"shift_accuracy_summary__{slug}.tex")
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(sentence + "\n")

    # Console preview (pretty)
    disp = table.copy()
    disp["shift_rate"] = disp["shift_rate"].map(lambda x: f"{x*100:.1f}%" if np.isfinite(x) else "--")
    disp["acc_shift"] = disp["acc_shift"].map(lambda x: f"{x*100:.1f}%" if np.isfinite(x) else "--")
    disp["acc_noshift"] = disp["acc_noshift"].map(lambda x: f"{x*100:.1f}%" if np.isfinite(x) else "--")
    disp["delta_pp"] = disp["delta_pp"].map(lambda x: f"{x:+.1f} pp" if np.isfinite(x) else "--")

    with pd.option_context("display.max_columns", None, "display.width", 120):
        print("\n== Shift frequency & conditional accuracy ==")
        print(disp.to_string(index=False))
        print("\nPreferred test (GLM + FE + cluster): p_shift =", p_glm)
        print("Two-proportion z-test:               p_shift =", p_z)
        print(f"\nSaved CSV  -> {out_csv}")
        print(f"Saved TEX  -> {out_tex}")
        print("\nLaTeX one-liner:\n" + sentence + "\n")

if __name__ == "__main__":
    main()
