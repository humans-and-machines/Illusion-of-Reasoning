#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aha Regression (Per-Domain)
---------------------------
For each provided folder/domain (e.g., --root_crossword, --root_math, --root_math2, --root_carpark):

  • Build per-example rows with:
        correct ∈ {0,1}
        shift   ∈ {0,1}           (LLM shift detector with domain-aware gating)
        step    ∈ ℤ               (training step)
        problem_id (string)       (problem fixed effect & clustering key)

  • Success definition:
        - Crossword/Math/Math2: correct = is_correct_pred
        - Carpark: correct = 1[ soft_reward OP threshold ] where OP ∈ {gt, ge, eq}

  • Standardize step within each domain: step_std = (step - mean)/std

  • Fit (per domain) a binomial GLM with logit link and problem fixed effects:
        correct ~ C(problem_id) + step_std + shift
    using cluster-robust SEs clustered on problem_id.

  • Report a one-row-per-domain table:
        domain, N, share_shift, acc_shift,
        delta_pp (acc_shift - acc_noshift in percentage points),
        AME (avg marginal effect of shift), p (p-value for shift)

Also saves CSV at: <out_dir>/aha_regression__<dataset>__<model>.csv

Notes
-----
- "Shift" = GPT/LLM-detected reasoning shift with domain-aware gating to avoid
  all-zeros on Crosswords (mirrors your figure code’s gating behavior).
- Inputs are pass1-style JSONL files.
- Step collection is hard-capped at step ≤ 1000 regardless of --max_step.
- Two GPT labeling modes:
    --gpt_mode=canonical : ["change_way_of_thinking","shift_in_reasoning_v1"]
    --gpt_mode=broad     : adds ["shift_llm","shift_gpt","pivot_llm","rechecked"]
- Use --no_gpt_subset_native to disable native-cue gating subset (keeps broader marks).
"""

import os
import re
import json
import argparse
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ---------- File / step parsing ----------

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
    out.sort()
    return out

# ---------- Small utils ----------

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
    try:
        return float(x)
    except Exception:
        return None

def get_problem_id(rec: Dict[str, Any]) -> Optional[str]:
    """
    Heuristics to extract a stable per-problem identifier.
    Tries several common keys seen in your JSONLs.
    """
    for k in ("problem_id","example_id","id","question","problem","clue","title","uid"):
        v = rec.get(k)
        if v is not None and not isinstance(v, (dict, list)):
            return str(v)
    v = rec.get("sample_idx")
    return None if v is None else f"sample_{v}"

# ---------- Domain-aware gating (mirrors your figure script) ----------

def _any_keys_true(p1: Dict[str, Any], rec: Dict[str, Any], keys: List[str]) -> int:
    for k in keys:
        v = p1.get(k, rec.get(k, None))
        if v is None: continue
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

def _make_carpark_success_fn(op: str, thr: float):
    def _cmp(val: Any) -> Optional[int]:
        x = coerce_float(val)
        if x is None:
            return None
        if op == "gt": return int(x >  thr)
        if op == "ge": return int(x >= thr)
        if op == "eq": return int(x == thr)
        return int(x > thr)  # safe default
    return _cmp

# ---------- Load & build dataframe ----------

def load_rows(files_by_domain: Dict[str, List[str]],
              gpt_keys: List[str],
              gpt_subset_native: bool,
              min_step: Optional[int],
              max_step: Optional[int],
              carpark_success_fn) -> pd.DataFrame:
    rows = []
    for dom, files in files_by_domain.items():
        dom_lower = str(dom).lower()
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
                    if not isinstance(p1, dict): continue

                    # step
                    step = rec.get("step", step_from_name if step_from_name is not None else None)
                    if step is None: continue
                    try:
                        step = int(step)
                    except Exception:
                        continue

                    if min_step is not None and step < min_step: continue
                    if max_step is not None and step > max_step: continue

                    # success → correct
                    if dom_lower.startswith("carpark"):
                        sr = _extract_soft_reward(rec, p1)
                        correct = carpark_success_fn(sr)  # 1/0 or None
                    else:
                        correct = coerce_bool(p1.get("is_correct_pred"))
                    if correct is None: continue

                    # problem id (for FE and clustering)
                    pid = get_problem_id(rec)
                    if pid is None:
                        continue

                    # shift (LLM, gated)
                    shift = aha_gpt_for_rec(p1, rec, gpt_subset_native, gpt_keys, dom)

                    rows.append({
                        "domain": str(dom),
                        "problem_id": pid,
                        "step": int(step),
                        "correct": int(correct),
                        "shift": int(shift),
                    })
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No rows found. Check roots/--split / step filters.")
    return df

# ---------- Per-domain regression & summary ----------

def ame_for_shift(result, sub: pd.DataFrame) -> float:
    """
    Average marginal effect of 'shift' on P(correct=1).
    For each row: predict with shift=1 and shift=0 (holding C(problem_id) and step_std fixed),
    then average the differences.
    """
    data1 = sub.copy()
    data0 = sub.copy()
    data1["shift"] = 1
    data0["shift"] = 0
    p1 = result.predict(data1)
    p0 = result.predict(data0)
    return float(np.mean(p1 - p0))

def summarize_domain(sub: pd.DataFrame, domain_name: str) -> Dict[str, Any]:
    # Groupwise stats
    N = int(len(sub))
    share_shift = float(sub["shift"].mean()) if N > 0 else np.nan
    acc_shift = float(sub.loc[sub["shift"] == 1, "correct"].mean()) if (sub["shift"] == 1).any() else np.nan
    acc_noshift = float(sub.loc[sub["shift"] == 0, "correct"].mean()) if (sub["shift"] == 0).any() else np.nan
    delta_pp = (acc_shift - acc_noshift) * 100.0 if np.isfinite(acc_shift) and np.isfinite(acc_noshift) else np.nan

    # GLM with FE and cluster-robust SEs
    model = smf.glm(
        formula="correct ~ C(problem_id) + step_std + shift",
        data=sub,
        family=sm.families.Binomial(link=sm.families.links.logit()),
    )
    # Be resilient to occasional separation / convergence hiccups
    try:
        res = model.fit(cov_type="cluster", cov_kwds={"groups": sub["problem_id"]}, maxiter=200)
    except Exception:
        # fallback: default fit then robustify
        res = model.fit()
        res = res.get_robustcov_results(cov_type="cluster", groups=sub["problem_id"])

    ame = ame_for_shift(res, sub)
    p_shift = float(res.pvalues.get("shift", np.nan))

    return {
        "domain": domain_name,
        "N": N,
        "share_shift": share_shift,
        "acc_shift": acc_shift,
        "delta_pp": delta_pp,
        "AME": ame,
        "p": p_shift,
    }

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_crossword", type=str, default=None)
    ap.add_argument("--root_math", type=str, default=None)
    # NEW: second Math dataset/model
    ap.add_argument("--root_math2", type=str, default=None,
                    help="Path to a second Math results folder (e.g., different model).")
    ap.add_argument("--label_math2", type=str, default="Math2",
                    help="Display label for the second Math row (default: 'Math2').")
    # NEW: Carpark with soft reward success
    ap.add_argument("--root_carpark", type=str, default=None)

    ap.add_argument("results_root", nargs="?", default=None,
                    help="Fallback single root if domain-specific roots are not provided.")
    ap.add_argument("--split", default=None, help="Substring filter for file names, e.g., 'test'")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--dataset_name", default="MIXED")
    ap.add_argument("--model_name", default="Qwen2.5-1.5B")

    # GPT label policy
    ap.add_argument("--gpt_mode", choices=["canonical","broad"], default="canonical")
    ap.add_argument("--no_gpt_subset_native", action="store_true")

    # Step filters (hard cap at 1000 is enforced below)
    ap.add_argument("--min_step", type=int, default=None)
    ap.add_argument("--max_step", type=int, default=None,
                    help="Upper bound on step. Note: the script imposes a hard cap of 1000.")

    # Carpark success policy
    ap.add_argument("--carpark_success_op", choices=["gt","ge","eq"], default="gt",
                    help="OP for Carpark success against soft_reward.")
    ap.add_argument("--carpark_soft_threshold", type=float, default=0.0,
                    help="Threshold for Carpark success (default 0.0).")

    args = ap.parse_args()

    files_by_domain: Dict[str, List[str]] = {}
    first_root = None
    if args.root_crossword:
        files_by_domain["Crossword"] = scan_files(args.root_crossword, args.split)
        first_root = first_root or args.root_crossword
    if args.root_math:
        files_by_domain["Math"] = scan_files(args.root_math, args.split)
        first_root = first_root or args.root_math
    if args.root_math2:
        label2 = (args.label_math2 or "Math2").strip() or "Math2"
        # Avoid accidental label collision
        name = label2
        i = 2
        while name in files_by_domain:
            name = f"{label2}-{i}"
            i += 1
        files_by_domain[name] = scan_files(args.root_math2, args.split)
        first_root = first_root or args.root_math2
    if args.root_carpark:
        files_by_domain["Carpark"] = scan_files(args.root_carpark, args.split)
        first_root = first_root or args.root_carpark

    if not files_by_domain:
        if not args.results_root:
            raise SystemExit("Provide --root_crossword/--root_math/--root_math2/--root_carpark, or a fallback results_root.")
        files_by_domain["All"] = scan_files(args.results_root, args.split)
        first_root = args.results_root

    total_files = sum(len(v) for v in files_by_domain.values())
    if total_files == 0:
        raise SystemExit("No JSONL files found. Check roots/--split.")

    out_dir = args.out_dir or os.path.join(first_root, "aha_regression")
    os.makedirs(out_dir, exist_ok=True)

    # GPT keys & policy
    gpt_subset_native = not args.no_gpt_subset_native
    gpt_keys = (["change_way_of_thinking", "shift_in_reasoning_v1"] if args.gpt_mode == "canonical"
                else ["change_way_of_thinking", "shift_in_reasoning_v1",
                      "shift_llm", "shift_gpt", "pivot_llm", "rechecked"])

    # Hard cap at step ≤ 1000
    HARD_MAX_STEP = 1000
    effective_max_step = HARD_MAX_STEP if args.max_step is None else min(args.max_step, HARD_MAX_STEP)
    if args.max_step is None or args.max_step > HARD_MAX_STEP:
        print(f"[info] Capping max_step to {effective_max_step} (hard cap = {HARD_MAX_STEP}).")

    # Carpark comparator
    carpark_success_fn = _make_carpark_success_fn(args.carpark_success_op, args.carpark_soft_threshold)

    # Load rows
    df = load_rows(files_by_domain, gpt_keys, gpt_subset_native,
                   min_step=args.min_step, max_step=effective_max_step,
                   carpark_success_fn=carpark_success_fn)

    # Standardize step per domain
    df["step_std"] = df.groupby("domain")["step"].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-12))

    # Per-domain regression & summary (for ALL present domains)
    rows = []
    for dom in sorted(df["domain"].unique(), key=str):
        sub = df.loc[df["domain"] == dom, ["correct","shift","step_std","problem_id"]].copy()
        # Guard against degenerate groups
        if sub.empty or sub["correct"].nunique() < 2 or sub["shift"].nunique() < 2:
            print(f"[warn] Domain {dom}: insufficient variation to fit model; skipping.")
            continue
        rows.append(summarize_domain(sub, dom))

    if not rows:
        raise SystemExit("No eligible domains to summarize.")

    table = pd.DataFrame(rows)[["domain","N","share_shift","acc_shift","delta_pp","AME","p"]]

    # Save CSV
    slug = f"{args.dataset_name}__{args.model_name}".replace(" ", "_")
    out_csv = os.path.join(out_dir, f"aha_regression__{slug}.csv")
    table.to_csv(out_csv, index=False)

    # Pretty print (rounded to match paper style)
    disp = table.copy()
    disp["N"] = disp["N"].astype(int)
    disp["share_shift"] = disp["share_shift"].map(lambda x: f"{x:.4f}")
    disp["acc_shift"]   = disp["acc_shift"].map(lambda x: f"{x:.4f}" if np.isfinite(x) else "--")
    disp["delta_pp"]    = disp["delta_pp"].map(lambda x: f"{x:+.2f}" if np.isfinite(x) else "--")
    disp["AME"]         = disp["AME"].map(lambda x: f"{x:+.4f}" if np.isfinite(x) else "--")
    def fmt_p(v: float) -> str:
        if not np.isfinite(v): return "--"
        if v < 1e-5: return f"{v:.2e}"
        return f"{v:.5f}".rstrip("0").rstrip(".")
    disp["p"] = disp["p"].map(fmt_p)

    with pd.option_context("display.max_columns", None, "display.width", 120):
        print("\nEffects of reasoning shift on success (per domain)")
        print("Model: correct ~ C(problem) + step_std + shift  [GLM Binomial(logit), cluster-robust SEs by problem]")
        print(disp.to_string(index=False))
        print(f"\nSaved CSV -> {out_csv}")

if __name__ == "__main__":
    main()
