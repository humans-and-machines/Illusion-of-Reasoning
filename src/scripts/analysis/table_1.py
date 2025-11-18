#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aha Conditionals (Domain-level) — robust to Llama-style rows
------------------------------------------------------------
For each provided folder/domain (e.g., --root_crossword, --root_math, --root_carpark),
compute:

  • P(success | shift) and P(success | no shift)
    - Crosswords/Math: success = accuracy (is_correct_pred)
    - Carpark: success = (soft_reward OP threshold), default OP='>' and threshold=0.0

Key robustness fixes
--------------------
• Read shift/cue fields from pass1 OR top level (handles dumps where fields moved).
• Optional: in --gpt_mode=broad, treat non-empty marker lists
  (shift_markers_v1, _shift_prefilter_markers) as SHIFT=1 even if booleans are false.
• Optional: --allow_judge_cues_non_xword makes judge-time cues open the gate on non-crossword.
• Success fallback: if pass1.is_correct_pred is missing, use top-level is_correct_pred.

Output
------
CSV at <out_dir>/aha_conditionals__<dataset>__<model>.csv with columns:
    domain, n_total,
    k_correct_shift, n_shift, p_correct_given_shift,
    k_correct_noshift, n_noshift, p_correct_given_noshift,
    shift_rate
"""

import os
import re
import json
import argparse
from typing import Optional, List, Dict, Any, Callable
import numpy as np
import pandas as pd

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


def coerce_bool(x) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, np.integer)):
        return int(bool(x))
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("1", "true", "t", "yes", "y"):
            return 1
        if s in ("0", "false", "f", "no", "n"):
            return 0
    try:
        # lists/dicts: non-empty -> True
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


# ---------- helpers to read from pass1 OR top-level ----------
def both_get(p1: Dict[str, Any], rec: Dict[str, Any], key: str, default=None):
    v = p1.get(key, None)
    return v if v is not None else rec.get(key, default)


# ----------------- Domain-aware gating -----------------

def _any_keys_true(p1: Dict[str, Any], rec: Dict[str, Any], keys: List[str]) -> int:
    """Any of the provided keys truthy (bool True, 1, or non-empty list/dict/string)."""
    for k in keys:
        v = both_get(p1, rec, k, None)
        if v is None:
            continue
        out = coerce_bool(v)
        if out is not None and out == 1:
            return 1
    return 0


def _cue_gate_for_llm(
    p1: Dict[str, Any],
    rec: Dict[str, Any],
    domain: Optional[str],
    allow_judge_non_xword: bool = False,
) -> int:
    has_reconsider = coerce_bool(both_get(p1, rec, "has_reconsider_cue")) == 1
    rec_marks = both_get(p1, rec, "reconsider_markers") or []
    injected = ("injected_cue" in rec_marks)
    reconsider_ok = has_reconsider and not injected

    prefilter_cues = both_get(p1, rec, "_shift_prefilter_markers") or []
    judge_cues = both_get(p1, rec, "shift_markers_v1") or []

    if str(domain).lower() == "crossword" or allow_judge_non_xword:
        return int(reconsider_ok or bool(prefilter_cues) or bool(judge_cues))
    else:
        return int(reconsider_ok or bool(prefilter_cues))


def _aha_gpt_for_rec(
    p1: Dict[str, Any],
    rec: Dict[str, Any],
    *,
    gpt_subset_native: bool,
    gpt_keys: List[str],
    domain: Optional[str],
    allow_judge_non_xword: bool,
    broad_counts_marker_lists: bool,
) -> int:
    """
    Shift-positive if any gpt_keys are truthy. In 'broad' mode (broad_counts_marker_lists=True),
    treat non-empty 'shift_markers_v1' / '_shift_prefilter_markers' as positive too.
    """
    raw = _any_keys_true(p1, rec, gpt_keys)

    if broad_counts_marker_lists and raw == 0:
        # Consider marker lists as positive signals in broad mode
        if coerce_bool(both_get(p1, rec, "shift_markers_v1")) == 1:
            raw = 1
        elif coerce_bool(both_get(p1, rec, "_shift_prefilter_markers")) == 1:
            raw = 1

    if not gpt_subset_native:
        return int(raw)

    gate = _cue_gate_for_llm(p1, rec, domain, allow_judge_non_xword)
    return int(raw & gate)


# ----------------- Core loading -----------------

def _make_carpark_success_fn(op: str, thr: float) -> Callable[[Any], Optional[int]]:
    def _cmp(val: Any) -> Optional[int]:
        x = coerce_float(val)
        if x is None:
            return None
        if op == "gt":
            return int(x > thr)
        if op == "ge":
            return int(x >= thr)
        if op == "eq":
            return int(x == thr)
        # default safe fallback
        return int(x > thr)
    return _cmp


def _extract_soft_reward(rec: Dict[str, Any], p1: Dict[str, Any]) -> Optional[float]:
    return coerce_float(both_get(p1, rec, "soft_reward"))


def load_rows(
    files_by_domain: Dict[str, List[str]],
    gpt_keys: List[str],
    gpt_subset_native: bool,
    min_step: Optional[int],
    max_step: Optional[int],
    carpark_success_fn: Callable[[Any], Optional[int]],
    *,
    allow_judge_cues_non_xword: bool,
    broad_counts_marker_lists: bool,
    debug: bool = False,
) -> pd.DataFrame:
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
                        p1 = {}

                    step = rec.get("step", step_from_name if step_from_name is not None else None)
                    if step is None:
                        continue
                    try:
                        step = int(step)
                    except Exception:
                        continue

                    if min_step is not None and step < min_step:
                        continue
                    if max_step is not None and step > max_step:
                        continue

                    # --- Domain-specific "success" ---
                    if dom_lower == "carpark":
                        sr = _extract_soft_reward(rec, p1)
                        correct = carpark_success_fn(sr)  # 1/0 or None
                    else:
                        correct = coerce_bool(both_get(p1, rec, "is_correct_pred"))

                    if correct is None:
                        continue

                    aha_llm = _aha_gpt_for_rec(
                        p1, rec,
                        gpt_subset_native=gpt_subset_native,
                        gpt_keys=gpt_keys,
                        domain=dom,
                        allow_judge_non_xword=allow_judge_cues_non_xword,
                        broad_counts_marker_lists=broad_counts_marker_lists,
                    )

                    rows.append(
                        {
                            "domain": str(dom),
                            "step": int(step),
                            "correct": int(correct),
                            "shift": int(aha_llm),
                        }
                    )
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No rows found. Check roots and/or --split / step filters.")
    if debug:
        for dom, sub in df.groupby("domain"):
            print(f"[debug] {dom:>12}: n={len(sub):6d}, shifts={int(sub['shift'].sum()):6d}")
    return df


# ----------------- Aggregation -----------------

def agg_domain_conditionals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per domain, preserving the input order.
    """
    out_rows = []
    domain_order = list(dict.fromkeys(df["domain"].tolist()))
    for dom in domain_order:
        sub = df[df["domain"] == dom]
        m_shift = (sub["shift"] == 1)
        m_noshift = (sub["shift"] == 0)

        n_shift = int(m_shift.sum())
        n_noshift = int(m_noshift.sum())
        n_total = int(len(sub))

        k_shift = int(sub.loc[m_shift, "correct"].sum()) if n_shift > 0 else 0
        k_noshift = int(sub.loc[m_noshift, "correct"].sum()) if n_noshift > 0 else 0

        p_c_given_shift = (k_shift / n_shift) if n_shift > 0 else np.nan
        p_c_given_noshift = (k_noshift / n_noshift) if n_noshift > 0 else np.nan
        shift_rate = (n_shift / n_total) if n_total > 0 else np.nan

        out_rows.append(
            {
                "domain": dom,
                "n_total": n_total,
                "k_correct_shift": k_shift,
                "n_shift": n_shift,
                "p_correct_given_shift": p_c_given_shift,
                "k_correct_noshift": k_noshift,
                "n_noshift": n_noshift,
                "p_correct_given_noshift": p_c_given_noshift,
                "shift_rate": shift_rate,
            }
        )
    return pd.DataFrame(out_rows)


# ----------------- CLI -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_crossword", type=str, default=None)
    ap.add_argument("--root_math", type=str, default=None)

    # Second & third Math roots (e.g., different models)
    ap.add_argument("--root_math2", type=str, default=None,
                    help="Path to a second Math results folder (e.g., Qwen-7B).")
    ap.add_argument("--label_math2", type=str, default="Math (Qwen-7B)",
                    help="Row label for the second Math root.")

    ap.add_argument("--root_math3", type=str, default=None,
                    help="Path to a third Math results folder (e.g., Llama-8B).")
    ap.add_argument("--label_math3", type=str, default="Math (Llama-8B)",
                    help="Row label for the third Math root.")

    ap.add_argument("--root_carpark", type=str, default=None)

    ap.add_argument(
        "results_root",
        nargs="?",
        default=None,
        help="Fallback single root if domain-specific roots are not provided.",
    )
    ap.add_argument("--split", default=None, help="Substring filter for file names, e.g., 'test'")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--dataset_name", default="MIXED")
    ap.add_argument("--model_name", default="MIXED_MODELS")

    # GPT label policy
    ap.add_argument("--gpt_mode", choices=["canonical", "broad"], default="canonical",
                    help="canonical: boolean keys only; broad: also count marker lists as positive.")
    ap.add_argument("--no_gpt_subset_native", action="store_true",
                    help="If set, DO NOT apply cue gating ('native subset').")

    # Allow judge-time cues for non-crossword (off by default to preserve prior behavior)
    ap.add_argument("--allow_judge_cues_non_xword", action="store_true",
                    help="Open the cue gate for judge-time cues in non-crossword domains.")

    # Step filters (hard cap at 1000 is enforced below)
    ap.add_argument("--min_step", type=int, default=None)
    ap.add_argument("--max_step", type=int, default=None,
                    help="Upper bound on step. Note: hard cap of 1000 is applied.")

    # --- Carpark success policy ---
    ap.add_argument("--carpark_soft_threshold", type=float, default=0.1,
                    help="Carpark success if soft_reward OP threshold (default OP='gt').")
    ap.add_argument("--carpark_success_op", choices=["gt", "ge", "eq"], default="gt",
                    help="Comparison operator for Carpark success against soft_reward.")

    ap.add_argument("--debug", action="store_true", help="Print domain counts after loading.")

    args = ap.parse_args()

    # Domain roots -> keep row order in the order we add them
    files_by_domain: Dict[str, List[str]] = {}
    first_root = None
    if args.root_crossword:
        files_by_domain["Crossword"] = scan_files(args.root_crossword, args.split)
        first_root = first_root or args.root_crossword
    if args.root_math:
        files_by_domain["Math"] = scan_files(args.root_math, args.split)
        first_root = first_root or args.root_math
    if args.root_math2:
        label2 = (args.label_math2 or "Math2").strip()
        if label2 in files_by_domain:
            base, i = label2, 2
            while label2 in files_by_domain:
                label2 = f"{base}-{i}"; i += 1
        files_by_domain[label2] = scan_files(args.root_math2, args.split)
        first_root = first_root or args.root_math2
    if args.root_math3:
        label3 = (args.label_math3 or "Math3").strip()
        if label3 in files_by_domain:
            base, i = label3, 2
            while label3 in files_by_domain:
                label3 = f"{base}-{i}"; i += 1
        files_by_domain[label3] = scan_files(args.root_math3, args.split)
        first_root = first_root or args.root_math3
    if args.root_carpark:
        files_by_domain["Carpark"] = scan_files(args.root_carpark, args.split)
        first_root = first_root or args.root_carpark

    if not files_by_domain:
        if not args.results_root:
            raise SystemExit("Provide --root_crossword/--root_math/--root_math2/--root_math3/--root_carpark, or a fallback results_root.")
        files_by_domain["All"] = scan_files(args.results_root, args.split)
        first_root = args.results_root

    total_files = sum(len(v) for v in files_by_domain.values())
    if total_files == 0:
        raise SystemExit("No JSONL files found. Check roots/--split.")

    out_dir = args.out_dir or os.path.join(first_root, "aha_conditionals")
    os.makedirs(out_dir, exist_ok=True)

    # GPT keys & policy
    gpt_subset_native = not args.no_gpt_subset_native
    if args.gpt_mode == "canonical":
        gpt_keys = ["change_way_of_thinking", "shift_in_reasoning_v1"]
        broad_counts_marker_lists = False
    else:
        # Broad: accept additional boolean keys AND also count non-empty marker lists as positive
        gpt_keys = [
            "change_way_of_thinking",
            "shift_in_reasoning_v1",
            "shift_llm",
            "shift_gpt",
            "pivot_llm",
            "rechecked",
        ]
        broad_counts_marker_lists = True

    # Enforce per-environment hard cap at step ≤ 1000
    HARD_MAX_STEP = 1000
    effective_max_step = HARD_MAX_STEP if args.max_step is None else min(args.max_step, HARD_MAX_STEP)
    if args.max_step is None or args.max_step > HARD_MAX_STEP:
        print(f"[info] Capping max_step to {effective_max_step} (hard cap = {HARD_MAX_STEP}).")

    # Carpark comparator
    carpark_success_fn = _make_carpark_success_fn(args.carpark_success_op, args.carpark_soft_threshold)

    # Load rows
    df = load_rows(
        files_by_domain,
        gpt_keys,
        gpt_subset_native,
        min_step=args.min_step,
        max_step=effective_max_step,
        carpark_success_fn=carpark_success_fn,
        allow_judge_cues_non_xword=args.allow_judge_cues_non_xword,
        broad_counts_marker_lists=broad_counts_marker_lists,
        debug=args.debug,
    )

    # Aggregate domain-level conditionals (preserve input order)
    table = agg_domain_conditionals(df)

    # Save + print
    slug = f"{args.dataset_name}__{args.model_name}".replace(" ", "_")
    out_csv = os.path.join(out_dir, f"aha_conditionals__{slug}.csv")
    table.to_csv(out_csv, index=False)

    with pd.option_context("display.max_columns", None, "display.width", 120):
        print("\nAha conditionals (domain-level):")
        print(table.to_string(index=False))
        print(f"\nSaved CSV -> {out_csv}")


if __name__ == "__main__":
    main()
