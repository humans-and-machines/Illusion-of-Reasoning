#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Formal Aha! JSON Exporter (with raw JSONL rows)
===============================================

What it does
------------
Scans one or more results roots, aggregates problem×step stats, marks Formal Aha!
events using thresholds (δ1, δ2, optional δ3), and exports all detected events to
JSON and JSONL.

NEW: For every event, we also include:
  • "raw_record": the fully parsed JSON object from the source JSONL line
  • "raw_jsonl" : the exact original JSONL line text

Default step range
------------------
By default this script only considers steps up to and including 1000
(via --max_step=1000). You can override on the CLI.

CLI examples
------------
  # Two roots (domain-aware), default thresholds:
  python formal_aha_export.py \
      --root_crossword /path/to/crossword/results \
      --root_math /path/to/math/results \
      --dataset_name MIXED --model_name Qwen2.5-1.5B

  # Single generic root, custom thresholds and min prior steps:
  python formal_aha_export.py \
      /path/to/all_results \
      --delta1 0.25 --delta2 0.25 --delta3 0.05 --min_prior_steps 3
"""

import os, re, json, argparse
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd

# ----------------- Utils -----------------

STEP_PAT = re.compile(r"step(\d+)", re.I)

def slugify(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")

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
        if s in ("1","true","t","yes","y"): return 1
        if s in ("0","false","f","no","n"): return 0
    return int(bool(x))

# ----------------- Label helpers -----------------

def _any_keys_true(p1: Dict[str, Any], rec: Dict[str, Any], keys: List[str]) -> int:
    for k in keys:
        v = p1.get(k, rec.get(k, None))
        if v is None:
            continue
        out = coerce_bool(v)
        if out is not None and out == 1:
            return 1
    return 0

def _aha_native(p1: Dict[str, Any]) -> int:
    aha_raw = coerce_bool(p1.get("has_reconsider_cue"))
    markers = p1.get("reconsider_markers") or []
    if isinstance(markers, list) and ("injected_cue" in markers):
        return 0
    return 0 if aha_raw is None else int(aha_raw)

def _cue_gate_for_llm(p1: Dict[str, Any], domain: Optional[str]) -> int:
    """Domain-aware gate so Crossword doesn't collapse to all zeros."""
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

def _aha_gpt_for_rec(p1: Dict[str, Any], rec: Dict[str, Any],
                     gpt_subset_native: bool, gpt_keys: List[str], domain: Optional[str]) -> int:
    gpt_raw = _any_keys_true(p1, rec, gpt_keys)
    if not gpt_subset_native:
        return int(gpt_raw)
    gate = _cue_gate_for_llm(p1, domain)
    return int(gpt_raw & gate)

# ----------------- Load samples (domain-aware) -----------------

def load_pass1_samples_multi(files_by_domain: Dict[str, List[str]],
                             gpt_keys: List[str],
                             gpt_subset_native: bool) -> pd.DataFrame:
    rows = []
    for dom, files in files_by_domain.items():
        if not files:
            continue
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
                    if not p1:
                        continue

                    step = rec.get("step", step_from_name if step_from_name is not None else None)
                    if step is None:
                        continue

                    prob = rec.get("problem") or rec.get("clue") or rec.get("row_key")
                    if prob is None:
                        di = rec.get("dataset_index")
                        prob = f"idx:{di}" if di is not None else "unknown"

                    correct = coerce_bool(p1.get("is_correct_pred"))
                    if correct is None:
                        continue

                    native = _aha_native(p1)
                    gpt_eff = _aha_gpt_for_rec(p1, rec, gpt_subset_native, gpt_keys, dom)

                    rows.append({
                        "domain": str(dom),
                        "step": int(step),
                        "problem": str(prob),
                        "aha_native": int(native),
                        "aha_gpt": int(gpt_eff),
                        "correct": int(correct),
                    })
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No PASS-1 rows found across provided roots. Check paths/--split.")
    return df

# ------------- Problem-step aggregation (domain-aware) -------------

def build_problem_step(df: pd.DataFrame) -> pd.DataFrame:
    group_keys = ["domain", "step", "problem"]

    base = (
        df.groupby(group_keys, as_index=False)
          .agg(
              n_samples=("correct", "size"),
              freq_correct=("correct", "mean"),
              aha_any_gpt=("aha_gpt", "max"),
              aha_rate_gpt=("aha_gpt", "mean"),
              aha_any_native=("aha_native", "max"),
              aha_rate_native=("aha_native", "mean"),
          )
          .sort_values(group_keys)
          .reset_index(drop=True)
    )

    def _pcs_row(g: pd.DataFrame) -> pd.Series:
        m = (g["aha_gpt"] == 1)
        if m.any():
            return pd.Series({"p_correct_given_shift": float(g.loc[m, "correct"].mean())})
        return pd.Series({"p_correct_given_shift": np.nan})

    pcs = df.groupby(group_keys).apply(_pcs_row).reset_index()
    ps = base.merge(pcs, on=group_keys, how="left")

    for c in ("n_samples", "aha_any_gpt", "aha_any_native"):
        ps[c] = ps[c].astype(int)
    for c in ("freq_correct", "aha_rate_gpt", "aha_rate_native", "p_correct_given_shift"):
        ps[c] = ps[c].astype(float)

    return ps

# ------------- Formal marking (δ1, δ2, optional δ3 gain-at-shift) -------------

def mark_formal_pairs(ps: pd.DataFrame,
                      delta1: float = 0.20,
                      delta2: float = 0.20,
                      min_prior_steps: int = 2,
                      delta3: Optional[float] = 0.0) -> pd.DataFrame:
    """
    A step is 'Formal Aha!' for a given problem if:
      1) max prior freq_correct < δ1   (no prior success),
      2) max prior aha_rate_gpt < δ2   (no prior frequent shifts),
      3) aha_any_gpt == 1 at this step (a shift occurs now),
      4) if δ3 is not None: (p_correct_given_shift - freq_correct) > δ3 at this step.
    """
    need = {"step","problem","freq_correct","aha_rate_gpt","aha_any_gpt","p_correct_given_shift"}
    if not need.issubset(ps.columns):
        missing = need - set(ps.columns)
        raise ValueError(f"mark_formal_pairs: missing columns {missing}")

    group_cols = ["domain", "problem"]
    ps = ps.sort_values(group_cols + ["step"]).reset_index(drop=True).copy()
    flags = np.zeros(len(ps), dtype=int)

    idx = 0
    for _, sub in ps.groupby(group_cols, sort=False):
        sub = sub.sort_values("step")
        freq      = sub["freq_correct"].to_numpy(float)
        rate      = sub["aha_rate_gpt"].to_numpy(float)
        shift_now = sub["aha_any_gpt"].to_numpy(int)
        p_plus    = sub["p_correct_given_shift"].to_numpy(float)

        for j in range(len(sub)):
            if j < min_prior_steps:
                flags[idx] = 0
            else:
                prior_fail   = (float(np.max(freq[:j])) < delta1)
                prior_stable = (float(np.max(rate[:j])) < delta2)
                shift_ok     = (shift_now[j] == 1)
                gain_ok      = True
                if delta3 is not None:
                    if np.isfinite(p_plus[j]):
                        gain_ok = (p_plus[j] - freq[j]) > delta3
                    else:
                        gain_ok = False
                flags[idx] = int(prior_fail and prior_stable and shift_ok and gain_ok)
            idx += 1

    ps["aha_formal"] = flags
    return ps

# ------------- Minimal pass1 text/answer extraction for JSON export -------------

_TAGS_ANSWER = re.compile(r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>", re.IGNORECASE | re.DOTALL)

def _extract_pass_text(pass_dict: Dict[str, Any]) -> Optional[str]:
    if not isinstance(pass_dict, dict): return None
    for k in ("output","text","content","response","model_output","assistant_text","prev_output"):
        v = pass_dict.get(k)
        if isinstance(v, str) and v.strip(): return v
    msgs = pass_dict.get("messages")
    if isinstance(msgs, list):
        for m in reversed(msgs):
            if isinstance(m, dict) and str(m.get("role","")).lower() == "assistant":
                c = m.get("content") or m.get("text")
                if isinstance(c, str) and c.strip():
                    return c
    ch = pass_dict.get("choices")
    if isinstance(ch, list) and ch and isinstance(ch[0], dict):
        msg = ch[0].get("message")
        if isinstance(msg, dict):
            c = msg.get("content") or msg.get("text")
            if isinstance(c, str) and c.strip():
                return c
    return None

def _extract_pass_answer(pass_dict: Dict[str, Any]) -> Optional[str]:
    if not isinstance(pass_dict, dict): return None
    for k in ("final_answer","answer","short_answer","pred","prediction","pred_text","parsed_answer","extracted_answer"):
        v = pass_dict.get(k)
        if isinstance(v, str) and v.strip():
            m = _TAGS_ANSWER.search(v)
            return (m.group(1).strip() if m else v.strip())
    out = pass_dict.get("output")
    if isinstance(out, str) and out.strip():
        m = _TAGS_ANSWER.findall(out)
        if m:
            cands = [seg.strip() for seg in m if seg and seg.strip()]
            if cands: return cands[-1]
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        if lines and len(lines[-1]) <= 200: return lines[-1]
    txt = _extract_pass_text(pass_dict)
    if isinstance(txt, str) and txt.strip():
        m = _TAGS_ANSWER.search(txt)
        if m: return m.group(1).strip()
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        if lines and len(lines[-1]) <= 200: return lines[-1]
    return None

# ------------- Export all Formal Aha events (domain-aware, WITH RAW ROW) -------------

def export_formal_aha_json_with_text(ps: pd.DataFrame,
                                     files_by_domain: Dict[str, List[str]],
                                     dataset: str, model: str,
                                     delta1: float, delta2: float, delta3: Optional[float],
                                     min_prior_steps: int,
                                     gpt_keys: List[str], gpt_subset_native: bool,
                                     out_dir: str, slug: str,
                                     max_chars: int = 4000) -> Tuple[str, str, int]:
    """
    For each formal event (domain, problem, step), write an entry that includes:
      - computed stats and thresholds
      - best-effort extracted pass1 answer text
      - raw_record: full parsed JSON row
      - raw_jsonl : exact line text
    """
    needed = {"step","problem","domain","aha_formal","n_samples","freq_correct","aha_rate_gpt","p_correct_given_shift"}
    if not needed.issubset(ps.columns):
        missing = needed - set(ps.columns)
        raise ValueError(f"export_formal_aha_json_with_text: missing columns: {missing}")

    targets = {(str(r["domain"]), str(r["problem"]), int(r["step"]))
               for _, r in ps.loc[ps["aha_formal"]==1, ["domain","problem","step"]].iterrows()}
    ps_key = {(str(r["domain"]), str(r["problem"]), int(r["step"])): r for _, r in ps.iterrows()}

    json_path  = os.path.join(out_dir, f"formal_aha_events__{slug}.json")
    jsonl_path = os.path.join(out_dir, f"formal_aha_events__{slug}.jsonl")

    if not targets:
        os.makedirs(out_dir, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f: json.dump([], f)
        with open(jsonl_path, "w", encoding="utf-8") as f: pass
        return json_path, jsonl_path, 0

    events: List[Dict[str, Any]] = []
    remaining = set(targets)

    for dom, files in files_by_domain.items():
        if not files:
            continue
        for path in files:
            if not remaining:
                break
            step_from_name = nat_step_from_path(path)
            with open(path, "r", encoding="utf-8") as f:
                for ln in f:
                    if not remaining:
                        break
                    raw_line = ln.rstrip("\n")
                    if not raw_line.strip():
                        continue
                    try:
                        rec = json.loads(raw_line)
                    except Exception:
                        continue

                    step = rec.get("step", step_from_name if step_from_name is not None else None)
                    if step is None:
                        continue
                    step = int(step)

                    prob_raw = rec.get("problem") or rec.get("clue") or rec.get("row_key")
                    if prob_raw is None:
                        di = rec.get("dataset_index")
                        prob_raw = f"idx:{di}" if di is not None else "unknown"
                    prob = str(prob_raw)

                    key = (dom, prob, step)
                    if key not in remaining:
                        continue

                    p1 = rec.get("pass1") or {}
                    if not isinstance(p1, dict):
                        continue

                    aha_gpt_now = _aha_gpt_for_rec(p1, rec, gpt_subset_native, gpt_keys, dom)
                    if aha_gpt_now != 1:
                        continue  # must actually shift now

                    r = ps_key.get(key)
                    if r is None:
                        continue

                    question = prob
                    answer = _extract_pass_answer(p1)
                    if answer and max_chars and len(answer) > max_chars:
                        answer = answer[:max_chars] + " …[truncated]"

                    p_plus = float(r["p_correct_given_shift"]) if np.isfinite(r["p_correct_given_shift"]) else None
                    p_base = float(r["freq_correct"])
                    delta_gain = (p_plus - p_base) if (p_plus is not None) else None

                    events.append({
                        "domain": dom,
                        "dataset": dataset,
                        "model": model,
                        "problem": prob,
                        "step": step,
                        "n_samples": int(r["n_samples"]),
                        "p_correct": p_base,
                        "p_shift_rate": float(r["aha_rate_gpt"]),
                        "p_correct_given_shift": (float(p_plus) if p_plus is not None else None),
                        "delta_gain_at_shift": (float(delta_gain) if delta_gain is not None else None),
                        "thresholds": {
                            "delta1": float(delta1),
                            "delta2": float(delta2),
                            "delta3": (float(delta3) if delta3 is not None else None),
                            "min_prior_steps": int(min_prior_steps),
                        },
                        "question": question,
                        "answer": answer,
                        # NEW: include the entire source row
                        "raw_record": rec,
                        "raw_jsonl": raw_line
                    })
                    remaining.discard(key)

    # For any still-remaining targets (couldn't pull text/row), add minimal entries
    for dom, prob, step in sorted(remaining):
        r = ps_key[(dom, prob, step)]
        p_plus = float(r["p_correct_given_shift"]) if np.isfinite(r["p_correct_given_shift"]) else None
        p_base = float(r["freq_correct"])
        delta_gain = (p_plus - p_base) if (p_plus is not None) else None
        events.append({
            "domain": dom,
            "dataset": dataset,
            "model": model,
            "problem": prob,
            "step": int(step),
            "n_samples": int(r["n_samples"]),
            "p_correct": p_base,
            "p_shift_rate": float(r["aha_rate_gpt"]),
            "p_correct_given_shift": (float(p_plus) if p_plus is not None else None),
            "delta_gain_at_shift": (float(delta_gain) if delta_gain is not None else None),
            "thresholds": {
                "delta1": float(delta1),
                "delta2": float(delta2),
                "delta3": (float(delta3) if delta3 is not None else None),
                "min_prior_steps": int(min_prior_steps),
            },
            "question": prob,
            "answer": None,
            # No raw row available in this fallback path:
            "raw_record": None,
            "raw_jsonl": None
        })

    os.makedirs(out_dir, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    return json_path, jsonl_path, len(events)

# ------------- Main -------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_crossword", type=str, default=None,
                    help="Root directory for Crossword results (domain='Crossword').")
    ap.add_argument("--root_math", type=str, default=None,
                    help="Root directory for Math results (domain='Math').")
    ap.add_argument("results_root", nargs="?", default=None,
                    help="Fallback single root (domain='All') if specific roots not given.")
    ap.add_argument("--split", default=None, help="Substring filter on filenames (e.g., 'validation').")
    ap.add_argument("--out_dir", default=None, help="Output dir (default: <first_root>/aha_events).")
    ap.add_argument("--dataset_name", default="MIXED")
    ap.add_argument("--model_name", default="Qwen2.5-1.5B")

    # Formal thresholds
    ap.add_argument("--delta1", type=float, default=0.13)
    ap.add_argument("--delta2", type=float, default=0.13)
    ap.add_argument("--delta3", type=str, default=0.12,
                    help="Gain-at-shift threshold; use 'none' to disable.")
    ap.add_argument("--min_prior_steps", type=int, default=1)

    # Optional step filters (DEFAULT: cap at 1000)
    ap.add_argument("--min_step", type=int, default=None)
    ap.add_argument("--max_step", type=int, default=1000)

    # GPT label policy
    ap.add_argument("--gpt_mode", choices=["canonical","broad"], default="canonical",
                    help="Which LLM shift keys to accept.")
    ap.add_argument("--no_gpt_subset_native", action="store_true",
                    help="If set, do NOT gate LLM labels by native cue presence (domain gate still applies).")

    args = ap.parse_args()

    # Parse delta3 (allow "none")
    d3s = (args.delta3.strip().lower() if isinstance(args.delta3, str) else args.delta3)
    if isinstance(d3s, str) and d3s in ("none","nan","null"):
        delta3_val: Optional[float] = None
    else:
        delta3_val = float(args.delta3) if args.delta3 is not None else None

    # Gather files (domain-aware if both roots provided)
    files_by_domain: Dict[str, List[str]] = {}
    first_root = None
    if args.root_crossword:
        files_by_domain["Crossword"] = scan_files(args.root_crossword, args.split)
        first_root = first_root or args.root_crossword
    if args.root_math:
        files_by_domain["Math"] = scan_files(args.root_math, args.split)
        first_root = first_root or args.root_math
    if not files_by_domain:
        if not args.results_root:
            raise SystemExit("Provide --root_crossword and/or --root_math, or a fallback results_root.")
        files_by_domain["All"] = scan_files(args.results_root, args.split)
        first_root = args.results_root

    total_files = sum(len(v) for v in files_by_domain.values())
    if total_files == 0:
        raise SystemExit("No JSONL files found. Check roots/--split.")

    out_dir = args.out_dir or os.path.join(first_root, "aha_events")
    os.makedirs(out_dir, exist_ok=True)
    slug = f"{slugify(args.dataset_name)}__{slugify(args.model_name)}"
    dataset, model = args.dataset_name, args.model_name

    # GPT label policy
    gpt_subset_native = not args.no_gpt_subset_native
    gpt_keys = (["change_way_of_thinking", "shift_in_reasoning_v1"] if args.gpt_mode == "canonical"
                else ["change_way_of_thinking", "shift_in_reasoning_v1",
                      "shift_llm", "shift_gpt", "pivot_llm", "rechecked"])

    # Load & step-filter (cap applies here)
    df = load_pass1_samples_multi(files_by_domain, gpt_keys=gpt_keys, gpt_subset_native=gpt_subset_native)
    if args.min_step is not None:
        df = df[df["step"] >= args.min_step]
    if args.max_step is not None:
        df = df[df["step"] <= args.max_step]
    if df.empty:
        raise SystemExit("No rows left after step filtering.")

    # Aggregate & mark formal
    ps_base = build_problem_step(df)
    ps_marked = mark_formal_pairs(ps_base.copy(),
                                  delta1=args.delta1, delta2=args.delta2,
                                  min_prior_steps=args.min_prior_steps,
                                  delta3=delta3_val)

    # Export JSON/JSONL (with raw rows embedded)
    json_path, jsonl_path, n_events = export_formal_aha_json_with_text(
        ps_marked, files_by_domain, dataset, model,
        delta1=args.delta1, delta2=args.delta2, delta3=delta3_val,
        min_prior_steps=args.min_prior_steps,
        gpt_keys=gpt_keys, gpt_subset_native=gpt_subset_native,
        out_dir=out_dir, slug=slug
    )

    print(f"[Formal Aha!] {n_events} events written to:\n  {json_path}\n  {jsonl_path}")

if __name__ == "__main__":
    main()
