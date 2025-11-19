#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aggregate crossword pass-1/pass-2 evaluation JSONLs by step, with optional:
  • Prompt-variant filtering and per-group caps (same semantics as your math tool)
  • Recompute correctness using crossword-friendly canon fields

Assumes crossword inference JSONLs where:
  - rec["problem"] holds the clue text
  - rec["gold_answer_canon"] is the canonical gold
  - rec["pass1"]["pred_answer_canon"], rec["pass2"]["pred_answer_canon"] exist
  - rec["pass1"]["soft_reward"] (optional) holds a soft similarity score in [0,1]
  - rec["pass2"]["soft_reward"] (optional) holds a soft similarity score in [0,1]
"""

import os
import re
import json
import argparse
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple, Optional

# ------------------------------ Utilities ------------------------------------

def mean_safe(xs: List[Optional[float]]) -> Optional[float]:
    vals = []
    for x in xs:
        if x is None:
            continue
        try:
            vals.append(float(x))
        except Exception:
            continue
    return sum(vals) / len(vals) if vals else None

def pct(n: int, d: int) -> str:
    return "-" if d == 0 else f"{100.0*n/d:5.1f}%"

def fmt_float(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:6.3f}"

def nat_step_from_path(path: str) -> Optional[int]:
    m = re.search(r"step(\d+)", path)
    return int(m.group(1)) if m else None

def scan_files(root: str, split: Optional[str]) -> List[str]:
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if not fn.endswith(".jsonl"):
                continue
            if split and split not in fn:
                continue
            out.append(os.path.join(dp, fn))
    out.sort(key=lambda p: (nat_step_from_path(p) or 0, p))
    return out

# --- Prompt & grouping helpers -----------------------------------------------

DEFAULT_PROMPT_KEYS = [
    "prompt", "prompt_text", "input_prompt", "input",
    "question_prompt", "fmt_prompt", "aug_prompt"
]

def _get_nested(d: Dict[str, Any], dotpath: str) -> Optional[Any]:
    cur = d
    for part in dotpath.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur

def extract_prompt(rec: Dict[str, Any], preferred_key: str, strict: bool) -> Optional[str]:
    v = _get_nested(rec, preferred_key) if preferred_key else None
    if v is not None or strict:
        return str(v) if v is not None else None
    for k in DEFAULT_PROMPT_KEYS:
        v = _get_nested(rec, k)
        if v is not None:
            return str(v)
    return None

def extract_group(rec: Dict[str, Any], key: str) -> str:
    """
    Grouping key for per-group capping.
    For crosswords, the default 'problem' is the clue text (good grouping).
    """
    if key == "problem":
        return str(rec.get("problem", ""))
    val = _get_nested(rec, key)
    if val is None and key == "prompt":
        # Friendly fallback if someone asks for 'prompt'
        return str(rec.get("problem", ""))
    return "" if val is None else str(val)

# ------------------------------ Correctness -----------------------------------

def _substr(hay: Optional[str], needle: Optional[str]) -> bool:
    if not hay or not needle:
        return False
    return needle in hay

def _exact(a: Optional[str], b: Optional[str]) -> bool:
    if a is None or b is None:
        return False
    return a == b

def maybe_recompute_correctness(pass_dict: Dict[str, Any],
                                gold_canon: Optional[str],
                                mode: str) -> Optional[bool]:
    """Return None to keep original; otherwise True/False."""
    if mode == "none":
        return None
    pred = pass_dict.get("pred_answer_canon")
    if mode == "substring":
        return _substr(pred, gold_canon)
    elif mode == "exact":
        return _exact(pred, gold_canon)
    else:
        return None

# ------------------------------ Aggregator -----------------------------------

class StepAgg:
    def __init__(self, step: int):
        self.step = step
        self.ex_correct_p1: Dict[str, bool] = {}
        self.ex_correct_p2: Dict[str, bool] = {}
        self.n_samp_p1 = 0
        self.n_samp_p2 = 0
        self.samp_correct_p1 = 0
        self.samp_correct_p2 = 0
        self.samp_improved_p2 = 0
        self.ex_improved_p2 = 0
        self.ent_p1_all: List[Optional[float]] = []
        self.ent_p1_think: List[Optional[float]] = []
        self.ent_p1_ans: List[Optional[float]] = []
        self.ent_p2_all: List[Optional[float]] = []
        self.ent_p2_think: List[Optional[float]] = []
        self.ent_p2_ans: List[Optional[float]] = []
        self.tok_p1_think: List[Optional[int]] = []
        self.tok_p1_ans: List[Optional[int]] = []
        self.tok_p2_think: List[Optional[int]] = []
        self.tok_p2_ans: List[Optional[int]] = []
        self.stop_think_p1 = Counter()
        self.stop_ans_p1   = Counter()
        self.stop_think_p2 = Counter()
        self.stop_ans_p2   = Counter()
        self.tag_ok_p1 = 0
        self.tag_ok_p2 = 0
        self.reconsider_rate_p1_numer = 0
        self.reconsider_rate_p2_numer = 0
        self.examples: set[str] = set()

        # soft rewards (sample-level) + per-example buckets (for E-level reduction)
        self.soft1_vals: List[Optional[float]] = []
        self.soft2_vals: List[Optional[float]] = []
        self.soft1_by_problem: Dict[str, List[Optional[float]]] = defaultdict(list)
        self.soft2_by_problem: Dict[str, List[Optional[float]]] = defaultdict(list)

    def add(self, rec: Dict[str, Any], recompute_mode: str):
        problem = rec.get("problem", "")
        self.examples.add(problem)
        gold_canon = rec.get("gold_answer_canon")

        p1 = rec.get("pass1") or {}
        if p1:
            self.n_samp_p1 += 1
            c1_orig = bool(p1.get("is_correct_pred"))
            c1_new = maybe_recompute_correctness(p1, gold_canon, recompute_mode)
            c1 = c1_orig if c1_new is None else bool(c1_new)
            self.samp_correct_p1 += int(c1)
            prev = self.ex_correct_p1.get(problem, False)
            self.ex_correct_p1[problem] = prev or c1
            self.ent_p1_all.append(p1.get("entropy"))
            self.ent_p1_think.append(p1.get("entropy_think"))
            self.ent_p1_ans.append(p1.get("entropy_answer"))
            self.tok_p1_think.append(p1.get("tokens_think"))
            self.tok_p1_ans.append(p1.get("tokens_answer"))
            self.stop_think_p1[(p1.get("stop_reason_think") or "unknown")] += 1
            self.stop_ans_p1[(p1.get("stop_reason_answer") or "unknown")] += 1
            if p1.get("valid_tag_structure"):
                self.tag_ok_p1 += 1
            if p1.get("has_reconsider_cue"):
                self.reconsider_rate_p1_numer += 1

            # soft1
            soft1 = p1.get("soft_reward")
            if soft1 is None and ("soft reward" in p1):
                soft1 = p1.get("soft reward")
            self.soft1_vals.append(soft1)
            self.soft1_by_problem[problem].append(soft1)

        p2 = rec.get("pass2") or {}
        if p2:
            self.n_samp_p2 += 1
            c2_orig = bool(p2.get("is_correct_pred"))
            c2_new = maybe_recompute_correctness(p2, gold_canon, recompute_mode)
            c2 = c2_orig if c2_new is None else bool(c2_new)
            self.samp_correct_p2 += int(c2)
            prev = self.ex_correct_p2.get(problem, False)
            self.ex_correct_p2[problem] = prev or c2
            self.ent_p2_all.append(p2.get("entropy"))
            self.ent_p2_think.append(p2.get("entropy_think"))
            self.ent_p2_ans.append(p2.get("entropy_answer"))
            self.tok_p2_think.append(p2.get("tokens_think"))
            self.tok_p2_ans.append(p2.get("tokens_answer"))
            self.stop_think_p2[(p2.get("stop_reason_think") or "unknown")] += 1
            self.stop_ans_p2[(p2.get("stop_reason_answer") or "unknown")] += 1
            if p2.get("valid_tag_structure"):
                self.tag_ok_p2 += 1
            if p2.get("has_reconsider_cue"):
                self.reconsider_rate_p2_numer += 1
            if p2.get("improved_over_pass1"):
                self.samp_improved_p2 += 1

            # soft2
            soft2 = p2.get("soft_reward")
            if soft2 is None and ("soft reward" in p2):
                soft2 = p2.get("soft reward")
            self.soft2_vals.append(soft2)
            self.soft2_by_problem[problem].append(soft2)

    def finalize(self):
        for prob in self.examples:
            p1_ok = self.ex_correct_p1.get(prob, False)
            p2_ok = self.ex_correct_p2.get(prob, False)
            if p2_ok and not p1_ok:
                self.ex_improved_p2 += 1

    def _example_level_soft_mean(self, by_problem: Dict[str, List[Optional[float]]]) -> Optional[float]:
        # Reduce samples → example via max (mirrors acc_E "any sample correct")
        per_example = []
        for prob in self.examples:
            vals = [float(v) for v in by_problem.get(prob, []) if v is not None]
            if vals:
                per_example.append(max(vals))
        return mean_safe(per_example)

    def row(self) -> str:
        nE = len(self.examples) if self.examples else 0
        acc1S = pct(self.samp_correct_p1, self.n_samp_p1)
        acc2S = pct(self.samp_correct_p2, self.n_samp_p2)
        acc1E = pct(sum(1 for v in self.ex_correct_p1.values() if v), nE) if nE else "-"
        acc2E = pct(sum(1 for v in self.ex_correct_p2.values() if v), nE) if nE else "-"
        ent1  = fmt_float(mean_safe(self.ent_p1_all))
        ent2  = fmt_float(mean_safe(self.ent_p2_all))
        t1    = fmt_float(mean_safe(self.ent_p1_think))
        a1    = fmt_float(mean_safe(self.ent_p1_ans))
        t2    = fmt_float(mean_safe(self.ent_p2_think))
        a2    = fmt_float(mean_safe(self.ent_p2_ans))
        impS  = pct(self.samp_improved_p2, self.n_samp_p2) if self.n_samp_p2 else "-"
        impE  = pct(self.ex_improved_p2, nE) if nE else "-"
        tag1 = pct(self.tag_ok_p1, self.n_samp_p1) if self.n_samp_p1 else "-"
        tag2 = pct(self.tag_ok_p2, self.n_samp_p2) if self.n_samp_p2 else "-"

        soft1S = fmt_float(mean_safe(self.soft1_vals))
        soft1E = fmt_float(self._example_level_soft_mean(self.soft1_by_problem))
        soft2S = fmt_float(mean_safe(self.soft2_vals))
        soft2E = fmt_float(self._example_level_soft_mean(self.soft2_by_problem))

        return (f"{self.step:6d} "
                f"{self.n_samp_p1:6d} {acc1S:>6} {acc1E:>6} {ent1:>8} {t1:>7} {a1:>7} {soft1S:>7} {soft1E:>7} "
                f"{self.n_samp_p2:6d} {acc2S:>6} {acc2E:>6} {ent2:>8} {t2:>7} {a2:>7} {soft2S:>7} {soft2E:>7} "
                f"{impS:>6} {impE:>6} {tag1:>6} {tag2:>6}")

    def footer(self) -> str:
        def fmt_counter(cnt: Counter, den: int) -> str:
            if den == 0:
                return "—"
            keys = ["stop_token", "eos", "max_new_tokens", "other", "unknown"]
            return ", ".join(f"{k}={pct(cnt.get(k,0), den)}" for k in keys)
        nE = len(self.examples)
        lines = [f"   • examples: {nE}"]
        if self.n_samp_p1:
            lines.append(f"   • p1 think stops: {fmt_counter(self.stop_think_p1, self.n_samp_p1)}")
            lines.append(f"   • p1 answer stops: {fmt_counter(self.stop_ans_p1, self.n_samp_p1)}")
        if self.n_samp_p2:
            lines.append(f"   • p2 think stops: {fmt_counter(self.stop_think_p2, self.n_samp_p2)}")
            lines.append(f"   • p2 answer stops: {fmt_counter(self.stop_ans_p2, self.n_samp_p2)}")
        if self.n_samp_p1:
            lines.append(f"   • p1 reconsider-markers rate: {pct(self.reconsider_rate_p1_numer, self.n_samp_p1)}")
        if self.n_samp_p2:
            lines.append(f"   • p2 reconsider-markers rate: {pct(self.reconsider_rate_p2_numer, self.n_samp_p2)}")

        if any(v is not None for v in self.tok_p1_think + self.tok_p2_think):
            mt1 = mean_safe([x for x in self.tok_p1_think if isinstance(x, (int, float))])
            ma1 = mean_safe([x for x in self.tok_p1_ans   if isinstance(x, (int, float))])
            mt2 = mean_safe([x for x in self.tok_p2_think if isinstance(x, (int, float))])
            ma2 = mean_safe([x for x in self.tok_p2_ans   if isinstance(x, (int, float))])
            lines.append("   • mean tokens — p1: think="
                         f"{'-' if mt1 is None else f'{mt1:.1f}'} answer="
                         f"{'-' if ma1 is None else f'{ma1:.1f}'}; "
                         "p2: think="
                         f"{'-' if mt2 is None else f'{mt2:.1f}'} answer="
                         f"{'-' if ma2 is None else f'{ma2:.1f}'}")

        ms1S = mean_safe(self.soft1_vals)
        ms1E = self._example_level_soft_mean(self.soft1_by_problem)
        if ms1S is not None or ms1E is not None:
            lines.append(f"   • mean soft (p1): samples={ '-' if ms1S is None else f'{ms1S:.3f}'}; "
                         f"examples[max]={ '-' if ms1E is None else f'{ms1E:.3f}'}")

        ms2S = mean_safe(self.soft2_vals)
        ms2E = self._example_level_soft_mean(self.soft2_by_problem)
        if ms2S is not None or ms2E is not None:
            lines.append(f"   • mean soft (p2): samples={ '-' if ms2S is None else f'{ms2S:.3f}'}; "
                         f"examples[max]={ '-' if ms2E is None else f'{ms2E:.3f}'}")
        return "\n".join(lines)

# ------------------------------ Main -----------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_root", help="Root directory containing step*/.../*.jsonl")
    ap.add_argument("--split", default=None, help="Filter filenames containing this split substring (e.g., 'test').")

    # Prompt-variant filtering
    ap.add_argument("--max_prompt_versions", type=int, default=None,
                    help="Drop any group that has more than this many distinct prompt variants.")
    ap.add_argument("--prompt_key", default="prompt", help="Dot-path to the prompt field (default: 'prompt').")
    ap.add_argument("--strict_prompt_key", action="store_true",
                    help="Only use --prompt_key; don't fall back to common alternatives.")
    ap.add_argument("--filter_scope", choices=["per_problem", "global"], default="per_problem",
                    help="Count prompt variants per problem (default) or globally.")
    ap.add_argument("--prompt_family_regex", default=None,
                    help="Regex to strip version markers before counting variants.")

    # Per-group hard cap
    ap.add_argument("--max_per_group", type=int, default=None,
                    help="Keep at most this many records per group (default grouping uses --group_key='problem').")
    ap.add_argument("--group_key", default="problem",
                    help="Dot-path field to group by for capping (default: 'problem' for clue text).")

    # Write filtered copies or rewrite in place
    ap.add_argument("--rewrite_filtered", action="store_true",
                    help="Rewrite input JSONLs IN PLACE (dangerous).")
    ap.add_argument("--write_filtered_to", default=None,
                    help="Write filtered JSONLs under this output root, mirroring the input tree.")
    ap.add_argument("--aggregate_from_filtered", action="store_true",
                    help="Aggregate from filtered copies instead of originals.")

    # Crossword-specific: recompute correctness
    ap.add_argument("--recompute_correctness", choices=["none","substring","exact"], default="none",
                    help="Override recorded correctness using pred_answer_canon vs gold_answer_canon.")

    # CSV outputs
    ap.add_argument("--save_csv", default=None, help="Optional CSV output path for the step table.")
    ap.add_argument("--per_example_csv", default=None,
                    help="Optional CSV with per-example correctness (p1, p2, improved).")
    args = ap.parse_args()

    files = scan_files(args.results_root, args.split)
    if not files:
        print("No JSONL files found. Check the path or split filter.")
        return

    # ---------------- PASS 1: prompt-variant pre-scan (optional) ----------------
    drop_groups: set[str] = set()
    normalize_re = (re.compile(args.prompt_family_regex) if args.prompt_family_regex else None)

    if args.max_prompt_versions is not None:
        variants_by_group: Dict[str, set] = defaultdict(set)

        def group_key_for_prompt_versions(rec: Dict[str, Any]) -> str:
            return str(rec.get("problem", "")) if args.filter_scope == "per_problem" else "__GLOBAL__"

        total_seen = 0
        total_with_prompt = 0

        for path in files:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        rec = json.loads(s)
                    except Exception:
                        continue
                    total_seen += 1
                    prompt = extract_prompt(rec, args.prompt_key, args.strict_prompt_key)
                    if prompt is None:
                        continue
                    total_with_prompt += 1
                    if normalize_re:
                        prompt = normalize_re.sub("", prompt)
                    g = group_key_for_prompt_versions(rec)
                    variants_by_group[g].add(prompt)

        for g, s in variants_by_group.items():
            if len(s) > args.max_prompt_versions:
                drop_groups.add(g)

        scope_msg = "per-problem" if args.filter_scope == "per_problem" else "global"
        print(f"[filter] Scope={scope_msg} | threshold={args.max_prompt_versions} | "
              f"records_seen={total_seen} | with_prompt={total_with_prompt} | "
              f"groups_dropped={len(drop_groups)}")

    # ---------------- Filtering/capping writer (optional) ----------------
    def should_drop_group(rec: Dict[str, Any]) -> bool:
        if not drop_groups:
            return False
        g = str(rec.get("problem", "")) if args.filter_scope == "per_problem" else "__GLOBAL__"
        return g in drop_groups

    wrote_any = False
    if args.rewrite_filtered or args.write_filtered_to:
        if args.rewrite_filtered and args.write_filtered_to:
            raise SystemExit("Choose only one of --rewrite_filtered or --write_filtered_to.")

        for src_path in files:
            if args.rewrite_filtered:
                out_path = src_path + ".tmp_filter"
            else:
                rel = os.path.relpath(src_path, args.results_root)
                out_path = os.path.join(args.write_filtered_to, rel)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)

            group_counts: Dict[str, int] = defaultdict(int)
            kept = 0
            total = 0
            exceeded_groups = set()

            with open(src_path, "r", encoding="utf-8") as src, open(out_path, "w", encoding="utf-8") as dst:
                for line in src:
                    s = line.strip()
                    if not s:
                        continue
                    total += 1
                    try:
                        rec = json.loads(s)
                    except Exception:
                        continue
                    if should_drop_group(rec):
                        continue

                    # Apply per-group cap (streaming; keep FIRST K)
                    if args.max_per_group is not None:
                        g = extract_group(rec, args.group_key)
                        c = group_counts[g]
                        if c >= args.max_per_group:
                            exceeded_groups.add(g)
                            continue
                        group_counts[g] = c + 1

                    dst.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    kept += 1

            if args.rewrite_filtered:
                os.replace(out_path, src_path)
                final_out = src_path
            else:
                final_out = out_path

            wrote_any = True
            print(f"[filter-write] {src_path} -> {final_out} | kept={kept} of {total} "
                  f"| unique_groups={len(group_counts)} | groups_truncated={len(exceeded_groups)}")

    # --------------- PASS 2: aggregate (supports virtual capping) ---------------
    files_for_agg = files
    if wrote_any and args.write_filtered_to and args.aggregate_from_filtered:
        files_for_agg = []
        for src_path in files:
            rel = os.path.relpath(src_path, args.results_root)
            files_for_agg.append(os.path.join(args.write_filtered_to, rel))

    steps: Dict[int, StepAgg] = {}

    for path in files_for_agg:
        group_counts: Dict[str, int] = defaultdict(int)
        step_from_name = nat_step_from_path(path)
        try:
            f = open(path, "r", encoding="utf-8")
        except FileNotFoundError:
            continue

        with f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    rec = json.loads(s)
                except Exception:
                    continue

                if should_drop_group(rec):
                    continue

                # Virtual cap if not writing filtered copies
                if args.max_per_group is not None and not wrote_any:
                    g = extract_group(rec, args.group_key)
                    c = group_counts[g]
                    if c >= args.max_per_group:
                        continue
                    group_counts[g] = c + 1

                step = rec.get("step", step_from_name if step_from_name is not None else 0)
                agg = steps.setdefault(step, StepAgg(step))
                agg.add(rec, args.recompute_correctness)

    for _, agg in steps.items():
        agg.finalize()

    ordered = [steps[k] for k in sorted(steps.keys())]

    header = ("  step   n1S  acc1S  acc1E    ent1      t1      a1  soft1S soft1E  "
              "n2S  acc2S  acc2E    ent2      t2      a2  soft2S soft2E  impS  impE  tag1  tag2")
    print(header)
    print("-" * len(header))
    for agg in ordered:
        print(agg.row())
    print()

    for agg in ordered:
        print(f"[step {agg.step}]")
        print(agg.footer())
        print()

    # Optional CSVs
    if args.save_csv:
        import csv
        with open(args.save_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["step","n1S","acc1S_pct","acc1E_pct","ent1","t1","a1","soft1S","soft1E",
                        "n2S","acc2S_pct","acc2E_pct","ent2","t2","a2","soft2S","soft2E",
                        "impS_pct","impE_pct","tag1_pct","tag2_pct"])
            for agg in ordered:
                nE = len(agg.examples) if agg.examples else 0
                acc1S = (100.0*agg.samp_correct_p1/agg.n_samp_p1) if agg.n_samp_p1 else None
                acc2S = (100.0*agg.samp_correct_p2/agg.n_samp_p2) if agg.n_samp_p2 else None
                acc1E = (100.0*sum(1 for v in agg.ex_correct_p1.values() if v)/nE) if nE else None
                acc2E = (100.0*sum(1 for v in agg.ex_correct_p2.values() if v)/nE) if nE else None
                impS  = (100.0*agg.samp_improved_p2/agg.n_samp_p2) if agg.n_samp_p2 else None
                impE  = (100.0*agg.ex_improved_p2/nE) if nE else None
                tag1p = (100.0*agg.tag_ok_p1/agg.n_samp_p1) if agg.n_samp_p1 else None
                tag2p = (100.0*agg.tag_ok_p2/agg.n_samp_p2) if agg.n_samp_p2 else None

                # soft means
                soft1S = mean_safe(agg.soft1_vals)
                soft2S = mean_safe(agg.soft2_vals)
                soft1E = mean_safe([
                    max([float(v) for v in vals if v is not None])
                    for prob, vals in agg.soft1_by_problem.items()
                    if any(v is not None for v in vals)
                ])
                soft2E = mean_safe([
                    max([float(v) for v in vals if v is not None])
                    for prob, vals in agg.soft2_by_problem.items()
                    if any(v is not None for v in vals)
                ])

                w.writerow([
                    agg.step,
                    agg.n_samp_p1, acc1S, acc1E,
                    mean_safe(agg.ent_p1_all),
                    mean_safe(agg.ent_p1_think),
                    mean_safe(agg.ent_p1_ans),
                    soft1S, soft1E,
                    agg.n_samp_p2, acc2S, acc2E,
                    mean_safe(agg.ent_p2_all),
                    mean_safe(agg.ent_p2_think),
                    mean_safe(agg.ent_p2_ans),
                    soft2S, soft2E,
                    impS, impE, tag1p, tag2p
                ])

    if args.per_example_csv:
        import csv
        with open(args.per_example_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["step","problem","p1_correct","p2_correct","improved"])
            for agg in ordered:
                for prob in sorted(agg.examples):
                    p1_ok = agg.ex_correct_p1.get(prob, False)
                    p2_ok = agg.ex_correct_p2.get(prob, False)
                    improved = (p2_ok and not p1_ok)
                    w.writerow([agg.step, prob, int(bool(p1_ok)), int(bool(p2_ok)), int(bool(improved))])


if __name__ == "__main__":
    main()
