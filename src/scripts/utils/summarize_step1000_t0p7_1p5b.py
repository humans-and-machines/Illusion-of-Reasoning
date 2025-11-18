#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Binned summary by pass-1 (think+answer) entropy with equal-count (quantile) bins per domain.

Features
--------
• Sample-level: p(correct_p1), p(correct_p2), optional p_soft@τ(...)
• 'none_right_p1' (aligned to HARD or SOFT depending on conditional setting)
• Example-level conditionals: p(p2|p1_none) and p(p2|p1_any) with counts
• Conditionals can use SOFT≥τ for selected domains (e.g., carpark)

Bin logic
---------
• Binning key: pass-1 entropy_think and entropy_answer combined via --combine {sum|mean}
• --binning {equal_count|equal_width}, default equal_count (quantiles per domain)
• Samples missing either entropy_think/entropy_answer are skipped from binning

"""
import os, re, json, argparse
from bisect import bisect_right
from typing import Dict, Any, Optional, Tuple, List, Set

DOM_MAP = {
    "carpark": "Carpark",
    "xword":   "Crossword",
    "math":    "Math",
}

# ---------- helpers ----------
def nat_step_from_path(path: str) -> Optional[int]:
    m = re.search(r"step(\d+)", path)
    return int(m.group(1)) if m else None

def _substr(hay: Optional[str], needle: Optional[str]) -> bool:
    if not hay or not needle:
        return False
    return needle in hay

def _exact(a: Optional[str], b: Optional[str]) -> bool:
    if a is None or b is None:
        return False
    return a == b

def compute_correct(pass_dict: Dict[str, Any], gold_canon: Optional[str], mode: str) -> Optional[bool]:
    if pass_dict is None:
        return None
    if mode == "none":
        v = pass_dict.get("is_correct_pred")
        return None if v is None else bool(v)
    pred = pass_dict.get("pred_answer_canon")
    if mode == "substring":
        return _substr(pred, gold_canon)
    if mode == "exact":
        return _exact(pred, gold_canon)
    return None

def get_soft(pass_dict: Dict[str, Any]) -> Optional[float]:
    if pass_dict is None:
        return None
    v = pass_dict.get("soft_reward", pass_dict.get("soft reward"))
    try:
        return float(v) if v is not None else None
    except Exception:
        return None

def domain_from_dirname(dname: str) -> Optional[str]:
    for key, label in DOM_MAP.items():
        if f"-{key}-" in dname:
            return label
    return None

def label_to_token(label: str) -> str:
    t = label.strip().lower()
    if t in ("carpark", "rush hour", "rush_hour", "rushhour"): return "carpark"
    if t in ("crossword", "xword", "crosswords"):              return "xword"
    if t in ("math", "math2"):                                 return "math"
    return t

def parse_cond_soft_domains(s: Optional[str]) -> Set[str]:
    if not s: return set()
    toks = set()
    for part in s.split(","):
        p = part.strip().lower()
        if p in ("carpark", "rush hour", "rush_hour", "rushhour"): toks.add("carpark")
        elif p in ("crossword", "xword", "crosswords"):            toks.add("xword")
        elif p in ("math", "math2"):                               toks.add("math")
        else:                                                      toks.add(p)
    return toks

def find_model_dirs(results_root: str, target_temp: float) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for name in os.listdir(results_root):
        if not name.startswith("GRPO-1.5B-"):
            continue
        m = re.search(r"-temp-([0-9.]+)$", name)
        if not m:
            continue
        try:
            temp_val = float(m.group(1))
        except ValueError:
            continue
        if abs(temp_val - target_temp) > 1e-12:
            continue
        dom = domain_from_dirname(name)
        if dom is None:
            continue
        full = os.path.join(results_root, name)
        if os.path.isdir(full):
            out.append((dom, full))
    chosen: Dict[str, str] = {}
    for dom, path in sorted(out, key=lambda x: x[1]):
        chosen[dom] = path
    return sorted(chosen.items(), key=lambda x: x[0])

def iter_jsonl_files(root: str, split_substr: Optional[str]) -> List[str]:
    paths = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if not fn.endswith(".jsonl"):
                continue
            if split_substr and split_substr not in fn:
                continue
            paths.append(os.path.join(dp, fn))
    return paths

def extract_problem_key(rec: Dict[str, Any]) -> str:
    for k in ("problem", "problem_id", "question", "clue", "title", "id", "uid", "example_id", "idx"):
        v = rec.get(k)
        if v is not None:
            return str(v)
    return f"__line_{id(rec)}"

# ---------- bin helpers ----------
def p1_think_answer_value(p1: Dict[str, Any], combine: str) -> Optional[float]:
    def _f(x):
        try: return float(x)
        except: return None
    t = _f(p1.get("entropy_think"))
    a = _f(p1.get("entropy_answer"))
    if t is None or a is None:
        return None
    return 0.5*(t + a) if combine == "mean" else (t + a)

def compute_equal_width_edges(vmin: float, vmax: float, bins: int) -> List[float]:
    if bins <= 0: return [vmin, vmax]
    if vmax <= vmin: return [vmin]*(bins) + [vmax]
    w = (vmax - vmin) / bins
    edges = [vmin + i*w for i in range(bins)]
    edges.append(vmax)
    edges[-1] = vmax
    return edges

def compute_equal_count_edges(values: List[float], bins: int) -> List[float]:
    # Quantile-style edges: edges[0]=min, edges[-1]=max; interior edges are bin starts
    vs = sorted(values)
    n = len(vs)
    if n == 0:
        return [0.0, 0.0]
    bins = max(1, min(bins, n))
    # boundaries are start indices of each bin in sorted order
    boundaries = [int(round(i * n / bins)) for i in range(bins + 1)]
    boundaries[0] = 0
    boundaries[-1] = n
    edges = [vs[boundaries[i]] for i in range(bins)]
    edges.append(vs[-1])
    return edges

def assign_bin_equal_width(value: float, edges: List[float]) -> int:
    B = len(edges) - 1
    if B <= 0: return 0
    vmin, vmax = edges[0], edges[-1]
    if vmax <= vmin: return 0
    w = (vmax - vmin) / B
    if w == 0: return 0
    idx = int((value - vmin) / w)
    if idx < 0: idx = 0
    if idx >= B: idx = B - 1
    return idx

def assign_bin_equal_count(value: float, edges: List[float]) -> int:
    # edges = [start0, start1, ..., start_{B-1}, max]; use interior starts as cutpoints
    if len(edges) <= 2:  # one bin
        return 0
    cut = edges[1:-1]  # starts of bins 1..B-1
    # number of starts <= value gives the bin index
    return bisect_right(cut, value)

def bin_label(i: int, edges: List[float]) -> str:
    B = len(edges) - 1
    lo = edges[i]; hi = edges[i+1]
    if i < B - 1:
        return f"[{lo:.3f}, {hi:.3f})"
    else:
        return f"[{lo:.3f}, {hi:.3f}]"

# ---------- per-bin aggregator ----------
class BinAgg:
    __slots__ = (
        "n1S","c1S","n2S","c2S",
        "n1_soft","c1_soft","n2_soft","c2_soft",
        "ex_seen","ex_has_p1","ex_p1_any_ok_cond","ex_p2_any_ok_cond"
    )
    def __init__(self):
        self.n1S = self.c1S = self.n2S = self.c2S = 0
        self.n1_soft = self.c1_soft = self.n2_soft = self.c2_soft = 0
        self.ex_seen: Set[str] = set()
        self.ex_has_p1: Set[str] = set()
        self.ex_p1_any_ok_cond: Dict[str, bool] = {}
        self.ex_p2_any_ok_cond: Dict[str, bool] = {}

# ---------- main summarizer ----------
def summarize_dir_binned(
    dirpath: str,
    domain_label: str,
    target_step: int,
    split_substr: Optional[str],
    recompute: str,
    soft_threshold: Optional[float],
    use_soft_for_conditionals: bool,
    bins: int,
    combine: str,
    binning: str
) -> Tuple[List[BinAgg], List[float], int]:
    """
    Returns:
      (per-bin BinAgg list, edges list, skipped_count_missing_entropy)
    """
    files = iter_jsonl_files(dirpath, split_substr)

    # pass 1: collect pass-1 think+answer values for quantiles (or min/max for equal-width)
    vals: List[float] = []
    vmin = float("inf"); vmax = float("-inf"); have_any = False
    for path in files:
        step_from_name = nat_step_from_path(path)
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if not s: continue
                try: rec = json.loads(s)
                except Exception: continue
                step = rec.get("step", step_from_name)
                if step != target_step: continue
                p1 = rec.get("pass1") or {}
                val = p1_think_answer_value(p1, combine)
                if val is None: continue
                have_any = True
                vals.append(val)
                if val < vmin: vmin = val
                if val > vmax: vmax = val

    if not have_any:
        return [BinAgg()], [0.0, 0.0], 0

    if binning == "equal_count":
        edges = compute_equal_count_edges(vals, bins)
    else:
        edges = compute_equal_width_edges(vmin, vmax, bins)

    # pass 2: populate per-bin aggregates
    aggs = [BinAgg() for _ in range(len(edges) - 1)]
    skipped_missing = 0

    for path in files:
        step_from_name = nat_step_from_path(path)
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if not s: continue
                try: rec = json.loads(s)
                except Exception: continue
                step = rec.get("step", step_from_name)
                if step != target_step: continue

                gold = rec.get("gold_answer_canon")
                prob_key = extract_problem_key(rec)

                p1 = rec.get("pass1") or {}
                val = p1_think_answer_value(p1, combine)
                if val is None:
                    skipped_missing += 1
                    continue

                if binning == "equal_count":
                    b = assign_bin_equal_count(val, edges)
                else:
                    b = assign_bin_equal_width(val, edges)

                agg = aggs[b]
                agg.ex_seen.add(prob_key)

                # ----- pass-1
                if p1:
                    agg.ex_has_p1.add(prob_key)

                    ok1_hard = compute_correct(p1, gold, recompute)
                    if ok1_hard is not None:
                        agg.n1S += 1
                        agg.c1S += int(bool(ok1_hard))

                    sr1 = get_soft(p1)
                    if sr1 is not None:
                        agg.n1_soft += 1
                        if (soft_threshold is not None) and (sr1 >= soft_threshold):
                            agg.c1_soft += 1

                    if use_soft_for_conditionals and (soft_threshold is not None):
                        ok1_cond = (sr1 is not None) and (sr1 >= soft_threshold)
                    else:
                        ok1_cond = bool(ok1_hard) if ok1_hard is not None else False
                    if ok1_cond:
                        agg.ex_p1_any_ok_cond[prob_key] = True

                # ----- pass-2 (categorized by the same p1 bin)
                p2 = rec.get("pass2") or {}
                if p2:
                    ok2_hard = compute_correct(p2, gold, recompute)
                    if ok2_hard is not None:
                        agg.n2S += 1
                        agg.c2S += int(bool(ok2_hard))

                    sr2 = get_soft(p2)
                    if sr2 is not None:
                        agg.n2_soft += 1
                        if (soft_threshold is not None) and (sr2 >= soft_threshold):
                            agg.c2_soft += 1

                    if use_soft_for_conditionals and (soft_threshold is not None):
                        ok2_cond = (sr2 is not None) and (sr2 >= soft_threshold)
                    else:
                        ok2_cond = bool(ok2_hard) if ok2_hard is not None else False
                    if ok2_cond:
                        agg.ex_p2_any_ok_cond[prob_key] = True

    return aggs, edges, skipped_missing

# ---------- pretty helpers ----------
def fmt_prob(num: int, den: int) -> str:
    return "-" if den == 0 else f"{num/den:.3f}"

def yes_no_na(n: int, k: int) -> str:
    if n == 0: return "NA"
    return "Yes" if k == 0 else "No"

def none_right_label(agg: BinAgg, use_soft_for_conditionals: bool, tau_set: bool) -> str:
    # Mirror conditional semantics: if conditionals use SOFT>=τ, reflect that here too
    if use_soft_for_conditionals and tau_set:
        return yes_no_na(agg.n1_soft, agg.c1_soft)
    else:
        return yes_no_na(agg.n1S, agg.c1S)

def domain_order_key(label: str) -> int:
    # Xword, Math, Rush Hour
    tok = label_to_token(label)
    order = {"xword": 0, "math": 1, "carpark": 2}
    return order.get(tok, 99)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results", help="Path containing GRPO-* result folders")
    ap.add_argument("--temperature", type=float, default=0.05, help="Temperature to select (e.g., 0.05, 0.3, 0.7)")
    ap.add_argument("--target-step", type=int, default=1000, help="Step to summarize (default: 1000)")
    ap.add_argument("--split", default=None, help="Only include files whose names contain this substring (e.g., 'test').")
    ap.add_argument("--recompute", choices=["none", "substring", "exact"], default="none",
                    help="Recompute correctness from *_canon fields (useful for crosswords).")
    ap.add_argument("--soft-threshold", type=float, default=None,
                    help="If set, also compute soft accuracy columns where soft_reward >= τ.")
    ap.add_argument("--none-right-col", action="store_true",
                    help="Add 'none_right_p1' (within-bin; semantics aligned with conditionals).")
    ap.add_argument("--cond-col", action="store_true",
                    help="Add example-level conditional columns.")
    ap.add_argument("--cond-soft-domains", default="",
                    help="Comma-separated domains whose conditionals use SOFT>=τ (e.g., 'carpark').")
    ap.add_argument("--bins", type=int, default=4, help="Number of bins (default: 4).")
    ap.add_argument("--combine", choices=["sum","mean"], default="sum",
                    help="Combine pass-1 entropy_think and entropy_answer via 'sum' or 'mean'.")
    ap.add_argument("--binning", choices=["equal_count","equal_width"], default="equal_count",
                    help="Bin strategy: equal_count (quantiles) or equal_width (ranges).")
    args = ap.parse_args()

    cond_soft_domains = parse_cond_soft_domains(args.cond_soft_domains)
    dirs = find_model_dirs(args.results_root, args.temperature)
    if not dirs:
        print(f"No matching GRPO-1.5B * temp-{args.temperature:g} result folders found.")
        return

    # Header
    hdr = [
        f"{'Domain':<10}", f"{'Bin':<16}",
        f"{'Number of Samples':>16}", f"{'p(correct_p1)':>16}", f"{'p(correct_p2)':>16}"
    ]
    if args.soft_threshold is not None:
        tau = args.soft_threshold
        hdr += [f"{f'p_soft@{tau:g}(p1)':>16}", f"{f'p_soft@{tau:g}(p2)':>16}", f"{'n_soft_p1':>16}", f"{'n_soft_p2':>16}"]
    if args.none_right_col:
        hdr += [f"{'none_right_p1':>16}"]
    if args.cond_col:
        hdr += [f"{'p(p2|p1_none)':>16}", f"{'nE_p1_none':>16}", f"{'nE_p1_none&p2':>16}",
                f"{'p(p2|p1_any)':>16}",  f"{'nE_p1_any':>16}",  f"{'nE_p1_any&p2':>16}"]
    print("  ".join(hdr))
    print("-" * (sum(len(h) for h in hdr) + 2*(len(hdr)-1)))

    # Per-domain, per-bin
    for domain_label, dpath in sorted(dirs, key=lambda x: domain_order_key(x[0])):
        token = label_to_token(domain_label)
        use_soft_for_conditionals = (token in cond_soft_domains)

        aggs, edges, skipped_missing = summarize_dir_binned(
            dpath, domain_label, args.target_step, args.split, args.recompute,
            args.soft_threshold, use_soft_for_conditionals,
            args.bins, args.combine, args.binning
        )

        for bi, agg in enumerate(aggs):
            # Example-level conditional tallies (bin-restricted)
            ex_p1_none_total = ex_p1_none_and_p2 = 0
            ex_p1_any_total  = ex_p1_any_and_p2  = 0
            for pk in agg.ex_seen:
                if pk in agg.ex_has_p1:
                    if agg.ex_p1_any_ok_cond.get(pk, False):
                        ex_p1_any_total += 1
                        if agg.ex_p2_any_ok_cond.get(pk, False):
                            ex_p1_any_and_p2 += 1
                    else:
                        ex_p1_none_total += 1
                        if agg.ex_p2_any_ok_cond.get(pk, False):
                            ex_p1_none_and_p2 += 1

            N = agg.n2S if agg.n2S > 0 else agg.n1S
            p1 = fmt_prob(agg.c1S, agg.n1S)
            p2 = fmt_prob(agg.c2S, agg.n2S)
            parts = [
                f"{domain_label:<10}", f"{bin_label(bi, edges):<16}",
                f"{N:>16d}", f"{p1:>16}", f"{p2:>16}",
            ]
            if args.soft_threshold is not None:
                p1s = fmt_prob(agg.c1_soft, agg.n1_soft)
                p2s = fmt_prob(agg.c2_soft, agg.n2_soft)
                parts += [f"{p1s:>16}", f"{p2s:>16}", f"{agg.n1_soft:>16d}", f"{agg.n2_soft:>16d}"]
            if args.none_right_col:
                parts += [f"{none_right_label(agg, use_soft_for_conditionals, args.soft_threshold is not None):>16}"]
            if args.cond_col:
                parts += [
                    f"{fmt_prob(ex_p1_none_and_p2, ex_p1_none_total):>16}",
                    f"{ex_p1_none_total:>16d}",
                    f"{ex_p1_none_and_p2:>16d}",
                    f"{fmt_prob(ex_p1_any_and_p2,  ex_p1_any_total):>16}",
                    f"{ex_p1_any_total:>16d}",
                    f"{ex_p1_any_and_p2:>16d}",
                ]
            print("  ".join(parts))

        if skipped_missing > 0:
            print(f"[{domain_label}] skipped {skipped_missing} rows missing pass-1 entropy_think or entropy_answer for binning.")

if __name__ == "__main__":
    main()
