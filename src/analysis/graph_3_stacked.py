#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
graph_3_stacked.py (PASS1 ONLY)
--------------------------------
Combine Carpark, Crossword, Math. Bin by PASS1 ANSWER ENTROPY only.
Plot a STACKED histogram per bin split into No Aha vs Aha.
Use --normalize to make each bin sum to 1.0 (proportions).

Usage
-----
python graph_3_stacked.py \
  --root_crossword artifacts/results/GRPO-1.5B-xword-temp-0.7 \
  --root_math      artifacts/results/GRPO-1.5B-math-temp-0.7 \
  --root_carpark   artifacts/results/GRPO-1.5B-carpark-temp-0.7 \
  --split test \
  --gpt_mode canonical \
  --bins 10 \
  --binning quantile \
  --outdir graphs \
  --outfile_tag combined \
  --normalize
"""

import argparse, json, os, re, sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import matplotlib.pyplot as plt

AHA_KEYS_CANON = ["change_way_of_thinking", "shift_in_reasoning_v1"]
AHA_KEYS_BROAD = AHA_KEYS_CANON + ["shift_llm", "shift_gpt", "pivot_llm", "rechecked"]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_crossword", type=str, default=None)
    ap.add_argument("--root_math",      type=str, default=None)
    ap.add_argument("--root_carpark",   type=str, default=None)
    ap.add_argument("--split",          type=str, default="test")
    ap.add_argument("--gpt_mode",       type=str, default="canonical", choices=["canonical","broad"])

    ap.add_argument("--bins",     type=int, default=20)
    ap.add_argument("--binning",  type=str, default="uniform", choices=["uniform","quantile"])
    ap.add_argument("--min_step", type=int, default=0)
    ap.add_argument("--max_step", type=int, default=1000)

    ap.add_argument("--outdir",      type=str, default="graphs")
    ap.add_argument("--outfile_tag", type=str, default=None)
    ap.add_argument("--title",       type=str, default="Counts by PASS1 Answer Entropy (Stacked No Aha vs Aha)")
    ap.add_argument("--dpi",         type=int, default=300)
    ap.add_argument("--width_in",    type=float, default=10.0)
    ap.add_argument("--height_in",   type=float, default=5.5)
    ap.add_argument("--normalize",   action="store_true", help="stack to proportions (each bin sums to 1)")
    return ap.parse_args()

def truthy(x):
    if x is True: return True
    if isinstance(x, (int, float)): return x != 0
    if isinstance(x, str): return x.strip().lower() in {"1","true","yes","y"}
    return False

def detect_aha_pass1(rec: Dict[str, Any], mode: str) -> bool:
    keys = AHA_KEYS_CANON if mode == "canonical" else AHA_KEYS_BROAD
    p1 = rec.get("pass1", {})
    if not isinstance(p1, dict): return False
    return any(truthy(p1.get(k, False)) for k in keys)

def extract_step(rec: Dict[str, Any], src_path: str) -> int:
    if isinstance(rec.get("step"), (int, float)): return int(rec["step"])
    m = re.search(r"step[-_]?(\d{1,5})", src_path) or re.search(r"global_step[-_]?(\d{1,5})", src_path)
    return int(m.group(1)) if m else 0

def extract_entropy_pass1(rec: Dict[str, Any]) -> float | None:
    p1 = rec.get("pass1", {})
    if not isinstance(p1, dict): return None
    for k in ("answer_entropy", "entropy_answer"):
        v = p1.get(k, None)
        if isinstance(v, (int, float)): return float(v)
    tok = p1.get("answer_token_entropies") or p1.get("token_entropies") or p1.get("entropies")
    if isinstance(tok, list) and tok:
        try:
            vals = [float(t) for t in tok]
            if len(vals): return float(np.mean(vals))
        except Exception:
            pass
    return None

def iter_jsonl_files(root: str) -> Iterable[str]:
    if not root: return []
    p = Path(root)
    if not p.exists(): return []
    return (str(f) for f in p.rglob("*.jsonl"))

def load_pass1_entropy_and_aha(root: str, split: str, min_step: int, max_step: int, gpt_mode: str) -> List[tuple]:
    out: List[tuple] = []
    if not root: return out
    for fp in iter_jsonl_files(root):
        try:
            with open(fp, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line: continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if split and rec.get("split") and str(rec["split"]) != split:
                        continue
                    step = extract_step(rec, fp)
                    if step < min_step or step > max_step:
                        continue
                    ent = extract_entropy_pass1(rec)
                    if ent is None: 
                        continue  # pass1-only
                    aha = 1 if detect_aha_pass1(rec, gpt_mode) else 0
                    out.append((float(ent), aha))
        except Exception:
            continue
    return out

def main():
    args = parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    rows: List[tuple] = []
    rows += load_pass1_entropy_and_aha(args.root_carpark,   args.split, args.min_step, args.max_step, args.gpt_mode)
    rows += load_pass1_entropy_and_aha(args.root_crossword, args.split, args.min_step, args.max_step, args.gpt_mode)
    rows += load_pass1_entropy_and_aha(args.root_math,      args.split, args.min_step, args.max_step, args.gpt_mode)

    if not rows:
        print("[error] No PASS1 records with answer entropy found.", file=sys.stderr)
        sys.exit(1)

    ent = np.array([r[0] for r in rows], dtype=float)
    aha = np.array([r[1] for r in rows], dtype=int)

    # binning
    if args.binning == "quantile":
        qs = np.linspace(0.0, 1.0, args.bins+1)
        edges = np.unique(np.quantile(ent, qs))
        if len(edges) < 3:
            edges = np.linspace(ent.min(), ent.max(), args.bins+1)
    else:
        edges = np.linspace(ent.min(), ent.max(), args.bins+1)

    nb = len(edges)-1
    bin_idx = np.digitize(ent, edges) - 1
    centers = 0.5*(edges[:-1] + edges[1:])
    w = (centers[1]-centers[0]) if len(centers)>1 else 0.1

    counts_no = np.zeros(nb, dtype=float)
    counts_yes= np.zeros(nb, dtype=float)
    for b in range(nb):
        m = (bin_idx == b)
        if not m.any(): 
            continue
        counts_yes[b] = float((aha[m] == 1).sum())
        counts_no[b]  = float((aha[m] == 0).sum())

    if args.normalize:
        totals = counts_no + counts_yes
        totals[totals == 0] = 1.0
        counts_no  = counts_no / totals
        counts_yes = counts_yes / totals
        ylab = "Proportion (per bin)"
    else:
        ylab = "Count"

    fig, ax = plt.subplots(figsize=(args.width_in, args.height_in), constrained_layout=True)
    ax.bar(centers, counts_no,  width=w*0.9, label="No Aha")
    ax.bar(centers, counts_yes, width=w*0.9, bottom=counts_no, label="Aha")
    ax.set_xlabel("PASS1 answer entropy (binned)")
    ax.set_ylabel(ylab)
    ax.set_title(args.title)
    if args.normalize:
        ax.set_ylim(0, 1.0)
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.3)

    tag = args.outfile_tag or "combined"
    outname = f"graph_3_pass1_stacked_{tag}" + ("_normalized" if args.normalize else "")
    out = os.path.join(args.outdir, f"{outname}.png")
    plt.savefig(out, dpi=args.dpi)
    print(f"[ok] wrote {out}")

if __name__ == "__main__":
    main()
