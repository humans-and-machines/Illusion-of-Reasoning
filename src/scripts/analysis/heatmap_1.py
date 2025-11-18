#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aha! Prevalence Heatmap (δ1 × δ2; δ3 = ε>0)
-------------------------------------------
Builds heatmaps showing the share (%) of problem–step pairs that qualify as a
formal "Aha!" event under thresholds:

  • δ1 ∈ {0, 1/8, 2/8}  (max prior per-checkpoint accuracy fraction)
  • δ2 ∈ {0, 1/8, 2/8}  (max prior per-checkpoint shift fraction)
  • δ3 = ε > 0          (strict improvement at current step over BEST prior acc)

This version:
- adds Math3 (e.g., Llama-8B) support,
- emits per-domain heatmaps for all provided domains by default (Crossword, Math, Math2, Math3, Carpark),
- adds a collective heatmap for 1.5B-only domains (default: Crossword, Math, Carpark),
- makes the figure 3/4 the previous height,
- removes the colorbar on the right,
- sets a concise title and axis labels.
"""

import os
import re
import json
import argparse
from typing import Optional, List, Dict, Any, Tuple, Callable

import numpy as np
import pandas as pd

# Matplotlib setup
import matplotlib as mpl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.image import imread
from tempfile import NamedTemporaryFile

# -------- Style params ----------
params = {
    "axes.titlesize" : 14,
    "axes.labelsize" : 14,
    "font.size"      : 14,
    "legend.fontsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "font.family"    : "serif",
    "font.serif"     : ["Times New Roman", "Times", "Nimbus Roman No9 L", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "mathtext.rm"     : "serif",
    "text.usetex"     : False,
    "pdf.use14corefonts": False,
}
mpl.rcParams.update(params)

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
def _make_carpark_success_fn(op: str, thr: float) -> Callable[[Any], Optional[int]]:
    def _cmp(val: Any) -> Optional[int]:
        x = coerce_float(val)
        if x is None: return None
        if op == "gt": return int(x >  thr)
        if op == "ge": return int(x >= thr)
        if op == "eq": return int(x == thr)
        return int(x > thr)
    return _cmp

def _extract_soft_reward(rec: Dict[str, Any], p1: Dict[str, Any]) -> Optional[float]:
    return coerce_float(rec.get("soft_reward", p1.get("soft_reward")))

# ---------- Load sample-level rows ----------
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
                    if not ln: continue
                    try:
                        rec = json.loads(ln)
                    except Exception:
                        continue
                    p1 = rec.get("pass1") or {}
                    if not isinstance(p1, dict): continue

                    step = rec.get("step", step_from_name if step_from_name is not None else None)
                    if step is None: continue
                    try:
                        step = int(step)
                    except Exception:
                        continue
                    if min_step is not None and step < min_step: continue
                    if max_step is not None and step > max_step: continue

                    if dom_lower.startswith("carpark"):
                        sr = _extract_soft_reward(rec, p1)
                        ok = carpark_success_fn(sr)
                        if ok is None: continue
                        correct = int(ok)
                    else:
                        cb = coerce_bool(p1.get("is_correct_pred"))
                        if cb is None: continue
                        correct = int(cb)

                    pid = get_problem_id(rec)
                    if pid is None: continue
                    pid_full = f"{dom}::{pid}"

                    shift = aha_gpt_for_rec(p1, rec, gpt_subset_native, gpt_keys, dom)

                    rows.append({
                        "domain_key": str(dom),
                        "problem_id": pid_full,
                        "step": int(step),
                        "correct": int(correct),
                        "shift": int(shift),
                    })
    return pd.DataFrame(rows)

# ---------- Aggregate to per-(problem, step) ----------
def make_step_level(df_samples: pd.DataFrame) -> pd.DataFrame:
    g = df_samples.groupby(["domain_key","problem_id","step"], as_index=False)
    out = g.agg(acc_frac=("correct","mean"),
                shift_frac=("shift","mean"),
                n_samples=("correct","size"))
    return out.sort_values(["domain_key","problem_id","step"]).reset_index(drop=True)

# ---------- Aha detection ----------
def count_ahas(step_df: pd.DataFrame,
               delta1: float, delta2: float) -> Tuple[int, int]:
    n_events = 0
    n_pairs = 0
    for (_, pid), sub in step_df.groupby(["domain_key","problem_id"], sort=False):
        vals = sub[["step","acc_frac","shift_frac"]].to_numpy()
        vals = vals[np.argsort(vals[:,0])]
        acc_hist, shift_hist = [], []
        for step, acc, sh in vals:
            if not acc_hist:
                acc_hist.append(acc); shift_hist.append(sh); continue
            prior_max_acc   = float(np.max(acc_hist))
            prior_max_shift = float(np.max(shift_hist))
            n_pairs += 1
            if (prior_max_acc <= delta1) and (prior_max_shift <= delta2) and (acc > prior_max_acc):
                n_events += 1
            acc_hist.append(acc); shift_hist.append(sh)
    return n_events, n_pairs

# ---------- Heatmap helpers ----------
def frac8_label(x: float) -> str:
    k = int(round(x * 8))
    if abs(x * 8 - k) < 1e-6:
        return f"{k}/8"
    return f"{x:.3f}"

def get_rendered_size(fig, dpi=200):
    with NamedTemporaryFile(suffix=".png") as f:
        fig.savefig(f.name, bbox_inches="tight", dpi=dpi)
        h_px, w_px, _ = imread(f.name).shape
        return w_px / dpi, h_px / dpi

def set_rendered_width(fig, target_width_in: float, dpi=200, eps=1e-3, max_iter=8):
    w0, h0 = fig.get_size_inches()
    target_height = h0 * (target_width_in / max(w0, 1e-6))
    w_set, h_set = target_width_in, target_height
    for _ in range(max_iter):
        fig.set_size_inches([w_set, h_set])
        w_act, h_act = get_rendered_size(fig, dpi=dpi)
        if abs(w_act - target_width_in) < eps:
            return True
        scale = target_width_in / max(w_act, 1e-9)
        w_set *= scale
        h_set *= scale
        if w_set * dpi < 10 or h_set * dpi < 10:
            return False
    return False

def sweep_grid(step_df: pd.DataFrame, deltas: List[float]) -> pd.DataFrame:
    rows = []
    for d1 in deltas:
        for d2 in deltas:
            k, n = count_ahas(step_df, d1, d2)
            pct = (100.0 * k / n) if n > 0 else np.nan
            rows.append({"delta1": d1, "delta2": d2, "n_events": k, "n_pairs": n, "pct": pct})
    return pd.DataFrame(rows)

def plot_heatmap(df_grid: pd.DataFrame, title: str, out_png: str, cmap_name: str = "YlGnBu"):
    # Build matrix with consistent δ ordering
    levels = sorted(df_grid["delta1"].unique())
    Z = np.zeros((len(levels), len(levels))) * np.nan
    for _, r in df_grid.iterrows():
        i = levels.index(r["delta2"])
        j = levels.index(r["delta1"])
        Z[i, j] = r["pct"]

    vmin = 0.0
    vmax = float(np.nanmax(Z)) if np.isfinite(np.nanmax(Z)) else 1.0
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Shorter canvas (3/4 height)
    fig_w, fig_h = 5.8, 4.9 * 0.75
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap=cmap, norm=norm)

    # Axis ticks & labels (k/8)
    ax.set_xticks(range(len(levels))); ax.set_yticks(range(len(levels)))
    ax.set_xticklabels([frac8_label(x) for x in levels])
    ax.set_yticklabels([frac8_label(x) for x in levels])

    ax.set_xlabel(r"$\delta_1$ (Max prior failures)")
    ax.set_ylabel(r"$\delta_2$ (Max prior stability)")
    ax.set_title(title, pad=4)

    # Annotations
    for i in range(len(levels)):
        for j in range(len(levels)):
            val = Z[i, j]
            if not np.isfinite(val):
                continue
            d1 = levels[j]; d2 = levels[i]
            row = df_grid[(df_grid["delta1"]==d1) & (df_grid["delta2"]==d2)].iloc[0]
            rC, gC, bC, _ = cmap(norm(val))
            luminance = 0.299*rC + 0.587*gC + 0.114*bC
            fg = "black" if luminance > 0.55 else "white"
            ax.text(j, i, f"{val:.2f}%\n({int(row['n_events'])}/{int(row['n_pairs'])})",
                    ha="center", va="center", fontsize=12, color=fg)

    fig.tight_layout()
    set_rendered_width(fig, target_width_in=5.0, dpi=500)

    fig.savefig(out_png, dpi=500, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_crossword", type=str, default=None)
    ap.add_argument("--root_math",     type=str, default=None)
    ap.add_argument("--root_math2",    type=str, default=None)
    ap.add_argument("--root_math3",    type=str, default=None, help="Third math root (e.g., Llama-8B).")
    ap.add_argument("--root_carpark",  type=str, default=None)

    ap.add_argument("--label_crossword", type=str, default="Crossword")
    ap.add_argument("--label_math",      type=str, default="Qwen1.5B-Math")
    ap.add_argument("--label_math2",     type=str, default="Qwen7B-Math")
    ap.add_argument("--label_math3",     type=str, default="Llama8B-Math")
    ap.add_argument("--label_carpark",   type=str, default="Carpark")

    ap.add_argument("--split", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--dataset_name", default="MIXED")
    ap.add_argument("--model_name",   default="MIXED_MODELS")

    # Per-domain heatmaps: ON by default (all provided domains)
    ap.add_argument("--per_domain", action="store_true", default=True,
                    help="Emit per-domain heatmaps/rows (default: True).")

    # Titles
    ap.add_argument("--title_overall", type=str,
                    default="Aha! Moment Prevalence (All provided domains)",
                    help="Overall heatmap title (all domains).")
    ap.add_argument("--title_15b", type=str,
                    default="Aha! Moment Prevalence (Qwen-1.5B; Crossword+Math+Carpark)",
                    help="1.5B-only collective heatmap title.")

    # 1.5B-only collective heatmap
    ap.add_argument("--make_15b_overall", action="store_true", default=True,
                    help="Also emit a 1.5B-only overall heatmap (default: True).")
    ap.add_argument("--domains_15b", type=str, default="Crossword,Math,Carpark",
                    help="Comma-separated domain keys to include in 1.5B overall.")

    ap.add_argument("--cmap", type=str, default="YlGnBu")

    ap.add_argument("--gpt_mode", choices=["canonical","broad"], default="canonical")
    ap.add_argument("--no_gpt_subset_native", action="store_true")

    ap.add_argument("--min_step", type=int, default=None)
    ap.add_argument("--max_step", type=int, default=None)

    ap.add_argument("--delta_values", nargs="*", type=float, default=[0.0, 1/8, 2/8])

    ap.add_argument("--carpark_success_op", choices=["gt","ge","eq"], default="gt")
    ap.add_argument("--carpark_soft_threshold", type=float, default=0.0)

    args = ap.parse_args()

    # Human-facing labels for plot titles
    label_map: Dict[str, str] = {
        "Crossword": (args.label_crossword or "Crossword").strip(),
        "Math":      (args.label_math or "Math").strip(),
        "Math2":     (args.label_math2 or "Math2").strip(),
        "Math3":     (args.label_math3 or "Math3").strip(),
        "Carpark":   (args.label_carpark or "Carpark").strip(),
    }

    # Wire roots
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
    if args.root_math3:
        files_by_domain["Math3"] = scan_files(args.root_math3, args.split)
        first_root = first_root or args.root_math3
    if args.root_carpark:
        files_by_domain["Carpark"] = scan_files(args.root_carpark, args.split)
        first_root = first_root or args.root_carpark
    if not files_by_domain:
        raise SystemExit("Provide at least one --root_* folder.")

    total_files = sum(len(v) for v in files_by_domain.values())
    if total_files == 0:
        raise SystemExit("No JSONL files found. Check roots/--split.")

    out_dir = args.out_dir or os.path.join(first_root, "aha_heatmaps")
    os.makedirs(out_dir, exist_ok=True)

    # GPT policy
    gpt_subset_native = not args.no_gpt_subset_native
    gpt_keys = (["change_way_of_thinking","shift_in_reasoning_v1"] if args.gpt_mode == "canonical"
                else ["change_way_of_thinking","shift_in_reasoning_v1","shift_llm","shift_gpt","pivot_llm","rechecked"])

    # Hard cap step ≤ 1000
    HARD_MAX_STEP = 1000
    max_step_eff = HARD_MAX_STEP if args.max_step is None else min(args.max_step, HARD_MAX_STEP)
    if args.max_step is None or args.max_step > HARD_MAX_STEP:
        print(f"[info] Capping max_step to {max_step_eff} (hard cap = {HARD_MAX_STEP}).")

    carpark_success_fn = _make_carpark_success_fn(args.carpark_success_op, args.carpark_soft_threshold)

    # Load & aggregate
    df_samples = load_rows(files_by_domain, gpt_keys, gpt_subset_native,
                           min_step=args.min_step, max_step=max_step_eff,
                           carpark_success_fn=carpark_success_fn)
    if df_samples.empty:
        raise SystemExit("No rows after filtering.")
    step_df = make_step_level(df_samples)

    # Overall (all provided domains)
    overall_grid = sweep_grid(step_df, args.delta_values)
    slug = f"{args.dataset_name}__{args.model_name}".replace(" ", "_")

    long_rows = []
    grid_overall_csv = overall_grid.copy()
    grid_overall_csv["scope"] = "overall_all"
    grid_overall_csv["domain_key"] = "ALL"
    grid_overall_csv["domain_label"] = "ALL"
    long_rows.append(grid_overall_csv)

    out_png_overall = os.path.join(out_dir, "aha_heatmap_overall.png")
    plot_heatmap(overall_grid, args.title_overall, out_png_overall, cmap_name=args.cmap)

    # Per-domain (all present) — on by default
    if args.per_domain:
        for dom_key in sorted(step_df["domain_key"].unique(), key=str):
            dom_df = step_df[step_df["domain_key"] == dom_key]
            dom_grid = sweep_grid(dom_df, args.delta_values)
            dom_grid_csv = dom_grid.copy()
            dom_grid_csv["scope"] = "domain"
            dom_grid_csv["domain_key"] = dom_key
            dom_grid_csv["domain_label"] = label_map.get(dom_key, dom_key)
            long_rows.append(dom_grid_csv)
            plot_heatmap(dom_grid,
                         f"Aha! Moment Prevalence ({label_map.get(dom_key, dom_key)})",
                         os.path.join(out_dir, f"aha_heatmap_{dom_key}.png"),
                         cmap_name=args.cmap)

    # 1.5B-only collective heatmap (default ON)
    if args.make_15b_overall:
        include_keys = [k.strip() for k in (args.domains_15b or "").split(",") if k.strip()]
        present = [k for k in include_keys if k in set(step_df["domain_key"].unique())]
        if not present:
            print("[warn] 1.5B overall requested, but none of the requested domains are present. Skipping.")
        else:
            step_15b = step_df[step_df["domain_key"].isin(present)].copy()
            grid_15b = sweep_grid(step_15b, args.delta_values)
            grid_15b_csv = grid_15b.copy()
            grid_15b_csv["scope"] = "group"
            grid_15b_csv["group_key"] = "1p5b"
            grid_15b_csv["group_domains"] = ",".join(present)
            long_rows.append(grid_15b_csv)

            out_png_15b = os.path.join(out_dir, "aha_heatmap_overall_1p5b.png")
            title_15b = args.title_15b
            plot_heatmap(grid_15b, title_15b, out_png_15b, cmap_name=args.cmap)

    # Concatenate + save CSV
    table = pd.concat(long_rows, axis=0, ignore_index=True)
    out_csv = os.path.join(out_dir, f"aha_heatmap__{slug}.csv")
    table.to_csv(out_csv, index=False)

    # Small LaTeX helper line at δ≈0.13
    def nearest_delta(x: float, vals: List[float]) -> float:
        arr = np.asarray(vals)
        return float(arr[np.argmin(np.abs(arr - x))])

    d1r = nearest_delta(0.13, args.delta_values)
    d2r = nearest_delta(0.13, args.delta_values)
    row = overall_grid[(overall_grid["delta1"]==d1r) & (overall_grid["delta2"]==d2r)]
    if not row.empty:
        k = int(row["n_events"].iloc[0]); n = int(row["n_pairs"].iloc[0]); pct = float(row["pct"].iloc[0])
        sentence = (
            r"Fig.~\ref{fig:aha-heatmap} shows the prevalence of formal ``Aha!'' moments "
            rf"for {args.model_name}. Using $(\delta_1={frac8_label(d1r)}, \delta_2={frac8_label(d2r)}, \delta_3=\epsilon>0)$, "
            rf"we find {k} events out of {n} problem--step pairs ({pct:.2f}\%)."
        )
        out_tex = os.path.join(out_dir, f"aha_heatmap_summary__{slug}.tex")
        with open(out_tex, "w", encoding="utf-8") as f:
            f.write(sentence + "\n")
        print("\nLaTeX one-liner:\n" + sentence + "\n")
        print(f"Saved TEX -> {out_tex}")

    with pd.option_context("display.max_columns", None, "display.width", 120):
        disp = overall_grid.copy()
        disp["delta1"] = disp["delta1"].map(frac8_label)
        disp["delta2"] = disp["delta2"].map(frac8_label)
        disp["pct"]    = disp["pct"].map(lambda x: f"{x:.2f}%" if np.isfinite(x) else "NaN")
        print("\n== Overall Aha! prevalence grid (ALL domains) ==")
        print(disp.to_string(index=False))
        print(f"\nSaved CSV  -> {out_csv}")
        print(f"Saved figs -> {out_png_overall} (+ PDF)")
        if args.per_domain:
            print(f"          + per-domain heatmaps in {out_dir}/")
        if args.make_15b_overall:
            print(f"          + 1.5B-only overall heatmap if requested")

if __name__ == "__main__":
    main()
