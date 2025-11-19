#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Raw shift effect per step (scatter only; no trend lines, no legend)
-------------------------------------------------------------------
Computes and plots the *raw* effect of LLM-detected "Aha!" (shift) on success
per training step, hard-capped at step ≤ 1000:

    raw_effect(step) = P(success | shift=1, step) - P(success | shift=0, step)

Success definition by domain:
  • Crossword/Math/Math2: success = is_correct_pred
  • Carpark: success = 1[ soft_reward OP threshold ] (OP ∈ {gt, ge, eq}; default: gt 0.0)

Outputs
-------
1) CSV  : <out_dir>/raw_effect_per_step__<dataset>__<model>.csv
2) PNGs : <out_dir>/raw_effect_per_step_panels_linear.png   (3 panels; Math overlays Math & Math2)
          <out_dir>/raw_effect_per_step_overlay_linear.png  (overlay across domains, not forced square)
          (PDFs saved alongside)

Plot units
----------
By default, plots are rendered in percentage points (pp), with ticks at
-20,-10,0,10,20 and axis padded slightly beyond ±20 ([-22,22]) so points/error
bars aren’t clipped. CSV remains in probability units (0..1).
"""

import os, re, json, argparse
from typing import Optional, List, Dict, Any, Callable, Tuple

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from textwrap import fill

try:
    # Package imports
    from .labels import aha_gpt_for_rec
    from .utils import coerce_bool, coerce_float, get_problem_id, nat_step_from_path
except ImportError:  # pragma: no cover - script fallback
    import os as _os
    import sys as _sys

    _ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _ROOT not in _sys.path:
        _sys.path.append(_ROOT)
    from analysis.labels import aha_gpt_for_rec  # type: ignore
    from analysis.utils import coerce_bool, coerce_float, get_problem_id, nat_step_from_path  # type: ignore

# --------- Style params ----------
STYLE_PARAMS = {
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
    "savefig.bbox": "standard",
    "savefig.pad_inches": 0.02,
}
mpl.rcParams.update(STYLE_PARAMS)

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

# ---------- Carpark helpers ----------
def _extract_soft_reward(rec: Dict[str, Any], p1: Dict[str, Any]) -> Optional[float]:
    return coerce_float(rec.get("soft_reward", p1.get("soft_reward")))

def _make_carpark_success_fn(op: str, thr: float):
    def _cmp(val: Any) -> Optional[int]:
        x = coerce_float(val)
        if x is None: return None
        if   op=="gt": return int(x >  thr)
        elif op=="ge": return int(x >= thr)
        elif op=="eq": return int(x == thr)
        return int(x > thr)
    return _cmp

# ---------- Load rows ----------
def load_rows(files_by_domain: Dict[str,List[str]],
              gpt_keys: List[str],
              gpt_subset_native: bool,
              min_step: Optional[int],
              max_step: Optional[int],
              carpark_success_fn: Callable[[Any], Optional[int]]) -> pd.DataFrame:
    rows=[]
    for dom, files in files_by_domain.items():
        dom_lower=str(dom).lower()
        for path in files:
            step_from_name=nat_step_from_path(path)
            with open(path,"r",encoding="utf-8") as f:
                for ln in f:
                    s=ln.strip()
                    if not s: continue
                    try: rec=json.loads(s)
                    except Exception: continue
                    p1=rec.get("pass1") or {}
                    if not isinstance(p1,dict): continue

                    step=rec.get("step", step_from_name if step_from_name is not None else None)
                    if step is None: continue
                    try: step=int(step)
                    except Exception: continue

                    if min_step is not None and step < min_step: continue
                    if max_step is not None and step > max_step: continue

                    if dom_lower.startswith("carpark"):
                        sr=_extract_soft_reward(rec,p1)
                        success=carpark_success_fn(sr)
                    else:
                        success=coerce_bool(p1.get("is_correct_pred"))
                    if success is None: continue

                    pid=get_problem_id(rec)
                    if pid is None: continue

                    shift=aha_gpt_for_rec(p1, rec, gpt_subset_native, gpt_keys, dom)

                    rows.append({
                        "domain":str(dom),
                        "problem_id":pid,
                        "step":int(step),
                        "correct":int(success),
                        "shift":int(shift),
                    })
    return pd.DataFrame(rows)

# ---------- Per-step aggregation ----------
def per_step_raw_effect(df: pd.DataFrame, domain: str, min_per_group: int=20) -> pd.DataFrame:
    sub=df[df["domain"]==domain].copy()
    if sub.empty:
        return pd.DataFrame(columns=["domain","step","n","n_shift","n_noshift",
                                     "p_correct_shift","p_correct_noshift","raw_effect"])
    out=[]
    for step,g in sub.groupby("step"):
        n=len(g); n_shift=int((g["shift"]==1).sum()); n_noshift=n-n_shift
        if n_shift==0 or n_noshift==0 or n<min_per_group: continue
        p_s=float(g.loc[g["shift"]==1,"correct"].mean())
        p_n=float(g.loc[g["shift"]==0,"correct"].mean())
        out.append({
            "domain":domain, "step":int(step), "n":int(n),
            "n_shift":int(n_shift), "n_noshift":int(n_noshift),
            "p_correct_shift":p_s, "p_correct_noshift":p_n,
            "raw_effect":p_s - p_n,
        })
    return pd.DataFrame(out).sort_values("step")

# ---------- Plotters ----------
_COLORS={"Crossword":"#1f77b4","Math":"#2ca02c","Math2":"#9467bd","Carpark":"#d62728"}
def _color_for(dom_key:str)->str: return _COLORS.get(dom_key,"C0")

def plot_panels(per_step: Dict[str,pd.DataFrame], label_map: Dict[str,str], out_png: str,
                dpi:int, width_in:float, marker_size:float, height_scale:float,
                y_scale: float, ylim: Tuple[float,float], yticks: Optional[List[float]], ylabel: str):
    base_w, base_h = 9.0, 4.2
    height_in = max(2.0, (width_in * (base_h/base_w)) * float(height_scale))
    fig,axes=plt.subplots(1,3, figsize=(width_in, height_in), sharey=True, constrained_layout=False)
    doms=["Crossword","Math","Carpark"]

    for ax,dom in zip(axes,doms):
        ax.set_title(dom)
        ax.axhline(0.0,lw=1,color="k",alpha=0.35)
        if dom=="Math":
            for key in ["Math","Math2"]:
                df=per_step.get(key)
                if df is None or df.empty: continue
                yvals = df["raw_effect"].values * y_scale
                ax.scatter(df["step"], yvals, s=marker_size, alpha=0.55,
                           edgecolor="none", color=_color_for(key))
        else:
            df=per_step.get(dom)
            if df is not None and not df.empty:
                yvals = df["raw_effect"].values * y_scale
                ax.scatter(df["step"], yvals, s=marker_size, alpha=0.55,
                           edgecolor="none", color=_color_for(dom))
        ax.set_xlabel("Step")
        ax.set_ylim(ylim[0], ylim[1])
        if yticks is not None:
            ax.set_yticks(yticks)

    axes[0].set_ylabel(ylabel)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.90, bottom=0.3)
    fig.set_size_inches(width_in, height_in, forward=True)
    fig.savefig(out_png, dpi=dpi)
    fig.savefig(out_png.replace(".png",".pdf"))
    plt.close(fig)

def _auto_wrap_title_to_two_lines(s:str, width:int=42)->str:
    wrapped=fill(s,width=width)
    lines=wrapped.splitlines()
    return "\n".join(lines[:2]) if len(lines)>2 else wrapped

def plot_overlay_all(per_step: Dict[str,pd.DataFrame], label_map: Dict[str,str], out_png: str,
                     dpi:int, overlay_width_in:float, overlay_height_scale:float,
                     marker_size:float, title: Optional[str],
                     y_scale: float, ylim: Tuple[float,float], yticks: Optional[List[float]], ylabel: str):
    base_w, base_h = 9.0, 7.0
    height_in = max(3, (overlay_width_in * (base_h/base_w)) * float(overlay_height_scale))
    fig,ax=plt.subplots(figsize=(overlay_width_in, height_in), constrained_layout=False)

    if title:
        t=_auto_wrap_title_to_two_lines(title, width=42)
        ax.set_title(t, pad=6)

    bottom_margin = 0.19
    top_margin = 0.86 if title and "\n" in t else 0.90
    fig.subplots_adjust(left=0.2, right=0.98, top=top_margin, bottom=bottom_margin)

    ax.axhline(0.0,lw=1,color="k",alpha=0.35)
    for dom in ["Crossword","Math","Math2","Carpark"]:
        df=per_step.get(dom)
        if df is None or df.empty: continue
        yvals = df["raw_effect"].values * y_scale
        ax.scatter(df["step"], yvals, s=marker_size, alpha=1,
                   edgecolor="none", color=_color_for(dom))

    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim[0], ylim[1])
    if yticks is not None:
        ax.set_yticks(yticks)

    fig.set_size_inches(overlay_width_in, height_in, forward=True)
    fig.savefig(out_png, dpi=dpi)
    fig.savefig(out_png.replace(".png",".pdf"))
    plt.close(fig)

# ---------- Main ----------
def _parse_float_list(s: Optional[str]) -> Optional[List[float]]:
    if s is None: return None
    s = s.strip()
    if not s: return None
    parts = re.split(r"[,\s]+", s)
    out = []
    for p in parts:
        if not p: continue
        try:
            out.append(float(p))
        except Exception:
            pass
    return out or None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root_crossword", type=str, default=None)
    ap.add_argument("--root_math", type=str, default=None)
    ap.add_argument("--root_math2", type=str, default=None)
    ap.add_argument("--label_math", type=str, default="Qwen1.5B-Math")
    ap.add_argument("--label_math2", type=str, default="Qwen7B-Math")
    ap.add_argument("--root_carpark", type=str, default=None)

    ap.add_argument("results_root", nargs="?", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--dataset_name", default="MIXED")
    ap.add_argument("--model_name", default="Qwen2.5-1.5B")
    ap.add_argument("--min_per_group", type=int, default=20)

    ap.add_argument("--gpt_mode", choices=["canonical","broad"], default="canonical")
    ap.add_argument("--no_gpt_subset_native", action="store_true")

    ap.add_argument("--min_step", type=int, default=None)
    ap.add_argument("--max_step", type=int, default=None)

    ap.add_argument("--carpark_success_op", choices=["gt","ge","eq"], default="gt")
    ap.add_argument("--carpark_soft_threshold", type=float, default=0.0)

    # Figure knobs
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--width_in", type=float, default=9.0,
                    help="Canvas width (inches) for 3-panel figure")
    ap.add_argument("--height_scale", type=float, default=0.6667,
                    help="Height scale for the 3-panel figure (width fixed)")
    ap.add_argument("--overlay_width_in", type=float, default=5,
                    help="Canvas width (inches) for overlay figure (not forced square)")
    ap.add_argument("--overlay_height_scale", type=float, default=0.8,
                    help="Height scale for overlay figure (relative to overlay_width_in)")
    ap.add_argument("--marker_size", type=float, default=28.0)
    ap.add_argument("--overlay_title", type=str, default=None)

    # Units, limits, ticks
    ap.add_argument("--plot_units", choices=["pp","prob"], default="pp",
                    help="Axis units: 'pp' = percentage points (×100), 'prob' = raw probabilities.")
    ap.add_argument("--ymin_pp", type=float, default=-50.0)
    ap.add_argument("--ymax_pp", type=float, default=5.0)
    ap.add_argument("--ymin_prob", type=float, default=-0.2)
    ap.add_argument("--ymax_prob", type=float, default=0.2)
    ap.add_argument("--ylim_pad_pp", type=float, default=2.0,
                    help="Extend beyond [ymin_pp,ymax_pp] by this many pp (default 2).")
    ap.add_argument("--ylim_pad_prob", type=float, default=0.02,
                    help="Extend beyond [ymin_prob,ymax_prob] by this amount (default 0.02).")
    ap.add_argument("--yticks_pp", type=str, default="-40,-20,0",
                    help="Comma/space-separated tick values when plot_units=pp.")
    ap.add_argument("--yticks_prob", type=str, default="-0.2,-0.1,0,0.1,0.2",
                    help="Comma/space-separated tick values when plot_units=prob.")

    args=ap.parse_args()

    label_map={"Crossword":"Xword",
               "Math": args.label_math.strip() or "Math",
               "Math2": args.label_math2.strip() or "Qwen7B-Math",
               "Carpark":"Rush Hour"}

    files_by_domain: Dict[str,List[str]]={}
    first_root=None
    if args.root_crossword:
        files_by_domain["Crossword"]=scan_files(args.root_crossword, args.split); first_root=first_root or args.root_crossword
    if args.root_math:
        files_by_domain["Math"]=scan_files(args.root_math, args.split); first_root=first_root or args.root_math
    if args.root_math2:
        files_by_domain["Math2"]=scan_files(args.root_math2, args.split); first_root=first_root or args.root_math2
    if args.root_carpark:
        files_by_domain["Carpark"]=scan_files(args.root_carpark, args.split); first_root=first_root or args.root_carpark
    if not files_by_domain:
        if not args.results_root: raise SystemExit("Provide --root_* or results_root.")
        files_by_domain["All"]=scan_files(args.results_root, args.split); first_root=args.results_root

    if sum(len(v) for v in files_by_domain.values())==0:
        raise SystemExit("No JSONL files found. Check roots/--split.")

    out_dir=args.out_dir or os.path.join(first_root, "raw_effect_plots")
    os.makedirs(out_dir, exist_ok=True)

    gpt_keys=(["change_way_of_thinking","shift_in_reasoning_v1"] if args.gpt_mode=="canonical"
              else ["change_way_of_thinking","shift_in_reasoning_v1","shift_llm","shift_gpt","pivot_llm","rechecked"])
    gpt_subset_native=not args.no_gpt_subset_native

    HARD_MAX_STEP=1000
    effective_max_step = HARD_MAX_STEP if args.max_step is None else min(args.max_step, HARD_MAX_STEP)
    if args.max_step is None or args.max_step > HARD_MAX_STEP:
        print(f"[info] Capping max_step to {effective_max_step} (hard cap = {HARD_MAX_STEP}).")

    carpark_success_fn=_make_carpark_success_fn(args.carpark_success_op, args.carpark_soft_threshold)

    df=load_rows(files_by_domain, gpt_keys, gpt_subset_native,
                 min_step=args.min_step, max_step=effective_max_step,
                 carpark_success_fn=carpark_success_fn)
    if df.empty: raise SystemExit("No rows after filtering.")

    per_step: Dict[str,pd.DataFrame]={}; rows_all=[]
    for dom in ["Crossword","Math","Math2","Carpark"]:
        if dom not in df["domain"].unique(): continue
        d=per_step_raw_effect(df, dom, min_per_group=args.min_per_group)
        per_step[dom]=d
        if not d.empty: rows_all.append(d)
    if not rows_all: raise SystemExit("No per-step groups met the minimum requirements.")

    table=pd.concat(rows_all, axis=0, ignore_index=True)
    table["domain"]=table["domain"].map(lambda d: label_map.get(d,d))
    slug=f"{args.dataset_name}__{args.model_name}".replace(" ","_")
    out_csv=os.path.join(out_dir, f"raw_effect_per_step__{slug}.csv")
    table.to_csv(out_csv, index=False)

    # Determine plotting units/limits/labels/ticks
    if args.plot_units == "pp":
        y_scale = 100.0
        base_lim = (args.ymin_pp, args.ymax_pp)
        pad = args.ylim_pad_pp
        yticks = _parse_float_list(args.yticks_pp) or [-20., -10., 0., 10., 20.]
        ylabel = "Raw effect of training \nstep on accuracy (pp)"
    else:
        y_scale = 1.0
        base_lim = (args.ymin_prob, args.ymax_prob)
        pad = args.ylim_pad_prob
        yticks = _parse_float_list(args.yticks_prob) or [-0.2, -0.1, 0.0, 0.1, 0.2]
        ylabel = "Raw effect on accuracy"

    ylim = (base_lim[0] - pad, base_lim[1] + pad)

    # 3-panel figure
    plot_panels(per_step, label_map,
                os.path.join(out_dir,"raw_effect_per_step_panels_linear.png"),
                dpi=args.dpi, width_in=args.width_in, marker_size=args.marker_size,
                height_scale=args.height_scale, y_scale=y_scale, ylim=ylim, yticks=yticks, ylabel=ylabel)

    # Overlay figure (NOT square unless you set matching width/height)
    overlay_path=os.path.join(out_dir,"raw_effect_per_step_overlay_linear.png")
    overlay_title=args.overlay_title or f"Raw Effect of LLM-Detected Shifts by Training Step ({args.model_name})"
    plot_overlay_all(per_step, label_map, overlay_path,
                     dpi=args.dpi, overlay_width_in=args.overlay_width_in,
                     overlay_height_scale=args.overlay_height_scale,
                     marker_size=args.marker_size, title=overlay_title,
                     y_scale=y_scale, ylim=ylim, yticks=yticks, ylabel=ylabel)

    # Back-compat filename
    plot_overlay_all(per_step, label_map,
                     os.path.join(out_dir,"raw_effect_per_step_crossword_math_linear.png"),
                     dpi=args.dpi, overlay_width_in=args.overlay_width_in,
                     overlay_height_scale=args.overlay_height_scale,
                     marker_size=args.marker_size, title=overlay_title,
                     y_scale=y_scale, ylim=ylim, yticks=yticks, ylabel=ylabel)

    with pd.option_context("display.max_columns", None, "display.width", 120):
        print("\nRaw effect per step (head):")
        print(table.head(12).to_string(index=False))
        print(f"\nSaved CSV -> {out_csv}")
        print(f"Saved figs -> {overlay_path} (+ PDF), "
              f"{out_dir}/raw_effect_per_step_panels_linear.[PNG|PDF], "
              f"{out_dir}/raw_effect_per_step_crossword_math_linear.[PNG|PDF]")
        if args.plot_units == "pp":
            print(f"[info] Plots use percentage points (pp). Ticks: {yticks}. "
                  f"Axis limits with pad: {ylim}. CSV remains in probabilities (0..1).")

if __name__ == "__main__":
    main()
