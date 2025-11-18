#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aha Ratios (Problem-level) — Dots-only + Trend Lines + Threshold Sweep
with Times (12 pt), domain-aware loading (Crossword + Math), independent y-axes,
and a Formal Aha! that includes δ3 (gain-at-shift)
---------------------------------------------------------------------------------

• Three-panel figure with per-domain overlay:
  – Dots only (no connecting lines) + 95% bootstrap CIs + WLS trend.
  – Independent y-axes (own ticks/limits).
  – RIGHT PANEL: dots turn GREEN at steps where the average gain-at-shift
    (p_correct_given_shift − p_correct) > 0 for that domain.

• LLM-detected shifts are gated (domain-aware) to avoid "all zeros" on Crosswords.

Panel titles (as requested):
  Left  : "Cue Phrases" Detection, Qwen2.5-1.5B
  Middle: LLM-Detected reasoning shifts, Qwen2.5-1.5B
  Right : Formal Reasoning Shifts (δ3 = 0), Qwen2.5-1.5B
  (Edit right title if you want to display a nonzero δ3.)

"""

import os, re, json, argparse
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ----------------- Global style: Times, size 12 -----------------

def set_global_fonts(font_family: str = "Times New Roman", font_size: int = 12):
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": [font_family, "Times", "DejaVu Serif", "Computer Modern Roman"],
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "figure.titlesize": font_size,
        "mathtext.fontset": "dejavuserif",
    })

def a4_size_inches(orientation: str = "landscape") -> Tuple[float, float]:
    if str(orientation).lower().startswith("p"):
        return (8.27, 11.69)
    return (11.69, 8.27)

# ----------------- Utils -----------------

STEP_PAT = re.compile(r"step(\d+)", re.I)

def slugify(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")

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

def coerce_bool(x) -> Optional[int]:
    if x is None: return None
    if isinstance(x, bool): return int(x)
    if isinstance(x, (int, np.integer)): return int(bool(x))
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("1","true","t","yes","y"): return 1
        if s in ("0","false","f","no","n"): return 0
    return int(bool(x))

def lighten_hex(hex_color: str, factor: float = 0.65) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16)/255.0 for i in (0,2,4))
    r = 1 - (1-r)*factor; g = 1 - (1-g)*factor; b = 1 - (1-b)*factor
    return "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))

# ----------------- Label helpers -----------------

def _any_keys_true(p1: Dict[str, Any], rec: Dict[str, Any], keys: List[str]) -> int:
    for k in keys:
        v = p1.get(k, rec.get(k, None))
        if v is None: continue
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
        if not files: continue
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
                    if not p1: continue

                    step = rec.get("step", step_from_name if step_from_name is not None else None)
                    if step is None: continue

                    prob = rec.get("problem") or rec.get("clue") or rec.get("row_key")
                    if prob is None:
                        di = rec.get("dataset_index")
                        prob = f"idx:{di}" if di is not None else "unknown"

                    correct = coerce_bool(p1.get("is_correct_pred"))
                    if correct is None: continue

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
    group_keys = ["domain", "step", "problem"] if "domain" in df.columns else ["step", "problem"]

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
                      delta3: Optional[float] = None) -> pd.DataFrame:
    need = {"step","problem","freq_correct","aha_rate_gpt","aha_any_gpt","p_correct_given_shift"}
    if not need.issubset(ps.columns):
        missing = need - set(ps.columns)
        raise ValueError(f"mark_formal_pairs: missing columns {missing}")

    group_cols = ["problem"]
    if "domain" in ps.columns:
        group_cols = ["domain"] + group_cols

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

# ------------- Problem-bootstrap ratios -------------

def bootstrap_problem_ratio(ps: pd.DataFrame, col: str, B: int = 1000, seed: int = 0) -> pd.DataFrame:
    if col not in ps.columns:
        raise KeyError(f"bootstrap_problem_ratio: column '{col}' not found in ps.")
    rng = np.random.default_rng(seed)
    rows = []
    for step, sub in ps.groupby("step"):
        vals = sub[col].astype(int).to_numpy()
        n = len(vals)
        if n == 0:
            rows.append({"step": int(step), "k": 0, "n": 0, "ratio": np.nan, "lo": np.nan, "hi": np.nan})
            continue
        mu = float(vals.mean())
        if B <= 0 or n == 1:
            lo = hi = np.nan
        else:
            bs = np.empty(B, dtype=float)
            for b in range(B):
                take = rng.integers(0, n, n)
                bs[b] = float(vals[take].mean())
            lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append({"step": int(step), "k": int(vals.sum()), "n": int(n),
                     "ratio": mu, "lo": float(lo), "hi": float(hi)})
    return pd.DataFrame(rows).sort_values("step")

# ------------- Trend (weighted LS) -------------

def fit_trend_wls(df: pd.DataFrame) -> Tuple[float,float,float,float,float,np.ndarray,np.ndarray]:
    x = df["step"].to_numpy(dtype=float)
    y = df["ratio"].to_numpy(dtype=float)
    w = df["n"].to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[m], y[m], w[m]
    if x.size < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan, x, y
    xbar = np.average(x, weights=w)
    ybar = np.average(y, weights=w)
    cov = np.average((x - xbar) * (y - ybar), weights=w)
    var = np.average((x - xbar) ** 2, weights=w)
    slope = cov / var if var > 0 else np.nan
    intercept = ybar - slope * xbar if np.isfinite(slope) else np.nan
    yhat = intercept + slope * x
    sse = np.sum(w * (y - yhat) ** 2)
    sst = np.sum(w * (y - ybar) ** 2)
    r2 = 1.0 - (sse / sst) if sst > 0 else np.nan
    slope_per_1k = slope * 1000.0
    delta_range = slope * (x.max() - x.min())
    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = intercept + slope * x_fit
    return float(slope), float(intercept), float(slope_per_1k), float(delta_range), float(r2), x_fit, y_fit

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

# ------------- Export all Formal Aha events (domain-aware) -------------

def export_formal_aha_json_with_text(ps: pd.DataFrame,
                                     files_by_domain: Dict[str, List[str]],
                                     dataset: str, model: str,
                                     delta1: float, delta2: float, delta3: Optional[float],
                                     min_prior_steps: int,
                                     gpt_keys: List[str], gpt_subset_native: bool,
                                     out_dir: str, slug: str,
                                     max_chars: int = 4000) -> Tuple[str, str, int]:
    needed = {"step","problem","aha_formal","n_samples","freq_correct","aha_rate_gpt","p_correct_given_shift"}
    if not needed.issubset(ps.columns):
        missing = needed - set(ps.columns)
        raise ValueError(f"export_formal_aha_json_with_text: missing columns: {missing}")

    if "domain" in ps.columns:
        targets = {(str(r["domain"]), str(r["problem"]), int(r["step"]))
                   for _, r in ps.loc[ps["aha_formal"]==1, ["domain","problem","step"]].iterrows()}
        ps_key = {(str(r["domain"]), str(r["problem"]), int(r["step"])): r for _, r in ps.iterrows()}
    else:
        targets = {("All", str(r["problem"]), int(r["step"]))
                   for _, r in ps.loc[ps["aha_formal"]==1, ["problem","step"]].iterrows()}
        ps = ps.copy(); ps["domain"] = "All"
        ps_key = {(str(r["domain"]), str(r["problem"]), int(r["step"])): r for _, r in ps.iterrows()}

    if not targets:
        json_path  = os.path.join(out_dir, f"formal_aha_events__{slug}.json")
        jsonl_path = os.path.join(out_dir, f"formal_aha_events__{slug}.jsonl")
        with open(json_path, "w", encoding="utf-8") as f: json.dump([], f)
        with open(jsonl_path, "w", encoding="utf-8") as f: pass
        return json_path, jsonl_path, 0

    events: List[Dict[str, Any]] = []
    remaining = set(targets)

    for dom, files in files_by_domain.items():
        if not files: continue
        for path in files:
            if not remaining: break
            step_from_name = nat_step_from_path(path)
            with open(path, "r", encoding="utf-8") as f:
                for ln in f:
                    if not remaining: break
                    ln = ln.strip()
                    if not ln: continue
                    try:
                        rec = json.loads(ln)
                    except Exception:
                        continue

                    step = rec.get("step", step_from_name if step_from_name is not None else None)
                    if step is None: continue
                    step = int(step)

                    prob_raw = rec.get("problem") or rec.get("clue") or rec.get("row_key")
                    if prob_raw is None:
                        di = rec.get("dataset_index")
                        prob_raw = f"idx:{di}" if di is not None else "unknown"
                    prob = str(prob_raw)

                    key = (dom, prob, step)
                    if key not in remaining: continue

                    p1 = rec.get("pass1") or {}
                    if not isinstance(p1, dict): continue

                    aha_gpt_now = _aha_gpt_for_rec(p1, rec, gpt_subset_native, gpt_keys, dom)
                    if aha_gpt_now != 1: continue

                    r = ps_key.get(key)
                    if r is None: continue

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
                        "answer": answer
                    })
                    remaining.discard(key)

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
            "answer": None
        })

    json_path  = os.path.join(out_dir, f"formal_aha_events__{slug}.json")
    jsonl_path = os.path.join(out_dir, f"formal_aha_events__{slug}.jsonl")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    return json_path, jsonl_path, len(events)

# ------------- Gain>0 flags for coloring Formal dots green -------------

def build_positive_delta_flags(ps: pd.DataFrame) -> Dict[str, Dict[int, bool]]:
    """
    For each (domain, step), return True if the average delta gain across problems
    with a shift is strictly positive: mean( p_correct_given_shift - p_correct ) > 0.
    If no problem at that step has a shift, mark False.
    """
    flags: Dict[str, Dict[int, bool]] = {}
    if "domain" in ps.columns:
        groups = ps.groupby(["domain","step"], sort=False)
    else:
        ps = ps.copy(); ps["domain"] = "All"
        groups = ps.groupby(["domain","step"], sort=False)

    for (dom, step), sub in groups:
        m = np.isfinite(sub["p_correct_given_shift"].to_numpy()) & (sub["aha_any_gpt"].to_numpy() == 1)
        if m.any():
            delta = sub.loc[m, "p_correct_given_shift"].to_numpy() - sub.loc[m, "freq_correct"].to_numpy()
            flag = bool(np.nanmean(delta) > 0.12)
        else:
            flag = False
        flags.setdefault(str(dom), {})[int(step)] = flag
    return flags

# ------------- Plot helpers (per-domain overlay) -------------

def _panel_auto_ylim(dfs_by_dom: Dict[str, pd.DataFrame],
                     pad: float = 0.05,
                     clamp: Tuple[Optional[float], Optional[float]] = (0.0, 1.0)
                     ) -> Tuple[float, float]:
    vals = []
    for df in dfs_by_dom.values():
        if df is None or df.empty: continue
        pieces = []
        for col in ("ratio", "lo", "hi"):
            if col in df:
                a = df[col].to_numpy(dtype=float)
                pieces.append(a[np.isfinite(a)])
        if pieces:
            a = np.concatenate(pieces)
            if a.size:
                vals.append((float(a.min()), float(a.max())))
    if not vals:
        return (0.0, 1.0)
    lo = min(v[0] for v in vals)
    hi = max(v[1] for v in vals)
    if hi == lo:
        hi = lo + 0.05
    rng = hi - lo
    lo -= pad * rng
    hi += pad * rng
    if clamp[0] is not None: lo = max(clamp[0], lo)
    if clamp[1] is not None: hi = min(clamp[1], hi)
    return (lo, hi)

def _plot_series_per_domain(ax,
                            dfs_by_dom: Dict[str, pd.DataFrame],
                            domain_colors: Dict[str, str],
                            alpha_ci: float,
                            ms: float = 5.0,
                            add_trend: bool = True,
                            trend_ls: str = "--",
                            highlight_by_dom: Optional[Dict[str, Dict[int, bool]]] = None,
                            highlight_color: str = "#2ca02c") -> List[Dict[str, Any]]:
    """
    Plot per-domain DOTS + CI + trend; optionally overlay GREEN dots on steps
    where highlight_by_dom[domain][step] is True. Return trend stats rows.
    """
    trend_rows = []
    for dom, df in dfs_by_dom.items():
        if df is None or df.empty: continue
        col = domain_colors.get(dom, None)
        ci  = lighten_hex(col, 0.65) if col else None

        # Base dots (domain color)
        ax.scatter(df["step"], df["ratio"], s=(ms**2), marker="o",
                   color=col, edgecolors="none", label=f"{dom}")

        # CI band
        if df["lo"].notna().any():
            ax.fill_between(df["step"], df["lo"], df["hi"], alpha=alpha_ci, color=ci)

        # Trend line
        slope, intercept, slope_k, delta, r2, x_fit, y_fit = fit_trend_wls(df)
        if add_trend and np.isfinite(slope) and x_fit.size >= 2:
            ax.plot(x_fit, y_fit, trend_ls, lw=2.0, color=col, alpha=0.95)

        # Overlay green markers for highlight steps
        if highlight_by_dom and str(dom) in highlight_by_dom:
            step_vals = df["step"].to_numpy(dtype=int)
            y_vals    = df["ratio"].to_numpy(dtype=float)
            mask = np.array([bool(highlight_by_dom[str(dom)].get(int(s), False)) for s in step_vals], dtype=bool)
            if mask.any():
                ax.scatter(step_vals[mask], y_vals[mask], s=(ms**2), marker="o",
                           color=highlight_color, edgecolors="none", zorder=3)

        trend_rows.append({
            "domain": dom,
            "slope_per_step": slope,
            "slope_per_1k": slope_k,
            "delta_over_range": delta,
            "intercept": intercept,
            "weighted_R2": r2,
        })
    return trend_rows

def _set_axes_box_aspect(ax, ratio: float):
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect(ratio)
    else:
        pos = ax.get_position()
        fig = ax.figure
        fw, fh = fig.get_size_inches()
        new_h = (pos.width * fw * ratio) / fh
        y0 = pos.y0 + 0.5 * (pos.height - new_h)
        ax.set_position([pos.x0, max(0.0, y0), pos.width, max(0.0, new_h)])

def plot_three_ratios_shared_axes_multi(
    native_by_dom: Dict[str, pd.DataFrame],
    gpt_by_dom: Dict[str, pd.DataFrame],
    formal_by_dom: Dict[str, pd.DataFrame],
    domain_colors: Dict[str, str],
    out_png: str,
    out_pdf: str,
    dataset: str,
    model: str,
    alpha_ci: float,
    a4_pdf: bool,
    a4_orientation: str,
    panel_box_aspect: float = 0.8,
    ms: float = 5.0,
    highlight_formal_by_dom: Optional[Dict[str, Dict[int, bool]]] = None,
    highlight_color: str = "#2ca02c",
) -> List[Dict[str, Any]]:
    fig_size = a4_size_inches(a4_orientation) if a4_pdf else (16.5, 4.8)
    fig, axes = plt.subplots(1, 3, figsize=fig_size, dpi=150, sharex=True, sharey=False)

    for ax in (axes if isinstance(axes, (list, tuple, np.ndarray)) else [axes]):
        _set_axes_box_aspect(ax, panel_box_aspect)

    left_title   = f"\"Cue Phrases\" Detection, {model}"
    middle_title = f"LLM-Detected reasoning shifts, {model}"
    right_title  = f"Formal Reasoning Shifts ($\\delta_3 = 0$), {model}"

    yl0 = _panel_auto_ylim(native_by_dom, pad=0.06, clamp=(0.0, 1.0))
    yl1 = _panel_auto_ylim(gpt_by_dom,    pad=0.06, clamp=(0.0, 1.0))
    yl2 = _panel_auto_ylim(formal_by_dom, pad=0.06, clamp=(0.0, 1.0))

    trend_rows_all: List[Dict[str, Any]] = []

    # Panel 0
    ax0 = axes[0]
    t0 = _plot_series_per_domain(ax0, native_by_dom, domain_colors, alpha_ci, ms=ms, add_trend=True)
    trend_rows_all += [{"series": "Words/Cue Phrases", **row} for row in t0]
    ax0.set_ylim(*yl0)
    ax0.set_xlabel("Training step"); ax0.set_ylabel("Ratio")
    ax0.set_title(left_title, pad=8); ax0.grid(True, alpha=0.35)
    ax0.tick_params(axis="y", labelleft=True)

    # Panel 1
    ax1 = axes[1]
    t1 = _plot_series_per_domain(ax1, gpt_by_dom, domain_colors, alpha_ci, ms=ms, add_trend=True)
    trend_rows_all += [{"series": "LLM-Detected Shifts", **row} for row in t1]
    ax1.set_ylim(*yl1)
    ax1.set_xlabel("Training step"); ax1.set_ylabel("Ratio")
    ax1.set_title(middle_title, pad=8); ax1.grid(True, alpha=0.35)
    ax1.tick_params(axis="y", labelleft=True)

    # Panel 2 (with green highlights)
    ax2 = axes[2]
    t2 = _plot_series_per_domain(ax2, formal_by_dom, domain_colors, alpha_ci, ms=ms,
                                 add_trend=True, highlight_by_dom=highlight_formal_by_dom,
                                 highlight_color=highlight_color)
    trend_rows_all += [{"series": "Formal Shifts", **row} for row in t2]
    ax2.set_ylim(*yl2)
    ax2.set_xlabel("Training step"); ax2.set_ylabel("Ratio")
    ax2.set_title(right_title, pad=8); ax2.grid(True, alpha=0.35)
    ax2.tick_params(axis="y", labelleft=True)

    # Legend (domains + green highlight key if present)
    handles, labels = [], []
    for dom, col in domain_colors.items():
        handles.append(Line2D([0],[0], marker="o", linestyle="None", color=col, markersize=ms, label=dom))
        labels.append(dom)
    any_green = bool(highlight_formal_by_dom) and any(
        any(flag for flag in stepmap.values()) for stepmap in highlight_formal_by_dom.values()
    )
    if any_green:
        handles.append(Line2D([0],[0], marker="o", linestyle="None", color=highlight_color,
                              markersize=ms, label="Δ > 0 at shift"))
        labels.append("Δ > 0 at shift")

    fig.legend(handles=handles, labels=labels, loc="lower center",
               ncol=min(4, max(2, len(domain_colors)+ (1 if any_green else 0))),
               frameon=False, bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.savefig(out_png)
    if a4_pdf:
        fig.set_size_inches(*a4_size_inches(a4_orientation))
    fig.savefig(out_pdf)
    plt.close(fig)
    return trend_rows_all

def plot_formal_sweep_grid(ps_base, delta1_list, delta2_list, min_prior_steps, B, seed,
                           out_png, out_pdf,
                           dataset, model, primary, ci, ymax, alpha_ci,
                           a4_pdf: bool, a4_orientation: str,
                           lw=2.0, ms=4.0, delta3: Optional[float] = None):
    if a4_pdf:
        fig_size = a4_size_inches(a4_orientation)
    else:
        fig_size = (4.8*max(1, len(delta2_list)), 3.2*max(1, len(delta1_list)))

    fig, axes = plt.subplots(len(delta1_list), len(delta2_list),
                             figsize=fig_size, dpi=140, sharex=True, sharey=True)
    axes = np.array(axes).reshape(len(delta1_list), len(delta2_list))

    for i, d1 in enumerate(delta1_list):
        for j, d2 in enumerate(delta2_list):
            ax = axes[i, j]
            ps = mark_formal_pairs(ps_base.copy(), delta1=float(d1), delta2=float(d2),
                                   min_prior_steps=min_prior_steps, delta3=delta3)
            formal_df = bootstrap_problem_ratio(ps, "aha_formal", B=B, seed=seed)
            ax.plot(formal_df["step"], formal_df["ratio"], marker="o", ms=ms, lw=lw, color=primary)
            if formal_df["lo"].notna().any():
                ax.fill_between(formal_df["step"], formal_df["lo"], formal_df["hi"], alpha=alpha_ci, color=ci)
            ax.set_ylim(0.0, ymax); ax.grid(True, alpha=0.35)
            if i == len(delta1_list) - 1: ax.set_xlabel("Training step")
            if j == 0:                     ax.set_ylabel('Formal "Aha!" ratio')
            title = f"δ1={d1:.2f}, δ2={d2:.2f}" + (f", δ3={delta3:.2f}" if delta3 is not None else "")
            ax.set_title(title, fontsize=12, pad=6)

    line = Line2D([0],[0], color=primary, lw=lw, marker="o", ms=ms, label="Ratio")
    patch = Patch(facecolor=ci, alpha=alpha_ci, label="95% CI (bootstrap)")
    fig.legend(handles=[line, patch], loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.15))
    fig.suptitle(f'Formal "Aha!" ratio sweep (problem-level)\n{dataset}, {model}', y=0.995, fontsize=12)

    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    fig.savefig(out_png)
    if a4_pdf:
        fig.set_size_inches(*a4_size_inches(a4_orientation))
    fig.savefig(out_pdf)
    plt.close(fig)

# ------------- Main -------------

def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_crossword", type=str, default=None)
    ap.add_argument("--root_math", type=str, default=None)
    ap.add_argument("results_root", nargs="?", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--dataset_name", default="MIXED")
    ap.add_argument("--model_name", default="Qwen2.5-1.5B")

    # Formal thresholds
    ap.add_argument("--delta1", type=float, default=0.20)
    ap.add_argument("--delta2", type=float, default=0.20)
    ap.add_argument("--delta3", type=float, default=None)
    ap.add_argument("--min_prior_steps", type=int, default=2)

    # Sweep grid
    ap.add_argument("--delta1_list", type=str, default="0.15,0.20,0.25")
    ap.add_argument("--delta2_list", type=str, default="0.15,0.20,0.25")

    # Step filters & panel
    ap.add_argument("--min_step", type=int, default=None)
    ap.add_argument("--max_step", type=int, default=None)
    ap.add_argument("--balanced_panel", action="store_true")

    # Colors
    ap.add_argument("--color_crossword", default="#4C78A8")
    ap.add_argument("--color_math", default="#E45756")

    # Bootstrap + style
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ymax", type=float, default=0.20)
    ap.add_argument("--ci_alpha", type=float, default=0.20)
    ap.add_argument("--ms", type=float, default=5.0)

    # Fonts
    ap.add_argument("--font_family", default="Times New Roman")
    ap.add_argument("--font_size", type=int, default=12)

    # GPT label policy
    ap.add_argument("--gpt_mode", choices=["canonical","broad"], default="canonical")
    ap.add_argument("--no_gpt_subset_native", action="store_true")

    # Output
    ap.add_argument("--out_basename", default=None)

    # A4 PDF
    ap.add_argument("--a4_pdf", action="store_true")
    ap.add_argument("--a4_orientation", choices=["landscape","portrait"], default="landscape")

    ap.add_argument("--panel_box_aspect", type=float, default=0.8)

    args = ap.parse_args()

    set_global_fonts(args.font_family, args.font_size)

    # Gather files
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

    out_dir = args.out_dir or os.path.join(first_root, "aha_ratios_bootstrap")
    os.makedirs(out_dir, exist_ok=True)
    slug = f"{slugify(args.dataset_name)}__{slugify(args.model_name)}"
    dataset, model = args.dataset_name, args.model_name

    # Colors
    domain_colors: Dict[str, str] = {}
    if "Crossword" in files_by_domain: domain_colors["Crossword"] = args.color_crossword
    if "Math" in files_by_domain:      domain_colors["Math"]      = args.color_math
    for d in files_by_domain:
        if d not in domain_colors:
            cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            domain_colors[d] = cycle[len(domain_colors) % len(cycle)]

    gpt_subset_native = not args.no_gpt_subset_native
    gpt_keys = (["change_way_of_thinking", "shift_in_reasoning_v1"] if args.gpt_mode == "canonical"
                else ["change_way_of_thinking", "shift_in_reasoning_v1",
                      "shift_llm", "shift_gpt", "pivot_llm", "rechecked"])

    # Load & filter
    df = load_pass1_samples_multi(files_by_domain, gpt_keys=gpt_keys, gpt_subset_native=gpt_subset_native)
    if args.min_step is not None: df = df[df["step"] >= args.min_step]
    if args.max_step is not None: df = df[df["step"] <= args.max_step]
    if df.empty: raise SystemExit("No rows left after step filtering.")

    if args.balanced_panel:
        parts = []
        for dom, sub in df.groupby("domain", sort=False):
            steps_d = np.sort(sub["step"].unique())
            have = (sub.groupby("problem")["step"].nunique() == len(steps_d))
            keep_probs = set(have[have].index.tolist())
            parts.append(sub[sub["problem"].isin(keep_probs)])
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if df.empty:
            raise SystemExit("Balanced panel filter removed all rows; relax filters.")

    # Aggregate once
    ps_base = build_problem_step(df)

    # ---------- Main 3-up ----------
    ps_main = mark_formal_pairs(ps_base.copy(),
                                delta1=args.delta1, delta2=args.delta2,
                                min_prior_steps=args.min_prior_steps,
                                delta3=args.delta3)

    # Domain bootstrap ratios
    native_by_dom: Dict[str, pd.DataFrame] = {}
    gpt_by_dom:    Dict[str, pd.DataFrame] = {}
    formal_by_dom: Dict[str, pd.DataFrame] = {}
    for dom, sub in ps_main.groupby("domain", sort=False):
        native_by_dom[dom] = bootstrap_problem_ratio(sub, "aha_any_native", B=args.B, seed=args.seed)
        gpt_by_dom[dom]    = bootstrap_problem_ratio(sub, "aha_any_gpt",    B=args.B, seed=args.seed)
        formal_by_dom[dom] = bootstrap_problem_ratio(sub, "aha_formal",     B=args.B, seed=args.seed)

    # Build green-dot flags for Formal panel
    highlight_flags = build_positive_delta_flags(ps_main)

    # Export Formal events (domain-aware)
    json_path, jsonl_path, n_events = export_formal_aha_json_with_text(
        ps_main, files_by_domain, dataset, model,
        delta1=args.delta1, delta2=args.delta2, delta3=args.delta3,
        min_prior_steps=args.min_prior_steps,
        gpt_keys=gpt_keys, gpt_subset_native=gpt_subset_native,
        out_dir=out_dir, slug=slug
    )
    print(f"Formal Aha events: {n_events} written to:\n  {json_path}\n  {jsonl_path}")

    # Output paths
    out_base = os.path.join(out_dir, args.out_basename) if args.out_basename \
               else os.path.join(out_dir, f"aha_ratios_problems_bootstrap__{slug}")
    out_png_main = out_base + ".png"
    out_pdf_main = out_base + ".pdf"

    # Draw main 3-up (independent y-axes; dots only + trend; green highlights on Formal)
    trend_rows = plot_three_ratios_shared_axes_multi(
        native_by_dom, gpt_by_dom, formal_by_dom, domain_colors,
        out_png=out_png_main, out_pdf=out_pdf_main,
        dataset=dataset, model=model, alpha_ci=args.ci_alpha,
        a4_pdf=bool(args.a4_pdf), a4_orientation=args.a4_orientation,
        panel_box_aspect=args.panel_box_aspect, ms=args.ms,
        highlight_formal_by_dom=highlight_flags, highlight_color="#2ca02c",
    )

    # Ratios CSV with domain column
    def _tag(df_, name, dom):
        d = df_.copy()
        d["series"] = name
        d["domain"] = dom
        return d[["domain","step","k","n","ratio","lo","hi","series"]]

    main_csv = os.path.join(out_dir, f"aha_ratios_problems_bootstrap__{slug}.csv")
    frames = []
    for dom in sorted(native_by_dom.keys()):
        frames.append(_tag(native_by_dom[dom], 'Words/Cue Phrases', dom))
        frames.append(_tag(gpt_by_dom[dom],    'LLM-Detected Shifts', dom))
        frames.append(_tag(formal_by_dom[dom], 'Formal Shifts',       dom))
    pd.concat(frames, ignore_index=True).sort_values(["series","domain","step"]).to_csv(main_csv, index=False)

    # Trend summary CSV (domain-tagged)
    trend_csv = os.path.join(out_dir, f"aha_trend_summary__{slug}.csv")
    pd.DataFrame(trend_rows).to_csv(trend_csv, index=False)

    # ---------- Threshold sweep (Formal only; pooled across domains) ----------
    d1_list = parse_float_list(args.delta1_list)
    d2_list = parse_float_list(args.delta2_list)

    out_png_sweep = os.path.join(out_dir, f"aha_formal_ratio_sweep__{slug}.png")
    out_pdf_sweep = os.path.join(out_dir, f"aha_formal_ratio_sweep__{slug}.pdf")

    plot_formal_sweep_grid(ps_base, d1_list, d2_list, args.min_prior_steps,
                           B=args.B, seed=args.seed,
                           out_png=out_png_sweep, out_pdf=out_pdf_sweep,
                           dataset=dataset, model=model,
                           primary="#2F5597", ci=lighten_hex("#2F5597", 0.65),
                           ymax=args.ymax, alpha_ci=args.ci_alpha,
                           a4_pdf=bool(args.a4_pdf), a4_orientation=args.a4_orientation,
                           delta3=args.delta3)

    # Sweep data CSV
    sweep_rows = []
    for d1 in d1_list:
        for d2 in d2_list:
            ps = mark_formal_pairs(ps_base.copy(), delta1=float(d1), delta2=float(d2),
                                   min_prior_steps=args.min_prior_steps, delta3=args.delta3)
            sdf = bootstrap_problem_ratio(ps, "aha_formal", B=args.B, seed=args.seed)
            sdf = sdf.assign(delta1=float(d1), delta2=float(d2),
                             delta3=args.delta3 if args.delta3 is not None else np.nan)
            sweep_rows.append(sdf)
    sweep_csv = os.path.join(out_dir, f"aha_formal_ratio_sweep__{slug}.csv")
    pd.concat(sweep_rows, ignore_index=True)\
      .sort_values(["delta1","delta2","step"]).to_csv(sweep_csv, index=False)

    # Console summary
    denom_note = ("(balanced panel per-domain; n constant within domain)" if args.balanced_panel
                  else "(unbalanced; n may vary by step and domain)")
    print("Wrote:")
    print("  3-up figure:", out_png_main, "and", out_pdf_main)
    print("  Ratios CSV :", main_csv, denom_note)
    print("  Trend CSV  :", trend_csv)
    print("  Sweep fig  :", out_png_sweep, "and", out_pdf_sweep)
    print("  Sweep CSV  :", sweep_csv)
    for row in trend_rows:
        print(f"[Trend] {row['series']} [{row['domain']}]: "
              f"slope/1k={row['slope_per_1k']:.4f}, Δ={row['delta_over_range']:.4f}, R^2={row['weighted_R2']:.3f}")

if __name__ == "__main__":
    main()
