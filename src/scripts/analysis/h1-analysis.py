#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
H1 GLM test (compact): Do Aha! moments help unconditionally?

We fit:     correct ~ C(problem) + step_std + aha
with Binomial GLM, cluster-robust SEs by problem, and report AME.

Aha variants (sample-level unless noted):
  • Words: pass1.has_reconsider_cue==1 (excludes injected cues)
  • GPT  : canonical shift labels (default) and, by default, GATED by Words (aha_gpt <= aha_words)
  • Formal (problem-step level): prior failure + prior shift-stability + shift now;
    merged back to samples and then AND-ed with sample's aha_gpt to enforce aha_formal <= aha_gpt.

Outputs
-------
CSV:
  - h1_glm_ame_summary.csv          # per-variant N, share_aha, Acc(aha=1), Acc(aha=0), Δ, AME, coef, p
  - h1_group_accuracy.csv           # overall acc for aha=0/1 by variant
  - h1_group_accuracy_delta.csv     # acc_aha1, acc_aha0, delta by variant
  - h1_group_accuracy_by_step.csv   # per-step acc for aha=0/1 by variant

Optional:
  - --tex_path h1_glm_ame_summary.tex   # compact LaTeX table with per-variant accuracies
  - --make_pdf                          # A4, 12pt summary PDF (h1_glm_summary.pdf)
"""

import os, re, json, argparse
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

# ---------- util ----------

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
    out.sort(); return out

def coerce_bool(x) -> Optional[int]:
    if x is None: return None
    if isinstance(x, bool): return int(x)
    if isinstance(x, (int, np.integer)): return int(bool(x))
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("1","true","t","yes","y"): return 1
        if s in ("0","false","f","no","n"): return 0
    return int(bool(x))

# ---------- labels ----------

def _aha_words(p1: Dict[str, Any]) -> int:
    v = coerce_bool(p1.get("has_reconsider_cue"))
    markers = p1.get("reconsider_markers") or []
    if isinstance(markers, list) and ("injected_cue" in markers):  # exclude injected
        return 0
    return 0 if v is None else int(v)

def _aha_gpt_canonical(p1: Dict[str, Any], rec: Dict[str, Any]) -> int:
    # canonical only
    for k in ("change_way_of_thinking", "shift_in_reasoning_v1"):
        v = p1.get(k, rec.get(k, None))
        if v is not None and coerce_bool(v) == 1:
            return 1
    return 0

def _aha_gpt_broad(p1: Dict[str, Any], rec: Dict[str, Any]) -> int:
    keys = ["change_way_of_thinking","shift_in_reasoning_v1","shift_llm","shift_gpt","pivot_llm","rechecked"]
    for k in keys:
        v = p1.get(k, rec.get(k, None))
        if v is not None and coerce_bool(v) == 1:
            return 1
    return 0

# ---------- load samples ----------

def load_samples(files: List[str], gpt_mode: str = "canonical", gate_gpt_by_words: bool = True) -> pd.DataFrame:
    """
    Returns per-sample rows with columns:
      problem, step, correct, aha_words, aha_gpt
    By default, enforces aha_gpt <= aha_words at the sample level.
    """
    rows = []
    for path in files:
        step_from_name = nat_step_from_path(path)
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                if not ln.strip(): continue
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

                words = _aha_words(p1)
                gpt_raw = (_aha_gpt_canonical if gpt_mode == "canonical" else _aha_gpt_broad)(p1, rec)
                gpt = int(gpt_raw and words) if gate_gpt_by_words else int(gpt_raw)

                rows.append({
                    "problem": str(prob),
                    "step": int(step),
                    "correct": int(correct),
                    "aha_words": int(words),
                    "aha_gpt": int(gpt),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No PASS-1 rows found.")
    return df

# ---------- problem-step table + Formal ----------

def build_problem_step(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["step","problem"], as_index=False)
    ps = grp.agg(
        n_samples=("correct","size"),
        freq_correct=("correct","mean"),
        aha_any_gpt=("aha_gpt", "max"),
        aha_rate_gpt=("aha_gpt","mean"),
    )
    for c in ("n_samples","aha_any_gpt"): ps[c] = ps[c].astype(int)
    ps["freq_correct"] = ps["freq_correct"].astype(float)
    ps["aha_rate_gpt"] = ps["aha_rate_gpt"].astype(float)
    return ps.sort_values(["problem","step"]).reset_index(drop=True)

def mark_formal(ps: pd.DataFrame, delta1: float, delta2: float, min_prior_steps: int) -> pd.DataFrame:
    """
    Formal at (problem,step): prior failure (max past freq_correct < δ1),
    prior shift-stability (max past aha_rate_gpt < δ2), and shift now (aha_any_gpt == 1).
    """
    need = {"step","problem","freq_correct","aha_rate_gpt","aha_any_gpt"}
    if not need.issubset(ps.columns): raise ValueError("mark_formal: missing cols")
    ps = ps.sort_values(["problem","step"]).copy()
    flags = np.zeros(len(ps), dtype=int); idx = 0
    for _, sub in ps.groupby("problem", sort=False):
        sub = sub.sort_values("step")
        freq = sub["freq_correct"].to_numpy(float)
        rate = sub["aha_rate_gpt"].to_numpy(float)
        shift = sub["aha_any_gpt"].to_numpy(int)
        for j in range(len(sub)):
            if j < min_prior_steps: flags[idx] = 0
            else:
                prior_ok = (float(np.max(freq[:j])) < delta1) and (float(np.max(rate[:j])) < delta2)
                flags[idx] = int(prior_ok and (shift[j] == 1))
            idx += 1
    ps["aha_formal_ps"] = flags  # problem-step level
    return ps

# ---------- GLM ----------

def _cov_spec(df: pd.DataFrame, cluster_by: str):
    if cluster_by == "problem":
        groups = pd.Categorical(df["problem"]).codes
        return "cluster", {"groups": groups, "use_correction": True, "df_correction": True}
    return "HC1", None

def fit_glm(df: pd.DataFrame, aha_col: str, out_txt: str,
            cluster_by: str = "problem") -> Dict[str, Any]:
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except Exception as e:
        raise RuntimeError("statsmodels is required (pip install statsmodels)") from e

    d = df.copy()
    d["step_std"] = (d["step"] - d["step"].mean()) / (d["step"].std(ddof=0) + 1e-8)
    if aha_col not in d.columns:
        raise ValueError(f"missing {aha_col}")

    model = smf.glm(f"correct ~ C(problem) + step_std + {aha_col}",
                    data=d, family=sm.families.Binomial())

    cov_type, cov_kwds = _cov_spec(d, cluster_by)
    try:
        res = model.fit(cov_type=cov_type, cov_kwds=(cov_kwds or {}))
    except TypeError:
        minimal_kw = {"groups": cov_kwds.get("groups")} if cov_kwds and "groups" in cov_kwds else {}
        res = model.fit(cov_type=cov_type, cov_kwds=minimal_kw)

    # AME: average p_hat(aha=1) - p_hat(aha=0)
    d1 = d.copy(); d1[aha_col] = 1
    d0 = d.copy(); d0[aha_col] = 0
    ame = float(np.mean(res.predict(d1) - res.predict(d0)))

    # Accuracies (overall and by aha)
    acc_overall = float(d["correct"].mean())
    m1 = d[aha_col] == 1
    m0 = d[aha_col] == 0
    acc_aha1 = float(d.loc[m1, "correct"].mean()) if m1.any() else float("nan")
    acc_aha0 = float(d.loc[m0, "correct"].mean()) if m0.any() else float("nan")
    delta_acc = (acc_aha1 - acc_aha0) if (np.isfinite(acc_aha1) and np.isfinite(acc_aha0)) else float("nan")

    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as fh:
        fh.write(res.summary().as_text())
        fh.write(f"\nCovariance: {cov_type}")
        if cov_kwds and "groups" in cov_kwds: fh.write(" (clustered by problem)")
        fh.write(f"\nAverage Marginal Effect (AME) of {aha_col}: {ame:.4f}\n")
        fh.write(f"\nAcc(overall)={acc_overall:.4f}  Acc(aha=1)={acc_aha1:.4f}  "
                 f"Acc(aha=0)={acc_aha0:.4f}  Δ={delta_acc:+.4f}\n")

    b  = float(res.params.get(aha_col, np.nan))
    se = float(res.bse.get(aha_col, np.nan))
    z  = b / se if se and np.isfinite(se) else np.nan
    p  = float(res.pvalues.get(aha_col, np.nan))

    return {
        "N": int(len(d)),
        "aha": aha_col,
        "share_aha": float(d[aha_col].mean()),
        "acc_overall": acc_overall,
        "acc_aha1": acc_aha1,
        "acc_aha0": acc_aha0,
        "delta_acc": delta_acc,
        "coef": b, "se": se, "z": z, "p": p, "AME": ame,
        "summary_path": out_txt,
    }

# ---------- Accuracy per grouping ----------

def compute_group_accuracy_tables(df: pd.DataFrame,
                                  out_dir: str) -> Tuple[str, str, str]:
    """
    For each variant (words/gpt/formal), compute:
      - overall accuracy for aha=0 and aha=1
      - delta table (acc_aha1 - acc_aha0)
      - per-step accuracy for aha=0/1
    """
    variants = [("aha_words","words"), ("aha_gpt","gpt"), ("aha_formal","formal")]

    rows_overall, rows_delta, rows_by_step = [], [], []

    for col, name in variants:
        if col not in df.columns:
            continue
        # Overall
        g = (df.groupby(df[col], as_index=False)
               .agg(n=("correct","size"), k=("correct","sum")))
        g["acc"] = g["k"] / g["n"]
        for _, r in g.iterrows():
            rows_overall.append({
                "variant": name,
                "aha": int(r[col]),
                "n": int(r["n"]),
                "k_correct": int(r["k"]),
                "accuracy": float(r["acc"])
            })
        # Delta
        acc1 = float(g.loc[g[col]==1, "acc"]) if (g[col]==1).any() else np.nan
        acc0 = float(g.loc[g[col]==0, "acc"]) if (g[col]==0).any() else np.nan
        n1   = int(g.loc[g[col]==1, "n"]) if (g[col]==1).any() else 0
        n0   = int(g.loc[g[col]==0, "n"]) if (g[col]==0).any() else 0
        rows_delta.append({
            "variant": name,
            "acc_aha1": acc1,
            "acc_aha0": acc0,
            "delta_acc": (acc1 - acc0) if (np.isfinite(acc1) and np.isfinite(acc0)) else np.nan,
            "n_aha1": n1,
            "n_aha0": n0
        })
        # By step
        gs = (df.groupby(["step", df[col]], as_index=False)
                .agg(n=("correct","size"), k=("correct","sum")))
        gs["accuracy"] = gs["k"] / gs["n"]
        gs = gs.rename(columns={col: "aha"})
        gs["variant"] = name
        rows_by_step.append(gs[["variant","step","aha","n","k","accuracy"]])

    overall_df = pd.DataFrame(rows_overall).sort_values(["variant","aha"]).reset_index(drop=True)
    delta_df = pd.DataFrame(rows_delta).sort_values(["variant"]).reset_index(drop=True)
    by_step_df = pd.concat(rows_by_step, ignore_index=True) if rows_by_step else pd.DataFrame()

    acc_csv = os.path.join(out_dir, "h1_group_accuracy.csv")
    delta_csv = os.path.join(out_dir, "h1_group_accuracy_delta.csv")
    step_csv = os.path.join(out_dir, "h1_group_accuracy_by_step.csv")

    overall_df.to_csv(acc_csv, index=False)
    delta_df.to_csv(delta_csv, index=False)
    by_step_df.to_csv(step_csv, index=False)

    return acc_csv, delta_csv, step_csv

# ---------- A4 PDF summary (Times New Roman 12pt) ----------

def write_a4_summary_pdf(summ_df: pd.DataFrame,
                         acc_delta_df: pd.DataFrame,
                         out_pdf: str,
                         dataset: str,
                         model: str,
                         font_family: str = "Times New Roman",
                         font_size: int = 12):
    """
    Create a one-page A4 PDF with 12pt text summarizing GLM and accuracy deltas.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": [font_family, "Times", "DejaVu Serif", "Computer Modern Roman"],
        "font.size": font_size,
        "axes.titlesize": font_size, "axes.labelsize": font_size,
        "xtick.labelsize": font_size, "ytick.labelsize": font_size,
        "legend.fontsize": font_size, "figure.titlesize": font_size,
    })

    A4 = (8.27, 11.69)
    fig = plt.figure(figsize=A4, dpi=300)
    ax = fig.add_axes([0.08, 0.08, 0.84, 0.84]); ax.axis("off")

    title = f"H1 GLM Summary — {dataset}, {model}"
    ax.text(0.0, 1.0, title, ha="left", va="top", weight="bold")

    cols = ["variant","N","share_aha","acc_aha1","acc_aha0","delta_acc","AME","coef","p"]
    hdr  = ["Variant","N","Share Aha","Acc(aha=1)","Acc(aha=0)","Δ (pp)","AME","Coef","p-value"]
    tab  = summ_df[cols].copy()
    def _fmt(x):
        if isinstance(x, (float, np.floating)):
            return f"{x:.4f}"
        return str(x)
    # convert delta to pp for display
    tab["delta_acc"] = tab["delta_acc"] * 100.0
    tab_disp = [[_fmt(v) for v in row] for row in tab.values.tolist()]

    y = 0.92
    line_h = 0.045
    ax.text(0.0, y, "GLM: correct ~ C(problem) + step_std + aha (cluster-robust SEs by problem)", ha="left", va="top")
    y -= 0.02

    x_positions = [0.00, 0.16, 0.28, 0.44, 0.59, 0.72, 0.82, 0.90, 0.97]
    for x, h in zip(x_positions, hdr):
        ax.text(x, y, h, ha="left", va="top", weight="bold")
    y -= 0.012
    ax.plot([0.00, 0.98], [y, y], color="black", lw=0.5)
    y -= 0.010
    for row in tab_disp:
        for x, cell in zip(x_positions, row):
            ax.text(x, y, cell, ha="left", va="top")
        y -= line_h

    # Accuracy delta (from file) quick recap
    y -= 0.02
    ax.text(0.0, y, "Group accuracy deltas (aha=1 vs aha=0):", ha="left", va="top", weight="bold")
    y -= 0.02
    hdr2 = ["Variant","acc(aha=1)","acc(aha=0)","Δ (pp)","n1","n0"]
    x2 = [0.00, 0.25, 0.45, 0.65, 0.82, 0.90]
    for x, h in zip(x2, hdr2):
        ax.text(x, y, h, ha="left", va="top", weight="bold")
    y -= 0.012
    ax.plot([0.00, 0.98], [y, y], color="black", lw=0.5)
    y -= 0.010

    acc_delta_df = acc_delta_df.copy()
    acc_delta_df["delta_pp"] = acc_delta_df["delta_acc"] * 100.0
    for _, r in acc_delta_df.sort_values("variant").iterrows():
        ax.text(x2[0], y, str(r["variant"]).title(), ha="left", va="top")
        ax.text(x2[1], y, f"{r['acc_aha1']:.4f}" if np.isfinite(r["acc_aha1"]) else "nan", ha="left", va="top")
        ax.text(x2[2], y, f"{r['acc_aha0']:.4f}" if np.isfinite(r["acc_aha0"]) else "nan", ha="left", va="top")
        ax.text(x2[3], y, f"{r['delta_pp']:+.2f}" if np.isfinite(r["delta_pp"]) else "nan", ha="left", va="top")
        ax.text(x2[4], y, f"{int(r['n_aha1'])}", ha="left", va="top")
        ax.text(x2[5], y, f"{int(r['n_aha0'])}", ha="left", va="top")
        y -= line_h

    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_root")
    ap.add_argument("--split", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--dataset_name", default="MATH-500")
    ap.add_argument("--model_name", default="Qwen2.5-1.5B")

    # GPT label policy
    ap.add_argument("--gpt_mode", choices=["canonical","broad"], default="canonical")
    # NOTE: default True to enforce aha_gpt ⊆ aha_words unless explicitly disabled.
    ap.add_argument("--no_gate_gpt_by_words", action="store_true",
                    help="If set, GPT shifts are NOT restricted to samples that also have Words-cue.")

    # Formal thresholds
    ap.add_argument("--delta1", type=float, default=0.13)
    ap.add_argument("--delta2", type=float, default=0.13)
    ap.add_argument("--min_prior_steps", type=int, default=2)

    # GLM cov
    ap.add_argument("--cluster_by", choices=["problem","none"], default="problem")

    # optional LaTeX table
    ap.add_argument("--tex_path", default=None, help="Write a LaTeX table here if provided (e.g., h1_glm_ame_summary.tex).")

    # Optional A4 PDF summary
    ap.add_argument("--make_pdf", action="store_true", help="Write an A4, 12pt summary PDF (h1_glm_summary.pdf).")
    ap.add_argument("--font_family", default="Times New Roman", help='PDF font family (default: "Times New Roman").')
    ap.add_argument("--font_size", type=int, default=12, help="PDF font size in points (default: 12).")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.results_root, "h1_glm_test")
    os.makedirs(out_dir, exist_ok=True)
    files = scan_files(args.results_root, args.split)
    if not files: raise SystemExit("No JSONL files found.")

    # Load per-sample rows (Words + GPT), with GPT gated by Words by default
    df = load_samples(files, gpt_mode=args.gpt_mode, gate_gpt_by_words=not args.no_gate_gpt_by_words)

    # Build Formal at problem-step, then merge back to samples AND gate by sample aha_gpt
    ps = build_problem_step(df)
    ps = mark_formal(ps, delta1=args.delta1, delta2=args.delta2, min_prior_steps=args.min_prior_steps)
    df = df.merge(ps[["step","problem","aha_formal_ps"]], on=["step","problem"], how="left").fillna({"aha_formal_ps":0})
    df["aha_formal_ps"] = df["aha_formal_ps"].astype(int)

    # Enforce strict subset: aha_formal <= aha_gpt (sample-level)
    df["aha_formal"] = (df["aha_formal_ps"] & df["aha_gpt"]).astype(int)

    # (Optional) sanity check / debug note on subset relations
    share_words  = float(df["aha_words"].mean())
    share_gpt    = float(df["aha_gpt"].mean())
    share_formal = float(df["aha_formal"].mean())
    if not (share_formal <= share_gpt + 1e-12 and share_gpt <= share_words + 1e-12):
        print("[warn] Subset relations violated (unexpected). shares:",
              f"words={share_words:.4f}, gpt={share_gpt:.4f}, formal={share_formal:.4f}")

    # Fit three GLMs
    rows = []
    for col, tag in [("aha_words","words"), ("aha_gpt","gpt"), ("aha_formal","formal")]:
        out_txt = os.path.join(out_dir, f"logit_pass1_correct_on_step_{col}.txt")
        r = fit_glm(df, aha_col=col, out_txt=out_txt, cluster_by=args.cluster_by)
        r["dataset"] = args.dataset_name; r["model"] = args.model_name; r["variant"] = tag
        rows.append(r)

    # Summary CSV
    summ = (pd.DataFrame(rows)[
        ["dataset","model","variant","N","share_aha",
         "acc_overall","acc_aha1","acc_aha0","delta_acc",
         "AME","coef","se","z","p","summary_path"]
    ].sort_values("variant"))
    csv_path = os.path.join(out_dir, "h1_glm_ame_summary.csv")
    summ.to_csv(csv_path, index=False)

    # Accuracy per grouping tables
    acc_csv, delta_csv, step_csv = compute_group_accuracy_tables(df, out_dir)

    # Optional LaTeX table (per-variant accuracies + Δ in pp)
    if args.tex_path:
        def fmt(x): 
            return f"{x:.4f}" if isinstance(x, (float, np.floating)) and np.isfinite(x) else str(x)
        lines = [
            "\\begin{tabular}{lrrrrrrrr}",
            "\\toprule",
            "Variant & N & Share Aha & Acc(aha=1) & Acc(aha=0) & $\\Delta$ (pp) & AME & Coef & $p$ \\\\",
            "\\midrule",
        ]
        for _, r in summ.iterrows():
            delta_pp = 100.0 * r["delta_acc"] if np.isfinite(r["delta_acc"]) else np.nan
            lines.append(
                f"{r['variant'].title()} & {int(r['N'])} & {fmt(r['share_aha'])} & "
                f"{fmt(r['acc_aha1'])} & {fmt(r['acc_aha0'])} & "
                f"{('+' if np.isfinite(delta_pp) and delta_pp>=0 else '')}{'' if not np.isfinite(delta_pp) else f'{delta_pp:.2f}'} & "
                f"{fmt(r['AME'])} & {fmt(r['coef'])} & {fmt(r['p'])} \\\\"
            )
        lines += ["\\bottomrule", "\\end{tabular}"]
        with open(args.tex_path, "w") as fh:
            fh.write("\n".join(lines))

    # Optional A4, 12pt summary PDF
    if args.make_pdf:
        pdf_path = os.path.join(out_dir, "h1_glm_summary.pdf")
        acc_delta_df = pd.read_csv(delta_csv)
        write_a4_summary_pdf(
            summ_df=summ,
            acc_delta_df=acc_delta_df,
            out_pdf=pdf_path,
            dataset=args.dataset_name,
            model=args.model_name,
            font_family=args.font_family,
            font_size=int(args.font_size),
        )
        print("A4 summary PDF:", pdf_path)

    # Console (variant-specific accuracies)
    print("Wrote:", csv_path)
    print("Group accuracy (overall):", acc_csv)
    print("Group accuracy (delta):  ", delta_csv)
    print("Group accuracy by step:  ", step_csv)
    for _, r in summ.iterrows():
        delta_pp = 100.0 * r["delta_acc"] if np.isfinite(r["delta_acc"]) else np.nan
        print(f"[{r['variant']}] N={int(r['N'])}, share_aha={r['share_aha']:.4f}, "
              f"acc(aha=1)={r['acc_aha1']:.4f}, acc(aha=0)={r['acc_aha0']:.4f}, "
              f"Δ={('+' if np.isfinite(delta_pp) and delta_pp>=0 else '')}{'' if not np.isfinite(delta_pp) else f'{delta_pp:.2f}'}pp, "
              f"AME={r['AME']:.4f}, coef={r['coef']:.4f}, p={r['p']:.4g}")
        print(f"  See: {r['summary_path']}")

if __name__ == "__main__":
    main()
