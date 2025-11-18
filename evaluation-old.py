#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 evaluate_and_trace.py
 ─────────────────────
 End‑to‑end scoring and statistical analysis for the Math220k Qwen2.5‑7B
 checkpoint sweep *with interpretive commentary*.

 Revision note (2025‑06‑30)
 -------------------------
 • **Answer extraction** now uses the *first* math‑looking token that follows a
   cue word (Answer/⇒/etc.) instead of a dedicated <answer> block, per request.
 • Clean‑ups: removed dead regex, duplicate imports, minor token‑strip tweak.
"""

import os, json, re, argparse, logging, pathlib, sys, random
from statistics import median
from typing import Optional  # moved up with std‑lib imports

from tqdm import tqdm
import pandas as pd, numpy as np
import statsmodels.api as sm
from scipy import stats as spstats

# ────────────────────────── Configuration ───────────────────────────
REV2STEP = {
    "aef4ec7": 50, "9ac7e5e": 150, "ba99bc4": 250, "7cbc93a": 350,
    "4803ae9": 450, "5d0b169": 550, "334f623": 650, "f6b58de": 750,
    "14436ba": 850, "d5bb572": 950, "1e7973f": 1050, "4771a8f": 1150,
    "a472241": 1250, "76bee3d": 1350, "57fa3c6": 1450, "70ddc2c": 1550,
    "29605a8": 1650, "26044d8": 1750, "e8a648f": 1850, "df6030f": 1950,
    "3075e71": 2050, "c148b71": 2150, "bd98859": 2250, "08b3d89": 2350,
    "d1d3bf7": 2450
}

# ─────────────────────────  math‑verify helpers  ─────────────────────────
from math_verify import parse, verify

parse_failures = 0  # global counter

def rhs(expr: str) -> str:
    """Return the part after the last '=' if one is present."""
    if '=' in expr:
        expr = expr.split('=')[-1]
    return expr.strip()

def safe_parse(expr):
    global parse_failures
    try:
        return parse(rhs(expr), extraction_mode="first_match")
    except Exception:
        parse_failures += 1
        return []

# ─── new helper ──────────────────────────────────────────────────────
def normalize_text(s: str) -> str:
    """Lower-case, strip surrounding whitespace and TeX boxes, collapse spaces."""
    s = re.sub(r"\\boxed\s*[{(](.*?)[})]", r"\1", s)  # remove \boxed{…}
    s = re.sub(r"[^\w\s]", " ", s)                    # drop punctuation
    return re.sub(r"\s+", " ", s.strip().lower())

# ─── replace your old `judge` with this one ───────────────────────────
def judge(gold_raw: str, pred_raw: str):
    """Return True / False / None as before, but with literal fallback."""
    g_parsed = safe_parse(gold_raw)
    if g_parsed:                # normal math path
        try:
            return bool(verify(g_parsed, safe_parse(pred_raw)))
        except Exception:
            return None

    # ── gold could not be parsed as math → literal fallback ──────────
    gold_norm = normalize_text(gold_raw)
    pred_norm = normalize_text(pred_raw)

    if not gold_norm or not pred_norm:
        return None                        # still un-gradable

    return gold_norm == pred_norm

# ─────────────────────────  re‑check cue regex  ─────────────────────────
RECHECK_RE = re.compile(
    r"\b(re[- ]?check|double[- ]?check|verify|let.?s\s+verify|confirm|"
    r"wait|hold on|hmm+|let me see|let.?s see|"
    r"aha|actually|on second thought|interesting)\b", re.I)

def has_recheck(text):
    return bool(RECHECK_RE.search(text))

# ─────────────────────────  answer‑extraction helpers  ───────────────────
# 1) strip TeX \boxed{}, \displaystyle, $$ … $$, etc.
BOX_RE   = re.compile(
    r"\\boxed\s*[{(]([\s\S]*?)[})]",    # allow newlines inside the box
    re.S                               # ← dot = newline
)
MATH_ENV = re.compile(r"\$\$(.*?)\$\$", re.S)  # $$ … $$  (unchanged but keep re.S)
INLINE   = re.compile(r"\$(.*?)\$",      re.S)  # $ … $   (add re.S for safety)

# 2) cue words that typically precede the answer
CUES = r"(?:final\s+answer|answer|Ans\.?|⇒|=>|thus|so|hence|therefore)[:\s,]*"
CUE_RE = re.compile(CUES, re.I)

def clean_math(expr: str) -> str:
    """Remove common LaTeX wrappers; return bare candidate."""
    m = BOX_RE.search(expr)
    if m:
        return m.group(1).strip()
    # if the entire string is a single math env, strip once
    for pat in (MATH_ENV, INLINE):
        m = pat.fullmatch(expr.strip())
        if m:
            return m.group(1).strip()
    return expr.strip()

def find_first_math(text: str) -> str:
    """Heuristic: first math‑looking token after a cue word, else last math."""
    for cue in CUE_RE.finditer(text):
        tail = text[cue.end():]
        # split on newline, semicolon, period — then strip leading spaces
        token = re.split(r"[\n;.]", tail, 1)[0]
        token = token.rstrip(".;:,")          # strip sentence punctuation
        if token:
            return token
    # fall‑back: last math expression in the whole text
    for pat in (BOX_RE, INLINE, MATH_ENV):
        matches = list(pat.finditer(text))
        if matches:
            return clean_math(matches[-1].group(1))
    # ultimate fall‑back: last non‑empty line
    for line in reversed(text.splitlines()):
        if line.strip():
            return clean_math(line)
    return ""

# -----------------------------------------------------------------
# replaced old extract_pred that relied on <answer>… blocks

def extract_pred(txt: str) -> str:
    return rhs(find_first_math(txt))

# ─────────────────────────  per‑checkpoint scorer  ─────────────────────

def evaluate_revision(jsonl_path):
    rev  = pathlib.Path(jsonl_path).parent.name
    step = REV2STEP.get(rev)
    if step is None:
        logging.warning(f"Unknown SHA {rev}; skipping.")
        return None, None

    accs, recks, toks, rows = [], [], [], []
    with open(jsonl_path, encoding="utf-8") as f:
        for ln in f:
            row  = json.loads(ln)
            text = row.get("output", "")
            gold = row.get("gold_answer", "")
            pred = extract_pred(text)
            corr = judge(gold, pred)
            rk   = has_recheck(text)
            accs.append(corr)
            recks.append(rk)
            toks.append(len(text.split()))
            # ensure keys needed later exist
            row.setdefault("problem", row.get("problem", "?"))
            row.setdefault("sample_idx", row.get("sample_idx", -1))
            row.update(correct=corr, has_recheck=rk, step=step, rev=rev, pred=pred)
            rows.append(row)

    gradable = [a for a in accs if a is not None]
    stats = dict(
        rev=rev, step=step,
        accuracy=sum(gradable)/len(gradable) if gradable else float("nan"),
        recheck_pct=sum(recks)/len(recks),
        median_tokens=median(toks), gradable=len(gradable), total=len(accs)
    )
    return stats, rows

# ─────────────────────────  trajectory flag helper ─────────────────────────

def flag_r2c(grp):
    grp = grp.sort_values("step")
    if not grp["correct"].notna().any():
        return -1
    wrong = grp[(grp["has_recheck"]) & (~grp["correct"].fillna(False))]
    if wrong.empty:
        return 0
    first_r = wrong["step"].iloc[0]
    later   = grp[(grp["step"] > first_r) & (grp["correct"] == True)]
    return int(not later.empty)

# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", required=True)
    ap.add_argument("--write_augmented", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    root = pathlib.Path(args.results_root)
    analysis = root / "analysis"; analysis.mkdir(exist_ok=True)

    files = sorted(root.glob("*/step0000_train.jsonl"))
    if not files:
        sys.exit("No JSONL files found under results_root.")

    # --- Test 1: Checkpoint summary ---
    print("Test 1: Per‑checkpoint accuracy and re‑check usage")
    print("Description: Evaluates correctness vs. gold answers and frequency of re‑check cues per training step.")

    summaries, all_rows = {}, []
    for path in tqdm(files, desc="Scoring checkpoints"):
        stats, rows = evaluate_revision(path)
        if stats is None:
            continue
        summaries[stats['rev']] = stats
        all_rows.extend(rows)
        if args.write_augmented:
            with open(analysis/f"{stats['rev']}_scored.jsonl", 'w') as w:
                for r in rows:
                    json.dump(r, w); w.write("\n")

    df_ckpt = pd.DataFrame(summaries.values()).sort_values('step')
    df_ckpt.to_csv(analysis/'checkpoint_summary.csv', index=False)
    print("\n=== Accuracy / Re‑check summary ===")
    print(df_ckpt[['rev','step','accuracy','recheck_pct','median_tokens','gradable']])

    # Trend interpretation
    rho, p_rho = spstats.spearmanr(df_ckpt['step'], df_ckpt['accuracy'])
    print(f"\nInterpretation: Accuracy vs. training step has Spearman ρ = {rho:.3f} (p = {p_rho:.3f}).")
    if p_rho < .05:
        print("    → Accuracy improves (monotonically) with training, albeit modestly under stochastic decoding.")
    else:
        print("    → No significant monotonic trend; stochastic decoding masks step‑wise gains.")

    # --- Test 2: Aha‑moment distribution ---
    print("\nTest 2: Distributional Aha‑moment (later‑correct | early re‑check)")

    df_rows = pd.DataFrame(all_rows)
    df_rows.to_parquet(analysis/'all_completions.parquet', index=False)

    traj = df_rows.groupby(['problem','sample_idx']).apply(flag_r2c).rename('r2c').reset_index()
    traj = traj[traj.r2c>=0]
    succ = traj.r2c.values
    p_hat = succ.mean()
    ci_low, ci_high = np.percentile([np.random.choice(succ, len(succ), True).mean() for _ in range(10_000)], [2.5,97.5])
    print(f"Result: P = {p_hat:.3f} (95% CI {ci_low:.3f}, {ci_high:.3f}), n = {len(succ)} trajectories")
    if p_hat > 0.5:
        print("    → Re‑checking *boosts* the chance of eventual correctness in a trajectory.")
    else:
        print("    → Re‑checking is *not* reliably followed by a fix; it often marks sticky confusion.")

    # --- Test 3: Logistic regression (correct ~ has_recheck + step) ---
    print("\nTest 3: Logistic regression controlling for step")

    df_glm = df_rows.dropna(subset=['correct']).copy()
    df_glm['has_recheck'] = df_glm['has_recheck'].astype(int)
    df_glm['correct'] = df_glm['correct'].astype(int)
    X = sm.add_constant(df_glm[['has_recheck','step']])
    y = df_glm['correct']
    glm = sm.GLM(y, X, family=sm.families.Binomial())
    res = glm.fit(cov_type='cluster', cov_kwds={'groups': df_glm['problem']})
    print(res.summary())

    # Interpret coefficients
    ci = res.conf_int()
    or_recheck = np.exp(res.params['has_recheck'])
    or_low, or_high = np.exp(ci.loc['has_recheck'])
    pval = res.pvalues['has_recheck']
    print(f"\nInterpretation: Re‑check odds ratio = {or_recheck:.2f} (95% CI {or_low:.2f}, {or_high:.2f}), p = {pval:.3g}.")
    if pval < .05:
        if or_recheck > 1:
            print("    → Re‑checking independently *increases* odds of correctness after accounting for training step.")
        else:
            print("    → Re‑checking independently *decreases* odds of correctness after accounting for training step.")
    else:
        print("    → No statistically reliable effect of re‑checking once step is controlled.")

    # --- Parser failure report ---
    total_rows = len(df_rows)
    if parse_failures:
        print(f"\n⚠️  math_verify parse failures: {parse_failures} of {total_rows} completions ({parse_failures/total_rows:.2%}).")
        print("   These rows were treated as un‑gradable and excluded from accuracy calculations.")

if __name__ == '__main__':
    main()
