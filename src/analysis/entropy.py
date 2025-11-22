"""Entropy vs re-check exploratory analyses and plotting helpers.

This module collects several small analysis scripts that were originally
developed as standalone files and runs them against the Math220k GRPO
results. The code is kept simple and side-effect focused (plots + prints),
but is now structured to satisfy linting rules.
"""

from __future__ import annotations

from pathlib import Path

import importlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from src.analysis.metrics import wilson_ci

# 1) Load
root = Path("artifacts/results/od2961/Math220k/GRPO/1.5B")
dfs = [pd.read_json(f, lines=True) for f in root.glob("**/*.jsonl")]


df  = pd.concat(dfs, ignore_index=True)

# 2) Summary
summary = (
    df.groupby('has_recheck')['entropy']
      .agg(['mean','std','count'])
      .rename(columns={'mean':'mean_entropy','std':'std_entropy'})
      .reset_index()
)
print(summary)

# 3) Prepare data **before** plotting
data0 = df[df['has_recheck']==0]['entropy']
data1 = df[df['has_recheck']==1]['entropy']

# 4) Ensure output dir
outdir = Path("analysis")
outdir.mkdir(exist_ok=True)

# 5) Boxplot
plt.figure(figsize=(6,4))
plt.boxplot([data0, data1], tick_labels=['No re-check','Re-check'])
plt.ylabel('Avg Token Entropy')
plt.title('Entropy vs Re-check Status')
plt.tight_layout()
plt.savefig(outdir/"entropy_vs_recheck_boxplot.png", dpi=300)
plt.close()

# 6) Scatter
plt.figure(figsize=(6,4))
plt.scatter(df['step'], df['entropy'], c=df['has_recheck'], cmap='coolwarm', alpha=0.6)
plt.colorbar(label='has_recheck')
plt.xlabel("Training Step")
plt.ylabel("Avg Token Entropy")
plt.title('Entropy by Step, colored by Re-check')
plt.tight_layout()
plt.savefig(outdir/"entropy_vs_step_scatter.png", dpi=300)
plt.close()

# 7) T-test
t, p = ttest_ind(data0, data1, equal_var=False)
print(f"T-test: t={t:.2f}, p={p:.3f}")

# viz_accuracy_entropy_recheck.py
# Load scored JSONLs, bucket by entropy, compute Wilson CIs,
# and plot accuracy ±95% CI with error bars.

files = sorted(Path("artifacts/results/od2961/Math220k/GRPO/1.5B/analysis").glob("*_scored.jsonl"))
print([f.name for f in files])

# ───────────────────────── Paths ─────────────────────────
ANALYSIS_DIR = Path("artifacts/results/od2961/Math220k/GRPO/1.5B/analysis")
files = sorted(ANALYSIS_DIR.glob("*_scored.jsonl"))
if not files:
    raise SystemExit(f"No scored files in {ANALYSIS_DIR!r}")

# ─────────────────────── Load & concat ───────────────────────
df = pd.concat([pd.read_json(f, lines=True) for f in files], ignore_index=True)

# ─────────────────── Sanity check / types ────────────────────
for c in ("entropy","correct","rechecked"):
    if c not in df.columns:
        raise KeyError(f"Missing column {c!r}")

df["correct"]     = df["correct"].astype(int)
df["has_recheck"] = df["rechecked"].astype(bool)

# ─────────────────── Bucket entropy ──────────────────────────
df["entropy_bucket"] = pd.qcut(
    df["entropy"],
    q=4,
    labels=["Low", "Med-Low", "Med-High", "High"],
)

# ────────────────── Compute summary + Wilson CIs ─────────────
grp = df.groupby(["entropy_bucket","has_recheck"])
summary = grp["correct"].agg(n="size", k="sum").reset_index()
summary["accuracy"] = summary["k"] / summary["n"]

# Wilson CIs via shared helper
ci_lower: list[float] = []
ci_upper: list[float] = []
for k_val, n_val in zip(summary["k"], summary["n"], strict=False):
    lo, hi = wilson_ci(int(k_val), int(n_val))
    ci_lower.append(lo)
    ci_upper.append(hi)
summary["ci_lower"] = ci_lower
summary["ci_upper"] = ci_upper

# Pivot for plotting
buckets = ["Low","Med-Low","Med-High","High"]
x = np.arange(len(buckets))

no_rc = summary[~summary["has_recheck"]].set_index("entropy_bucket")
yes_rc = summary[summary["has_recheck"]].set_index("entropy_bucket")

# ───────────────────── Plotting ────────────────────────────
plt.figure(figsize=(8,5))
# No re-check
plt.errorbar(
    x - 0.05,
    no_rc.loc[buckets,"accuracy"],
    yerr=[
        no_rc.loc[buckets,"accuracy"] - no_rc.loc[buckets,"ci_lower"],
        no_rc.loc[buckets,"ci_upper"]   - no_rc.loc[buckets,"accuracy"]
    ],
    fmt="-o",
    capsize=5,
    label="No re-check",
    color="C0"
)
# With re-check
plt.errorbar(
    x + 0.05,
    yes_rc.loc[buckets,"accuracy"],
    yerr=[
        yes_rc.loc[buckets,"accuracy"] - yes_rc.loc[buckets,"ci_lower"],
        yes_rc.loc[buckets,"ci_upper"]   - yes_rc.loc[buckets,"accuracy"]
    ],
    fmt="-s",
    capsize=5,
    label="With re-check",
    color="C1"
)

plt.xticks(x, buckets)
plt.ylim(0,1)
plt.xlabel("Entropy Quartile")
plt.ylabel("Accuracy")
plt.title("Accuracy by Entropy Quartile  ±95% Wilson CI")
plt.legend()
plt.tight_layout()

# ───────────────── Save ────────────────────────────────────
outpath = ANALYSIS_DIR / "accuracy_entropy_recheck_ci.png"
plt.savefig(outpath, dpi=300)
print(f"Saved plot to {outpath}")


# ─── Load your scored JSONLs ───────────────────────────────────────────────
ANALYSIS_DIR = Path("artifacts/results/od2961/Math220k/GRPO/1.5B/analysis")
files = sorted(ANALYSIS_DIR.glob("*_scored.jsonl"))
df = pd.concat([pd.read_json(f, lines=True) for f in files], ignore_index=True)

# ─── Prep & bucket entropy ─────────────────────────────────────────────────
df["has_recheck"]     = df["rechecked"].astype(bool)
df["correct"]         = df["correct"].astype(int)
df["entropy_bucket"] = pd.qcut(
    df["entropy"],
    4,
    labels=["Low", "Med-Low", "Med-High", "High"],
)

# ─── Compute n, k, accuracy & 95% Wilson CIs ───────────────────────────────
grp = df.groupby(["entropy_bucket","has_recheck"])["correct"]
summary = grp.agg(n="size", k="sum").reset_index()
summary["acc"] = summary["k"] / summary["n"]
ci_lo_list: list[float] = []
ci_hi_list: list[float] = []
for k_val, n_val in zip(summary["k"], summary["n"], strict=False):
    lo, hi = wilson_ci(int(k_val), int(n_val))
    ci_lo_list.append(lo)
    ci_hi_list.append(hi)
summary["ci_lo"] = ci_lo_list
summary["ci_hi"] = ci_hi_list

# Pivot to make plotting easy
buckets = ["Low","Med-Low","Med-High","High"]
no_rc = summary[~summary["has_recheck"]].set_index("entropy_bucket")
yes_rc = summary[summary["has_recheck"]].set_index("entropy_bucket")

# ─── Plot slope‐graph ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8,5))

for b in buckets:
    y0 = no_rc.at[b, "acc"]
    y1 = yes_rc.at[b,"acc"]
    lo0, hi0 = no_rc.at[b,"ci_lo"], no_rc.at[b,"ci_hi"]
    lo1, hi1 = yes_rc.at[b,"ci_lo"], yes_rc.at[b,"ci_hi"]
    # draw line between points
    ax.plot([0,1], [y0,y1], '-o', color="C1" if b=="High" else "gray", alpha=0.7)
    # draw error bars at each end
    ax.errorbar(0, y0, yerr=[[y0-lo0],[hi0-y0]], color="C0", capsize=5)
    ax.errorbar(1, y1, yerr=[[y1-lo1],[hi1-y1]], color="C1", capsize=5)
    # label the line at midpoint
    ax.text(1.02, (y0+y1)/2, b, va="center")

ax.set_xlim(-0.2, 1.4)
ax.set_xticks([0,1], ["No re-check","With re-check"])
ax.set_ylabel("Accuracy")
ax.set_title("Re-check Lift by Entropy Quartile (±95% CI)")
plt.tight_layout()
fig.savefig(ANALYSIS_DIR / "accuracy_entropy_recheck_slope.png", dpi=300)

ANALYSIS_DIR = Path("artifacts/results/od2961/Math220k/GRPO/1.5B/analysis")
files = sorted(ANALYSIS_DIR.glob("*_scored.jsonl"))
df = pd.concat([pd.read_json(f, lines=True) for f in files], ignore_index=True)

# ensure types
df["has_recheck"] = df["rechecked"].astype(bool)
df["correct"]     = df["correct"].astype(int)

# 1) Bucket entropy into 10 deciles
decile_labels = [f"D{i+1}" for i in range(10)]
df["entropy_bucket"] = pd.qcut(
    df["entropy"],
    q=10,
    labels=decile_labels,
)

# 2) Compute counts, accuracy, Wilson CIs
grp = df.groupby(
    ["entropy_bucket","has_recheck"],
    observed=True
)["correct"]
summary = grp.agg(n="size", k="sum").reset_index()
summary["acc"] = summary["k"] / summary["n"]
ci_lo_list: list[float] = []
ci_hi_list: list[float] = []
for k_val, n_val in zip(summary["k"], summary["n"], strict=False):
    lo, hi = wilson_ci(int(k_val), int(n_val))
    ci_lo_list.append(lo)
    ci_hi_list.append(hi)
summary["ci_lo"], summary["ci_hi"] = ci_lo_list, ci_hi_list

# 3) Pivot tables for easy lookup
pivot_acc = summary.pivot(index="entropy_bucket", columns="has_recheck", values="acc")
pivot_n   = summary.pivot(index="entropy_bucket", columns="has_recheck", values="n")

# 4) Draw barplot + annotate sample sizes
BAR_WIDTH = 0.35
fig, ax = plt.subplots(figsize=(14, 6))
indices = np.arange(len(decile_labels))

no_values = pivot_acc.loc[decile_labels, False]
yes_values = pivot_acc.loc[decile_labels, True]

ax.bar(indices - BAR_WIDTH / 2, no_values.values, BAR_WIDTH, label="No re-check", color="C0")
ax.bar(indices + BAR_WIDTH / 2, yes_values.values, BAR_WIDTH, label="With re-check", color="C1")

for i, label in enumerate(decile_labels):
    y0 = no_values.iloc[i]
    n0 = int(pivot_n.at[label, False])
    ax.text(i - 0.2, y0 + 0.02, f"n={n0}", ha="center", color="C0")
    y1 = yes_values.iloc[i]
    n1 = int(pivot_n.at[label, True])
    ax.text(i + 0.2, y1 + 0.02, f"n={n1}", ha="center", color="C1")

ax.set_xlabel("Entropy Decile")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy by Entropy Decile & Re-check (95% CI)")
ax.legend(title="Has Re-check", loc="upper left")
plt.tight_layout()
plt.savefig(ANALYSIS_DIR / "accuracy_by_entropy_decile_ci_with_n.png", dpi=300)
plt.close()

# standardize & encode
df["entropy_z"] = (df["entropy"] - df["entropy"].mean()) / df["entropy"].std()
df["has_recheck_i"] = df["rechecked"].astype(int)

# fit with clustered SEs (requires statsmodels)
statsmodels_formula_api = importlib.import_module("statsmodels.formula.api")
model = statsmodels_formula_api.logit("correct ~ entropy_z * has_recheck_i + step", data=df)
res = model.fit(
    disp=False,
    cov_type='cluster',
    cov_kwds={'groups': df['problem']}
)

print(res.summary())

# ── 1) Load your scored data ───────────────────────────────────────────
ANALYSIS_DIR = Path("artifacts/results/od2961/Math220k/GRPO/1.5B/analysis")
files = sorted(ANALYSIS_DIR.glob("*_scored.jsonl"))
df_all = pd.concat([pd.read_json(f, lines=True) for f in files],
                   ignore_index=True)

# ensure types
df_all['correct']     = df_all['correct'].astype(bool)
df_all['has_recheck'] = df_all['rechecked'].astype(bool)

# ── 2) Find “flip” trajectories ───────────────────────────────────────
#   – those whose first record is incorrect and last record is correct
# sort by step, then group
df_sorted = df_all.sort_values('step')
stats = df_sorted.groupby(['problem','sample_idx'])['correct'].agg(
    first_corr='first', last_corr='last'
).reset_index()

# select only the flips
flip_mask = (~stats["first_corr"]) & (stats["last_corr"])
flip_ids = stats.loc[flip_mask, ["problem", "sample_idx"]]
# merge back to get only those rows
flip_df = df_all.merge(flip_ids, on=['problem','sample_idx'], how='inner')

# ── 3) Aggregate per-step metrics ──────────────────────────────────────
agg = flip_df.groupby('step').agg(
    mean_entropy   = ('entropy','mean'),
    recheck_rate   = ('has_recheck','mean'),
    n_trajectories = ('correct','size')
).reset_index()

print("\nFlip trajectories per step:")
print(agg)

# ── 4) Plot entropy & re-check rate over steps ─────────────────────────
fig, ax1 = plt.subplots(figsize=(8,5))

COLOR_ENTROPY = "C0"
ax1.set_xlabel('Training Step')
ax1.set_ylabel("Mean Token Entropy", color=COLOR_ENTROPY)
ax1.plot(agg["step"], agg["mean_entropy"], marker="o", color=COLOR_ENTROPY, label="Entropy")
ax1.tick_params(axis="y", labelcolor=COLOR_ENTROPY)

ax2 = ax1.twinx()
COLOR_RECHECK = "C1"
ax2.set_ylabel("Re-check Rate", color=COLOR_RECHECK)
ax2.plot(
    agg["step"],
    agg["recheck_rate"],
    marker="s",
    color=COLOR_RECHECK,
    label="Re-check rate",
)
ax2.tick_params(axis="y", labelcolor=COLOR_RECHECK)
ax2.set_ylim(0, 1)

# optional: annotate sample-count
for x,y in zip(agg['step'], agg['n_trajectories']):
    ax1.text(x, agg['mean_entropy'].max()*1.02, f"n={y}", ha='center', fontsize=8)

fig.tight_layout()
plt.title('Flip Trajectories: Entropy & Re-check Rate over Steps')
out = ANALYSIS_DIR / "flip_entropy_recheck_over_steps.png"
plt.savefig(out, dpi=300)
plt.close()

print(f"\nSaved flip‐trajectory plot to {out}")

# 1) Load all scored JSONL files
analysis_dir = Path("artifacts/results/od2961/Math220k/GRPO/1.5B/analysis")
scored_files = list(analysis_dir.glob("*_scored.jsonl"))
df = pd.concat([pd.read_json(f, lines=True) for f in scored_files], ignore_index=True)

# 2) Focus on step 50 (initial) and step 850 (later)
init = df[df['step'] == 50].set_index(['problem', 'sample_idx'])
final = df[df['step'] == 850].set_index(['problem', 'sample_idx'])

# 3) Join on (problem, sample_idx)
joined = init.join(final, lsuffix='_50', rsuffix='_850', how='inner')

# 4) Filter: wrong at 50, correct at 850
mask = (~joined['correct_50']) & (joined['correct_850'])
flips = joined[mask]

# 5) Select relevant columns
result = flips.reset_index()[[
    'problem', 'sample_idx',
    'output_50', 'entropy_50', 'has_recheck_50',
    'output_850', 'entropy_850', 'has_recheck_850'
]]

#!/usr/bin/env python
#   – those whose first record is incorrect and last record is correct
# sort by step, then group
df_sorted = df_all.sort_values('step')
stats = df_sorted.groupby(['problem','sample_idx'])['correct'].agg(
    first_corr='first', last_corr='last'
).reset_index()

# select only the flips
flip_mask = (~stats["first_corr"]) & (stats["last_corr"])
flip_ids = stats.loc[flip_mask, ["problem", "sample_idx"]]
# merge back to get only those rows
flip_df = df_all.merge(flip_ids, on=['problem','sample_idx'], how='inner')

# ── 3) Aggregate per-step metrics ──────────────────────────────────────
agg = flip_df.groupby('step').agg(
    mean_entropy   = ('entropy','mean'),
    recheck_rate   = ('has_recheck','mean'),
    n_trajectories = ('correct','size')
).reset_index()

print("\nFlip trajectories per step:")
print(agg)

# ── 4) Plot entropy & re-check rate over steps ─────────────────────────
fig, ax1 = plt.subplots(figsize=(8,5))

COLOR_ENTROPY_2 = "C0"
ax1.set_xlabel("Training Step")
ax1.set_ylabel("Mean Token Entropy", color=COLOR_ENTROPY_2)
ax1.plot(agg["step"], agg["mean_entropy"], marker="o", color=COLOR_ENTROPY_2, label="Entropy")
ax1.tick_params(axis="y", labelcolor=COLOR_ENTROPY_2)

ax2 = ax1.twinx()
COLOR_RECHECK_2 = "C1"
ax2.set_ylabel("Re-check Rate", color=COLOR_RECHECK_2)
ax2.plot(
    agg["step"],
    agg["recheck_rate"],
    marker="s",
    color=COLOR_RECHECK_2,
    label="Re-check rate",
)
ax2.tick_params(axis="y", labelcolor=COLOR_RECHECK_2)
ax2.set_ylim(0, 1)

# optional: annotate sample-count
for x,y in zip(agg['step'], agg['n_trajectories']):
    ax1.text(x, agg['mean_entropy'].max()*1.02, f"n={y}", ha='center', fontsize=8)

fig.tight_layout()
plt.title('Flip Trajectories: Entropy & Re-check Rate over Steps')
out = ANALYSIS_DIR / "flip_entropy_recheck_over_steps.png"
plt.savefig(out, dpi=300)
plt.close()

print(f"\nSaved flip‐trajectory plot to {out}")
