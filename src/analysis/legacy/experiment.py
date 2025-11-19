import argparse
import json
import logging
import pathlib
import sys
from statistics import median

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as spstats
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint

# Mapping from revision hash to step number
REV2STEP = {
    "aef4ec7":50,  "9ac7e5e":150, "ba99bc4":250, "7cbc93a":350,
    "4803ae9":450, "5d0b169":550, "334f623":650, "f6b58de":750,
    "14436ba":850, "d5bb572":950, "1e7973f":1050,"4771a8f":1150,
    "a472241":1250,"76bee3d":1350,"57fa3c6":1450,"70ddc2c":1550,
    "29605a8":1650,"26044d8":1750,"e8a648f":1850,"df6030f":1950,
    "3075e71":2050,"c148b71":2150,"bd98859":2250,"08b3d89":2350,
    "d1d3bf7":2450,
}

# Helper to detect self-correction trajectories
def flag_r2c(grp: pd.DataFrame) -> int:
    grp = grp.sort_values("step")
    wrong = grp[(grp.has_recheck) & (~grp.correct)]
    if wrong.empty:
        return 0
    first = wrong.step.iloc[0]
    later_ok = grp[(grp.step > first) & (grp.correct)]
    return int(not later_ok.empty)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", required=True,
                    help="Root directory of GRPO outputs")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    root = pathlib.Path(args.results_root)
    analysis = root / "analysis"
    if not analysis.exists():
        sys.exit("Error: analysis/ folder not found. Run grading first.")

    # Load all graded files
    graded_files = sorted(analysis.glob("*_scored.jsonl"))
    if not graded_files:
        sys.exit("No *_scored.jsonl files found in analysis/.")

    all_rows = []
    summaries = {}
    for gf in graded_files:
        rev = gf.stem.split("_")[0]
        step = REV2STEP.get(rev)
        if step is None:
            logging.warning("Unknown rev %s; skipping.", rev)
            continue
        rows = []
        for line in gf.open(encoding="utf-8"):
            r = json.loads(line)
            r['rev'] = rev
            r['step'] = step
            rows.append(r)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        summaries[rev] = dict(
            rev=rev,
            step=step,
            accuracy=df['correct'].mean(),
            recheck_pct=df['rechecked'].mean(),
            median_tokens=df['output'].str.split().str.len().median(),
            num_samples=len(df)
        )
        all_rows.extend(rows)

    # Test 1: Checkpoint summary
    df_ckpt = pd.DataFrame(summaries.values()).sort_values('step')
    df_ckpt.to_csv(analysis / "checkpoint_summary.csv", index=False)
    print("\n=== Accuracy / Re-check summary ===")
    print(df_ckpt[['rev','step','accuracy','recheck_pct','median_tokens','num_samples']])

    # Prepare DataFrame of all completions
    df_rows = pd.DataFrame(all_rows)
    df_rows['has_recheck'] = df_rows['rechecked'].astype(bool)
    df_rows['correct']     = df_rows['correct'].astype(bool)
    df_rows['has_recheck_int'] = df_rows['has_recheck'].astype(int)
    df_rows['correct_int']     = df_rows['correct'].astype(int)

    # Test 2: Aha-moment distribution
    total = len(df_rows)
    num_re = df_rows['has_recheck'].sum()
    print(f"\nTotal completions:           {total}")
    print(f"Completions with re-check:   {num_re}")
    print(f"Completions without re-check:{total - num_re}")
    traj = (df_rows.groupby(['problem','sample_idx'])
                        .apply(flag_r2c, include_groups=False)
                        .rename('r2c')
                        .reset_index())
    succ = traj['r2c'].values
    p_hat = succ.mean()
    ci = np.percentile([np.random.choice(succ, len(succ), True).mean() for _ in range(10000)], [2.5,97.5])
    print(f"\nP(later-correct | early re-check) = {p_hat:.3f} (95% CI {ci[0]:.3f}, {ci[1]:.3f}), n={len(succ)}")

    # Test 3: Logistic regression
    X = sm.add_constant(df_rows[['has_recheck_int','step']])
    res = sm.GLM(df_rows['correct_int'], X, family=sm.families.Binomial())
    fit = res.fit(cov_type='cluster', cov_kwds={'groups': df_rows['problem']})
    print("\n=== Logistic regression ===")
    print(fit.summary())

    # Test 4: Self-correction rate by step
    df_sc = df_rows.sort_values(['problem','sample_idx','step'])
    df_sc['ever_correct_after'] = (
        df_sc.iloc[::-1]
             .groupby(['problem','sample_idx'])['correct']
             .cummax()
             .iloc[::-1]
    )
    re_rows = df_sc[df_sc['has_recheck']]
    summary4 = (
        re_rows.groupby('step')
               .agg(P_correct_after=('ever_correct_after','mean'),
                    N_rechecks=('ever_correct_after','count'))
               .reset_index()
    )
    print("=== P(correct at or after step | re-check) ===")
    print(summary4.to_string(index=False))
    plt.figure(figsize=(6,4))
    plt.plot(summary4['step'], summary4['P_correct_after'], marker='o')
    plt.xlabel('Step'); plt.ylabel('P(correct after)')
    plt.title('Self-correction Rate')
    plt.tight_layout()
    plt.savefig(analysis / 'self_correction_by_step.png', dpi=300)

    # Test 5: Correctness at same step
    p_same = re_rows['correct'].mean()
    ci2 = np.percentile([np.random.choice(re_rows['correct'], len(re_rows), True).mean() for _ in range(10000)], [2.5,97.5])
    print(f"=== P(correct at same step | re-check) ===")
    print(f"P = {p_same:.3f} (95% CI {ci2[0]:.3f}, {ci2[1]:.3f}), n={len(re_rows)}")

    # === Entropy experiments ===
    data0 = df_rows[df_rows['has_recheck']==0]['entropy']
    data1 = df_rows[df_rows['has_recheck']==1]['entropy']
    ent_summary = df_rows.groupby('has_recheck')['entropy'].agg(mean='mean', std='std', count='size')
    print("\n=== Entropy by re-check status ===")
    print(ent_summary)
    # boxplot (use tick_labels to silence MatplotlibDeprecationWarning)
    plt.figure(figsize=(6,4))
    plt.boxplot([data0, data1], tick_labels=['No re-check','Re-check'])
    plt.ylabel('Avg Token Entropy')
    plt.title('Entropy vs Re-check')
    plt.tight_layout()
    plt.savefig(analysis / 'entropy_vs_recheck_boxplot.png', dpi=300)
    plt.close()
    # scatter
    plt.figure(figsize=(6,4))
    plt.scatter(df_rows['step'], df_rows['entropy'], c=df_rows['has_recheck_int'], cmap='coolwarm', alpha=0.6)
    plt.colorbar(label='has_recheck')
    plt.xlabel('Step'); plt.ylabel('Entropy')
    plt.title('Entropy by Step')
    plt.tight_layout()
    plt.savefig(analysis / 'entropy_vs_step_scatter.png', dpi=300)
    plt.close()
    # t-test
    t, p = ttest_ind(data0, data1, equal_var=False)
    print(f"\nT-test on entropy: t={t:.2f}, p={p:.3f}")

    # === Accuracy by entropy quartile ±95% CI ===
    df_rows['entropy_bucket'] = pd.qcut(df_rows['entropy'],4,labels=['Low','Med-Low','Med-High','High'])
    # include observed=False to silence FutureWarning
    grp = df_rows.groupby(['entropy_bucket','has_recheck'], observed=False)['correct']
    summ = grp.agg(n='size', k='sum').reset_index()
    summ['acc'] = summ['k']/summ['n']
    ci_lo, ci_hi = proportion_confint(summ['k'], summ['n'], method='wilson')
    summ['ci_lo'], summ['ci_hi'] = ci_lo, ci_hi
    buckets=['Low','Med-Low','Med-High','High']
    no_rc = summ[summ.has_recheck==0].set_index('entropy_bucket')
    yes_rc= summ[summ.has_recheck==1].set_index('entropy_bucket')
    plt.figure(figsize=(8,5))
    x = np.arange(len(buckets))
    plt.errorbar(x-0.05, no_rc.loc[buckets,'acc'],
                 yerr=[no_rc.loc[buckets,'acc']-no_rc.loc[buckets,'ci_lo'],
                       no_rc.loc[buckets,'ci_hi']-no_rc.loc[buckets,'acc']],
                 fmt='-o', capsize=5, label='No re-check')
    plt.errorbar(x+0.05, yes_rc.loc[buckets,'acc'],
                 yerr=[yes_rc.loc[buckets,'acc']-yes_rc.loc[buckets,'ci_lo'],
                       yes_rc.loc[buckets,'ci_hi']-yes_rc.loc[buckets,'acc']],
                 fmt='-s', capsize=5, label='Re-check')
    plt.xticks(x, buckets); plt.ylim(0,1)
    plt.xlabel('Entropy Quartile'); plt.ylabel('Accuracy')
    plt.title('Accuracy by Entropy Quartile')
    plt.legend(); plt.tight_layout()
    plt.savefig(analysis / 'accuracy_entropy_quartile_ci.png', dpi=300)

    # === Random sampling experiment ===
    # (draw 15 examples with recheck=True and 15 with recheck=False)
    sample_true  = df_rows[df_rows['has_recheck']].sample(n=15, random_state=42)
    sample_false = df_rows[~df_rows['has_recheck']].sample(n=15, random_state=42)
    sample_df    = pd.concat([sample_true, sample_false]).sample(frac=1, random_state=42)

    sample_path = analysis / 'random_30_samples.csv'
    sample_df.to_csv(sample_path, index=False)
    print(f"\nSaved random sample (15 recheck / 15 no-recheck) to {sample_path}\n")

if __name__ == "__main__":
    main()
