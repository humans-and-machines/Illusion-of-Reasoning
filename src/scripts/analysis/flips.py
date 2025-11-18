#!/usr/bin/env python
# show_flips.py
# ──────────────────────────────────────────────────────────────────────────────
# Usage:
#   python show_flips.py --results_root path/to/GRPO_outputs \
#       [--init_step 50] [--final_step 850] [--output flips.csv]
# ──────────────────────────────────────────────────────────────────────────────

import argparse
import pandas as pd
from pathlib import Path

def main():
    p = argparse.ArgumentParser(
        description="List problems that flip from wrong at init_step to correct at final_step"
    )
    p.add_argument(
        "--results_root", required=True,
        help="Root directory of your GRPO outputs (contains 'analysis/' subfolder)"
    )
    p.add_argument(
        "--init_step", type=int, default=50,
        help="First step to check (default: 50)"
    )
    p.add_argument(
        "--final_step", type=int, default=850,
        help="Final step to check (default: 850)"
    )
    p.add_argument(
        "--output", default="flips.csv",
        help="CSV file to write the flip examples (default: flips.csv)"
    )
    args = p.parse_args()

    analysis_dir = Path(args.results_root) / "analysis"
    if not analysis_dir.is_dir():
        raise SystemExit(f"Error: {analysis_dir!r} not found")

    # 1) Load all scored JSONLs
    files = sorted(analysis_dir.glob("*_scored.jsonl"))
    if not files:
        raise SystemExit(f"No scored JSONL files in {analysis_dir!r}")
    df = pd.concat((pd.read_json(f, lines=True) for f in files),
                   ignore_index=True)

    # 2) pick the init and final steps
    df_init = df[df["step"] == args.init_step].copy()
    df_final= df[df["step"] == args.final_step].copy()

    # key on problem + sample_idx
    df_init.set_index(["problem","sample_idx"], inplace=True)
    df_final.set_index(["problem","sample_idx"], inplace=True)

    # 3) join
    joined = df_init.join(
        df_final,
        lsuffix=f"_{args.init_step}", rsuffix=f"_{args.final_step}",
        how="inner"
    )

    # 4) filter flips: wrong -> correct
    mask_wrong_init   = joined[f"correct_{args.init_step}"] == False
    mask_right_final  = joined[f"correct_{args.final_step}"] == True
    flips = joined[mask_wrong_init & mask_right_final]

    total_candidates = joined.shape[0]
    total_flips      = flips.shape[0]
    print(f"Found {total_flips} flips out of {total_candidates} trajectories "
          f"(wrong@{args.init_step} → correct@{args.final_step})\n")

    if total_flips == 0:
        return

    # 5) select and rename columns for clarity
    cols = [
        "output", "entropy", "has_recheck", "reason"
    ]
    out = flips.reset_index()[[
        "problem","sample_idx"
    ] + [f"{col}_{args.init_step}" for col in cols]
      + [f"{col}_{args.final_step}" for col in cols]]

    # print the first few
    pd.set_option("display.max_colwidth", 50)
    print(out.head(20).to_string(index=False))

    # 6) save to CSV
    out.to_csv(args.output, index=False)
    print(f"\nSaved full flip list ({total_flips} rows) to {args.output}")

if __name__ == "__main__":
    main()
