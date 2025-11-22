#!/usr/bin/env python
"""
List problems that flip from wrong at the initial step to correct at the final step.

Usage example:

  python flips.py --results_root path/to/GRPO_outputs \\
      --init_step 50 --final_step 850 --output flips.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def _load_all_scores(analysis_dir: Path) -> pd.DataFrame:
    """
    Load and concatenate all scored JSONL files from an analysis directory.
    """
    files = sorted(analysis_dir.glob("*_scored.jsonl"))
    if not files:
        raise SystemExit(f"No scored JSONL files in {analysis_dir!r}")
    return pd.concat(
        (pd.read_json(path, lines=True) for path in files),
        ignore_index=True,
    )


def _compute_flips(
    df_all: pd.DataFrame,
    init_step: int,
    final_step: int,
) -> tuple[pd.DataFrame, int]:
    """
    Compute wrong→correct flips between two training steps.
    """
    df_init = df_all[df_all["step"] == init_step].copy()
    df_final = df_all[df_all["step"] == final_step].copy()

    # key on problem + sample_idx
    df_init.set_index(["problem", "sample_idx"], inplace=True)
    df_final.set_index(["problem", "sample_idx"], inplace=True)

    joined = df_init.join(
        df_final,
        lsuffix=f"_{init_step}",
        rsuffix=f"_{final_step}",
        how="inner",
    )

    # filter flips: wrong -> correct
    correct_init = joined[f"correct_{init_step}"].astype(bool)
    correct_final = joined[f"correct_{final_step}"].astype(bool)
    mask_wrong_init = ~correct_init
    mask_right_final = correct_final
    flips = joined[mask_wrong_init & mask_right_final]

    total_candidates = joined.shape[0]
    return flips, total_candidates


def main() -> None:
    """CLI entry point for computing wrong→correct flip trajectories."""
    parser = argparse.ArgumentParser(
        description="List problems that flip from wrong at init_step to correct at final_step"
    )
    parser.add_argument(
        "--results_root", required=True,
        help="Root directory of your GRPO outputs (contains 'analysis/' subfolder)"
    )
    parser.add_argument(
        "--init_step", type=int, default=50,
        help="First step to check (default: 50)"
    )
    parser.add_argument(
        "--final_step", type=int, default=850,
        help="Final step to check (default: 850)"
    )
    parser.add_argument(
        "--output", default="flips.csv",
        help="CSV file to write the flip examples (default: flips.csv)"
    )
    args = parser.parse_args()

    analysis_dir = Path(args.results_root) / "analysis"
    if not analysis_dir.is_dir():
        raise SystemExit(f"Error: {analysis_dir!r} not found")

    df_all = _load_all_scores(analysis_dir)
    flips, total_candidates = _compute_flips(
        df_all=df_all,
        init_step=args.init_step,
        final_step=args.final_step,
    )
    total_flips = flips.shape[0]
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
