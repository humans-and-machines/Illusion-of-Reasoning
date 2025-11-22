#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Uncertainty → Correctness: Counts, Densities, Accuracy, and Regression (All in One)
Times (12 pt) + optional exact A4 PDFs
-------------------------------------------------------------------------------

Figures written:
  1) unc_vs_correct_4hists__<DS>__<MODEL>.png/.pdf
     # 4-panel COUNT(CORRECT) histograms
  2) unc_vs_correct_overlaid__<DS>__<MODEL>.png/.pdf
     # overlaid densities (CORRECT only)
  3) unc_vs_corr_incorr_by_type__<DS>__<MODEL>.png/.pdf
     # 2×2 CORRECT vs INCORRECT (All, Words, LLM, Formal)
  4) unc_accuracy_by_bin__<DS>__<MODEL>.png/.pdf
     # per-bin accuracy with Wilson 95% CIs
  5) acc_vs_uncertainty_regression__<DS>__<MODEL>.png/.pdf
     # GLM: correct ~ C(problem)+C(step)+aha:C(perplexity_bucket)

CSVs:
  - unc_vs_correct_4hists.csv
  - unc_vs_correct_overlaid.csv
  - unc_vs_corr_incorr_by_type.csv
  - unc_accuracy_by_bin.csv
  - acc_vs_uncertainty_regression.csv

Use --a4_pdf to save exact A4 PDFs; text is Times/Times New Roman (12 pt).
"""

from src.analysis.figure_2_helpers import build_arg_parser, run_uncertainty_figures


def main() -> None:
    """CLI entry point for the Figure 2 uncertainty/correctness pipeline."""
    parser = build_arg_parser()
    args = parser.parse_args()
    run_uncertainty_figures(args)


if __name__ == "__main__":
    main()
