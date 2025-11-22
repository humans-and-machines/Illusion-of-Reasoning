#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
H3: Can Aha! Moments Help When the Model is Uncertain?
+ Re-asking (pass2) at QUESTION & PROMPT levels, with Aha variants
+ Wilson CIs, step-wise & bucket-wise plots/tables
---------------------------------------------------------------

Uncertainty measure
-------------------
Default is --measure entropy (nats). Buckets are built from entropy values.
(If you set --measure perplexity, the loader converts entropy->perplexity via exp(entropy).)
We keep the column name "perplexity_bucket" for backward compatibility with your H2/H3 formula.

What this script does
---------------------
1) Robustly loads your JSONL (pass1 & pass2), extracts entropy/ppx, Aha labels:
   - Words (excludes injected/forced cues)
   - GPT-4o canonical (optionally gated by Words)
   - Formal (problem-step, gated by GPT)
2) H3 GLMs per Aha variant:
   correct ~ C(problem) + step_std + aha * C(perplexity_bucket)
3) Re-asking analysis (pass2 vs pass1), with **Wilson 95% CIs**:
   • QUESTION level (per problem): fraction solved ≥1 time (overall, by step, by bucket)
   • PROMPT level (per pair): accuracy (overall, by step, by bucket)
   • PROMPT-level Δ histogram (with/without forced-insight split)
4) Answers written to **h3_answers.txt**:
   (A) Given ≥1 correct in pass1, does pass2 increase overall accuracy?
   (B) Given 0 correct in pass1, does pass2 achieve ≥1 correct?
   Each answered overall, step-wise, and bucket-wise.

Outputs
-------
CSV:
  - h3_glm_bucket_margins.csv
  - h3_bucket_group_accuracy.csv
  - h3_pass2_prompt_level.csv                 # per-pair aligned rows
  - h3_pass2_question_level.csv               # per-problem aligned rows
  - h3_pass2_conditioned_summary.csv          # overall A & B answers (macro & micro)
  - h3_pass2_conditioned_by_forcing.csv       # (optional) split by forced insight
  - h3_prompt_level_overall.csv               # k/n with Wilson CIs
  - h3_prompt_level_by_step.csv               # step-wise k/n with CIs
  - h3_prompt_level_by_bucket.csv             # bucket-wise k/n with CIs
  - h3_question_level_overall.csv             # k/n (any-correct) with Wilson CIs
  - h3_question_level_by_step.csv             # step-wise any-correct with CIs
  - h3_question_level_by_bucket.csv           # bucket-wise any-correct with CIs
  - (optional) h3_pass2_prompt_level_by_aha.csv
  - (optional) h3_pass2_question_level_by_aha.csv

Plots (if --make_plots):
  - h3_plot_question_overall.png              # bars + Wilson CIs
  - h3_plot_question_by_step.png              # lines + CIs by step
  - h3_plot_question_by_bucket.png            # points/bars + CIs by bucket
  - h3_plot_prompt_overall.png                # bars + CIs
  - h3_plot_prompt_by_step.png                # lines + CIs
  - h3_plot_prompt_by_bucket.png              # points/bars + CIs
  - h3_plot_prompt_level_delta.png            # Δ histogram (± forced)

PDF (if --make_pdf):
  - h3_summary.pdf                            # GLM bucket margins summary page

Text:
  - h3_answers.txt                            # answers to (A) and (B), overall/step/bucket

"""

import argparse
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from src.analysis.core import (
    build_problem_step_from_samples,
    make_formal_thresholds,
    mark_formal_pairs_with_gain,
)
from src.analysis.h3_uncertainty.glm import (
    bucket_group_accuracy,
    fit_glm_bucket_interaction,
)
from src.analysis.h3_uncertainty.ingest import add_perplexity_buckets, load_rows
from src.analysis.h3_uncertainty.plotting import (
    plot_prompt_by_bucket_ci,
    plot_prompt_by_step_ci,
    plot_prompt_level_deltas,
    plot_prompt_overall_ci,
    plot_question_by_bucket_ci,
    plot_question_by_step_ci,
    plot_question_overall_ci,
)
from src.analysis.h3_uncertainty.reasking import (
    compute_reasking_tables,
    prompt_level_acc_with_ci,
    question_level_any_with_ci,
    split_reasking_by_aha,
)
from src.analysis.h3_uncertainty.reporting import (
    AnswersTables,
    PdfSummaryConfig,
    write_a4_summary_pdf,
    write_answers_txt,
)
from src.analysis.io import scan_jsonl_files

GLM_VARIANTS = [
    ("aha_words", "words"),
    ("aha_gpt", "gpt"),
    ("aha_formal", "formal"),
]


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root")
    parser.add_argument("--split", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--dataset_name", default="MATH-500")
    parser.add_argument("--model_name", default="Qwen2.5-1.5B")

    parser.add_argument("--unc_field", choices=["answer", "overall", "think"], default="answer")
    parser.add_argument("--measure", choices=["perplexity", "entropy"], default="entropy")
    parser.add_argument("--n_buckets", type=int, default=5)
    parser.add_argument(
        "--bucket_method",
        choices=["quantile", "fixed"],
        default="quantile",
    )
    parser.add_argument(
        "--bucket_edges",
        default=None,
        help="Comma-separated edges for fixed bins (entropy values, e.g., 0.2,0.5,1,1.5,2,3)",
    )

    parser.add_argument("--gpt_mode", choices=["canonical", "broad"], default="canonical")
    parser.add_argument(
        "--no_gate_gpt_by_words",
        action="store_true",
        help="If set, GPT shifts are NOT restricted to samples with Words-cue.",
    )

    parser.add_argument("--delta1", type=float, default=0.13)
    parser.add_argument("--delta2", type=float, default=0.13)
    parser.add_argument("--min_prior_steps", type=int, default=2)

    parser.add_argument("--cluster_by", choices=["problem", "none"], default="problem")
    parser.add_argument(
        "--strict_interaction_only",
        action="store_true",
        help="Fit correct ~ C(problem) + step_std + aha:C(perplexity_bucket).",
    )
    parser.add_argument(
        "--split_reasking_by_aha",
        action="store_true",
        help="Additionally write re-asking summaries split by Words/GPT/Formal Aha.",
    )

    parser.add_argument("--tex_path", default=None)
    parser.add_argument("--make_pdf", action="store_true")
    parser.add_argument(
        "--make_plots",
        action="store_true",
        help=(
            "Emit PNG plots for question- and prompt-level re-asking effects "
            "(overall/step/bucket)."
        ),
    )
    parser.add_argument("--also_pdf_plots", action="store_true")
    parser.add_argument("--font_family", default="Times New Roman")
    parser.add_argument("--font_size", type=int, default=12)
    return parser


def parse_bucket_edges(edge_string: Optional[str]) -> Optional[List[float]]:
    """Parse comma-separated bucket edges."""
    if not edge_string:
        return None
    try:
        return [
            float(piece.strip())
            for piece in edge_string.split(",")
            if piece.strip()
        ]
    except ValueError as exc:
        raise SystemExit("Failed to parse --bucket_edges.") from exc


def prepare_pass1_dataframe(
    samples_df: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Add formal labels and entropy buckets to the pass-1 subset."""
    pass1_df = samples_df[samples_df["pass_id"] == 1].copy()
    problem_steps = build_problem_step_from_samples(pass1_df)
    thresholds = make_formal_thresholds(
        delta1=float(args.delta1),
        delta2=float(args.delta2),
        min_prior_steps=int(args.min_prior_steps),
        delta3=None,
    )
    problem_steps = mark_formal_pairs_with_gain(problem_steps, thresholds)
    pass1_df = pass1_df.merge(
        problem_steps[["step", "problem", "aha_formal_pair"]],
        on=["step", "problem"],
        how="left",
    ).fillna({"aha_formal_pair": 0})
    pass1_df["aha_formal"] = (
        pass1_df["aha_formal_pair"].astype(int)
        & pass1_df["aha_gpt"].astype(int)
    ).astype(int)

    custom_edges = None
    if args.bucket_method == "fixed":
        custom_edges = parse_bucket_edges(args.bucket_edges)
    return add_perplexity_buckets(
        pass1_df,
        n_buckets=int(args.n_buckets),
        method=args.bucket_method,
        custom_edges=custom_edges,
    )


def run_glm_variants(
    pass1_df: pd.DataFrame,
    args: argparse.Namespace,
    out_dir: str,
) -> pd.DataFrame:
    """Fit GLMs per variant and write associated CSVs."""
    summary_dir = os.path.join(out_dir, "h3_glm_fit_summaries")
    os.makedirs(summary_dir, exist_ok=True)
    margin_rows: List[Dict[str, Any]] = []

    for aha_col, tag in GLM_VARIANTS:
        subset = pass1_df[~pass1_df[aha_col].isna()].copy()
        if subset.empty:
            continue
        out_txt = os.path.join(
            summary_dir,
            f"logit_pass1_correct_on_{aha_col}_by_bucket.txt",
        )
        model_info, _ = fit_glm_bucket_interaction(
            data_frame=subset,
            aha_col=aha_col,
            strict_interaction_only=args.strict_interaction_only,
            cluster_by=args.cluster_by,
            out_txt=out_txt,
        )
        for bucket_row in model_info["bucket_rows"]:
            margin_rows.append(
                {
                    "dataset": args.dataset_name,
                    "model": args.model_name,
                    "variant": tag,
                    "perplexity_bucket": bucket_row["bucket"],
                    "N": bucket_row["N"],
                    "share_aha": bucket_row["share_aha"],
                    "AME_bucket": bucket_row["AME_bucket"],
                    "glm_summary_path": out_txt,
                }
            )

        accuracy_df = bucket_group_accuracy(subset, aha_col=aha_col)
        accuracy_df.insert(0, "variant", tag)
        accuracy_df.to_csv(
            os.path.join(out_dir, f"h3_bucket_group_accuracy__{tag}.csv"),
            index=False,
        )

    margins_df = (
        pd.DataFrame(margin_rows)
        .sort_values(["variant", "perplexity_bucket"])
        .reset_index(drop=True)
    )
    margins_df.to_csv(os.path.join(out_dir, "h3_glm_bucket_margins.csv"), index=False)

    accuracy_parts = []
    for _, tag in GLM_VARIANTS:
        accuracy_path = os.path.join(out_dir, f"h3_bucket_group_accuracy__{tag}.csv")
        if os.path.exists(accuracy_path):
            accuracy_parts.append(pd.read_csv(accuracy_path))
    if accuracy_parts:
        pd.concat(accuracy_parts, ignore_index=True).to_csv(
            os.path.join(out_dir, "h3_bucket_group_accuracy.csv"),
            index=False,
        )
    return margins_df


def _write_dataframe_if_any(dataframe: pd.DataFrame, destination: str) -> None:
    """Persist ``dataframe`` to CSV when non-empty."""
    if dataframe.empty:
        return
    dataframe.to_csv(destination, index=False)


def _empty_prompt_tables() -> Dict[str, pd.DataFrame]:
    """Return placeholder prompt-level tables when no data exists."""
    empty = pd.DataFrame()
    return {"p_overall": empty, "p_by_step": empty, "p_by_bucket": empty}


def _empty_question_tables() -> Dict[str, pd.DataFrame]:
    """Return placeholder question-level tables when no data exists."""
    empty = pd.DataFrame()
    return {"q_overall": empty, "q_by_step": empty, "q_by_bucket": empty}


def _write_core_reasking_tables(
    out_dir: str,
    pairs_df: pd.DataFrame,
    probs_df: pd.DataFrame,
    cond_df: pd.DataFrame,
    cond_forced_df: pd.DataFrame,
) -> None:
    """Write the base re-asking CSVs."""
    _write_dataframe_if_any(pairs_df, os.path.join(out_dir, "h3_pass2_prompt_level.csv"))
    _write_dataframe_if_any(probs_df, os.path.join(out_dir, "h3_pass2_question_level.csv"))
    _write_dataframe_if_any(cond_df, os.path.join(out_dir, "h3_pass2_conditioned_summary.csv"))
    _write_dataframe_if_any(
        cond_forced_df,
        os.path.join(out_dir, "h3_pass2_conditioned_by_forcing.csv"),
    )


def _prepare_prompt_tables(
    pairs_df: pd.DataFrame,
    out_dir: str,
) -> Dict[str, pd.DataFrame]:
    """Compute and persist prompt-level aggregates."""
    if pairs_df.empty:
        return _empty_prompt_tables()
    prompt_tables = {
        "p_overall": prompt_level_acc_with_ci(pairs_df, []),
        "p_by_step": prompt_level_acc_with_ci(pairs_df, ["step"]),
    }
    if "perplexity_bucket" in pairs_df.columns:
        prompt_tables["p_by_bucket"] = prompt_level_acc_with_ci(
            pairs_df,
            ["perplexity_bucket"],
        )
    else:
        prompt_tables["p_by_bucket"] = pd.DataFrame()

    _write_dataframe_if_any(
        prompt_tables["p_overall"],
        os.path.join(out_dir, "h3_prompt_level_overall.csv"),
    )
    _write_dataframe_if_any(
        prompt_tables["p_by_step"],
        os.path.join(out_dir, "h3_prompt_level_by_step.csv"),
    )
    _write_dataframe_if_any(
        prompt_tables["p_by_bucket"],
        os.path.join(out_dir, "h3_prompt_level_by_bucket.csv"),
    )
    return prompt_tables


def _prepare_question_tables(
    pairs_df: pd.DataFrame,
    out_dir: str,
) -> Dict[str, pd.DataFrame]:
    """Compute and persist question-level aggregates."""
    if pairs_df.empty:
        return _empty_question_tables()
    question_tables = {
        "q_overall": question_level_any_with_ci(pairs_df, []),
        "q_by_step": question_level_any_with_ci(pairs_df, ["step"]),
    }
    if "perplexity_bucket" in pairs_df.columns:
        question_tables["q_by_bucket"] = question_level_any_with_ci(
            pairs_df,
            ["perplexity_bucket"],
        )
    else:
        question_tables["q_by_bucket"] = pd.DataFrame()

    _write_dataframe_if_any(
        question_tables["q_overall"],
        os.path.join(out_dir, "h3_question_level_overall.csv"),
    )
    _write_dataframe_if_any(
        question_tables["q_by_step"],
        os.path.join(out_dir, "h3_question_level_by_step.csv"),
    )
    _write_dataframe_if_any(
        question_tables["q_by_bucket"],
        os.path.join(out_dir, "h3_question_level_by_bucket.csv"),
    )
    return question_tables


def _write_split_by_aha(
    pairs_df: pd.DataFrame,
    probs_df: pd.DataFrame,
    pass1_df: pd.DataFrame,
    out_dir: str,
) -> None:
    """Optionally write prompt/problem splits grouped by Aha variant."""
    if pairs_df.empty or probs_df.empty:
        return
    prompt_by_aha, problem_by_aha = split_reasking_by_aha(
        pairs_df,
        probs_df,
        pass1_df,
    )
    _write_dataframe_if_any(
        prompt_by_aha,
        os.path.join(out_dir, "h3_pass2_prompt_level_by_aha.csv"),
    )
    _write_dataframe_if_any(
        problem_by_aha,
        os.path.join(out_dir, "h3_pass2_question_level_by_aha.csv"),
    )


def run_reasking_analysis(
    samples_df: pd.DataFrame,
    pass1_df: pd.DataFrame,
    args: argparse.Namespace,
    out_dir: str,
) -> Dict[str, pd.DataFrame]:
    """Produce re-asking tables, aggregates, and optional split-by-aha files."""
    pass1_buckets = pass1_df[
        ["pair_id", "perplexity_bucket", "aha_words", "aha_gpt", "aha_formal"]
    ].drop_duplicates()
    aligned_df = samples_df.merge(pass1_buckets, on="pair_id", how="left")
    pairs_df, probs_df, cond_df, cond_forced_df = compute_reasking_tables(
        aligned_df,
        pass1_bucket_col="perplexity_bucket",
    )
    _write_core_reasking_tables(out_dir, pairs_df, probs_df, cond_df, cond_forced_df)

    prompt_tables = _prepare_prompt_tables(pairs_df, out_dir)
    question_tables = _prepare_question_tables(pairs_df, out_dir)

    if args.split_reasking_by_aha:
        _write_split_by_aha(pairs_df, probs_df, pass1_df, out_dir)

    results = {
        "pairs_df": pairs_df,
        "probs_df": probs_df,
        "cond_df": cond_df,
        "cond_forced_df": cond_forced_df,
    }
    results.update(prompt_tables)
    results.update(question_tables)
    return results


def maybe_plot_results(
    args: argparse.Namespace,
    out_dir: str,
    results: Dict[str, pd.DataFrame],
) -> None:
    """Generate plots when requested."""
    if not args.make_plots:
        return
    pairs_df = results["pairs_df"]
    q_overall = results["q_overall"]
    q_by_step = results["q_by_step"]
    q_by_bucket = results["q_by_bucket"]
    p_overall = results["p_overall"]
    p_by_step = results["p_by_step"]
    p_by_bucket = results["p_by_bucket"]

    if not q_overall.empty:
        plot_question_overall_ci(
            q_overall,
            os.path.join(out_dir, "h3_plot_question_overall.png"),
            args.also_pdf_plots,
        )
    if not q_by_step.empty:
        plot_question_by_step_ci(
            q_by_step,
            os.path.join(out_dir, "h3_plot_question_by_step.png"),
            args.also_pdf_plots,
        )
    if not q_by_bucket.empty:
        plot_question_by_bucket_ci(
            q_by_bucket,
            os.path.join(out_dir, "h3_plot_question_by_bucket.png"),
            args.also_pdf_plots,
        )

    if not p_overall.empty:
        plot_prompt_overall_ci(
            p_overall,
            os.path.join(out_dir, "h3_plot_prompt_overall.png"),
            args.also_pdf_plots,
        )
    if not p_by_step.empty:
        plot_prompt_by_step_ci(
            p_by_step,
            os.path.join(out_dir, "h3_plot_prompt_by_step.png"),
            args.also_pdf_plots,
        )
    if not p_by_bucket.empty:
        plot_prompt_by_bucket_ci(
            p_by_bucket,
            os.path.join(out_dir, "h3_plot_prompt_by_bucket.png"),
            args.also_pdf_plots,
        )
    if not pairs_df.empty:
        plot_prompt_level_deltas(
            pairs_df,
            os.path.join(out_dir, "h3_plot_prompt_level_delta.png"),
            by_forced=True,
            also_pdf=args.also_pdf_plots,
        )


def summarize_outputs(out_dir: str) -> None:
    """Print pointers to the key outputs."""
    print("\nWROTE:")
    for file_name in [
        "h3_glm_bucket_margins.csv",
        "h3_bucket_group_accuracy.csv",
        "h3_pass2_prompt_level.csv",
        "h3_pass2_question_level.csv",
        "h3_pass2_conditioned_summary.csv",
        "h3_pass2_conditioned_by_forcing.csv",
        "h3_prompt_level_overall.csv",
        "h3_prompt_level_by_step.csv",
        "h3_prompt_level_by_bucket.csv",
        "h3_question_level_overall.csv",
        "h3_question_level_by_step.csv",
        "h3_question_level_by_bucket.csv",
        "h3_plot_question_overall.png",
        "h3_plot_question_by_step.png",
        "h3_plot_question_by_bucket.png",
        "h3_plot_prompt_overall.png",
        "h3_plot_prompt_by_step.png",
        "h3_plot_prompt_by_bucket.png",
        "h3_plot_prompt_level_delta.png",
        "h3_summary.pdf",
        "h3_answers.txt",
    ]:
        candidate = os.path.join(out_dir, file_name)
        if os.path.exists(candidate):
            print(" ", candidate)


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the H3 uncertainty buckets analysis."""
    out_dir = args.out_dir or os.path.join(args.results_root, "h3_uncertainty_buckets")
    os.makedirs(out_dir, exist_ok=True)
    files = scan_jsonl_files(args.results_root, args.split)
    if not files:
        raise SystemExit("No JSONL files found.")

    samples_df = load_rows(
        files,
        gpt_mode=args.gpt_mode,
        gate_gpt_by_words=not args.no_gate_gpt_by_words,
        unc_field=args.unc_field,
        measure=args.measure,
    )
    if samples_df["pass_id"].min() > 1 or samples_df["pass_id"].max() < 1:
        raise SystemExit("No pass1 rows present; cannot proceed.")

    pass1_df = prepare_pass1_dataframe(samples_df, args)
    margins_df = run_glm_variants(pass1_df, args, out_dir)
    reask_results = run_reasking_analysis(samples_df, pass1_df, args, out_dir)

    if args.make_pdf:
        pdf_path = os.path.join(out_dir, "h3_summary.pdf")
        pdf_config = PdfSummaryConfig(
            dataset=args.dataset_name,
            model=args.model_name,
            font_family=args.font_family,
            font_size=int(args.font_size),
        )
        write_a4_summary_pdf(margins_df=margins_df, out_pdf=pdf_path, config=pdf_config)
        print("A4 summary PDF:", pdf_path)

    answers_path = os.path.join(out_dir, "h3_answers.txt")
    answers_tables = AnswersTables(
        cond=reask_results["cond_df"],
        q_overall=reask_results["q_overall"],
        q_by_step=reask_results["q_by_step"],
        q_by_bucket=reask_results["q_by_bucket"],
        p_overall=reask_results["p_overall"],
        p_by_step=reask_results["p_by_step"],
        p_by_bucket=reask_results["p_by_bucket"],
    )
    write_answers_txt(answers_path, answers_tables)
    print("Answers written to:", answers_path)

    maybe_plot_results(args, out_dir, reask_results)
    summarize_outputs(out_dir)


def main() -> None:
    """CLI entry point."""
    args = build_arg_parser().parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
