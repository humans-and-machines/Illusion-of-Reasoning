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
  - h1_glm_ame_summary.csv          # per-variant N, share_aha, Acc(aha=1),
                                    # Acc(aha=0), Δ, AME, coef, p
  - h1_group_accuracy.csv           # overall acc for aha=0/1 by variant
  - h1_group_accuracy_delta.csv     # acc_aha1, acc_aha0, delta by variant
  - h1_group_accuracy_by_step.csv   # per-step acc for aha=0/1 by variant

Optional:
  - --tex_path h1_glm_ame_summary.tex   # compact LaTeX table with per-variant
                                       # accuracies
  - --make_pdf                          # A4, 12pt summary PDF
                                       # (h1_glm_summary.pdf)
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


try:
    import statsmodels.api as sm  # type: ignore[import]
    import statsmodels.formula.api as smf  # type: ignore[import]
except ImportError:  # pragma: no cover - optional dependency
    sm = None  # type: ignore[assignment]
    smf = None  # type: ignore[assignment]

from src.analysis.common.parser_helpers import standard_results_parser
from src.analysis.core.plotting_helpers import create_a4_figure, set_global_fonts
from src.analysis.io import iter_records_from_file, scan_jsonl_files
from src.analysis.labels import aha_gpt_broad, aha_gpt_canonical, aha_words
from src.analysis.metrics import glm_fit_with_covariance, write_glm_summary_header
from src.analysis.utils import (
    add_formal_threshold_args,
    add_gpt_label_policy_args,
    coerce_bool,
    extract_pass1_and_step,
    nat_step_from_path,
    problem_key_from_record,
)


matplotlib.use("Agg")

# ---------- load samples ----------


def _build_sample_row(
    record: Dict[str, Any],
    step_from_name: int | None,
    gpt_mode: str,
    gate_gpt_by_words: bool,
) -> Dict[str, Any] | None:
    """
    Build a single per-sample row with correctness and Aha labels.

    :param record: JSON record from a PASS-1-style log.
    :param step_from_name: Optional step inferred from the filename.
    :param gpt_mode: GPT label mode (``\"canonical\"`` or ``\"broad\"``).
    :param gate_gpt_by_words: Whether to enforce ``aha_gpt <= aha_words``.
    :returns: Row dictionary or ``None`` if the record should be skipped.
    """
    pass1, step_value = extract_pass1_and_step(record, step_from_name)
    if not pass1 or step_value is None:
        return None

    problem = problem_key_from_record(record, missing_default="unknown")

    correct = coerce_bool(pass1.get("is_correct_pred"))
    if correct is None:
        return None

    words = aha_words(pass1)
    gpt_base = aha_gpt_canonical if gpt_mode == "canonical" else aha_gpt_broad
    gpt_raw = gpt_base(pass1, record)
    gpt = int(gpt_raw and words) if gate_gpt_by_words else int(gpt_raw)

    return {
        "problem": str(problem),
        "step": int(step_value),
        "correct": int(correct),
        "aha_words": int(words),
        "aha_gpt": int(gpt),
    }


def load_samples(
    files: List[str],
    gpt_mode: str = "canonical",
    gate_gpt_by_words: bool = True,
) -> pd.DataFrame:
    """
    Return per-sample rows with correctness and Words/GPT Aha labels.

    The output has columns: ``problem``, ``step``, ``correct``, ``aha_words``,
    and ``aha_gpt``. By default, ``aha_gpt`` is gated so that
    ``aha_gpt <= aha_words`` at the sample level.
    """
    rows: List[Dict[str, Any]] = []
    for path in files:
        step_from_name = nat_step_from_path(path)
        for record in iter_records_from_file(path):
            row = _build_sample_row(
                record,
                step_from_name,
                gpt_mode,
                gate_gpt_by_words,
            )
            if row is not None:
                rows.append(row)

    samples_df = pd.DataFrame(rows)
    if samples_df.empty:
        raise SystemExit("No PASS-1 rows found.")
    return samples_df


# ---------- problem-step table + Formal ----------


def build_problem_step(samples_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sample-level rows to per-(step, problem) summaries.

    The result includes counts, accuracy, and GPT Aha rate per (step, problem).
    """
    group = samples_df.groupby(["step", "problem"], as_index=False)
    problem_step_df = group.agg(
        n_samples=("correct", "size"),
        freq_correct=("correct", "mean"),
        aha_any_gpt=("aha_gpt", "max"),
        aha_rate_gpt=("aha_gpt", "mean"),
    )
    for col in ("n_samples", "aha_any_gpt"):
        problem_step_df[col] = problem_step_df[col].astype(int)
    problem_step_df["freq_correct"] = problem_step_df["freq_correct"].astype(float)
    problem_step_df["aha_rate_gpt"] = problem_step_df["aha_rate_gpt"].astype(float)
    return problem_step_df.sort_values(["problem", "step"]).reset_index(drop=True)


def mark_formal(
    problem_step_df: pd.DataFrame,
    delta1: float,
    delta2: float,
    min_prior_steps: int,
) -> pd.DataFrame:
    """
    Mark formal Aha at the (problem, step) level.

    A formal Aha requires:
      - Prior failure: max past ``freq_correct`` < ``delta1``.
      - Prior shift-stability: max past ``aha_rate_gpt`` < ``delta2``.
      - Shift now: ``aha_any_gpt == 1`` at the current step.
    """
    required_cols = {"step", "problem", "freq_correct", "aha_rate_gpt", "aha_any_gpt"}
    if not required_cols.issubset(problem_step_df.columns):
        raise ValueError("mark_formal: missing required columns")

    problem_step_df = problem_step_df.sort_values(["problem", "step"]).copy()
    flags = np.zeros(len(problem_step_df), dtype=int)
    idx = 0
    for _, sub in problem_step_df.groupby("problem", sort=False):
        sub_sorted = sub.sort_values("step")
        freq = sub_sorted["freq_correct"].to_numpy(float)
        rate = sub_sorted["aha_rate_gpt"].to_numpy(float)
        shift = sub_sorted["aha_any_gpt"].to_numpy(int)
        for j in range(len(sub_sorted)):
            if j < min_prior_steps:
                flags[idx] = 0
            else:
                prior_ok = float(np.max(freq[:j])) < delta1 and float(np.max(rate[:j])) < delta2
                flags[idx] = int(prior_ok and (shift[j] == 1))
            idx += 1
    problem_step_df["aha_formal_ps"] = flags
    return problem_step_df


# ---------- GLM ----------


def _compute_glm_metrics(
    glm_data: pd.DataFrame,
    glm_result,
    aha_column: str,
) -> Dict[str, float]:
    """
    Compute AME and accuracy metrics for a fitted GLM and Aha column.
    """
    data_with_aha1 = glm_data.copy()
    data_with_aha1[aha_column] = 1
    data_with_aha0 = glm_data.copy()
    data_with_aha0[aha_column] = 0
    ame = float(
        np.mean(
            glm_result.predict(data_with_aha1) - glm_result.predict(data_with_aha0),
        ),
    )

    acc_overall = float(glm_data["correct"].mean())
    mask_aha1 = glm_data[aha_column] == 1
    mask_aha0 = glm_data[aha_column] == 0
    acc_aha1 = float(glm_data.loc[mask_aha1, "correct"].mean()) if mask_aha1.any() else float("nan")
    acc_aha0 = float(glm_data.loc[mask_aha0, "correct"].mean()) if mask_aha0.any() else float("nan")
    if np.isfinite(acc_aha1) and np.isfinite(acc_aha0):
        delta_acc = acc_aha1 - acc_aha0
    else:
        delta_acc = float("nan")

    return {
        "ame": ame,
        "acc_overall": acc_overall,
        "acc_aha1": acc_aha1,
        "acc_aha0": acc_aha0,
        "delta_acc": delta_acc,
    }


def fit_glm(
    samples_df: pd.DataFrame,
    aha_col: str,
    out_txt: str,
    cluster_by: str = "problem",
) -> Dict[str, Any]:
    """
    Fit a Binomial GLM for correctness with an Aha term and report summary stats.

    The model is ``correct ~ C(problem) + step_std + aha_col`` with cluster-robust
    covariance by problem (or as specified by ``cluster_by``).
    """
    if sm is None or smf is None:
        raise RuntimeError("statsmodels is required (pip install statsmodels).")

    glm_data = samples_df.copy()
    glm_data["step_std"] = (glm_data["step"] - glm_data["step"].mean()) / (glm_data["step"].std(ddof=0) + 1e-8)
    if aha_col not in glm_data.columns:
        raise ValueError(f"missing {aha_col}")

    model = smf.glm(
        f"correct ~ C(problem) + step_std + {aha_col}",
        data=glm_data,
        family=sm.families.Binomial(),
    )

    try:
        glm_result, cov_type, cov_kwds = glm_fit_with_covariance(
            model,
            glm_data,
            cluster_by,
        )
    except np.linalg.LinAlgError:
        print(
            "[warn] Singular covariance when clustering; refitting without cluster-robust SEs.",
        )
        glm_result = model.fit()
        cov_type = "nonrobust"
        cov_kwds = None

    metrics = _compute_glm_metrics(glm_data, glm_result, aha_col)

    write_glm_summary_header(out_txt, glm_result, cov_type, cov_kwds)
    with open(out_txt, "a", encoding="utf-8") as summary_file:
        summary_file.write(
            f"\nAverage Marginal Effect (AME) of {aha_col}: {metrics['ame']:.4f}\n",
        )
        summary_file.write(
            f"\nAcc(overall)={metrics['acc_overall']:.4f}  "
            f"Acc(aha=1)={metrics['acc_aha1']:.4f}  "
            f"Acc(aha=0)={metrics['acc_aha0']:.4f}  "
            f"Δ={metrics['delta_acc']:+.4f}\n",
        )

    coef_value = float(glm_result.params.get(aha_col, np.nan))
    std_err = float(glm_result.bse.get(aha_col, np.nan))
    z_score = coef_value / std_err if std_err and np.isfinite(std_err) else np.nan
    p_value = float(glm_result.pvalues.get(aha_col, np.nan))

    return {
        "N": int(len(glm_data)),
        "aha": aha_col,
        "share_aha": float(glm_data[aha_col].mean()),
        "acc_overall": metrics["acc_overall"],
        "acc_aha1": metrics["acc_aha1"],
        "acc_aha0": metrics["acc_aha0"],
        "delta_acc": metrics["delta_acc"],
        "coef": coef_value,
        "se": std_err,
        "z": z_score,
        "p": p_value,
        "AME": metrics["ame"],
        "summary_path": out_txt,
    }


# ---------- Accuracy per grouping ----------


def _append_accuracy_rows_for_variant(
    samples_df: pd.DataFrame,
    aha_column: str,
    variant_name: str,
    rows: Dict[str, Any],
) -> None:
    """
    Append overall, delta, and per-step accuracy rows for a single Aha variant.
    """
    grouped = samples_df.groupby(samples_df[aha_column], as_index=False).agg(
        n=("correct", "size"), k=("correct", "sum")
    )
    grouped["acc"] = grouped["k"] / grouped["n"]

    for _, grouped_row in grouped.iterrows():
        rows["overall"].append(
            {
                "variant": variant_name,
                "aha": int(grouped_row[aha_column]),
                "n": int(grouped_row["n"]),
                "k_correct": int(grouped_row["k"]),
                "accuracy": float(grouped_row["acc"]),
            },
        )

    def _extract_acc_and_count(label: int) -> tuple[float, int]:
        subset = grouped.loc[grouped[aha_column] == label]
        if subset.empty:
            return np.nan, 0
        return float(subset["acc"].iloc[0]), int(subset["n"].iloc[0])

    acc_aha1, num_aha1 = _extract_acc_and_count(1)
    acc_aha0, num_aha0 = _extract_acc_and_count(0)
    rows["delta"].append(
        {
            "variant": variant_name,
            "acc_aha1": acc_aha1,
            "acc_aha0": acc_aha0,
            "delta_acc": (acc_aha1 - acc_aha0 if (np.isfinite(acc_aha1) and np.isfinite(acc_aha0)) else np.nan),
            "n_aha1": num_aha1,
            "n_aha0": num_aha0,
        },
    )

    grouped_step = samples_df.groupby(["step", samples_df[aha_column]], as_index=False).agg(
        n=("correct", "size"), k=("correct", "sum")
    )
    grouped_step["accuracy"] = grouped_step["k"] / grouped_step["n"]
    grouped_step = grouped_step.rename(columns={aha_column: "aha"})
    grouped_step["variant"] = variant_name
    rows["by_step"].append(
        grouped_step[["variant", "step", "aha", "n", "k", "accuracy"]],
    )


def compute_group_accuracy_tables(
    samples_df: pd.DataFrame,
    out_dir: str,
) -> Tuple[str, str, str]:
    """
    Compute grouped accuracy tables for the Words, GPT, and Formal variants.

    Returns three CSV paths:
      - overall accuracy by variant and Aha value,
      - delta table (acc_aha1 - acc_aha0) by variant,
      - per-step accuracy for aha=0/1 by variant.
    """
    variants = [
        ("aha_words", "words"),
        ("aha_gpt", "gpt"),
        ("aha_formal", "formal"),
    ]
    rows: Dict[str, Any] = {
        "overall": [],
        "delta": [],
        "by_step": [],
    }

    for aha_column, variant_name in variants:
        if aha_column not in samples_df.columns:
            continue
        _append_accuracy_rows_for_variant(
            samples_df=samples_df,
            aha_column=aha_column,
            variant_name=variant_name,
            rows=rows,
        )

    overall_df = pd.DataFrame(rows["overall"]).sort_values(["variant", "aha"]).reset_index(drop=True)
    delta_df = pd.DataFrame(rows["delta"]).sort_values(["variant"]).reset_index(drop=True)
    by_step_df = pd.concat(rows["by_step"], ignore_index=True) if rows["by_step"] else pd.DataFrame()

    acc_csv = os.path.join(out_dir, "h1_group_accuracy.csv")
    delta_csv = os.path.join(out_dir, "h1_group_accuracy_delta.csv")
    step_csv = os.path.join(out_dir, "h1_group_accuracy_by_step.csv")

    overall_df.to_csv(acc_csv, index=False)
    delta_df.to_csv(delta_csv, index=False)
    by_step_df.to_csv(step_csv, index=False)

    return acc_csv, delta_csv, step_csv


# ---------- A4 PDF summary (Times New Roman 12pt) ----------


def _fmt_float_or_str(value: Any) -> str:
    """
    Format floats to 4 decimal places; leave other values as strings.
    """
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        return f"{value:.4f}"
    return str(value)


def _draw_glm_table(
    axis: plt.Axes,
    summary_df: pd.DataFrame,
) -> float:
    """
    Draw the GLM summary table and return the updated y position.
    """
    columns = [
        "variant",
        "N",
        "share_aha",
        "acc_aha1",
        "acc_aha0",
        "delta_acc",
        "AME",
        "coef",
        "p",
    ]
    headers = [
        "Variant",
        "N",
        "Share Aha",
        "Acc(aha=1)",
        "Acc(aha=0)",
        "Δ (pp)",
        "AME",
        "Coef",
        "p-value",
    ]
    table_df = summary_df[columns].copy()
    table_df["delta_acc"] = table_df["delta_acc"] * 100.0
    table_rows = [[_fmt_float_or_str(value) for value in row_values] for row_values in table_df.values.tolist()]

    y_cursor = 0.92
    line_height = 0.045
    axis.text(
        0.0,
        y_cursor,
        ("GLM: correct ~ C(problem) + step_std + aha (cluster-robust SEs by problem)"),
        ha="left",
        va="top",
    )
    y_cursor -= 0.02

    x_positions = [0.00, 0.16, 0.28, 0.44, 0.59, 0.72, 0.82, 0.90, 0.97]
    for x_pos, header in zip(x_positions, headers):
        axis.text(x_pos, y_cursor, header, ha="left", va="top", weight="bold")
    y_cursor -= 0.012
    axis.plot([0.00, 0.98], [y_cursor, y_cursor], color="black", lw=0.5)
    y_cursor -= 0.010
    for row_values in table_rows:
        for x_pos, cell in zip(x_positions, row_values):
            axis.text(x_pos, y_cursor, cell, ha="left", va="top")
        y_cursor -= line_height
    return y_cursor


def _draw_accuracy_delta_table(
    axis: plt.Axes,
    acc_delta_df: pd.DataFrame,
    start_y: float,
) -> None:
    """
    Draw the summary of group accuracy deltas on the given axis.
    """
    y_cursor = start_y - 0.02
    axis.text(
        0.0,
        y_cursor,
        "Group accuracy deltas (aha=1 vs aha=0):",
        ha="left",
        va="top",
        weight="bold",
    )
    y_cursor -= 0.02
    headers = ["Variant", "acc(aha=1)", "acc(aha=0)", "Δ (pp)", "n1", "n0"]
    x_positions = [0.00, 0.25, 0.45, 0.65, 0.82, 0.90]
    for x_pos, header in zip(x_positions, headers):
        axis.text(x_pos, y_cursor, header, ha="left", va="top", weight="bold")
    y_cursor -= 0.012
    axis.plot([0.00, 0.98], [y_cursor, y_cursor], color="black", lw=0.5)
    y_cursor -= 0.010

    acc_delta_df = acc_delta_df.copy()
    acc_delta_df["delta_pp"] = acc_delta_df["delta_acc"] * 100.0
    line_height = 0.045
    for _, delta_row in acc_delta_df.sort_values("variant").iterrows():
        axis.text(
            x_positions[0],
            y_cursor,
            str(delta_row["variant"]).title(),
            ha="left",
            va="top",
        )
        acc_aha1 = f"{delta_row['acc_aha1']:.4f}" if np.isfinite(delta_row["acc_aha1"]) else "nan"
        acc_aha0 = f"{delta_row['acc_aha0']:.4f}" if np.isfinite(delta_row["acc_aha0"]) else "nan"
        delta_pp = f"{delta_row['delta_pp']:+.2f}" if np.isfinite(delta_row["delta_pp"]) else "nan"
        axis.text(x_positions[1], y_cursor, acc_aha1, ha="left", va="top")
        axis.text(x_positions[2], y_cursor, acc_aha0, ha="left", va="top")
        axis.text(x_positions[3], y_cursor, delta_pp, ha="left", va="top")
        axis.text(
            x_positions[4],
            y_cursor,
            f"{int(delta_row['n_aha1'])}",
            ha="left",
            va="top",
        )
        axis.text(
            x_positions[5],
            y_cursor,
            f"{int(delta_row['n_aha0'])}",
            ha="left",
            va="top",
        )
        y_cursor -= line_height


def write_a4_summary_pdf(
    summary_df: pd.DataFrame,
    acc_delta_df: pd.DataFrame,
    out_pdf: str,
    args: argparse.Namespace,
) -> None:
    """
    Create a one-page A4 PDF with 12pt text summarizing GLM and accuracy deltas.
    """
    set_global_fonts(
        font_family=args.font_family,
        font_size=int(args.font_size),
    )

    fig = create_a4_figure(dpi=300)
    pdf_axis = fig.add_axes([0.08, 0.08, 0.84, 0.84])
    pdf_axis.axis("off")

    title = f"H1 GLM Summary — {args.dataset_name}, {args.model_name}"
    pdf_axis.text(0.0, 1.0, title, ha="left", va="top", weight="bold")

    y_position = _draw_glm_table(pdf_axis, summary_df)
    _draw_accuracy_delta_table(pdf_axis, acc_delta_df, start_y=y_position)

    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def scan_files(root: str, split_substr: str | None) -> list[str]:
    """
    Recursively collect ``.jsonl`` files under a root directory.

    :param root: Root directory under which to search.
    :param split_substr: Optional substring that must appear in the filename.
    :returns: Sorted list of matching file paths.
    """
    return scan_jsonl_files(root, split_substr=split_substr)


# ---------- main helpers ----------


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build and return the command-line argument parser for the H1 analysis.
    """
    parser = standard_results_parser()

    # GPT label policy
    # NOTE: default True to enforce aha_gpt ⊆ aha_words unless explicitly disabled.
    add_gpt_label_policy_args(parser)
    parser.add_argument("--min_step", type=int, default=None)
    parser.add_argument("--max_step", type=int, default=None)

    # Formal thresholds
    add_formal_threshold_args(parser)

    # GLM covariance clustering
    parser.add_argument("--cluster_by", choices=["problem", "none"], default="problem")

    parser.add_argument(
        "--aha_variant",
        choices=["words", "gpt", "formal", "all"],
        default="all",
        help=(
            "Which Aha indicator(s) to include in the GLMs. "
            '"all" runs Words, GPT, and Formal; otherwise only the selected variant is fit.'
        ),
    )

    parser.add_argument(
        "--no_gate_formal_by_gpt",
        dest="gate_formal_by_gpt",
        action="store_false",
        help=(
            "By default, formal Aha indicators are intersected with GPT-labeled shifts "
            "(aha_formal <= aha_gpt). Pass this flag to use the raw formal signal without "
            "requiring an accompanying GPT shift label."
        ),
    )
    parser.set_defaults(gate_formal_by_gpt=True)

    # optional LaTeX table
    parser.add_argument(
        "--tex_path",
        default=None,
        help=("Write a LaTeX table here if provided (e.g., h1_glm_ame_summary.tex)."),
    )

    # Optional A4 PDF summary
    parser.add_argument(
        "--make_pdf",
        action="store_true",
        help="Write an A4, 12pt summary PDF (h1_glm_summary.pdf).",
    )
    parser.add_argument(
        "--font_family",
        default="Times New Roman",
        help='PDF font family (default: "Times New Roman").',
    )
    parser.add_argument(
        "--font_size",
        type=int,
        default=12,
        help="PDF font size in points (default: 12).",
    )
    return parser


def _load_and_prepare_samples(args: argparse.Namespace) -> pd.DataFrame:
    """
    Load per-sample rows and attach Words/GPT/Formal Aha labels.
    """
    files = scan_files(args.results_root, args.split)
    if not files:
        raise SystemExit("No JSONL files found.")

    samples_df = load_samples(
        files,
        gpt_mode=args.gpt_mode,
        gate_gpt_by_words=not args.no_gate_gpt_by_words,
    )
    if args.min_step is not None:
        samples_df = samples_df[samples_df["step"] >= args.min_step]
    if args.max_step is not None:
        samples_df = samples_df[samples_df["step"] <= args.max_step]
    if samples_df.empty:
        raise SystemExit("No rows remain after step filtering.")

    problem_step_df = build_problem_step(samples_df)
    problem_step_df = mark_formal(
        problem_step_df,
        delta1=args.delta1,
        delta2=args.delta2,
        min_prior_steps=args.min_prior_steps,
    )
    samples_df = samples_df.merge(
        problem_step_df[["step", "problem", "aha_formal_ps"]],
        on=["step", "problem"],
        how="left",
    ).fillna({"aha_formal_ps": 0})
    samples_df["aha_formal_ps"] = samples_df["aha_formal_ps"].astype(int)

    # Optionally enforce strict subset: aha_formal <= aha_gpt (sample-level)
    if getattr(args, "gate_formal_by_gpt", True):
        samples_df["aha_formal"] = (samples_df["aha_formal_ps"] & samples_df["aha_gpt"]).astype(int)
    else:
        samples_df["aha_formal"] = samples_df["aha_formal_ps"]

    # (Optional) sanity check / debug note on subset relations
    share_words = float(samples_df["aha_words"].mean())
    share_gpt = float(samples_df["aha_gpt"].mean())
    share_formal = float(samples_df["aha_formal"].mean())
    enforce_gpt_subset = not getattr(args, "no_gate_gpt_by_words", False)
    enforce_formal_subset = getattr(args, "gate_formal_by_gpt", True)
    tol = 1e-12
    warn_needed = False
    if enforce_gpt_subset and share_gpt > share_words + tol:
        warn_needed = True
    if enforce_formal_subset and share_formal > share_gpt + tol:
        warn_needed = True
    if warn_needed:
        print(
            "[warn] Subset relations violated (unexpected). shares:",
            f"words={share_words:.4f}, gpt={share_gpt:.4f}, formal={share_formal:.4f}",
        )

    return samples_df


def _fit_glms_and_write_summary(
    args: argparse.Namespace,
    samples_df: pd.DataFrame,
    out_dir: str,
) -> Tuple[pd.DataFrame, str]:
    """
    Fit GLMs for Words/GPT/Formal Aha variants and write the summary CSV.
    """
    rows: list[dict[str, Any]] = []
    variant_specs = [
        ("aha_words", "words"),
        ("aha_gpt", "gpt"),
        ("aha_formal", "formal"),
    ]
    if args.aha_variant != "all":
        variant_specs = [
            spec for spec in variant_specs if spec[1] == args.aha_variant
        ]
    for aha_column, variant_tag in variant_specs:
        out_txt = os.path.join(
            out_dir,
            f"logit_pass1_correct_on_step_{aha_column}.txt",
        )
        glm_result = fit_glm(
            samples_df,
            aha_col=aha_column,
            out_txt=out_txt,
            cluster_by=args.cluster_by,
        )
        glm_result["dataset"] = args.dataset_name
        glm_result["model"] = args.model_name
        glm_result["variant"] = variant_tag
        rows.append(glm_result)

    summary_df = pd.DataFrame(rows)[
        [
            "dataset",
            "model",
            "variant",
            "N",
            "share_aha",
            "acc_overall",
            "acc_aha1",
            "acc_aha0",
            "delta_acc",
            "AME",
            "coef",
            "se",
            "z",
            "p",
            "summary_path",
        ]
    ].sort_values("variant")
    csv_path = os.path.join(out_dir, "h1_glm_ame_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    return summary_df, csv_path


def _write_latex_table(summary_df: pd.DataFrame, tex_path: str) -> None:
    """
    Write an optional LaTeX summary table for the GLM variants.
    """

    def fmt(value: Any) -> str:
        if isinstance(value, (float, np.floating)) and np.isfinite(value):
            return f"{value:.4f}"
        return str(value)

    lines = [
        "\\begin{tabular}{lrrrrrrrr}",
        "\\toprule",
        ("Variant & N & Share Aha & Acc(aha=1) & Acc(aha=0) & $\\Delta$ (pp) & AME & Coef & $p$ \\\\"),
        "\\midrule",
    ]
    for _, summary_row in summary_df.iterrows():
        delta_pp = 100.0 * summary_row["delta_acc"] if np.isfinite(summary_row["delta_acc"]) else np.nan
        lines.append(
            f"{summary_row['variant'].title()} & {int(summary_row['N'])} & "
            f"{fmt(summary_row['share_aha'])} & "
            f"{fmt(summary_row['acc_aha1'])} & {fmt(summary_row['acc_aha0'])} & "
            f"{'+' if np.isfinite(delta_pp) and delta_pp >= 0 else ''}"
            f"{'' if not np.isfinite(delta_pp) else f'{delta_pp:.2f}'} & "
            f"{fmt(summary_row['AME'])} & {fmt(summary_row['coef'])} & {fmt(summary_row['p'])} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(tex_path, "w", encoding="utf-8") as file_handle:
        file_handle.write("\n".join(lines))


def _write_pdf_summary(
    args: argparse.Namespace,
    out_dir: str,
    summary_df: pd.DataFrame,
    delta_csv: str,
) -> None:
    """
    Write the optional A4, 12pt summary PDF for the H1 GLM results.
    """
    pdf_path = os.path.join(out_dir, "h1_glm_summary.pdf")
    acc_delta_df = pd.read_csv(delta_csv)
    write_a4_summary_pdf(
        summary_df=summary_df,
        acc_delta_df=acc_delta_df,
        out_pdf=pdf_path,
        args=args,
    )
    print("A4 summary PDF:", pdf_path)


def _print_console_summary(
    summary_df: pd.DataFrame,
    summary_csv_path: str,
    acc_csv: str,
    delta_csv: str,
    step_csv: str,
) -> None:
    """
    Print variant-specific accuracies and file locations to the console.
    """
    print("Wrote:", summary_csv_path)
    print("Group accuracy (overall):", acc_csv)
    print("Group accuracy (delta):  ", delta_csv)
    print("Group accuracy by step:  ", step_csv)
    for _, summary_row in summary_df.iterrows():
        delta_pp = 100.0 * summary_row["delta_acc"] if np.isfinite(summary_row["delta_acc"]) else np.nan
        if np.isfinite(delta_pp):
            sign = "+" if delta_pp >= 0 else ""
            delta_str = f"{sign}{delta_pp:.2f}"
        else:
            delta_str = "nan"
        print(
            f"[{summary_row['variant']}] N={int(summary_row['N'])}, "
            f"share_aha={summary_row['share_aha']:.4f}, "
            f"acc(aha=1)={summary_row['acc_aha1']:.4f}, "
            f"acc(aha=0)={summary_row['acc_aha0']:.4f}, "
            f"Δ={delta_str}pp, "
            f"AME={summary_row['AME']:.4f}, coef={summary_row['coef']:.4f}, "
            f"p={summary_row['p']:.4g}",
        )
        print(f"  See: {summary_row['summary_path']}")


# ---------- main ----------


def main() -> None:
    """
    CLI entry point for the H1 formal/GPT/Words GLM analysis.
    """
    parser = build_arg_parser()
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(
        args.results_root,
        "h1_glm_test",
    )
    os.makedirs(out_dir, exist_ok=True)

    samples_df = _load_and_prepare_samples(args)
    print(f"[info] Loaded {len(samples_df):,} rows from {args.results_root}")
    summary_df, summary_csv_path = _fit_glms_and_write_summary(args, samples_df, out_dir)

    acc_csv, delta_csv, step_csv = compute_group_accuracy_tables(samples_df, out_dir)

    if args.tex_path:
        _write_latex_table(summary_df, args.tex_path)

    if args.make_pdf:
        _write_pdf_summary(args, out_dir, summary_df, delta_csv)

    _print_console_summary(summary_df, summary_csv_path, acc_csv, delta_csv, step_csv)


if __name__ == "__main__":
    main()
