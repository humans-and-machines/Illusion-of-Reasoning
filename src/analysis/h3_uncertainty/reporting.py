"""
Reporting helpers for exporting tables and text summaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.plotting import apply_paper_font_style


@dataclass
class PdfSummaryConfig:
    """User-configurable styling metadata for the PDF summary."""

    dataset: str
    model: str
    font_family: str = "Times New Roman"
    font_size: int = 12


@dataclass
class AnswersTables:
    """Collection of prompt/question aggregates used for answers.txt."""

    cond: pd.DataFrame
    q_overall: pd.DataFrame
    q_by_step: pd.DataFrame
    q_by_bucket: pd.DataFrame
    p_overall: pd.DataFrame
    p_by_step: pd.DataFrame
    p_by_bucket: pd.DataFrame


_GROUP_SECTION_CONFIGS = [
    (
        "Question-level by step (any-correct):",
        "q_by_step",
        ["step"],
        [
            ("  P1", "any_pass1", ("any_pass1_lo", "any_pass1_hi")),
            ("  P2", "any_pass2", ("any_pass2_lo", "any_pass2_hi")),
        ],
    ),
    (
        "Question-level by (entropy) bucket (any-correct):",
        "q_by_bucket",
        ["perplexity_bucket"],
        [
            ("  P1", "any_pass1", ("any_pass1_lo", "any_pass1_hi")),
            ("  P2", "any_pass2", ("any_pass2_lo", "any_pass2_hi")),
        ],
    ),
    (
        "Prompt-level by step (accuracy):",
        "p_by_step",
        ["step"],
        [
            ("  P1", "acc_pass1", ("acc_pass1_lo", "acc_pass1_hi")),
            ("  P2", "acc_pass2", ("acc_pass2_lo", "acc_pass2_hi")),
        ],
    ),
    (
        "Prompt-level by (entropy) bucket (accuracy):",
        "p_by_bucket",
        ["perplexity_bucket"],
        [
            ("  P1", "acc_pass1", ("acc_pass1_lo", "acc_pass1_hi")),
            ("  P2", "acc_pass2", ("acc_pass2_lo", "acc_pass2_hi")),
        ],
    ),
]


def _format_overall_block(
    title: str,
    table: pd.DataFrame,
    pass1_col: str,
    pass2_col: str,
) -> List[str]:
    if table.empty:
        return []
    overall_row = table.iloc[0]
    return [
        title,
        (
            f"  Pass1 {pass1_col}: {overall_row[pass1_col]:.4f}  "
            f"(95% CI: {overall_row[pass1_col + '_lo']:.4f}, "
            f"{overall_row[pass1_col + '_hi']:.4f})"
        ),
        (
            f"  Pass2 {pass2_col}: {overall_row[pass2_col]:.4f}  "
            f"(95% CI: {overall_row[pass2_col + '_lo']:.4f}, "
            f"{overall_row[pass2_col + '_hi']:.4f})"
        ),
        f"  Δ = {overall_row['delta']:+.4f}",
        "",
    ]


def _format_table_lines(
    data_frame: pd.DataFrame,
    keys: List[str],
    metric_col: str,
    ci_cols: Tuple[str, str],
    label: str,
) -> List[str]:
    formatted: List[str] = []
    lower_col, upper_col = ci_cols
    for _, row in data_frame.iterrows():
        key_text = ", ".join(f"{key}={row[key]}" for key in keys)
        formatted.append(
            f"{label} [{key_text}]  pass1={row[metric_col]:.4f} "
            f"(CI {row[lower_col]:.4f},{row[upper_col]:.4f})"
        )
    return formatted


def _append_condition_sections(lines: List[str], cond_df: pd.DataFrame) -> None:
    """Append condition-based findings for pass1/pass2 accuracy."""
    if cond_df.empty:
        return
    pass1_condition = cond_df[cond_df["condition"] == "pass1_any_correct==1"]
    pass0_condition = cond_df[cond_df["condition"] == "pass1_any_correct==0"]
    if not pass1_condition.empty:
        pass1_row = pass1_condition.iloc[0]
        lines.extend(
            [
                "A) Given ≥1 correct in pass1, does pass2 increase overall accuracy?",
                f"  Problems: {int(pass1_row['n_problems'])}",
                (
                    "  Macro mean acc: "
                    f"pass1={pass1_row['macro_mean_acc_pass1']:.4f}, "
                    f"pass2={pass1_row['macro_mean_acc_pass2']:.4f}, "
                    f"Δ={pass1_row['macro_delta_mean']:+.4f}"
                ),
                (
                    "  Micro mean acc: "
                    f"pass1={pass1_row['micro_acc_pass1']:.4f}, "
                    f"pass2={pass1_row['micro_acc_pass2']:.4f}, "
                    f"Δ={pass1_row['micro_delta']:+.4f}"
                ),
                (
                    "  Reliability (stay any-correct in pass2): "
                    f"{pass1_row['share_keep_any_correct_in_pass2']:.4f}"
                ),
                "",
            ]
        )
    if not pass0_condition.empty:
        pass0_row = pass0_condition.iloc[0]
        lines.extend(
            [
                "B) Given wrong every time in pass1, does pass2 get it correct at least once?",
                f"  Problems: {int(pass0_row['n_problems'])}",
                f"  Share any-correct in pass2: {pass0_row['share_any_pass2']:.4f}",
                (
                    "  Macro mean acc: "
                    f"pass1={pass0_row['macro_mean_acc_pass1']:.4f}, "
                    f"pass2={pass0_row['macro_mean_acc_pass2']:.4f}, "
                    f"Δ={pass0_row['macro_delta_mean']:+.4f}"
                ),
                (
                    "  Micro mean acc: "
                    f"pass1={pass0_row['micro_acc_pass1']:.4f}, "
                    f"pass2={pass0_row['micro_acc_pass2']:.4f}, "
                    f"Δ={pass0_row['micro_delta']:+.4f}"
                ),
                "",
            ]
        )


def _append_group_sections(lines: List[str], tables: AnswersTables) -> None:
    """Append grouped tables (step/bucket by prompt/question)."""
    for title, table_attr, key_cols, metric_specs in _GROUP_SECTION_CONFIGS:
        table = getattr(tables, table_attr)
        if table.empty:
            continue
        lines.append(title)
        for label, metric, ci_cols in metric_specs:
            lines.extend(_format_table_lines(table, key_cols, metric, ci_cols, label))
        lines.append("")


def write_a4_summary_pdf(
    margins_df: pd.DataFrame,
    out_pdf: str,
    config: PdfSummaryConfig,
) -> None:
    """
    Render a bucket summary table onto an A4 PDF with Times-like fonts.
    """
    apply_paper_font_style(font_family=config.font_family, font_size=config.font_size)
    a4_size = (8.27, 11.69)
    fig = plt.figure(figsize=a4_size, dpi=300)
    axis = fig.add_axes([0.08, 0.08, 0.84, 0.84])
    axis.axis("off")
    vertical_pos = 0.98
    axis.text(
        0.0,
        vertical_pos,
        f"H3 Summary — {config.dataset}, {config.model}",
        ha="left",
        va="top",
        weight="bold",
    )
    vertical_pos -= 0.04
    axis.text(
        0.0,
        vertical_pos,
        "Bucket-wise Aha Margins (Δ Acc):",
        ha="left",
        va="top",
        weight="bold",
    )
    vertical_pos -= 0.02
    headers = ["Variant", "Bucket", "N", "Share Aha", "Δ Acc"]
    column_positions = [0.00, 0.28, 0.56, 0.72, 0.86]
    for position, header in zip(column_positions, headers):
        axis.text(
            position,
            vertical_pos,
            header,
            ha="left",
            va="top",
            weight="bold",
        )
    vertical_pos -= 0.012
    axis.plot([0.00, 0.98], [vertical_pos, vertical_pos], color="black", lw=0.5)
    vertical_pos -= 0.01
    for row in margins_df.head(20).itertuples(index=False):
        axis.text(column_positions[0], vertical_pos, str(row.variant).title(), ha="left", va="top")
        axis.text(column_positions[1], vertical_pos, row.perplexity_bucket, ha="left", va="top")
        axis.text(column_positions[2], vertical_pos, f"{int(row.N)}", ha="left", va="top")
        axis.text(column_positions[3], vertical_pos, f"{row.share_aha:.3f}", ha="left", va="top")
        axis.text(column_positions[4], vertical_pos, f"{row.AME_bucket:+.3f}", ha="left", va="top")
        vertical_pos -= 0.028
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def write_answers_txt(
    path: str,
    tables: AnswersTables,
) -> None:
    """
    Emit the textual summary of the main H3 findings.
    """

    lines = ["H3 Answers\n==========\n"]
    _append_condition_sections(lines, tables.cond)

    lines.extend(
        _format_overall_block(
            "Question-level (any-correct, overall)",
            tables.q_overall,
            "any_pass1",
            "any_pass2",
        ),
    )
    lines.extend(
        _format_overall_block(
            "Prompt-level (accuracy, overall)",
            tables.p_overall,
            "acc_pass1",
            "acc_pass2",
        ),
    )

    _append_group_sections(lines, tables)

    with open(path, "w", encoding="utf-8") as file_handle:
        file_handle.write("\n".join(lines))
