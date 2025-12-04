import pandas as pd

import src.analysis.h3_uncertainty.reporting as rep


def test_format_overall_block_and_empty():
    table = pd.DataFrame(
        [
            {
                "any_pass1": 0.5,
                "any_pass1_lo": 0.4,
                "any_pass1_hi": 0.6,
                "any_pass2": 0.7,
                "any_pass2_lo": 0.6,
                "any_pass2_hi": 0.8,
                "delta": 0.2,
            }
        ]
    )
    lines = rep._format_overall_block("Title", table, "any_pass1", "any_pass2")
    assert "Title" in lines[0]
    assert "Pass1 any_pass1" in lines[1]
    assert "Δ = +0.2000" in lines[3]
    assert rep._format_overall_block("X", pd.DataFrame(), "a", "b") == []


def test_format_table_lines():
    df = pd.DataFrame({"step": [1, 2], "metric": [0.1, 0.2], "lo": [0.0, 0.1], "hi": [0.2, 0.3]})
    lines = rep._format_table_lines(df, keys=["step"], metric_col="metric", ci_cols=("lo", "hi"), label="L")
    assert len(lines) == 2
    assert "step=1" in lines[0] and "step=2" in lines[1]


def test_append_condition_sections_and_groups(tmp_path):
    cond_df = pd.DataFrame(
        [
            {
                "condition": "pass1_any_correct==1",
                "n_problems": 3,
                "macro_mean_acc_pass1": 0.1,
                "macro_mean_acc_pass2": 0.2,
                "macro_delta_mean": 0.1,
                "micro_acc_pass1": 0.3,
                "micro_acc_pass2": 0.6,
                "micro_delta": 0.3,
                "share_keep_any_correct_in_pass2": 0.8,
            },
            {
                "condition": "pass1_any_correct==0",
                "n_problems": 2,
                "macro_mean_acc_pass1": 0.0,
                "macro_mean_acc_pass2": 0.5,
                "macro_delta_mean": 0.5,
                "micro_acc_pass1": 0.0,
                "micro_acc_pass2": 0.5,
                "micro_delta": 0.5,
                "share_any_pass2": 0.5,
            },
        ]
    )
    lines = []
    rep._append_condition_sections(lines, cond_df)
    assert any("Given ≥1 correct" in line for line in lines)
    assert any("Given wrong every time" in line for line in lines)

    tables = rep.AnswersTables(
        cond=cond_df,
        q_overall=pd.DataFrame(
            [
                {
                    "any_pass1": 0.5,
                    "any_pass1_lo": 0.4,
                    "any_pass1_hi": 0.6,
                    "any_pass2": 0.6,
                    "any_pass2_lo": 0.5,
                    "any_pass2_hi": 0.7,
                    "delta": 0.1,
                }
            ]
        ),
        q_by_step=pd.DataFrame(
            [
                {
                    "step": 1,
                    "any_pass1": 0.5,
                    "any_pass1_lo": 0.4,
                    "any_pass1_hi": 0.6,
                    "any_pass2": 0.7,
                    "any_pass2_lo": 0.6,
                    "any_pass2_hi": 0.8,
                }
            ]
        ),
        q_by_bucket=pd.DataFrame(
            [
                {
                    "perplexity_bucket": "low",
                    "any_pass1": 0.5,
                    "any_pass1_lo": 0.4,
                    "any_pass1_hi": 0.6,
                    "any_pass2": 0.7,
                    "any_pass2_lo": 0.6,
                    "any_pass2_hi": 0.8,
                }
            ]
        ),
        p_overall=pd.DataFrame(
            [
                {
                    "acc_pass1": 0.5,
                    "acc_pass1_lo": 0.4,
                    "acc_pass1_hi": 0.6,
                    "acc_pass2": 0.6,
                    "acc_pass2_lo": 0.5,
                    "acc_pass2_hi": 0.7,
                    "delta": 0.1,
                }
            ]
        ),
        p_by_step=pd.DataFrame(
            [
                {
                    "step": 1,
                    "acc_pass1": 0.5,
                    "acc_pass1_lo": 0.4,
                    "acc_pass1_hi": 0.6,
                    "acc_pass2": 0.7,
                    "acc_pass2_lo": 0.6,
                    "acc_pass2_hi": 0.8,
                }
            ]
        ),
        p_by_bucket=pd.DataFrame(
            [
                {
                    "perplexity_bucket": "low",
                    "acc_pass1": 0.5,
                    "acc_pass1_lo": 0.4,
                    "acc_pass1_hi": 0.6,
                    "acc_pass2": 0.7,
                    "acc_pass2_lo": 0.6,
                    "acc_pass2_hi": 0.8,
                }
            ]
        ),
    )
    lines_final = []
    rep._append_group_sections(lines_final, tables)
    assert lines_final  # at least some content was appended


def test_write_answers_txt(tmp_path):
    tables = rep.AnswersTables(
        cond=pd.DataFrame(),
        q_overall=pd.DataFrame(),
        q_by_step=pd.DataFrame(),
        q_by_bucket=pd.DataFrame(),
        p_overall=pd.DataFrame(),
        p_by_step=pd.DataFrame(),
        p_by_bucket=pd.DataFrame(),
    )
    out_path = tmp_path / "answers.txt"
    rep.write_answers_txt(str(out_path), tables)
    assert out_path.exists()
    content = out_path.read_text()
    assert "H3 Answers" in content


def test_write_a4_summary_pdf(tmp_path, monkeypatch):
    margins = pd.DataFrame(
        [
            {"variant": "words", "perplexity_bucket": "low", "N": 10, "share_aha": 0.5, "AME_bucket": 0.1},
            {"variant": "gpt", "perplexity_bucket": "high", "N": 8, "share_aha": 0.4, "AME_bucket": -0.2},
        ]
    )
    monkeypatch.setattr(rep, "apply_paper_font_style", lambda **kwargs: None)
    out_pdf = tmp_path / "summary.pdf"
    rep.write_a4_summary_pdf(margins, str(out_pdf), rep.PdfSummaryConfig(dataset="D", model="M"))
    assert out_pdf.exists()
