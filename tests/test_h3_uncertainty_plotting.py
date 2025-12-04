import matplotlib
import pandas as pd

from src.analysis.h3_uncertainty import plotting


matplotlib.use("Agg", force=True)


def test_plot_question_by_step_ci_noop_on_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(plotting, "apply_paper_font_style", lambda: None)
    out_png = tmp_path / "out.png"
    plotting.plot_question_by_step_ci(pd.DataFrame(), str(out_png), also_pdf=False)
    assert not out_png.exists()


def test_plot_question_by_bucket_ci_creates_png(tmp_path, monkeypatch):
    monkeypatch.setattr(plotting, "apply_paper_font_style", lambda: None)
    df = pd.DataFrame(
        {
            "group": ["g1", "g1"],
            "perplexity_bucket": [0, 1],
            "any_pass1": [0.2, 0.4],
            "any_pass1_lo": [0.1, 0.3],
            "any_pass1_hi": [0.3, 0.5],
            "any_pass2": [0.5, 0.6],
            "any_pass2_lo": [0.4, 0.5],
            "any_pass2_hi": [0.6, 0.7],
        }
    )
    out_png = tmp_path / "bucket.png"
    plotting.plot_question_by_bucket_ci(df, str(out_png), also_pdf=False)
    assert out_png.exists()


def test_plot_question_by_bucket_ci_noop_on_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(plotting, "apply_paper_font_style", lambda: None)
    out_png = tmp_path / "bucket.png"
    plotting.plot_question_by_bucket_ci(pd.DataFrame(), str(out_png), also_pdf=True)
    assert not out_png.exists()


def test_plot_prompt_overall_ci_writes_pdf(tmp_path, monkeypatch):
    monkeypatch.setattr(plotting, "apply_paper_font_style", lambda: None)
    df = pd.DataFrame(
        {
            "group": ["g1", "g2"],
            "acc_pass1": [0.3, 0.4],
            "acc_pass1_lo": [0.2, 0.3],
            "acc_pass1_hi": [0.4, 0.5],
            "acc_pass2": [0.5, 0.6],
            "acc_pass2_lo": [0.4, 0.5],
            "acc_pass2_hi": [0.6, 0.7],
        }
    )
    out_png = tmp_path / "prompt.png"
    plotting.plot_prompt_overall_ci(df, str(out_png), also_pdf=True)
    assert out_png.exists()
    assert out_png.with_suffix(".pdf").exists()
