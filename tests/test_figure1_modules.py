import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest


matplotlib.use("Agg")

import src.analysis.core.figure_1_helpers as fig1_helpers
import src.analysis.core.figure_1_plotting as fig1_plotting
import src.analysis.core.figure_1_style as fig1_style


def _basic_problem_frame():
    return pd.DataFrame(
        {
            "domain": ["math"] * 3,
            "problem": ["p1"] * 3,
            "step": [1, 2, 3],
            "freq_correct": [0.1, 0.15, 0.1],
            "aha_rate_gpt": [0.0, 0.05, 0.05],
            "aha_any_gpt": [0, 0, 1],
            "p_correct_given_shift": [0.1, 0.1, 0.4],
        },
    )


def test_mark_formal_pairs_sets_flags():
    df = _basic_problem_frame()
    marked = fig1_helpers.mark_formal_pairs(
        df,
        delta1=0.2,
        delta2=0.2,
        min_prior_steps=2,
    )
    assert "aha_formal" in marked.columns
    assert list(marked["aha_formal"]) == [0, 0, 1]


def test_bootstrap_problem_ratio_counts():
    df = fig1_helpers.mark_formal_pairs(_basic_problem_frame())
    ratios = fig1_helpers.bootstrap_problem_ratio(
        df,
        "aha_formal",
        num_bootstrap_samples=0,
        seed=42,
    )
    assert ratios["step"].tolist() == [1, 2, 3]
    assert ratios.loc[ratios["step"] == 1, "k"].item() == 0
    assert ratios.loc[ratios["step"] == 3, "ratio"].item() == pytest.approx(1.0)
    assert np.isnan(ratios.loc[ratios["step"] == 1, "lo"]).all()


def test_export_formal_aha_json_with_text(monkeypatch, tmp_path):
    problem_step_df = pd.DataFrame(
        [
            {
                "domain": "math",
                "problem": "p1",
                "step": 1,
                "freq_correct": 0.2,
                "aha_rate_gpt": 0.3,
                "aha_any_gpt": 1,
                "p_correct_given_shift": 0.6,
                "aha_formal": 1,
                "n_samples": 5,
            },
            {
                "domain": "crossword",
                "problem": "p2",
                "step": 2,
                "freq_correct": 0.1,
                "aha_rate_gpt": 0.2,
                "aha_any_gpt": 1,
                "p_correct_given_shift": 0.4,
                "aha_formal": 1,
                "n_samples": 3,
            },
        ],
    )
    config = fig1_helpers.FormalAhaExportConfig(
        meta=fig1_helpers.FormalExportMeta(dataset="ds", model="model"),
        thresholds=fig1_helpers.make_formal_thresholds(0.2, 0.2, 1, None),
        gpt_keys=["k1"],
        gpt_subset_native=False,
        out_dir=str(tmp_path),
        slug="unit",
        max_chars=12,
    )
    records = {
        "math_file": [
            {
                "problem": "p1",
                "step": 1,
                "pass1": {"answer": "A" * 50, "is_correct_pred": True},
            },
        ],
        "xword_file": [],
    }
    monkeypatch.setattr(
        fig1_helpers,
        "iter_records_from_file",
        lambda path: records.get(path, []),
    )
    monkeypatch.setattr(
        fig1_helpers,
        "aha_gpt_for_rec",
        lambda *args, **kwargs: 1,
    )

    json_path, jsonl_path, count = fig1_helpers.export_formal_aha_json_with_text(
        problem_step_df,
        {"math": ["math_file"], "crossword": ["xword_file"]},
        config,
    )
    assert count == 2
    data = json.loads(Path(json_path).read_text())
    assert Path(json_path).exists()
    assert Path(jsonl_path).exists()
    assert {event["problem"] for event in data} == {"p1", "p2"}
    p1_event = next(event for event in data if event["problem"] == "p1")
    assert p1_event["answer"].endswith("…[truncated]")
    p2_event = next(event for event in data if event["problem"] == "p2")
    assert p2_event["answer"] is None


def test_build_positive_delta_flags_handles_domains():
    df_with_domain = pd.DataFrame(
        [
            {
                "domain": "math",
                "step": 1,
                "aha_any_gpt": 1,
                "p_correct_given_shift": 0.5,
                "freq_correct": 0.2,
            },
            {
                "domain": "math",
                "step": 1,
                "aha_any_gpt": 1,
                "p_correct_given_shift": 0.4,
                "freq_correct": 0.1,
            },
        ],
    )
    df_no_domain = pd.DataFrame(
        [
            {
                "step": 2,
                "problem": "p3",
                "aha_any_gpt": 0,
                "p_correct_given_shift": 0.0,
                "freq_correct": 0.0,
            },
        ],
    )

    flags_with_domain = fig1_helpers.build_positive_delta_flags(df_with_domain)
    flags_no_domain = fig1_helpers.build_positive_delta_flags(df_no_domain)

    assert flags_with_domain["math"][1] is True
    assert flags_no_domain["All"][2] is False


def test_style_helpers(monkeypatch):
    captured = {}

    def fake_apply_paper_font_style(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(fig1_style, "apply_paper_font_style", fake_apply_paper_font_style)
    monkeypatch.setattr(fig1_style, "base_a4_size_inches", lambda orientation="landscape": (1.0, 2.0))

    fig1_style.set_global_fonts(font_family="Serif", font_size=11)
    assert captured["font_family"] == "Serif"
    assert captured["font_size"] == 11
    assert fig1_style.a4_size_inches("portrait") == (1.0, 2.0)
    assert fig1_style.lighten_hex("#000000", 0.5) == "#7f7f7f"


def test_plot_three_ratios_shared_axes_multi(tmp_path):
    base_df = pd.DataFrame(
        {
            "step": [1, 2],
            "ratio": [0.2, 0.4],
            "lo": [0.1, 0.3],
            "hi": [0.3, 0.5],
            "n": [10, 12],
        },
    )
    native_by_dom = {"math": base_df}
    gpt_by_dom = {"math": base_df.assign(ratio=[0.1, 0.3])}
    formal_by_dom = {"math": base_df.assign(ratio=[0.05, 0.2])}
    config = {
        "domain_colors": {"math": "#1f77b4"},
        "dataset": "DS",
        "model": "Model",
        "alpha_ci": 0.2,
        "out_png": str(tmp_path / "ratios.png"),
        "out_pdf": str(tmp_path / "ratios.pdf"),
        "a4_pdf": False,
        "a4_orientation": "landscape",
        "highlight_formal_by_dom": {"math": {2: True}},
        "highlight_color": "#2ca02c",
        "panel_box_aspect": 0.9,
    }

    trends = fig1_plotting.plot_three_ratios_shared_axes_multi(
        native_by_dom,
        gpt_by_dom,
        formal_by_dom,
        config,
    )

    assert Path(config["out_png"]).exists()
    assert Path(config["out_pdf"]).exists()
    assert {row["series"] for row in trends} == {
        "Words/Cue Phrases",
        "LLM-Detected Shifts",
        "Formal Shifts",
    }
    assert all(np.isfinite(row["slope_per_1k"]) for row in trends)


def test_plot_formal_sweep_grid(tmp_path):
    pair_stats = pd.DataFrame(
        {
            "problem": ["p1", "p1"],
            "step": [1, 2],
            "freq_correct": [0.0, 0.0],
            "aha_rate_gpt": [0.0, 0.0],
            "aha_any_gpt": [0, 1],
            "p_correct_given_shift": [0.0, 0.5],
        },
    )
    config = fig1_plotting.FormalSweepPlotConfig(
        min_prior_steps=1,
        n_bootstrap=0,
        seed=0,
        out_png=str(tmp_path / "sweep.png"),
        out_pdf=str(tmp_path / "sweep.pdf"),
        dataset="DS",
        model="Model",
        primary_color="#000000",
        ci_color="#cccccc",
        ymax=1.0,
        alpha_ci=0.2,
        a4_pdf=False,
        orientation="landscape",
    )

    fig1_plotting.plot_formal_sweep_grid(
        pair_stats,
        delta1_list=[0.1],
        delta2_list=[0.1],
        config=config,
    )

    assert Path(config.out_png).exists()
    assert Path(config.out_pdf).exists()
