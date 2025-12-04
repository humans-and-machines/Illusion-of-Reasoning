import argparse
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import src.analysis.core.figure_1_components as fig1


def test_build_arg_parser_defaults():
    parser = fig1.build_arg_parser()
    args = parser.parse_args([])
    assert args.dataset_name == "MIXED"
    assert args.model_name == "Qwen2.5-1.5B"
    assert args.delta1 == pytest.approx(0.20)
    assert args.gpt_mode == "canonical"
    assert args.no_gpt_subset_native is False


def test_collect_files_with_roots(monkeypatch):
    monkeypatch.setattr(fig1, "scan_files", lambda root, split: [f"{root}/a"])
    args = argparse.Namespace(root_crossword="cw_root", root_math="m_root", results_root=None, split=None)
    files_by_domain, first_root = fig1._collect_files(args)
    assert files_by_domain == {
        "Crossword": ["cw_root/a"],
        "Math": ["m_root/a"],
    }
    assert first_root == "cw_root"


def test_collect_files_with_results_root(monkeypatch):
    monkeypatch.setattr(fig1, "scan_files", lambda root, split: [f"{root}/only"])
    args = argparse.Namespace(root_crossword=None, root_math=None, results_root="base", split=None)
    files_by_domain, first_root = fig1._collect_files(args)
    assert files_by_domain == {"All": ["base/only"]}
    assert first_root == "base"


def test_collect_files_raises_when_empty(monkeypatch):
    monkeypatch.setattr(fig1, "scan_files", lambda root, split: [])
    args = argparse.Namespace(root_crossword="cw_root", root_math=None, results_root=None, split=None)
    with pytest.raises(SystemExit):
        fig1._collect_files(args)


def test_configure_domain_colors_respects_overrides(monkeypatch):
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=["#123456", "#abcdef"])
    args = argparse.Namespace(color_crossword="#111111", color_math="#222222")
    files_by_domain = {"Crossword": ["a"], "Math": ["b"], "Extra": ["c"]}
    colors = fig1._configure_domain_colors(files_by_domain, args)
    assert colors["Crossword"] == "#111111"
    assert colors["Math"] == "#222222"
    assert colors["Extra"] == "#123456"


def test_select_gpt_config_canonical_and_broad():
    canonical_keys, subset_native = fig1._select_gpt_config(
        argparse.Namespace(gpt_mode="canonical", no_gpt_subset_native=False)
    )
    assert canonical_keys == ["change_way_of_thinking", "shift_in_reasoning_v1"]
    assert subset_native is True

    broad_keys, subset_native = fig1._select_gpt_config(
        argparse.Namespace(gpt_mode="broad", no_gpt_subset_native=True)
    )
    assert "shift_llm" in broad_keys and "rechecked" in broad_keys
    assert subset_native is False


def test_parse_delta_grid():
    args = argparse.Namespace(delta1_list="0.1,0.2", delta2_list="0.3")
    grid = fig1._parse_delta_grid(args)
    assert grid.delta1 == [0.1, 0.2]
    assert grid.delta2 == [0.3]


def test_prepare_formal_sweep_config_uses_paths_and_colors():
    args = argparse.Namespace(
        min_prior_steps=2,
        B=10,
        seed=1,
        ymax=0.9,
        ci_alpha=0.1,
        a4_pdf=False,
        a4_orientation="portrait",
        dataset_name="Data",
        model_name="Model",
        delta3=None,
    )
    config = fig1._prepare_formal_sweep_config(args, "/tmp/out", "slug")
    assert config.out_png.endswith("aha_formal_ratio_sweep__slug.png")
    assert config.out_pdf.endswith("aha_formal_ratio_sweep__slug.pdf")
    assert config.dataset == "Data"
    assert config.model == "Model"
    assert config.ci_color != config.primary_color


def test_write_domain_ratios_csv(tmp_path):
    native_by_dom = {
        "Crossword": pd.DataFrame({"step": [1], "k": [1], "n": [2], "ratio": [0.5], "lo": [0.1], "hi": [0.9]})
    }
    gpt_by_dom = {
        "Crossword": pd.DataFrame({"step": [1], "k": [2], "n": [2], "ratio": [1.0], "lo": [0.8], "hi": [1.0]})
    }
    formal_by_dom = {
        "Crossword": pd.DataFrame({"step": [1], "k": [1], "n": [2], "ratio": [0.5], "lo": [0.2], "hi": [0.7]})
    }
    csv_path = fig1._write_domain_ratios_csv(str(tmp_path), "slug", native_by_dom, gpt_by_dom, formal_by_dom)
    df = pd.read_csv(csv_path)
    assert set(df["series"]) == {"Words/Cue Phrases", "LLM-Detected Shifts", "Formal Shifts"}
    assert set(df["domain"]) == {"Crossword"}


def test_write_trend_csv(tmp_path):
    path = fig1._write_trend_csv(str(tmp_path), "slug", [{"series": "s", "domain": "d"}])
    df = pd.read_csv(path)
    assert df.iloc[0]["series"] == "s"


def test_build_domain_bootstraps(monkeypatch):
    bootstrap_df = pd.DataFrame({"step": [1], "k": [1], "n": [2], "ratio": [0.5], "lo": [0.1], "hi": [0.9]})
    monkeypatch.setattr(fig1, "bootstrap_problem_ratio", lambda df, col, num_bootstrap, seed: bootstrap_df)
    monkeypatch.setattr(fig1, "build_positive_delta_flags", lambda df: {"Math": {1: True}})

    ps_main = pd.DataFrame(
        {
            "domain": ["Math", "Math"],
            "step": [1, 1],
            "aha_any_native": [1, 0],
            "aha_any_gpt": [1, 0],
            "aha_formal": [1, 0],
        }
    )
    result = fig1._build_domain_bootstraps(ps_main, SimpleNamespace(B=5, seed=0))
    assert isinstance(result.native["Math"], pd.DataFrame)
    assert result.highlights == {"Math": {1: True}}


def test_write_sweep_csv(monkeypatch, tmp_path):
    monkeypatch.setattr(
        fig1,
        "mark_formal_pairs",
        lambda df, delta1, delta2, min_prior_steps, delta3: df.assign(aha_formal=1),
    )
    monkeypatch.setattr(
        fig1,
        "bootstrap_problem_ratio",
        lambda df, col, num_bootstrap, seed: pd.DataFrame(
            {"step": [1], "k": [1], "n": [1], "ratio": [1.0], "lo": [1.0], "hi": [1.0]}
        ),
    )
    ps_base = pd.DataFrame({"step": [1], "problem": ["p1"]})
    args = SimpleNamespace(min_prior_steps=1, delta3=None, B=1, seed=0)
    sweep_path = fig1._write_sweep_csv(str(tmp_path), "slug", ps_base, ((0.1, 0.2), (0.3, 0.4)), args)
    df = pd.read_csv(sweep_path)
    assert set(df["delta1"]) == {0.1, 0.3}
    assert set(df["delta2"]) == {0.2, 0.4}


def test_load_samples_raises_on_filters(monkeypatch):
    sample_df = pd.DataFrame({"domain": ["Math"], "step": [1], "problem": ["p1"]})
    monkeypatch.setattr(fig1, "load_pass1_samples_multi", lambda *_args, **_kwargs: sample_df)
    args = argparse.Namespace(min_step=None, max_step=0, balanced_panel=False)
    with pytest.raises(SystemExit):
        fig1._load_samples({"Math": ["path"]}, [], False, args)

    # Balanced panel removing all rows triggers a separate SystemExit
    sample_df2 = pd.DataFrame({"domain": ["Math", "Math"], "step": [1, 2], "problem": ["p1", "p2"]})
    monkeypatch.setattr(fig1, "load_pass1_samples_multi", lambda *_args, **_kwargs: sample_df2)
    args_balanced = argparse.Namespace(min_step=None, max_step=None, balanced_panel=True)
    with pytest.raises(SystemExit):
        fig1._load_samples({"Math": ["path"]}, [], False, args_balanced)


def test_export_formal_events_builds_config_and_prints(monkeypatch, capsys, tmp_path):
    captured = {}

    def fake_export(ps_main, files_by_domain, config):
        captured["config"] = config
        return (str(tmp_path / "out.json"), str(tmp_path / "out.jsonl"), 3)

    monkeypatch.setattr(fig1, "export_formal_aha_json_with_text", fake_export)
    ctx = fig1.Figure1Context(
        args=SimpleNamespace(
            dataset_name="DS",
            model_name="Model",
            delta1=0.1,
            delta2=0.2,
            delta3=None,
            min_prior_steps=2,
            gpt_keys=["k"],
            gpt_subset_native=True,
        ),
        files_by_domain={"Math": ["p"]},
        out_dir=str(tmp_path),
        slug="slug",
        domain_colors={},
        gpt_keys=["k"],
        gpt_subset_native=True,
    )
    fig1._export_formal_events(pd.DataFrame(), {"Math": ["p"]}, ctx)
    out = capsys.readouterr().out
    assert "Formal Aha events: 3" in out
    config = captured["config"]
    assert config.thresholds.delta1 == 0.1
    assert config.destinations.slug == "slug"


def test_print_summary_includes_trend_rows(capsys):
    summary = fig1.SummaryArtifacts(
        main_outputs=("main.png", "main.pdf"),
        ratios_csv="ratios.csv",
        trend_csv="trend.csv",
        sweep_outputs=("sweep.png", "sweep.pdf"),
        sweep_csv="sweep.csv",
        denom_note="note",
    )
    trend_rows = [
        {"series": "All", "domain": "Math", "slope_per_1k": 0.1, "delta_over_range": 0.2, "weighted_R2": 0.3}
    ]
    fig1._print_summary(summary, trend_rows)
    out = capsys.readouterr().out
    assert "main.png" in out and "ratios.csv" in out
    assert "[Trend] All [Math]" in out


def test_script_entry_help_exits(monkeypatch):
    import runpy
    import sys

    monkeypatch.setattr(sys, "argv", ["prog", "--help"])
    sys.modules.pop("src.analysis.core.figure_1_components", None)
    with pytest.raises(SystemExit):
        runpy.run_module("src.analysis.core.figure_1_components", run_name="__main__")


def test_main_runs_with_stubbed_pipeline(monkeypatch, tmp_path):
    import sys

    # Build a lightweight context
    ctx = fig1.Figure1Context(
        args=SimpleNamespace(
            dataset_name="DS",
            model_name="Model",
            delta1=0.1,
            delta2=0.2,
            delta3=None,
            min_prior_steps=1,
            out_basename=None,
            font_family="Times",
            font_size=12,
            ci_alpha=0.1,
            a4_pdf=False,
            a4_orientation="landscape",
            panel_box_aspect=1.0,
            ms=4.0,
            balanced_panel=False,
            split=None,
        ),
        files_by_domain={"Math": ["path"]},
        out_dir=str(tmp_path),
        slug="slug",
        domain_colors={"Math": "#000"},
        gpt_keys=["k"],
        gpt_subset_native=True,
    )
    monkeypatch.setattr(fig1, "_build_context", lambda *_args, **_kwargs: ctx)
    monkeypatch.setattr(sys, "argv", ["prog", "root"])
    monkeypatch.setattr(fig1, "set_global_fonts", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        fig1,
        "_load_samples",
        lambda *_args, **_kwargs: pd.DataFrame(
            {"domain": ["Math"], "step": [1], "problem": ["p1"], "aha_native": [1], "aha_gpt": [1], "correct": [1]}
        ),
    )
    monkeypatch.setattr(
        fig1,
        "build_problem_step",
        lambda df: pd.DataFrame(
            {
                "domain": ["Math"],
                "step": [1],
                "problem": ["p1"],
                "freq_correct": [1.0],
                "aha_rate_gpt": [1.0],
                "aha_any_gpt": [1],
                "p_correct_given_shift": [1.0],
                "aha_any_native": [1],
                "aha_rate_native": [1.0],
                "aha_formal": [1],
            }
        ),
    )
    monkeypatch.setattr(fig1, "mark_formal_pairs", lambda df, **_kwargs: df)
    dummy_bootstrap = fig1.DomainBootstrapResult(
        native={"Math": pd.DataFrame({"step": [1], "k": [1], "n": [1], "ratio": [1.0], "lo": [1.0], "hi": [1.0]})},
        gpt={"Math": pd.DataFrame({"step": [1], "k": [1], "n": [1], "ratio": [1.0], "lo": [1.0], "hi": [1.0]})},
        formal={"Math": pd.DataFrame({"step": [1], "k": [1], "n": [1], "ratio": [1.0], "lo": [1.0], "hi": [1.0]})},
        highlights={"Math": {1: True}},
    )
    monkeypatch.setattr(fig1, "_build_domain_bootstraps", lambda *_args, **_kwargs: dummy_bootstrap)
    monkeypatch.setattr(fig1, "_export_formal_events", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        fig1,
        "plot_three_ratios_shared_axes_multi",
        lambda *_args, **_kwargs: [
            {"series": "s", "domain": "Math", "slope_per_1k": 0.0, "delta_over_range": 0.0, "weighted_R2": 0.0}
        ],
    )
    monkeypatch.setattr(fig1, "_parse_delta_grid", lambda *_args, **_kwargs: fig1.DeltaGrid([0.1], [0.2]))
    monkeypatch.setattr(
        fig1,
        "_prepare_formal_sweep_config",
        lambda *_args, **_kwargs: SimpleNamespace(out_png="sweep.png", out_pdf="sweep.pdf"),
    )
    monkeypatch.setattr(fig1, "plot_formal_sweep_grid", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(fig1, "_write_domain_ratios_csv", lambda *_args, **_kwargs: "ratios.csv")
    monkeypatch.setattr(fig1, "_write_trend_csv", lambda *_args, **_kwargs: "trend.csv")
    monkeypatch.setattr(fig1, "_write_sweep_csv", lambda *_args, **_kwargs: "sweep.csv")
    captured = {}
    monkeypatch.setattr(
        fig1, "_print_summary", lambda summary, trend: captured.update({"summary": summary, "trend": trend})
    )

    fig1.main()
    assert captured["summary"].ratios_csv == "ratios.csv"
    assert captured["trend"][0]["series"] == "s"
