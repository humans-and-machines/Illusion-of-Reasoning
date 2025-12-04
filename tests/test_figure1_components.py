import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib import cycler


matplotlib.use("Agg")

import src.analysis.core.figure_1_components as comp


def _args(**kwargs):
    defaults = dict(
        root_crossword=None,
        root_math=None,
        results_root=None,
        split=None,
        out_dir=None,
        dataset_name="DS Name",
        model_name="Model Name",
        delta1=0.2,
        delta2=0.2,
        delta3=None,
        min_prior_steps=2,
        delta1_list="0.1,0.2",
        delta2_list="0.3,0.4",
        min_step=None,
        max_step=None,
        balanced_panel=False,
        color_crossword="#111111",
        color_math="#222222",
        B=5,
        seed=0,
        ymax=1.0,
        ci_alpha=0.2,
        ms=4.0,
        font_family="Serif",
        font_size=10,
        gpt_mode="canonical",
        no_gpt_subset_native=False,
        out_basename=None,
        a4_pdf=False,
        a4_orientation="landscape",
        panel_box_aspect=0.8,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_summary_artifacts_properties():
    art = comp.SummaryArtifacts(
        main_outputs=("m.png", "m.pdf"),
        ratios_csv="ratios.csv",
        trend_csv="trend.csv",
        sweep_outputs=("s.png", "s.pdf"),
        sweep_csv="sweep.csv",
        denom_note="note",
    )
    assert art.main_png == "m.png"
    assert art.main_pdf == "m.pdf"
    assert art.sweep_png == "s.png"
    assert art.sweep_pdf == "s.pdf"


def test_collect_files_with_roots(monkeypatch):
    monkeypatch.setattr(comp, "scan_files", lambda root, split=None: [f"{root}/a.jsonl"])
    args = _args(root_crossword="cross", root_math=None, results_root=None)
    files, first = comp._collect_files(args)
    assert files == {"Crossword": ["cross/a.jsonl"]}
    assert first == "cross"


def test_collect_files_requires_root_or_results(monkeypatch):
    args = _args()
    with pytest.raises(SystemExit):
        comp._collect_files(args)

    monkeypatch.setattr(comp, "scan_files", lambda *args, **kwargs: [])
    args_res = _args(results_root="root")
    with pytest.raises(SystemExit):
        comp._collect_files(args_res)


def test_configure_domain_colors(monkeypatch):
    matplotlib.rcParams["axes.prop_cycle"] = cycler(color=["#aaa", "#bbb"])
    args = _args()
    files = {"Crossword": [], "Math": [], "Other": []}
    colors = comp._configure_domain_colors(files, args)
    assert colors["Crossword"] == "#111111"
    assert colors["Math"] == "#222222"
    assert colors["Other"] == "#aaa"


def test_select_gpt_config_modes():
    keys, subset = comp._select_gpt_config(_args(gpt_mode="canonical"))
    assert subset is True
    assert keys == ["change_way_of_thinking", "shift_in_reasoning_v1"]

    keys_broad, subset_off = comp._select_gpt_config(_args(gpt_mode="broad", no_gpt_subset_native=True))
    assert subset_off is False
    assert "shift_llm" in keys_broad


def test_build_context_uses_slug_and_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(comp, "_collect_files", lambda args: ({"Math": ["f1"]}, str(tmp_path)))
    monkeypatch.setattr(comp, "_configure_domain_colors", lambda fbd, a: {"Math": "#abc"})
    monkeypatch.setattr(comp, "_select_gpt_config", lambda a: (["k1"], True))
    ctx = comp._build_context(_args(out_dir=None, dataset_name="My Data", model_name="Model One"))
    assert ctx.out_dir == str(tmp_path / "aha_ratios_bootstrap")
    assert ctx.slug.lower() == "my_data__model_one"
    assert ctx.domain_colors["Math"] == "#abc"
    assert ctx.gpt_keys == ["k1"]


def test_load_samples_filters_and_balances(monkeypatch):
    df = pd.DataFrame(
        {
            "domain": ["Math", "Math", "Math"],
            "problem": ["p1", "p1", "p2"],
            "step": [1, 2, 1],
            "aha_any_native": [0, 1, 0],
            "aha_any_gpt": [0, 1, 0],
        },
    )
    monkeypatch.setattr(comp, "load_pass1_samples_multi", lambda *args, **kwargs: df)
    # Filter to step>=2 drops two rows
    filtered = comp._load_samples({"Math": []}, ["k"], True, _args(min_step=2))
    assert len(filtered) == 1

    balanced = comp._load_samples({"Math": []}, ["k"], True, _args(balanced_panel=True))
    assert set(balanced["problem"]) == {"p1"}


def test_prepare_formal_sweep_config_uses_lighten_hex(tmp_path):
    args = _args(
        min_prior_steps=3, B=10, seed=4, ymax=0.9, ci_alpha=0.3, delta3=0.05, a4_pdf=True, a4_orientation="portrait"
    )
    cfg = comp._prepare_formal_sweep_config(args, str(tmp_path), "slug")
    assert cfg.min_prior_steps == 3
    assert cfg.delta3 == 0.05
    assert cfg.out_png.endswith("slug.png")
    assert cfg.ci_color != "#2F5597"  # lightened


def test_write_domain_and_trend_csv(tmp_path):
    df = pd.DataFrame({"step": [1], "k": [1], "n": [2], "ratio": [0.5], "lo": [0.2], "hi": [0.8]})
    native = {"Math": df}
    gpt = {"Math": df}
    formal = {"Math": df}
    ratios_path = comp._write_domain_ratios_csv(str(tmp_path), "slug", native, gpt, formal)
    assert Path(ratios_path).exists()
    content = Path(ratios_path).read_text()
    assert "Words/Cue Phrases" in content
    trend_path = comp._write_trend_csv(str(tmp_path), "slug", [{"domain": "Math", "slope_per_1k": 0.0}])
    assert Path(trend_path).exists()


def test_build_domain_bootstraps(monkeypatch):
    calls = []

    def fake_bootstrap(sub, col, num_bootstrap, seed):
        calls.append((col, num_bootstrap, seed, sub["domain"].iloc[0]))
        return pd.DataFrame({"step": [1], "k": [1], "n": [1], "ratio": [1.0], "lo": [np.nan], "hi": [np.nan]})

    monkeypatch.setattr(comp, "bootstrap_problem_ratio", fake_bootstrap)
    monkeypatch.setattr(comp, "build_positive_delta_flags", lambda df: {"Math": {1: True}})
    ps_main = pd.DataFrame(
        {
            "domain": ["Math"],
            "step": [1],
            "aha_any_native": [1],
            "aha_any_gpt": [1],
            "aha_formal": [1],
            "freq_correct": [0.5],
            "p_correct_given_shift": [0.6],
        },
    )
    res = comp._build_domain_bootstraps(ps_main, _args(B=3, seed=9))
    assert res.highlights["Math"][1] is True
    assert len(calls) == 3


def test_write_sweep_csv(monkeypatch, tmp_path):
    monkeypatch.setattr(comp, "mark_formal_pairs", lambda df, **kwargs: df.assign(aha_formal=1))
    monkeypatch.setattr(
        comp,
        "bootstrap_problem_ratio",
        lambda df, col, num_bootstrap, seed: df.assign(k=1, n=1, ratio=1.0, lo=np.nan, hi=np.nan)[
            ["step", "ratio", "lo", "hi"]
        ],
    )
    ps_base = pd.DataFrame(
        {
            "step": [1, 2],
            "problem": ["p1", "p2"],
            "freq_correct": [0.1, 0.2],
            "aha_rate_gpt": [0.0, 0.0],
            "aha_any_gpt": [1, 1],
            "p_correct_given_shift": [0.2, 0.3],
        },
    )
    path = comp._write_sweep_csv(
        str(tmp_path), "slug", ps_base, ((0.1, 0.2), (0.3, 0.4)), _args(B=0, seed=1, delta3=0.05)
    )
    content = Path(path).read_text()
    assert "delta1" in content and "delta2" in content


def test_export_formal_events(monkeypatch, capsys):
    captured = {}

    def fake_export(ps_main, files_by_domain, cfg):
        captured["dataset"] = cfg.dataset
        return ("a.json", "b.jsonl", 2)

    monkeypatch.setattr(comp, "export_formal_aha_json_with_text", fake_export)
    ctx = comp.Figure1Context(
        args=_args(dataset_name="DS", model_name="Model", delta1=0.2, delta2=0.2, delta3=None, min_prior_steps=2),
        files_by_domain={"Math": ["f.jsonl"]},
        out_dir="out",
        slug="slug",
        domain_colors={},
        gpt_keys=["k"],
        gpt_subset_native=True,
    )
    comp._export_formal_events(pd.DataFrame(), {"Math": ["f.jsonl"]}, ctx)
    out = capsys.readouterr().out
    assert "Formal Aha events" in out
    assert captured["dataset"] == "DS"


def test_parse_delta_grid(monkeypatch):
    monkeypatch.setattr(comp, "parse_float_list", lambda s: [9.0])
    grid = comp._parse_delta_grid(_args(delta1_list="x", delta2_list="y"))
    assert grid.delta1 == [9.0] and grid.delta2 == [9.0]


def test_print_summary(capsys):
    artifacts = comp.SummaryArtifacts(
        main_outputs=("a.png", "a.pdf"),
        ratios_csv="rat.csv",
        trend_csv="tr.csv",
        sweep_outputs=("s.png", "s.pdf"),
        sweep_csv="sw.csv",
        denom_note="note",
    )
    trend_rows = [
        {"series": "Words", "domain": "Math", "slope_per_1k": 0.1, "delta_over_range": 0.2, "weighted_R2": 0.3},
    ]
    comp._print_summary(artifacts, trend_rows)
    out = capsys.readouterr().out
    assert "a.png" in out and "Trend" in out
