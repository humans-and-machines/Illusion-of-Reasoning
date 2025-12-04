import argparse
import importlib
import sys
import types
from pathlib import Path

import matplotlib
import pandas as pd


matplotlib.use("Agg")

import src.analysis.figure_2_uncertainty as fig2_uncertainty
import src.analysis.final_plot as final_plot


def test_figure2_uncertainty_invokes_legacy_main(monkeypatch):
    called = {}

    def fake_load(path, import_name):
        called["path"] = Path(path)
        called["import_name"] = import_name

        def _main():
            called["ran"] = True

        return _main

    monkeypatch.setattr(fig2_uncertainty, "load_legacy_main_from_path", fake_load)
    fig2_uncertainty.main()

    assert called["ran"] is True
    assert called["path"].name == "figure_2.py"
    assert called["import_name"] == "analysis_figure_2_legacy"


def test_figure2_uncertainty_entrypoint(monkeypatch):
    import runpy
    import types

    called = {}

    def fake_loader(path, import_name):
        called["loader_called"] = True

        def _main():
            called["legacy_ran"] = True

        return _main

    fake_utils = types.SimpleNamespace(load_legacy_main_from_path=fake_loader)
    monkeypatch.setitem(sys.modules, "src.analysis.utils", fake_utils)
    monkeypatch.setattr(sys, "argv", ["prog", "root"])
    sys.modules.pop("src.analysis.figure_2_uncertainty", None)
    runpy.run_module("src.analysis.figure_2_uncertainty", run_name="__main__")
    assert called["loader_called"] and called["legacy_ran"]


def test_figures_package_exports_dummies(monkeypatch):
    pkg = importlib.reload(importlib.import_module("src.analysis.figures"))
    expected = [
        "figure_1",
        "figure_2_uncertainty",
        "final_plot_uncertainty_gate",
        "math_plots",
        "table_1",
        "graph_1",
        "graph_2",
        "graph_3",
        "graph_3_stacked",
        "graph_4",
        "heatmap_1",
        "temp_graph",
        "flips",
        "forced_aha_effect",
        "h2_temp_aha_eval",
        "h3_uncertainty_buckets",
        "h4_analysis",
    ]
    assert pkg.__all__ == expected
    for name in expected:
        assert hasattr(pkg, name)


def test_pass2_triggered_logic():
    assert (
        final_plot.pass2_triggered(
            {"has_reconsider_cue": True, "reconsider_markers": ["injected_cue"]},
        )
        == 1
    )
    assert (
        final_plot.pass2_triggered(
            {"has_reconsider_cue": False, "reconsider_markers": ["injected_cue"]},
        )
        == 0
    )
    assert final_plot.pass2_triggered({"has_reconsider_cue": True, "reconsider_markers": []}) == 0


def test_build_row_for_record_and_filters():
    cfg = final_plot.CarparkEvalConfig(
        carpark_op="ge",
        carpark_thr=0.0,
        min_step=0,
        max_step=10,
    )
    rec = {
        "example_id": "ex1",
        "step": 5,
        "pass1": {
            "entropy": 0.5,
            "entropy_think": None,
            "is_correct_pred": True,
            "shift_in_reasoning_v1": True,
        },
        "pass2": {
            "has_reconsider_cue": True,
            "reconsider_markers": ["injected_cue"],
            "is_correct_pred": False,
        },
    }
    row = final_plot._build_row_for_record("fake/path.jsonl", rec, cfg)
    assert row is not None
    assert row["group_id"].startswith("example_id:ex1")
    assert row["p1_correct"] == 1
    assert row["p2_triggered"] == 1
    assert row["p2_correct"] == 0
    assert row["p1_shift"] == 1

    cfg_min_filter = final_plot.CarparkEvalConfig("ge", 0.0, min_step=6, max_step=None)
    assert final_plot._build_row_for_record("fake/path.jsonl", rec, cfg_min_filter) is None


def test_load_dataframe_filters_missing(monkeypatch):
    good_rec = {
        "step": 1,
        "example_id": "ex-good",
        "pass1": {"entropy": 0.2, "is_correct_pred": True, "shift_in_reasoning_v1": False},
        "pass2": {"has_reconsider_cue": True, "reconsider_markers": ["injected_cue"], "is_correct_pred": True},
    }
    bad_rec = {
        "step": 2,
        "pass1": {"entropy": None, "is_correct_pred": True},
        "pass2": {"has_reconsider_cue": True, "reconsider_markers": ["injected_cue"], "is_correct_pred": True},
    }

    def fake_iter(path):
        return [good_rec] if "one" in path else [bad_rec]

    monkeypatch.setattr(final_plot, "iter_records_from_file", fake_iter)
    df = final_plot.load_dataframe(
        files=["file_one.jsonl", "file_two.jsonl"],
        carpark_op="ge",
        carpark_thr=0.0,
        min_step=None,
        max_step=None,
    )
    assert len(df) == 1
    assert df.iloc[0]["p2_triggered"] == 1
    assert df.iloc[0]["p2_correct"] == 1


def test_summarize_and_plot(tmp_path):
    data = pd.DataFrame(
        [
            {
                "group_id": "g1",
                "entropy": 0.4,
                "p1_correct": 1,
                "p2_triggered": 1,
                "p2_correct": 1,
                "p1_shift": 1,
            },
            {
                "group_id": "g2",
                "entropy": 0.6,
                "p1_correct": 0,
                "p2_triggered": 1,
                "p2_correct": 0,
                "p1_shift": 0,
            },
        ],
    )
    bins = [0, 1, 2]
    df_top, df_bot = final_plot.summarize_for_figure(data, bins)

    assert set(df_top.columns) == {"_bin", "N", "shift_share"}
    assert df_top["N"].sum() == 2
    assert {"baseline ≥1/8", "baseline 0/8"} == set(df_bot["_stratum"])

    config = final_plot.FigureOutputConfig(
        out_png=str(tmp_path / "plot.png"),
        out_pdf=str(tmp_path / "plot.pdf"),
        title_suffix="Model",
        dpi=120,
    )
    final_plot.plot_figure(df_top, df_bot, bins, config)
    assert Path(config.out_png).exists()
    assert Path(config.out_pdf).exists()


def test_maybe_run_rq3(monkeypatch):
    calls = []
    fake_module = types.SimpleNamespace(main=lambda: calls.append("ran"))
    monkeypatch.setattr(final_plot, "_rq3_analysis_module", fake_module)
    monkeypatch.setattr(final_plot, "_RQ3_IMPORT_ERROR", None)
    args = argparse.Namespace(run_rq3=True, scan_root="root", split=None)

    final_plot._maybe_run_rq3(args)
    assert calls == ["ran"]
