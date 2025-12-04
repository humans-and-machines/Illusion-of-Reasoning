import argparse
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.analysis import graph_1 as g1


def test_iter_rows_and_load_rows(monkeypatch):
    samples = [
        ("Math", 1, {"pid": "p1"}, True, False),
        ("Math", 2, {"pid": "p2"}, False, True),
    ]
    monkeypatch.setattr(g1, "iter_correct_and_shift_samples_for_config", lambda files, cfg: samples)
    monkeypatch.setattr(g1, "get_problem_id", lambda rec: rec.get("pid"))

    cfg = g1.LoadRowsConfig(
        gpt_keys=[],
        gpt_subset_native=False,
        min_step=None,
        max_step=None,
        carpark_success_fn=lambda rec, p1=None: None,
    )
    rows = list(g1._iter_rows({"Math": ["f"]}, cfg))
    assert rows[0]["problem_id"] == "p1" and rows[0]["shift"] == 0

    df = g1.load_rows({"Math": ["f"]}, cfg)
    assert set(df.columns) == {"domain", "problem_id", "step", "correct", "shift"}
    assert len(df) == 2

    # pid missing should be skipped
    monkeypatch.setattr(
        g1, "iter_correct_and_shift_samples_for_config", lambda files, cfg: [("Math", 1, {}, True, True)]
    )
    monkeypatch.setattr(g1, "get_problem_id", lambda rec: None)
    assert list(g1._iter_rows({"Math": ["f"]}, cfg)) == []


def test_per_step_raw_effect_filters_and_computes():
    df_empty = pd.DataFrame(columns=["domain", "step", "correct", "shift"])
    assert g1.per_step_raw_effect(df_empty, "Math").empty

    df = pd.DataFrame(
        {
            "domain": ["Math"] * 4,
            "step": [1, 1, 1, 1],
            "correct": [1, 0, 1, 0],
            "shift": [1, 1, 0, 0],
        }
    )
    result = g1.per_step_raw_effect(df, "Math", min_per_group=1)
    assert result.iloc[0]["raw_effect"] == pytest.approx(0.0)
    assert result.iloc[0]["n_shift"] == 2 and result.iloc[0]["n_noshift"] == 2

    # insufficient counts should skip
    skip_df = pd.DataFrame({"domain": ["Math"], "step": [1], "correct": [1], "shift": [1]})
    with pytest.raises(KeyError):
        g1.per_step_raw_effect(skip_df, "Math", min_per_group=2)


def test_compute_and_label_maps_and_parse_float_list():
    args = SimpleNamespace(label_math="  LM  ", label_math2="", model_name="m")
    labels = g1._build_label_map(args)
    assert labels["Math"] == "LM" and labels["Math2"] == "Qwen7B-Math"

    assert g1._parse_float_list(None) is None
    assert g1._parse_float_list("  ") is None
    assert g1._parse_float_list("1, 2 3") == [1.0, 2.0, 3.0]
    # Extra separators should be skipped gracefully
    assert g1._parse_float_list("1,,2") == [1.0, 2.0]


def test_compute_per_step_collects_domains():
    df = pd.DataFrame(
        {
            "domain": ["Math", "Math", "Crossword", "Crossword"],
            "step": [1, 1, 2, 2],
            "correct": [1, 0, 1, 0],
            "shift": [1, 0, 1, 0],
        }
    )
    per_step, rows_all = g1._compute_per_step(df, min_per_group=1)
    assert set(per_step.keys()) == {"Math", "Crossword"}
    assert len(rows_all) == 2


def test_determine_plot_config_and_wrap_title():
    args = SimpleNamespace(
        plot_units="pp",
        ymin_pp=-10.0,
        ymax_pp=10.0,
        ylim_pad_pp=1.0,
        yticks_pp="",
        ymin_prob=-0.1,
        ymax_prob=0.1,
        ylim_pad_prob=0.01,
        yticks_prob="",
    )
    cfg_pp = g1._determine_plot_config(args)
    assert cfg_pp.y_scale == 100.0 and cfg_pp.ylim == (-11.0, 11.0)

    args.plot_units = "prob"
    cfg_prob = g1._determine_plot_config(args)
    assert cfg_prob.y_scale == 1.0 and cfg_prob.ylim == (-0.11, 0.11)

    long_title = "This is a very long title that should wrap into multiple lines automatically for readability"
    wrapped = g1._auto_wrap_title_to_two_lines(long_title, width=20)
    assert wrapped.count("\n") <= 1


def test_plotters_produce_files(tmp_path):
    per_step = {
        "Crossword": pd.DataFrame({"step": [1], "raw_effect": [0.1]}),
        "Math": pd.DataFrame({"step": [1], "raw_effect": [0.2]}),
        "Math2": pd.DataFrame({"step": [1], "raw_effect": [0.3]}),
        "Carpark": pd.DataFrame({"step": [1], "raw_effect": [0.0]}),
    }
    label_map = {"Crossword": "X", "Math": "M", "Math2": "M2", "Carpark": "C"}
    units = g1.PlotUnitsConfig(y_scale=100.0, ylim=(-50, 50), yticks=[-50, 0, 50], ylabel="y")

    panels_png = tmp_path / "panels.png"
    g1.plot_panels(
        per_step,
        label_map,
        str(panels_png),
        g1.PanelFigureConfig(dpi=50, width_in=4.0, height_scale=0.5, marker_size=10),
        units,
    )
    assert panels_png.exists() and panels_png.with_suffix(".pdf").exists()

    overlay_png = tmp_path / "overlay.png"
    g1.plot_overlay_all(
        per_step,
        label_map,
        str(overlay_png),
        g1.OverlayFigureConfig(
            dpi=50,
            width_in=4.0,
            height_scale=0.5,
            marker_size=8,
            title="Overlay Title",
        ),
        units,
    )
    assert overlay_png.exists() and overlay_png.with_suffix(".pdf").exists()

    # overlay skips missing domains without error
    per_step.pop("Math2")
    overlay_png2 = tmp_path / "overlay2.png"
    g1.plot_overlay_all(
        per_step,
        label_map,
        str(overlay_png2),
        g1.OverlayFigureConfig(
            dpi=50,
            width_in=4.0,
            height_scale=0.5,
            marker_size=8,
            title=None,
        ),
        units,
    )
    assert overlay_png2.exists()


def test_build_arg_parser_defaults(monkeypatch):
    # Patch builder deps to avoid importing full CLI machinery.
    monkeypatch.setattr(g1, "build_mixed_root_arg_parser", lambda: argparse.ArgumentParser())
    monkeypatch.setattr(g1, "add_gpt_step_and_carpark_args", lambda parser: parser)
    parser = g1._build_arg_parser()
    args = parser.parse_args([])
    assert args.dpi == 600
    assert args.plot_units == "pp"


def test_main_error_paths(monkeypatch):
    args = SimpleNamespace(
        gpt_mode="canonical",
        no_gpt_subset_native=False,
        min_step=None,
        max_step=None,
        carpark_success_op="gt",
        carpark_soft_threshold=0.0,
        out_dir=None,
        dataset_name="ds",
        model_name="m",
        min_per_group=1,
        label_math="M1",
        label_math2="M2",
        width_in=4.0,
        height_scale=0.5,
        overlay_width_in=3.0,
        overlay_height_scale=0.5,
        marker_size=5.0,
        overlay_title=None,
        plot_units="pp",
        ymin_pp=-10.0,
        ymax_pp=10.0,
        ylim_pad_pp=1.0,
        yticks_pp="0",
        ymin_prob=-0.1,
        ymax_prob=0.1,
        ylim_pad_prob=0.01,
        yticks_prob="",
        dpi=50,
    )
    monkeypatch.setattr(g1, "_build_arg_parser", lambda: SimpleNamespace(parse_args=lambda: args))
    monkeypatch.setattr(g1, "build_files_by_domain_for_args", lambda a: ({"Math": ["f.jsonl"]}, "root"))
    monkeypatch.setattr(g1, "compute_effective_max_step", lambda a, hard_max_step: 5)
    monkeypatch.setattr(g1, "make_carpark_success_fn", lambda *a, **k: lambda *a2, **k2: None)

    # empty rows_df triggers SystemExit
    monkeypatch.setattr(g1, "load_rows", lambda files, cfg: pd.DataFrame())
    with pytest.raises(SystemExit):
        g1.main()

    # no per-step groups triggers SystemExit
    monkeypatch.setattr(
        g1,
        "load_rows",
        lambda files, cfg: pd.DataFrame({"domain": ["Math"], "step": [1], "correct": [1], "shift": [1]}),
    )
    monkeypatch.setattr(g1, "_compute_per_step", lambda df, min_per_group: ({}, []))
    with pytest.raises(SystemExit):
        g1.main()


def test_main_happy_path(tmp_path, monkeypatch):
    args = SimpleNamespace(
        gpt_mode="canonical",
        no_gpt_subset_native=False,
        min_step=None,
        max_step=None,
        carpark_success_op="gt",
        carpark_soft_threshold=0.0,
        out_dir=str(tmp_path),
        dataset_name="ds",
        model_name="m",
        min_per_group=1,
        label_math="M1",
        label_math2="M2",
        width_in=3.0,
        height_scale=0.5,
        overlay_width_in=3.0,
        overlay_height_scale=0.5,
        marker_size=2.0,
        overlay_title="Overlay",
        plot_units="pp",
        ymin_pp=-10.0,
        ymax_pp=10.0,
        ylim_pad_pp=1.0,
        yticks_pp="0",
        ymin_prob=-0.1,
        ymax_prob=0.1,
        ylim_pad_prob=0.01,
        yticks_prob="",
        dpi=50,
    )
    monkeypatch.setattr(g1, "_build_arg_parser", lambda: SimpleNamespace(parse_args=lambda: args))
    monkeypatch.setattr(g1, "build_files_by_domain_for_args", lambda a: ({"Math": ["f.jsonl"]}, "root"))
    monkeypatch.setattr(g1, "compute_effective_max_step", lambda a, hard_max_step: 5)
    monkeypatch.setattr(g1, "make_carpark_success_fn", lambda *a, **k: lambda *a2, **k2: None)

    rows_df = pd.DataFrame({"domain": ["Math", "Math"], "step": [1, 1], "correct": [1, 0], "shift": [1, 0]})
    monkeypatch.setattr(g1, "load_rows", lambda files, cfg: rows_df)

    per_step = {"Math": pd.DataFrame({"domain": ["Math"], "step": [1], "raw_effect": [0.0]})}
    rows_all = [per_step["Math"]]
    monkeypatch.setattr(g1, "_compute_per_step", lambda df, min_per_group: (per_step, rows_all))

    # stub plotters to avoid matplotlib dependency
    monkeypatch.setattr(g1, "plot_panels", lambda *a, **k: None)
    monkeypatch.setattr(g1, "plot_overlay_all", lambda *a, **k: None)

    g1.main()
    csvs = list(tmp_path.glob("raw_effect_per_step__*.csv"))
    assert csvs, "CSV output should be written"


def test_main_prints_pp_info(tmp_path, monkeypatch, capsys):
    # Reuse happy-path stubs but capture console output for pp info line.
    args = SimpleNamespace(
        gpt_mode="canonical",
        no_gpt_subset_native=False,
        min_step=None,
        max_step=None,
        carpark_success_op="gt",
        carpark_soft_threshold=0.0,
        out_dir=str(tmp_path),
        dataset_name="ds",
        model_name="m",
        min_per_group=1,
        label_math="M1",
        label_math2="M2",
        width_in=3.0,
        height_scale=0.5,
        overlay_width_in=3.0,
        overlay_height_scale=0.5,
        marker_size=2.0,
        overlay_title="Overlay",
        plot_units="pp",
        ymin_pp=-10.0,
        ymax_pp=10.0,
        ylim_pad_pp=1.0,
        yticks_pp="0",
        ymin_prob=-0.1,
        ymax_prob=0.1,
        ylim_pad_prob=0.01,
        yticks_prob="",
        dpi=50,
    )
    monkeypatch.setattr(g1, "_build_arg_parser", lambda: SimpleNamespace(parse_args=lambda: args))
    monkeypatch.setattr(g1, "build_files_by_domain_for_args", lambda a: ({"Math": ["f.jsonl"]}, "root"))
    monkeypatch.setattr(g1, "compute_effective_max_step", lambda a, hard_max_step: 5)
    monkeypatch.setattr(g1, "make_carpark_success_fn", lambda *a, **k: lambda *a2, **k2: None)
    rows_df = pd.DataFrame({"domain": ["Math", "Math"], "step": [1, 1], "correct": [1, 0], "shift": [1, 0]})
    monkeypatch.setattr(g1, "load_rows", lambda files, cfg: rows_df)
    per_step = {"Math": pd.DataFrame({"domain": ["Math"], "step": [1], "raw_effect": [0.0]})}
    rows_all = [per_step["Math"]]
    monkeypatch.setattr(g1, "_compute_per_step", lambda df, min_per_group: (per_step, rows_all))
    monkeypatch.setattr(g1, "plot_panels", lambda *a, **k: None)
    monkeypatch.setattr(g1, "plot_overlay_all", lambda *a, **k: None)

    g1.main()
    out = capsys.readouterr().out
    assert "percentage points (pp)" in out


def test_dunder_main_executes_with_stubs(monkeypatch, tmp_path):
    import argparse
    import runpy
    import sys
    import types

    class _AxisStub:
        def __init__(self):
            self.scatter_calls = []

        def set_title(self, *a, **k):
            return None

        def axhline(self, *a, **k):
            return None

        def scatter(self, *a, **k):
            self.scatter_calls.append(a)

        def set_xlabel(self, *a, **k):
            return None

        def set_ylabel(self, *a, **k):
            return None

        def set_ylim(self, *a, **k):
            return None

        def set_yticks(self, *a, **k):
            return None

        def legend(self, *a, **k):
            return None

    class _FigStub:
        def __init__(self, axes):
            self.axes = axes

        def subplots_adjust(self, *a, **k):
            return None

        def set_size_inches(self, *a, **k):
            return None

        def savefig(self, path, dpi=None):
            Path(path).write_text("fig")

    def fake_subplots(*args, **kwargs):
        nrows = args[0] if args else kwargs.get("nrows", 1)
        ncols = args[1] if len(args) > 1 else kwargs.get("ncols", 1)
        n_axes = max(1, int(nrows or 1) * int(ncols or 1))
        axes = [_AxisStub() for _ in range(n_axes)]
        fig = _FigStub(axes)
        return fig, axes if n_axes > 1 else axes[0]

    plt_stub = types.ModuleType("matplotlib.pyplot")
    plt_stub.subplots = fake_subplots
    plt_stub.close = lambda fig: None

    matplotlib_stub = types.ModuleType("matplotlib")
    matplotlib_stub.use = lambda backend: None
    matplotlib_stub.pyplot = plt_stub

    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib_stub)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", plt_stub)
    monkeypatch.setattr("src.analysis.plotting.apply_default_style", lambda *a, **k: None)

    def fake_parser_builder():
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset_name", default="ds")
        parser.add_argument("--model_name", default="m")
        parser.add_argument("--out_dir", default=None)
        parser.add_argument("--split", default=None)
        parser.add_argument("--root_dir", default=str(tmp_path))
        return parser

    monkeypatch.setattr("src.analysis.utils.build_mixed_root_arg_parser", fake_parser_builder)

    def fake_add_gpt_and_carpark(parser):
        parser.add_argument("--gpt_mode", default="canonical")
        parser.add_argument("--no_gpt_subset_native", action="store_true")
        parser.add_argument("--min_step", type=int, default=None)
        parser.add_argument("--max_step", type=int, default=None)
        parser.add_argument("--carpark_success_op", default="gt")
        parser.add_argument("--carpark_soft_threshold", type=float, default=0.0)
        return parser

    monkeypatch.setattr("src.analysis.utils.add_gpt_step_and_carpark_args", fake_add_gpt_and_carpark)
    monkeypatch.setattr("src.analysis.utils.gpt_keys_for_mode", lambda mode: ["g"])
    monkeypatch.setattr("src.analysis.utils.get_problem_id", lambda rec: rec.get("pid"))
    monkeypatch.setattr("src.analysis.core.plotting_helpers.compute_effective_max_step", lambda *a, **k: 3)
    monkeypatch.setattr("src.analysis.metrics.make_carpark_success_fn", lambda *a, **k: (lambda *a2, **k2: None))
    monkeypatch.setattr(
        "src.analysis.io.build_files_by_domain_for_args", lambda args: ({"Math": ["f"]}, str(tmp_path))
    )
    monkeypatch.setattr(
        "src.analysis.core.iter_correct_and_shift_samples_for_config",
        lambda files, cfg: [
            ("Math", 1, {"pid": "p1"}, True, True),
            ("Math", 1, {"pid": "p2"}, False, False),
        ],
    )

    monkeypatch.delitem(sys.modules, "src.analysis.graph_1", raising=False)
    monkeypatch.setattr(sys, "argv", ["graph_1", "--min_per_group", "1"])
    runpy.run_module("src.analysis.graph_1", run_name="__main__")
    csvs = list((tmp_path / "raw_effect_plots").glob("raw_effect_per_step__*.csv"))
    assert csvs
