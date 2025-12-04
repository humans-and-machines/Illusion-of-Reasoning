import builtins
import importlib.util
import runpy
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest


matplotlib.use("Agg")

# Provide minimal matplotlib fallbacks only if real pyplot/axes are unavailable.
import types


try:
    import matplotlib.axes as _axes  # noqa: F401
    import matplotlib.pyplot as _plt  # noqa: F401
except Exception:  # pragma: no cover - fallback for environments without matplotlib
    plt_stub = types.SimpleNamespace()

    def _stub_subplots(*_args, **_kwargs):
        fig = types.SimpleNamespace()
        axis = types.SimpleNamespace(
            spines={
                "top": types.SimpleNamespace(set_visible=lambda *_a, **_k: None),
                "right": types.SimpleNamespace(set_visible=lambda *_a, **_k: None),
            },
            grid=lambda *_a, **_k: None,
            scatter=lambda *_a, **_k: None,
            bar=lambda *_a, **_k: None,
            errorbar=lambda *_a, **_k: None,
            text=lambda *_a, **_k: None,
            set_ylim=lambda *_a, **_k: None,
            set_ylabel=lambda *_a, **_k: None,
            set_title=lambda *_a, **_k: None,
            set_xticks=lambda *_a, **_k: None,
            set_xticklabels=lambda *_a, **_k: None,
            axhline=lambda *_a, **_k: None,
        )
        return fig, axis

    plt_stub.subplots = _stub_subplots
    plt_stub.close = lambda *_a, **_k: None
    plt_stub.switch_backend = lambda *_a, **_k: None
    sys.modules["matplotlib.pyplot"] = plt_stub
    matplotlib.pyplot = plt_stub

    axes_stub = types.SimpleNamespace(Axes=type("Axes", (), {"name": "Axes"}))
    sys.modules["matplotlib.axes"] = axes_stub
    matplotlib.axes = axes_stub

import src.analysis.graph_4 as g4


def test_extract_step_and_group_key_variants():
    rec = {"step": 5}
    assert g4.extract_step(rec, "path") == 5
    assert g4.extract_step({}, "step-123.jsonl") == 123
    assert g4.extract_step({}, "no_step") == 0

    assert g4.group_key_for({"problem": "p"}, 1) == "problem:p"
    assert g4.group_key_for({"uid": 7}, 2) == "uid:7"
    assert g4.group_key_for({}, 9).startswith("__LINE__:")


def test_correctness_helpers():
    carpark_pass = {"soft_1": 0.2}
    assert g4.carpark_correct(carpark_pass, "ge", 0.1) is True
    assert g4.carpark_correct(carpark_pass, "lt", 0.5) is True
    assert g4.carpark_correct(carpark_pass, "le", 0.2) is True
    assert g4.carpark_correct({"is_correct": False}, "ge", 0.0) is False
    # Unknown op should fall through to False even with numeric input.
    assert g4.carpark_correct({"soft_reward": 1.0}, "unknown", 0.0) is False
    assert g4.carpark_correct({}, "ge", 0.0) is False

    assert g4.general_correct({"correct": 1}) is True
    assert g4.pass_correct_for_domain({"correct": True}, "Math", "ge", 0.0) is True
    assert g4.pass_correct_for_domain({"soft_reward": 0.0}, "Carpark", "gt", 0.1) is False
    # Non-dict pass data yields False
    assert g4.pass_correct_for_domain(None, "Math", "ge", 0.0) is False


def test_expand_dirs_and_iter_jsonl(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "f.jsonl").write_text("{}", encoding="utf-8")

    dirs = g4.expand_dirs([str(tmp_path / "a"), str(tmp_path / "missing"), str(tmp_path / "*")])
    assert (tmp_path / "a") in dirs and (tmp_path / "b") in dirs

    pairs = list(g4.iter_jsonl([str(tmp_path)], split=None))
    assert pairs and pairs[0][1].endswith("f.jsonl")

    pairs_filtered = list(g4.iter_jsonl([str(tmp_path)], split="nope"))
    assert pairs_filtered == []


def test_accumulate_and_build_rows(monkeypatch):
    bucket = g4.defaultdict(lambda: {"p1": [], "p2": []})
    cfg = g4.AggregationConfig(domain_name="Math", comparison_op="ge", threshold=0.0, min_step=0, max_step=10)
    rec = {"pass1": {"correct": True}, "pass2": {"correct": False}, "problem": "p"}
    g4._accumulate_record(bucket, "file_step-5.jsonl", rec, 0, cfg)
    rows = g4._build_per_problem_rows(bucket, "Math")
    assert rows[0]["p1_acc"] == 1.0 and rows[0]["p2_acc"] == 0.0
    assert rows[0]["p1_any_correct"] == 1

    # Missing pass2 should be skipped
    bucket = g4.defaultdict(lambda: {"p1": [], "p2": []})
    rec2 = {"pass1": {"correct": True}, "problem": "p"}
    g4._accumulate_record(bucket, "file_step-5.jsonl", rec2, 1, cfg)
    assert g4._build_per_problem_rows(bucket, "Math") == []
    # Step outside bounds should not populate bucket
    bucket = g4.defaultdict(lambda: {"p1": [], "p2": []})
    cfg_far = g4.AggregationConfig(domain_name="Math", comparison_op="ge", threshold=0.0, min_step=0, max_step=1)
    g4._accumulate_record(bucket, "file_step-5.jsonl", rec, 0, cfg_far)
    assert bucket == {}


def test_load_per_problem_uses_iter_helpers(monkeypatch):
    monkeypatch.setattr(g4, "iter_jsonl", lambda roots, split: [("root", "path.jsonl")])
    record = {"pass1": {"correct": True}, "pass2": {"correct": False}, "problem": "p"}
    monkeypatch.setattr(g4, "iter_records_from_file", lambda path: [record])
    args = type(
        "A",
        (),
        {"carpark_success_op": "ge", "carpark_soft_threshold": 0.0, "min_step": 0, "max_step": 10, "split": None},
    )
    df = g4.load_per_problem(args, "Math", ["root"])
    assert df.iloc[0]["raw_effect"] == -1.0


def test_bootstrap_ci_empty_and_nonempty():
    assert all(np.isnan(val) for val in g4.bootstrap_ci(np.array([])))
    lo, hi = g4.bootstrap_ci(np.array([0.0, 1.0]), n_bootstrap=10, seed=0)
    assert lo <= hi


def test_plot_helpers(monkeypatch):
    # minimal_axes should hide spines and enable grid
    fig, axis = matplotlib.pyplot.subplots()
    g4.minimal_axes(axis)
    assert not axis.spines["top"].get_visible()
    assert not axis.spines["right"].get_visible()

    rng = np.random.default_rng(0)
    cfg = g4.DomainPlotConfig(labels={0: "A", 1: "B"}, colors={0: "blue", 1: "red"}, rng=rng)
    df = pd.DataFrame(
        {
            "domain": ["Math", "Math", "Math", "Math"],
            "p1_any_correct": [0, 0, 1, 1],
            "raw_effect": [0.1, 0.2, -0.2, -0.1],
        }
    )
    # scatter should not raise
    g4._scatter_raw_effects(axis, df, cfg)

    # plot_domain_panel handles empty and populated data
    fig2, ax2 = matplotlib.pyplot.subplots()
    g4._plot_domain_panel(ax2, "Other", df, cfg)
    fig3, ax3 = matplotlib.pyplot.subplots()
    monkeypatch.setattr(matplotlib.axes.Axes, "errorbar", lambda self, *a, **k: None)
    g4._plot_domain_panel(ax3, "Math", df, cfg)
    assert ax3.get_ylim()[0] <= -0.2

    # Empty bucket branch for scatter (group with no values)
    scatter_calls = []
    axis_scatter = matplotlib.pyplot.subplots()[1]
    monkeypatch.setattr(axis_scatter, "scatter", lambda *a, **k: scatter_calls.append(a))
    df_single_bucket = pd.DataFrame({"domain": ["Math"], "p1_any_correct": [0], "raw_effect": [0.1]})
    g4._scatter_raw_effects(axis_scatter, df_single_bucket, cfg)
    assert len(scatter_calls) == 1

    # Force size==0 path for ylim fallback
    axis_ylim = matplotlib.pyplot.subplots()[1]
    monkeypatch.setattr(pd.Series, "size", property(lambda self: 0))
    g4._plot_domain_panel(axis_ylim, "Math", df_single_bucket, cfg)
    lo, hi = axis_ylim.get_ylim()
    assert lo == pytest.approx(-0.05) and hi == pytest.approx(0.05)


def test_plot_pass2_effects(tmp_path, monkeypatch):
    monkeypatch.setattr(matplotlib.axes.Axes, "errorbar", lambda self, *a, **k: None)
    out_base = tmp_path / "out_plot"
    df = pd.DataFrame(
        {
            "domain": ["Carpark", "Carpark", "Crossword", "Crossword", "Math", "Math"],
            "p1_any_correct": [0, 1, 0, 1, 0, 1],
            "raw_effect": [0.1, 0.2, 0.0, -0.1, 0.05, -0.05],
        }
    )
    g4.plot_pass2_effects(df, str(out_base), dpi=10, title="T")
    assert (tmp_path / "out_plot.png").exists()
    # When a domain has no matching data, _plot_domain_panel should emit placeholder text.
    df_empty = pd.DataFrame(columns=["domain", "p1_any_correct", "raw_effect"])
    fig, axis = matplotlib.pyplot.subplots()
    g4._plot_domain_panel(
        axis,
        "Nope",
        df_empty,
        g4.DomainPlotConfig(labels={0: "A", 1: "B"}, colors={0: "b", 1: "r"}, rng=np.random.default_rng(1)),
    )
    # Matplotlib stores text artists on the axis; ensure one was added.
    assert axis.texts


def test_build_summary_table():
    df = pd.DataFrame(
        {
            "domain": ["Carpark", "Carpark"],
            "p1_any_correct": [0, 1],
            "raw_effect": [0.1, 0.2],
        }
    )
    summary = g4._build_summary_table(df)
    assert set(summary["bucket"]) == {"P1_NONE", "P1_ANY"}


def test_main_runs_with_mocked_io(monkeypatch, tmp_path, capsys):
    # Mock CLI args
    outdir = tmp_path / "graphs"
    argv = [
        "prog",
        "--roots_carpark",
        "car",
        "--roots_crossword",
        "cross",
        "--roots_math",
        "math",
        "--outdir",
        str(outdir),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    # Provide deterministic records
    rec = {"pass1": {"correct": True}, "pass2": {"correct": False}, "problem": "p", "step": 1}
    monkeypatch.setattr(g4, "iter_jsonl", lambda roots, split: [(roots[0], "path.jsonl")])
    monkeypatch.setattr(g4, "iter_records_from_file", lambda path: [rec])
    monkeypatch.setattr(g4, "plot_pass2_effects", lambda df, out_base, dpi, title: None)

    g4.main()
    out = capsys.readouterr().out
    assert "[ok] wrote" in out
    assert (outdir / "tables" / "pass2_per_problem_combined.csv").exists()
    assert (outdir / "tables" / "pass2_summary_combined.csv").exists()


def test_main_handles_no_data_and_pooling(monkeypatch, tmp_path):
    # No roots -> expect sys.exit with stderr message
    args_empty = type(
        "Args",
        (),
        dict(
            roots_carpark=[],
            roots_crossword=[],
            roots_math=[],
            outdir=str(tmp_path / "empty"),
            outfile_tag=None,
            carpark_success_op="ge",
            carpark_soft_threshold=0.1,
            min_step=0,
            max_step=1,
            split=None,
            pool_across_steps=False,
            dpi=10,
            title=None,
        ),
    )()
    monkeypatch.setattr(g4, "parse_args", lambda: args_empty)
    with pytest.raises(SystemExit):
        g4.main()

    # Pooling branch aggregates across steps
    outdir = tmp_path / "graphs2"
    rows = [
        {"pass1": {"correct": True}, "pass2": {"correct": False}, "problem": "p", "step": 1},
        {"pass1": {"correct": False}, "pass2": {"correct": True}, "problem": "p", "step": 2},
    ]
    monkeypatch.setattr(sys, "argv", ["prog", "--roots_math", "root", "--outdir", str(outdir), "--pool_across_steps"])
    monkeypatch.setattr(
        g4,
        "parse_args",
        lambda: type(
            "Args",
            (),
            dict(
                roots_carpark=[],
                roots_crossword=[],
                roots_math=["root"],
                outdir=str(outdir),
                outfile_tag=None,
                carpark_success_op="ge",
                carpark_soft_threshold=0.0,
                min_step=0,
                max_step=10,
                split=None,
                pool_across_steps=True,
                dpi=10,
                title=None,
            ),
        )(),
    )
    monkeypatch.setattr(g4, "iter_jsonl", lambda roots, split: [(roots[0], "path.jsonl")])
    monkeypatch.setattr(g4, "iter_records_from_file", lambda path: rows)
    monkeypatch.setattr(g4, "plot_pass2_effects", lambda df, out_base, dpi, title: None)
    g4.main()
    pooled = pd.read_csv(outdir / "tables" / "pass2_per_problem_combined.csv")
    # raw_effect should be averaged across steps: ( -1 + 1 ) / 2 = 0
    assert pytest.approx(pooled.iloc[0]["raw_effect"]) == 0


def test_plot_pass2_effects_handles_nonlist_axes(monkeypatch, tmp_path):
    # Force list(...) to return the axis object so the non-list branch is exercised.
    orig_list = builtins.list

    def fake_list(obj=None):
        try:
            return orig_list() if obj is None else orig_list(obj)
        except TypeError:
            return obj

    monkeypatch.setattr(builtins, "list", fake_list)
    fig = types.SimpleNamespace(
        savefig=lambda path, **_k: Path(path).write_bytes(b""),
        suptitle=lambda *_a, **_k: None,
        legend=lambda *_a, **_k: None,
    )
    axis = types.SimpleNamespace(set_xlabel=lambda *_a, **_k: None)
    monkeypatch.setattr(g4.plt, "subplots", lambda *a, **k: (fig, axis))
    monkeypatch.setattr(g4, "_plot_domain_panel", lambda **kwargs: None)

    df = pd.DataFrame(
        {
            "domain": ["Carpark", "Crossword", "Math"],
            "p1_any_correct": [0, 0, 0],
            "raw_effect": [0.1, 0.2, 0.3],
        },
    )
    g4.plot_pass2_effects(df, str(tmp_path / "out_base"), dpi=10, title=None)


def test_main_guard_hit(monkeypatch, capsys, tmp_path):
    # Run the module as a script to exercise the __main__ guard and error exit.
    monkeypatch.delitem(sys.modules, "src.analysis.graph_4", raising=False)
    monkeypatch.setattr(sys, "argv", ["prog", "--outdir", str(tmp_path / "graphs")])
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("src.analysis.graph_4", run_name="__main__")
    assert excinfo.value.code == 2
    assert "No data found" in capsys.readouterr().err


def test_matplotlib_axes_shims_reload(monkeypatch):
    """Reload graph_4 with a dynamic matplotlib.axes stub to hit shim branches."""
    import types as _types

    import src.analysis.plotting as plotting

    monkeypatch.setattr(plotting, "apply_entropy_plot_style", lambda *_a, **_k: None)

    class DynamicAxesModule(_types.ModuleType):
        def __init__(self, name):
            super().__init__(name)
            self.created = []

        def __getattr__(self, item):
            if item == "Axes":
                cls = type("DynAxes", (), {})
                self.created.append(cls)
                return cls
            raise AttributeError(item)

    axes_mod = DynamicAxesModule("matplotlib.axes")
    pyplot_mod = _types.ModuleType("matplotlib.pyplot")
    pyplot_mod.subplots = lambda *_a, **_k: (
        _types.SimpleNamespace(
            savefig=lambda *a, **k: None,
            suptitle=lambda *a, **k: None,
            legend=lambda *a, **k: None,
        ),
        _types.SimpleNamespace(),
    )
    pyplot_mod.close = lambda *_a, **_k: None
    pyplot_mod.switch_backend = lambda *_a, **_k: None

    matplotlib_mod = _types.ModuleType("matplotlib")
    matplotlib_mod.axes = axes_mod
    matplotlib_mod.pyplot = pyplot_mod

    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib_mod)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot_mod)
    monkeypatch.setitem(sys.modules, "matplotlib.axes", axes_mod)

    spec = importlib.util.spec_from_file_location("graph_4_shim_test", Path(g4.__file__))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert len(axes_mod.created) >= 3
    for cls in axes_mod.created[:3]:
        assert hasattr(cls, "errorbar")
        assert hasattr(cls, "bar")
