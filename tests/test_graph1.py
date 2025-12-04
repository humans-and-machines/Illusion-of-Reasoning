import types
from pathlib import Path

import pandas as pd

import src.analysis.graph_1 as graph1


def test_parse_float_list_and_color_for():
    assert graph1._parse_float_list(None) is None
    assert graph1._parse_float_list("  ") is None
    assert graph1._parse_float_list("1, 2 3") == [1.0, 2.0, 3.0]
    assert graph1._parse_float_list("x,4") == [4.0]
    assert graph1._parse_float_list("1,, ,bad,3") == [1.0, 3.0]
    assert graph1._color_for("Math") != "C0"
    assert graph1._color_for("Unknown") == "C0"


def test_per_step_raw_effect_and_compute_per_step():
    df = pd.DataFrame(
        {
            "domain": ["Math", "Math", "Math"],
            "step": [1, 1, 1],
            "correct": [1, 0, 1],
            "shift": [1, 0, 1],
        }
    )
    out = graph1.per_step_raw_effect(df, "Math", min_per_group=2)
    assert not out.empty
    assert out.iloc[0]["n_shift"] == 2
    empty_out = graph1.per_step_raw_effect(df[df["domain"] == "None"], "Math")
    assert empty_out.empty

    per_step, rows_all = graph1._compute_per_step(df, min_per_group=1)
    assert "Math" in per_step and rows_all


def test_auto_wrap_title_limits_lines():
    title = " ".join(["word"] * 50)
    wrapped = graph1._auto_wrap_title_to_two_lines(title, width=10)
    assert wrapped.count("\n") <= 1


def test_determine_plot_config_branches():
    args_pp = types.SimpleNamespace(
        plot_units="pp",
        ymin_pp=-10.0,
        ymax_pp=10.0,
        ylim_pad_pp=1.0,
        yticks_pp="-10,0,10",
        ymin_prob=-0.1,
        ymax_prob=0.1,
        ylim_pad_prob=0.01,
        yticks_prob=None,
    )
    cfg_pp = graph1._determine_plot_config(args_pp)
    assert cfg_pp.y_scale == 100.0
    assert cfg_pp.ylim == (-11.0, 11.0)
    assert cfg_pp.yticks == [-10.0, 0.0, 10.0]

    args_prob = types.SimpleNamespace(
        plot_units="prob",
        ymin_pp=-10.0,
        ymax_pp=10.0,
        ylim_pad_pp=1.0,
        yticks_pp="-10,0,10",
        ymin_prob=-0.2,
        ymax_prob=0.2,
        ylim_pad_prob=0.02,
        yticks_prob="-0.2,0,0.2",
    )
    cfg_prob = graph1._determine_plot_config(args_prob)
    assert cfg_prob.y_scale == 1.0
    assert cfg_prob.yticks == [-0.2, 0.0, 0.2]


class _AxisStub:
    def __init__(self):
        self.scatter_calls = []
        self.ticks = None
        self.title = None
        self.yticks = None

    def set_title(self, title, pad=None):
        self.title = title

    def axhline(self, *a, **k):
        return None

    def scatter(self, x, y, **kwargs):
        self.scatter_calls.append((list(x), list(y), kwargs))

    def set_xlabel(self, label):
        self.xlabel = label

    def set_ylabel(self, label):
        self.ylabel = label

    def set_ylim(self, low, high):
        self.ylim = (low, high)

    def set_yticks(self, ticks):
        self.yticks = list(ticks)

    def legend(self):
        return None


class _FigStub:
    def __init__(self, axes):
        self.axes = axes
        self.size = None

    def subplots_adjust(self, *a, **k):
        return None

    def set_size_inches(self, w, h, forward=None):
        self.size = (w, h)

    def savefig(self, path, dpi=None):
        Path(path).write_text("fig")

    def add_axes(self, rect):
        return self.axes[0]

    def add_subplot(self, *_):
        ax = _AxisStub()
        self.axes.append(ax)
        return ax


def test_plot_panels_and_overlay(monkeypatch, tmp_path):
    axes = [_AxisStub(), _AxisStub(), _AxisStub()]

    def fake_subplots(rows, cols=None, figsize=None, constrained_layout=None, sharey=None):
        return _FigStub(axes), axes

    monkeypatch.setattr(graph1, "plt", types.SimpleNamespace(subplots=fake_subplots, close=lambda fig: None))

    per_step = {
        "Crossword": pd.DataFrame({"step": [1], "raw_effect": [0.1]}),
        "Math": pd.DataFrame({"step": [1], "raw_effect": [0.2]}),
        "Math2": pd.DataFrame({"step": [1], "raw_effect": [0.3]}),
        "Carpark": pd.DataFrame({"step": [1], "raw_effect": [-0.1]}),
    }
    label_map = {"Crossword": "X", "Math": "M1", "Math2": "M2", "Carpark": "C"}
    units = graph1.PlotUnitsConfig(y_scale=100.0, ylim=(-10, 10), yticks=[-5, 0, 5], ylabel="Y")
    fig_cfg = graph1.PanelFigureConfig(dpi=100, width_in=4, height_scale=1.0, marker_size=10)
    out_png = tmp_path / "panels.png"
    graph1.plot_panels(per_step, label_map, str(out_png), fig_cfg, units)
    assert out_png.exists()

    # overlay plotting
    def fake_subplots_overlay(figsize=None, constrained_layout=None):
        ax = _AxisStub()
        return _FigStub([ax]), ax

    monkeypatch.setattr(graph1, "plt", types.SimpleNamespace(subplots=fake_subplots_overlay, close=lambda fig: None))
    overlay_cfg = graph1.OverlayFigureConfig(
        dpi=100, width_in=4, height_scale=1.0, marker_size=8, title="long title " * 10
    )
    overlay_path = tmp_path / "overlay.png"
    graph1.plot_overlay_all(per_step, label_map, str(overlay_path), overlay_cfg, units)
    assert overlay_path.exists()


def test_plot_panels_skips_missing_math(monkeypatch, tmp_path):
    axes = [_AxisStub(), _AxisStub(), _AxisStub()]

    def fake_subplots(rows, cols=None, figsize=None, constrained_layout=None, sharey=None):
        return _FigStub(axes), axes

    monkeypatch.setattr(
        graph1,
        "plt",
        types.SimpleNamespace(subplots=fake_subplots, close=lambda fig: None),
    )

    per_step = {
        "Carpark": pd.DataFrame({"step": [1], "raw_effect": [0.05]}),
    }
    label_map = {"Crossword": "X", "Math": "M", "Carpark": "C"}
    units = graph1.PlotUnitsConfig(y_scale=100.0, ylim=(-5, 5), yticks=[-5, 0, 5], ylabel="Y")
    fig_cfg = graph1.PanelFigureConfig(dpi=50, width_in=3.0, height_scale=1.0, marker_size=5.0)
    out_png = tmp_path / "panels_missing.png"

    graph1.plot_panels(per_step, label_map, str(out_png), fig_cfg, units)

    # Math axis should skip scatter when both Math/Math2 missing; Carpark scatters once.
    assert axes[1].scatter_calls == []
    assert len(axes[2].scatter_calls) == 1
