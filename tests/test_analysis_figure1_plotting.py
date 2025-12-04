from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.analysis.core.figure_1_plotting as figplot


def test_panel_auto_ylim_handles_empty_and_clamp():
    empty = {"A": pd.DataFrame()}
    assert figplot._panel_auto_ylim(empty) == (0.0, 1.0)

    df = pd.DataFrame({"ratio": [0.2, 0.4], "lo": [0.1, 0.3], "hi": [0.3, 0.5]})
    bounds = figplot._panel_auto_ylim({"A": df}, pad=0.0, clamp=(0.15, 0.45))
    assert bounds == (0.15, 0.45)


def test_lighten_for_ci_and_highlights(monkeypatch):
    assert figplot._lighten_for_ci(None) is None
    assert figplot._lighten_for_ci("#000000").startswith("#")

    axis_calls = []

    class _Axis:
        def scatter(self, *args, **kwargs):
            axis_calls.append(("scatter", args, kwargs))

    df = pd.DataFrame({"step": [1, 2], "ratio": [0.1, 0.2]})
    opts = figplot.PlotSeriesOptions(alpha_ci=0.2, highlight_map={"Dom": {2: True}})
    figplot._plot_highlights(_Axis(), df, opts, domain="Dom")
    assert axis_calls and axis_calls[0][0] == "scatter"


def test_build_panel_specs_uses_highlight():
    axes = np.array([object(), object(), object()])
    base_df = pd.DataFrame({"ratio": [0.1], "lo": [0.05], "hi": [0.15], "step": [1]})
    ctx = figplot.PanelBuilderContext(
        axes=axes,
        native_by_dom={"D": base_df},
        gpt_by_dom={"D": base_df},
        formal_by_dom={"D": base_df},
        base_options=figplot.PlotSeriesOptions(alpha_ci=0.2),
        config={"model": "M", "highlight_formal_by_dom": {"D": {1: True}}, "highlight_color": "#0f0"},
    )
    specs = figplot._build_panel_specs(ctx)
    assert specs[2].options.highlight_map == {"D": {1: True}}


def test_setup_ratio_figure_box_aspect(monkeypatch):
    # Patch plt.subplots to return fake axes with set_box_aspect missing to hit fallback.
    class _FakeAxis:
        def __init__(self):
            self.pos = [0, 0, 1, 1]

        def get_position(self):
            return SimpleNamespace(x0=0, y0=0, width=1, height=1)

        @property
        def figure(self):
            return SimpleNamespace(get_size_inches=lambda: (1.0, 1.0))

        def set_position(self, pos):
            self.pos = pos

    fake_axes = [_FakeAxis(), _FakeAxis(), _FakeAxis()]

    def fake_subplots(nrows, ncols, figsize, dpi, sharex, sharey):
        return "fig", fake_axes

    monkeypatch.setattr(figplot.plt, "subplots", fake_subplots)
    fig, axes = figplot._setup_ratio_figure({"a4_pdf": False, "a4_orientation": "landscape"})
    assert axes.shape == (3,)
    assert fig == "fig"


def test_setup_ratio_figure_uses_a4_size(monkeypatch):
    captured = {}

    class FakeAxis:
        def __init__(self):
            self.aspect = None

        def set_box_aspect(self, value):
            self.aspect = value

    axes = [FakeAxis(), FakeAxis(), FakeAxis()]

    monkeypatch.setattr(figplot, "a4_size_inches", lambda orient: (11.0, 8.0))

    def fake_subplots(nrows, ncols, figsize, dpi, sharex, sharey):
        captured["figsize"] = figsize
        captured["args"] = (nrows, ncols, dpi, sharex, sharey)
        return "figure", axes

    monkeypatch.setattr(figplot.plt, "subplots", fake_subplots)
    fig, arr = figplot._setup_ratio_figure({"a4_pdf": True, "a4_orientation": "portrait", "panel_box_aspect": 1.2})
    assert captured["figsize"] == (11.0, 8.0)
    assert fig == "figure"
    assert all(axis.aspect == 1.2 for axis in arr)


def test_plot_series_per_domain_handles_empty_and_trend(monkeypatch):
    calls = []

    class FakeAxis:
        def scatter(self, *args, **kwargs):
            calls.append(("scatter", args, kwargs))

        def fill_between(self, *args, **kwargs):
            calls.append(("fill_between", args, kwargs))

        def plot(self, *args, **kwargs):
            calls.append(("plot", args, kwargs))

    df = pd.DataFrame(
        {
            "step": [0, 1],
            "ratio": [0.1, 0.2],
            "lo": [0.05, 0.15],
            "hi": [0.15, 0.25],
        }
    )
    monkeypatch.setattr(
        figplot,
        "fit_trend_wls",
        lambda d: (0.1, 0.0, 0.1, 0.05, 0.9, np.array([0, 1]), np.array([0.1, 0.2])),
    )
    trend_rows = figplot._plot_series_per_domain(
        FakeAxis(),
        {"empty": pd.DataFrame(), "dom": df},
        {"dom": "#123456"},
        figplot.PlotSeriesOptions(alpha_ci=0.4),
    )
    assert any(name == "fill_between" for name, _, _ in calls)
    assert any(name == "plot" for name, _, _ in calls)
    domains = {row["domain"] for row in trend_rows}
    assert domains == {"dom"}


def test_add_ratio_legend_includes_highlight():
    called = {}

    def fake_legend(handles, labels, loc, ncol, frameon, bbox_to_anchor):
        called["handles"] = handles
        called["labels"] = labels
        called["ncol"] = ncol

    fig = SimpleNamespace(legend=fake_legend)
    highlight_map = {"A": {1: True}}
    figplot._add_ratio_legend(
        fig,
        {"A": "red"},
        marker_size=5.0,
        highlight_map=highlight_map,
        highlight_color="#0f0",
    )
    labels = [h.get_label() for h in called["handles"]]
    assert "Δ > 0 at shift" in labels
    assert called["ncol"] == 2


def test_finalize_ratio_figure_adjusts_a4(monkeypatch):
    calls = []

    class FakeFig:
        def tight_layout(self, rect=None):
            calls.append(("layout", rect))

        def suptitle(self, *args, **kwargs):
            calls.append(("title", args, kwargs))

        def savefig(self, path):
            calls.append(("save", path))

        def set_size_inches(self, *args):
            calls.append(("size", args))

    monkeypatch.setattr(figplot, "a4_size_inches", lambda orient: (7.0, 9.0))
    closed = {}
    monkeypatch.setattr(figplot.plt, "close", lambda fig: closed.setdefault("closed", fig))
    figplot._finalize_ratio_figure(
        FakeFig(),
        {
            "dataset": "ds",
            "model": "m",
            "out_png": "png",
            "out_pdf": "pdf",
            "a4_pdf": True,
            "a4_orientation": "portrait",
        },
    )
    assert ("size", (7.0, 9.0)) in calls
    assert closed["closed"] is not None


def test_compute_formal_ratio_series_validates(monkeypatch):
    pair_stats = pd.DataFrame({"step": [1], "aha_formal": [1], "ratio": [0.5], "lo": [0.4], "hi": [0.6]})
    monkeypatch.setattr(figplot, "mark_formal_pairs", lambda df, **kwargs: df)

    # If bootstrap_problem_ratio returns non-DataFrame, TypeError is raised.
    monkeypatch.setattr(figplot, "bootstrap_problem_ratio", lambda df, col, num_bootstrap, seed: "not_df")
    cfg = figplot.FormalSweepPlotConfig(
        min_prior_steps=1,
        n_bootstrap=1,
        seed=0,
        out_png="p.png",
        out_pdf="p.pdf",
        dataset="d",
        model="m",
        primary_color="#000",
        ci_color="#111",
        ymax=1.0,
        alpha_ci=0.2,
        a4_pdf=False,
        orientation="landscape",
    )
    with pytest.raises(TypeError):
        figplot._compute_formal_ratio_series(pair_stats, (0.1, 0.2), cfg)


def test_init_formal_grid_a4(monkeypatch):
    captured = {}

    monkeypatch.setattr(figplot, "a4_size_inches", lambda orient: (5.0, 6.0))

    def fake_subplots(nrows, ncols, figsize, dpi, sharex, sharey):
        captured["figsize"] = figsize
        captured["args"] = (nrows, ncols, dpi, sharex, sharey)
        return "fig", [["a"] * ncols for _ in range(nrows)]

    monkeypatch.setattr(figplot.plt, "subplots", fake_subplots)
    fig, axes = figplot._init_formal_grid(
        [0.1],
        [0.2, 0.3],
        figplot.FormalSweepPlotConfig(
            min_prior_steps=1,
            n_bootstrap=1,
            seed=0,
            out_png="p",
            out_pdf="p2",
            dataset="d",
            model="m",
            primary_color="#010101",
            ci_color="#020202",
            ymax=1.0,
            alpha_ci=0.5,
            a4_pdf=True,
            orientation="portrait",
        ),
    )
    assert captured["figsize"] == (5.0, 6.0)
    assert fig == "fig"
    assert axes.shape == (1, 2)


def test_plot_formal_ratio_with_ci():
    calls = []

    class FakeAxis:
        def plot(self, *args, **kwargs):
            calls.append(("plot", args, kwargs))

        def fill_between(self, *args, **kwargs):
            calls.append(("fill_between", args, kwargs))

    ratio_series = figplot.FormalRatioSeries(
        step=pd.Series([1, 2]),
        ratio=pd.Series([0.1, 0.2]),
        lower_ci=pd.Series([0.05, 0.15]),
        upper_ci=pd.Series([0.15, 0.25]),
    )
    cfg = figplot.FormalSweepPlotConfig(
        min_prior_steps=1,
        n_bootstrap=1,
        seed=0,
        out_png="p",
        out_pdf="p2",
        dataset="d",
        model="m",
        primary_color="#010101",
        ci_color="#020202",
        ymax=1.0,
        alpha_ci=0.5,
        a4_pdf=False,
        orientation="portrait",
    )
    figplot._plot_formal_ratio(FakeAxis(), ratio_series, cfg)
    assert any(name == "fill_between" for name, _, _ in calls)


def test_format_delta_title_includes_delta3():
    title = figplot._format_delta_title((0.1, 0.2), delta3=0.3)
    assert "δ3=0.30" in title


def test_finalize_formal_grid_sets_a4(monkeypatch):
    calls = []

    class FakeFig:
        def legend(self, *args, **kwargs):
            calls.append(("legend", args, kwargs))

        def suptitle(self, *args, **kwargs):
            calls.append(("suptitle", args, kwargs))

        def tight_layout(self, *args, **kwargs):
            calls.append(("layout", args, kwargs))

        def savefig(self, path):
            calls.append(("savefig", path))

        def set_size_inches(self, *args):
            calls.append(("size", args))

    monkeypatch.setattr(figplot, "a4_size_inches", lambda orient: (8.5, 11.0))
    closed = {}
    monkeypatch.setattr(figplot.plt, "close", lambda fig: closed.setdefault("closed", fig))
    cfg = figplot.FormalSweepPlotConfig(
        min_prior_steps=1,
        n_bootstrap=1,
        seed=0,
        out_png="p",
        out_pdf="p2",
        dataset="d",
        model="m",
        primary_color="#010101",
        ci_color="#020202",
        ymax=1.0,
        alpha_ci=0.5,
        a4_pdf=True,
        orientation="portrait",
    )
    figplot._finalize_formal_grid(FakeFig(), cfg)
    assert any(call[0] == "size" for call in calls)
    assert closed["closed"] is not None
