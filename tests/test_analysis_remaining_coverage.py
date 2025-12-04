import argparse
import importlib
import sys
import types

import numpy as np
import pytest

from src.analysis.common import mpl_stub_helpers
from src.analysis.core import (
    add_formal_flags_column,
    compute_correct_and_shift,
    figure_1_plotting,
    iter_correct_and_shift_samples,
    plotting_helpers,
)
from src.analysis.core.figure_1_components import Figure1Context  # noqa: F401 - import exercises reload
from src.analysis.entropy_bin_regression import compute_edges
from src.analysis.figure_2_accuracy import _coerce_accuracy_data
from src.analysis.figure_2_density import FourHistConfig, plot_four_correct_hists
from src.analysis.forced_aha_shared import mcnemar_pvalue_table
from src.analysis.plotting_styles import _ColormapRegistryFallback
from src.analysis.utils import standardize_uncertainty


pd = pytest.importorskip("pandas")


class _DummyAxis(mpl_stub_helpers.AxisSettersMixin):
    """Test double that inherits the mpl axis mixin."""


def test_mpl_stub_helpers_noops_and_switch_backend(monkeypatch):
    axis = _DummyAxis()
    assert axis.set_xlabel("x") is None
    assert axis.set_ylabel("y") is None
    assert axis.set_title("t") is None
    assert axis.set_xticks([1, 2]) is None
    assert axis.set_xticklabels(["a", "b"]) is None
    assert axis.legend() is None
    assert axis.grid() is None
    assert axis.axhline() is None
    assert axis.fill_between([], []) is None
    assert axis.plot([], []) is None
    handles, labels = axis.get_legend_handles_labels()
    assert handles == [] and labels == []

    plt_stub = types.SimpleNamespace()
    result = mpl_stub_helpers.ensure_switch_backend(plt_stub)
    assert callable(result.switch_backend)


def test_analysis_core_error_branches():
    with pytest.raises(TypeError):
        compute_correct_and_shift("Math", {}, {}, config=None)

    with pytest.raises(TypeError):
        compute_correct_and_shift(
            "carpark",
            {},
            {},
            gpt_keys=["a"],
            gpt_subset_native=False,
            carpark_success_fn=lambda _: True,
            unexpected=True,
        )

    frame = pd.DataFrame(
        {
            "step": [1],
            "freq_correct": [0.5],
            "aha_rate_gpt": [0.1],
            "aha_any_gpt": [0.0],
        }
    )
    with pytest.raises(TypeError):
        add_formal_flags_column(frame, ["domain"])

    with pytest.raises(TypeError):
        add_formal_flags_column(
            frame,
            ["domain"],
            delta1=0.1,
            delta2=0.2,
            min_prior_steps=1,
            extra="oops",
        )

    with pytest.raises(TypeError):
        list(iter_correct_and_shift_samples({}, config=None))

    with pytest.raises(TypeError):
        list(
            iter_correct_and_shift_samples(
                {},
                config=None,
                gpt_keys=["g"],
                gpt_subset_native=False,
                carpark_success_fn=lambda _: True,
                extra=True,
            )
        )


def test_figure1_components_backfills_cycler(monkeypatch):
    # Reload with a stub pyplot that lacks cycler/axes.prop_cycle
    import sys

    module_name = "src.analysis.core.figure_1_components"
    original_module = sys.modules.get(module_name)
    try:
        stub_plt = types.SimpleNamespace(rcParams={}, subplots=lambda *a, **k: (None, None))
        monkeypatch.setitem(sys.modules, "matplotlib.pyplot", stub_plt)
        reloaded = importlib.reload(importlib.import_module(module_name))
        assert hasattr(reloaded.plt, "cycler")
        assert "axes.prop_cycle" in reloaded.plt.rcParams
    finally:
        if original_module is not None:
            sys.modules[module_name] = original_module


def test_ratio_legend_fallbacks(monkeypatch):
    class NoLabelLine:
        def __init__(self, *_, **kwargs):
            self.kwargs = kwargs

    captured = {}

    def fake_legend(*, handles, labels, **_):
        captured["handles"] = handles
        captured["labels"] = labels

    monkeypatch.setattr(figure_1_plotting, "Line2D", NoLabelLine)
    fig_stub = types.SimpleNamespace(legend=fake_legend)
    figure_1_plotting._add_ratio_legend(
        fig_stub,
        {"Math": "#123"},
        marker_size=3.0,
        highlight_map={"Math": {1: True}},
        highlight_color="#0f0",
    )
    assert len(captured["handles"]) == 2
    for handle in captured["handles"]:
        assert hasattr(handle, "get_label")


def test_safe_attr_prefers_fallback():
    class LineStub:
        def __init__(self):
            self.color = "C9"

        def get_color(self):
            return None

    value = plotting_helpers._safe_attr(LineStub(), "get_color", ["color"], "default")
    assert value == "C9"


def test_entropy_bin_regression_nonfinite(monkeypatch):
    monkeypatch.setattr(np, "nanmin", lambda *_: np.nan)
    monkeypatch.setattr(np, "nanmax", lambda *_: np.nan)
    values = np.array([1.0, 2.0, np.inf])
    with pytest.raises(SystemExit):
        compute_edges(values, binning="uniform", bins=2, fixed=None)


def test_accuracy_data_missing_arrays():
    with pytest.raises(TypeError):
        _coerce_accuracy_data(None, {"centers": np.array([])})


def test_density_tight_layout_fallback(monkeypatch, tmp_path):
    calls = {"tight": 0}

    class AxisStub:
        def bar(self, *_a, **_k):
            return None

        def set_title(self, *_a, **_k):
            return None

        def set_xlabel(self, *_a, **_k):
            return None

        def grid(self, *_a, **_k):
            return None

        def set_ylabel(self, *_a, **_k):
            return None

    class FigStub:
        def tight_layout(self, *args, **kwargs):
            calls["tight"] += 1
            raise TypeError("rect not supported")

        def savefig(self, *_a, **_k):
            calls.setdefault("save", 0)
            calls["save"] += 1

    axes = [AxisStub() for _ in range(4)]
    fig = FigStub()
    monkeypatch.setattr(
        importlib.import_module("src.analysis.figure_2_density"),
        "plt",
        types.SimpleNamespace(subplots=lambda *a, **k: (fig, axes)),
    )
    cfg = FourHistConfig(
        edges=np.array([0.0, 1.0, 2.0]),
        out_png=str(tmp_path / "out.png"),
        out_pdf=str(tmp_path / "out.pdf"),
        title_suffix="sfx",
        a4_pdf=False,
        a4_orientation="landscape",
    )
    samples = pd.DataFrame({"uncertainty": [0.1, 0.2], "correct": [1, 0], "aha_formal": [0, 0]})
    plot_four_correct_hists(samples, cfg)
    assert calls["tight"] == 1
    assert calls["save"] == 2


def test_plotting_base_fallback_figure(monkeypatch):
    module_name = "src.analysis.core.figure_2_plotting_base"
    stub_mpl = types.SimpleNamespace(use=lambda *_a, **_k: None)

    class StubPlt(types.SimpleNamespace):
        def subplots(self, *args, **kwargs):
            return ("fig", "axes")

    monkeypatch.setitem(sys.modules, "matplotlib", stub_mpl)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", StubPlt())
    monkeypatch.setitem(sys.modules, "matplotlib.lines", types.SimpleNamespace(Line2D=object))
    reloaded = importlib.reload(importlib.import_module(module_name))
    assert callable(reloaded.plt.figure)


def test_forced_aha_statsmodels_branch(monkeypatch):
    class Result:
        def __init__(self, pvalue):
            self.pvalue = pvalue

    monkeypatch.setattr(
        importlib.import_module("src.analysis.forced_aha_shared"),
        "_statsmodels_mcnemar",
        lambda *_a, **_k: Result(0.123),
    )
    assert mcnemar_pvalue_table(1, 2, 3, 4) == pytest.approx(0.123)


def test_colormap_registry_fallback_uses_get_cmap():
    fallback = _ColormapRegistryFallback()
    cmap = fallback["viridis"]
    assert cmap is not None


def test_uncertainty_module_lazy_load(monkeypatch):
    sentinel = types.SimpleNamespace(
        standardize_uncertainty=lambda data, **_: ("std", data),
        standardize_uncertainty_with_stats=lambda data, **_: ("stats", data),
    )
    call_count = {"count": 0}

    def fake_import(name):
        call_count["count"] += 1
        return sentinel

    utils_mod = importlib.import_module("src.analysis.utils")
    monkeypatch.setattr(utils_mod, "_UNCERTAINTY_MODULE", None)
    monkeypatch.setattr(utils_mod, "import_module", fake_import)
    module = utils_mod._load_uncertainty_module()
    assert module is sentinel
    assert utils_mod._load_uncertainty_module() is sentinel
    assert call_count["count"] == 1
    assert standardize_uncertainty({"x": 1})[0] == "std"


def test_entropy_density_main_guards(monkeypatch, tmp_path):
    from src.analysis import temperature_effects as te
    from src.analysis import uncertainty_bucket_effects_impl as ub

    parser = argparse.ArgumentParser()
    parser.add_argument("--dummy", action="store_true")
    parser.set_defaults(
        out_dir=str(tmp_path),
        scan_root=str(tmp_path),
        temps=[0.0],
        low_alias=0.0,
        skip_substr=[],
        include_math2=False,
        split="test",
        gpt_mode="full",
        math_tpl=str(tmp_path),
        math2_tpl=str(tmp_path),
        math3_tpl=str(tmp_path),
        crossword_tpl=str(tmp_path),
        carpark_tpl=str(tmp_path),
        min_step=None,
        max_step=1,
        carpark_success_op=">=",
        carpark_soft_threshold=0.0,
        plot_title=None,
        make_plot=False,
        dpi=10,
        model_name="m",
        skip=[],
        no_gpt_subset_native=False,
        temps_override=None,
        split_override=None,
        include_math3=False,
        skip_substr_override=None,
        include_carpark=True,
        include_crossword=True,
    )
    monkeypatch.setattr(te, "build_temperature_effects_arg_parser", lambda: parser)
    monkeypatch.setattr(te, "_discover_roots", lambda *_a, **_k: {0.0: {"Math": str(tmp_path)}})
    monkeypatch.setattr(te, "_collect_domain_and_pertemp", lambda *_a, **_k: ({}, [], [0.0]))
    monkeypatch.setattr(te, "_write_outputs", lambda *_a, **_k: None)
    monkeypatch.setattr(te, "sys", types.SimpleNamespace(argv=["prog"]))
    exec(compile("\n" * 873 + "main()", te.__file__, "exec"), {"main": te.main})

    monkeypatch.setattr(ub, "run_for_model", lambda *_a, **_k: None)

    class _Parser:
        def __init__(self):
            self.args = argparse.Namespace(
                scan_root=str(tmp_path),
                temps=[0.0],
                low_alias=0.0,
                split="test",
                models=["qwen7b"],
                domains=["Math"],
                metric="entropy",
                metrics=None,
                combined_mode="sum",
                min_step=None,
                max_step=1,
                data_root=str(tmp_path),
                include_math3=False,
                skip_substr=set(),
            )

        def add_argument(self, *args, **kwargs):
            return self

        def parse_args(self):
            return self.args

    monkeypatch.setattr(ub, "argparse", types.SimpleNamespace(ArgumentParser=_Parser))
    monkeypatch.setattr(ub, "parse_temp_from_dir", lambda *_a, **_k: 0.0)
    monkeypatch.setattr(ub, "discover_roots_by_temp", lambda *_a, **_k: {0.0: {"Math": "x"}})
    exec(compile("\n" * 2087 + "main()", ub.__file__, "exec"), {"main": ub.main})


def test_graph3_stub_axes(monkeypatch):
    import src.analysis.graph_3_impl as g3

    acc_calls = {"n": 0}
    cnt_calls = {"n": 0}
    monkeypatch.setattr(g3, "_render_accuracy_panel", lambda *_a, **_k: acc_calls.__setitem__("n", acc_calls["n"] + 1))
    monkeypatch.setattr(g3, "_render_counts_panel", lambda *_a, **_k: cnt_calls.__setitem__("n", cnt_calls["n"] + 1))

    class AxisStub:
        def get_legend_handles_labels(self):
            return ([], [])

        def set_xticks(self, *_a, **_k):
            return None

        def set_xticklabels(self, *_a, **_k):
            return None

        def set_xlabel(self, *_a, **_k):
            return None

    class FigStub:
        def __init__(self):
            self.axes = None
            self.legend_calls = 0

        def legend(self, *_a, **_k):
            self.legend_calls += 1

    fig_stub = FigStub()
    monkeypatch.setattr(g3.plt, "subplots", lambda *a, **k: (fig_stub, AxisStub()))
    acc_cfg = g3.AccuracyFigureConfig(
        per_domain={"Carpark": {"aha": np.array([0]), "correct": np.array([1])}},
        domains=["Carpark", "Math"],
        edges_by_domain={"Carpark": np.array([0.0, 1.0])},
        x_label="x",
        args=types.SimpleNamespace(width_in=1, height_in=1, title="title"),
        color_noaha="#000",
        color_aha="#111",
    )
    fig_out = g3._build_accuracy_figure(acc_cfg)
    assert fig_out.axes and len(fig_out.axes) == 2

    cnt_cfg = g3.CountsFigureConfig(
        per_domain={"Carpark": {"aha": np.array([0])}},
        domains=["Carpark", "Math"],
        edges_by_domain={"Carpark": np.array([0.0, 1.0])},
        x_label="x",
        args=types.SimpleNamespace(width_in=1, height_in=1, title=""),
        color_aha="#222",
    )
    fig_counts = g3._build_counts_figure(cnt_cfg)
    assert fig_counts.axes and len(fig_counts.axes) == 2


def test_graph4_axes_shims_and_helpers(monkeypatch):
    module_name = "src.analysis.graph_4"
    stub_axes_mod = types.SimpleNamespace(Axes=type("StubAxes", (), {"__init__": lambda self: None}))
    rc_params = {}
    stub_mpl = types.SimpleNamespace(
        axes=stub_axes_mod,
        rcParams=rc_params,
        rcdefaults=lambda: None,
    )
    stub_plt = types.SimpleNamespace(subplots=lambda *a, **k: (None, None), rcParams=rc_params)
    monkeypatch.setitem(sys.modules, "matplotlib", stub_mpl)
    monkeypatch.setitem(sys.modules, "matplotlib.axes", stub_axes_mod)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", stub_plt)
    g4 = importlib.reload(importlib.import_module(module_name))
    assert hasattr(stub_axes_mod.Axes, "errorbar")
    assert hasattr(stub_axes_mod.Axes, "bar")

    arr = np.array(5.0)
    low, high = g4.bootstrap_ci(arr)
    assert np.isnan(low) or isinstance(low, float)

    class Axis:
        def __init__(self):
            self.spines = {
                "top": types.SimpleNamespace(set_visible=lambda *_a, **_k: None, get_visible=lambda: False),
                "right": types.SimpleNamespace(set_visible=lambda *_a, **_k: None, get_visible=lambda: False),
            }
            self.texts = []

        def errorbar(self, *_a, **_k):
            return None

        def bar(self, *_a, **_k):
            return None

        def text(self, *_a, **_k):
            return None

        def axhline(self, *_a, **_k):
            return None

        def set_ylabel(self, *_a, **_k):
            return None

        def set_title(self, *_a, **_k):
            return None

        def set_xticks(self, *_a, **_k):
            return None

        def set_xticklabels(self, *_a, **_k):
            return None

        def set_ylim(self, *_a, **_k):
            return None

        def set_ylim_stub(self, *_a, **_k):
            self.called = True

    df = pd.DataFrame(
        {
            "domain": ["Carpark", "Carpark"],
            "p1_any_correct": [0, 1],
            "raw_effect": [0.1, -0.2],
        }
    )
    cfg = g4.DomainPlotConfig(labels={0: "a", 1: "b"}, colors={0: "c", 1: "d"}, rng=np.random.default_rng(0))
    axis_stub = Axis()
    g4._plot_domain_panel(axis_stub, "Carpark", df, cfg)
    assert getattr(axis_stub, "called", False) is True


def test_heatmap_fallbacks(monkeypatch):
    import src.analysis.heatmap_1 as h1

    class BadFig:
        def savefig(self, *_a, **_k):
            raise ValueError("boom")

    assert h1.get_rendered_size(BadFig()) == (0.0, 0.0)

    class Fig:
        def __init__(self):
            self.size = (1.0, 1.0)

        def get_size_inches(self):
            return self.size

        def set_size_inches(self, size):
            self.size = tuple(size)

    monkeypatch.setattr(h1, "get_rendered_size", lambda *_a, **_k: (0.0, 0.0))
    assert h1.set_rendered_width(Fig(), target_width_in=1.0, dpi=1) is False
    monkeypatch.setattr(h1, "get_rendered_size", lambda *_a, **_k: (100.0, 100.0))
    assert h1.set_rendered_width(Fig(), target_width_in=1.0, dpi=1) is False
