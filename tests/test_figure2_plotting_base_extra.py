import importlib
import sys
import types


def test_fallback_figure_added_when_missing(monkeypatch):
    # Remove cached module so reload uses our stubs.
    monkeypatch.setitem(sys.modules, "src.analysis.figure_2_plotting_base", None, raising=False)

    mpl_stub = types.ModuleType("matplotlib")
    mpl_stub.use = lambda *_a, **_k: None

    pyplot_stub = types.ModuleType("matplotlib.pyplot")

    def fake_subplots(*_args, **_kwargs):
        return "fig", "axes"

    pyplot_stub.subplots = fake_subplots

    lines_stub = types.ModuleType("matplotlib.lines")

    class DummyLine:
        pass

    lines_stub.Line2D = DummyLine

    monkeypatch.setitem(sys.modules, "matplotlib", mpl_stub)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot_stub)
    monkeypatch.setitem(sys.modules, "matplotlib.lines", lines_stub)

    module = importlib.reload(importlib.import_module("src.analysis.figure_2_plotting_base"))

    assert hasattr(module.plt, "figure")
    assert module.plt.figure() == "fig"
