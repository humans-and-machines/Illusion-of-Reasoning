import sys
import types

import pandas as pd

import src.analysis.math_plots_impl as math_plots


def test_get_seaborn_returns_imported_module(monkeypatch):
    stub = types.SimpleNamespace()

    def fake_import(name):
        assert name == "seaborn"
        return stub

    monkeypatch.setattr(math_plots.importlib, "import_module", fake_import)
    assert math_plots._get_seaborn() is stub


def test_main_runs_noop_plots_when_columns_missing(monkeypatch, tmp_path):
    csv_path = tmp_path / "summary.csv"
    pd.DataFrame({"step": [0, 1]}).to_csv(csv_path, index=False)

    seaborn_stub = types.SimpleNamespace(
        set_theme=lambda **_: None,
        set_palette=lambda *_, **__: None,
    )
    monkeypatch.setattr(math_plots, "_get_seaborn", lambda: seaborn_stub)
    outdir = tmp_path / "plots_out"
    monkeypatch.setattr(sys, "argv", ["prog", str(csv_path), "--outdir", str(outdir)])

    math_plots.main()

    assert outdir.exists()


def test_plot_entropy_phase_adds_pass2_lines(monkeypatch):
    seaborn_stub = types.SimpleNamespace(
        set_theme=lambda **_: None,
        set_palette=lambda *_, **__: None,
        lineplot=lambda **kwargs: kwargs,
    )
    monkeypatch.setattr(math_plots, "_get_seaborn", lambda: seaborn_stub)

    class Axis:
        def __init__(self):
            self.legend_calls = 0

        def get_legend_handles_labels(self):
            return ["h"], ["l"]

        def legend(self, *a, **k):
            self.legend_calls += 1

        def set_xlabel(self, *_):
            return None

        def set_ylabel(self, *_):
            return None

        def set_title(self, *_):
            return None

        def grid(self, *a, **k):
            return None

    class Fig:
        def __init__(self):
            self.axis = Axis()

        def tight_layout(self):
            return None

        def savefig(self, *a, **k):
            return None

    class FakePlt:
        def __init__(self, fig):
            self.fig = fig

        def subplots(self, **kwargs):
            return self.fig, self.fig.axis

        def close(self, *a, **k):
            return None

    fig = Fig()
    monkeypatch.setattr(math_plots, "plt", FakePlt(fig))

    df = pd.DataFrame(
        {
            "step": [1, 2],
            "t1": [1.0, 1.1],
            "a1": [0.9, 1.0],
            "t2": [1.2, 1.3],
            "a2": [1.1, 1.2],
        }
    )
    math_plots._plot_entropy_phase(df, lambda fig_obj, name: None)
    assert fig.axis.legend_calls >= 1
