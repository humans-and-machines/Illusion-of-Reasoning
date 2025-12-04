import types
from pathlib import Path

import numpy as np
import pandas as pd

import src.analysis.h2_plotting as h2_plot


class _AxisStub:
    def __init__(self):
        self.calls = []

    def plot(self, *args, **kwargs):
        self.calls.append(("plot", args, kwargs))
        return None

    def fill_between(self, *args, **kwargs):
        self.calls.append(("fill_between", args, kwargs))
        return None

    def boxplot(self, *args, **kwargs):
        self.calls.append(("boxplot", args, kwargs))
        return None

    def set_xlabel(self, val):
        self.calls.append(("xlabel", val))

    def set_ylabel(self, val):
        self.calls.append(("ylabel", val))

    def set_title(self, val):
        self.calls.append(("title", val))

    def set_xticks(self, *args, **kwargs):
        self.calls.append(("xticks", args, kwargs))

    def set_xticklabels(self, *args, **kwargs):
        self.calls.append(("xticklabels", args, kwargs))

    def grid(self, *args, **kwargs):
        self.calls.append(("grid", args, kwargs))


def _figure_stub(save_cb=None):
    def _savefig(path, *args, **kwargs):
        if save_cb:
            save_cb(path)

    return types.SimpleNamespace(
        tight_layout=lambda *a, **k: None,
        savefig=_savefig,
    )


def test_lineplot_saves(tmp_path, monkeypatch):
    axis = _AxisStub()
    fig = _figure_stub()
    monkeypatch.setattr(h2_plot.plt, "subplots", lambda *a, **k: (fig, axis))
    monkeypatch.setattr(h2_plot.plt, "close", lambda fig_obj: setattr(fig_obj, "closed", True))

    out = tmp_path / "line.png"
    h2_plot.lineplot([1, 2], [3, 4], ("x", "y", "t", str(out)))

    # Save was attempted and axis methods called
    assert ("xlabel", "x") in axis.calls
    assert ("ylabel", "y") in axis.calls
    assert ("title", "t") in axis.calls


def test_plot_diag_panel_writes(tmp_path, monkeypatch):
    axes = [_AxisStub(), _AxisStub(), _AxisStub()]
    fig = _figure_stub(save_cb=lambda path: Path(path).touch())
    monkeypatch.setattr(h2_plot.plt, "subplots", lambda *a, **k: (fig, axes))
    monkeypatch.setattr(h2_plot.plt, "close", lambda *a, **k: None)

    df = pd.DataFrame(
        {
            "uncertainty_std": [0.1, 0.2, 0.3, 0.4],
            "aha": [0, 1, 0, 1],
            "step": [1, 1, 2, 2],
        }
    )
    h2_plot.plot_diag_panel(df, out_dir=str(tmp_path))
    assert (tmp_path / "h2_diag_panel.png").exists()
    # Axis interactions happened
    assert any(call[0] == "boxplot" for call in axes[0].calls)
    assert any(call[0] == "plot" for call in axes[1].calls)
    assert any(call[0] == "plot" for call in axes[2].calls)


def test_plot_diag_panel_falls_back_legacy_labels(monkeypatch, tmp_path):
    class LegacyAxis(_AxisStub):
        def __init__(self):
            super().__init__()
            self.boxplot_calls = 0

        def boxplot(self, *args, **kwargs):
            self.boxplot_calls += 1
            self.calls.append(("boxplot", args, kwargs))
            if self.boxplot_calls == 1:
                raise TypeError("tick_labels unsupported")
            return {"boxes": [1]}

    legacy_axis = LegacyAxis()
    axes = [legacy_axis, _AxisStub(), _AxisStub()]
    fig = _figure_stub(save_cb=lambda path: Path(path).touch())
    monkeypatch.setattr(h2_plot.plt, "subplots", lambda *a, **k: (fig, axes))
    monkeypatch.setattr(h2_plot.plt, "close", lambda *a, **k: None)

    df = pd.DataFrame(
        {
            "uncertainty_std": [0.1, 0.2],
            "aha": [0, 1],
            "step": [1, 1],
        }
    )
    h2_plot.plot_diag_panel(df, out_dir=str(tmp_path))
    assert (tmp_path / "h2_diag_panel.png").exists()
    assert legacy_axis.boxplot_calls == 2
    assert legacy_axis.calls[1][2]["labels"] == ["aha=0", "aha=1"]


def test_plot_ame_with_ci_handles_missing_columns(tmp_path):
    df = pd.DataFrame({"step": [1, 2], "aha_ame": [0.1, 0.2]})
    # Missing required lower/upper -> early return, no file created.
    h2_plot.plot_ame_with_ci(df[["step", "aha_ame"]], out_dir=str(tmp_path))
    assert not (tmp_path / "aha_ame_with_ci.png").exists()


def test_plot_ame_with_ci_early_exit_on_empty_after_drop(monkeypatch, tmp_path):
    df = pd.DataFrame(
        {
            "step": [1, 2],
            "aha_ame": [np.nan, np.nan],
            "aha_ame_lo": [0.0, 0.1],
            "aha_ame_hi": [0.2, 0.3],
        }
    )

    def fail_subplots(*args, **kwargs):
        raise AssertionError("subplots should not be called when dataframe empties")

    monkeypatch.setattr(h2_plot.plt, "subplots", fail_subplots)
    h2_plot.plot_ame_with_ci(df, out_dir=str(tmp_path))
    assert not (tmp_path / "aha_ame_with_ci.png").exists()


def test_plot_ame_with_ci_full_path(tmp_path, monkeypatch):
    axis = _AxisStub()
    fig = _figure_stub(save_cb=lambda path: Path(path).touch())
    monkeypatch.setattr(h2_plot.plt, "subplots", lambda *a, **k: (fig, axis))
    monkeypatch.setattr(h2_plot.plt, "close", lambda *a, **k: None)

    called_style = {}
    monkeypatch.setattr(h2_plot, "style_ame_axis", lambda ax: called_style.setdefault("ran", True))

    df = pd.DataFrame(
        {
            "step": [1, 2],
            "aha_ame": [0.1, 0.2],
            "aha_ame_lo": [0.0, 0.1],
            "aha_ame_hi": [0.2, 0.3],
        }
    )
    h2_plot.plot_ame_with_ci(df, out_dir=str(tmp_path))
    assert (tmp_path / "aha_ame_with_ci.png").exists()
    assert called_style.get("ran")
    assert any(call[0] == "fill_between" for call in axis.calls)
