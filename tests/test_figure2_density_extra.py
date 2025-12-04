import numpy as np
import pandas as pd

import src.analysis.figure_2_density as fig2


class _DummyAxis:
    def __init__(self):
        self.bar_called = False
        self.titles = []
        self.labels = []
        self.grid_called = False

    def bar(self, *args, **kwargs):
        self.bar_called = True

    def set_title(self, title):
        self.titles.append(title)

    def set_xlabel(self, label):
        self.labels.append(label)

    def grid(self, *_args, **_kwargs):
        self.grid_called = True

    def set_ylabel(self, *_args, **_kwargs):
        self.labels.append("y")


class _DummyFig:
    def tight_layout(self, *args, **kwargs):
        return None


def test_plot_four_correct_hists_calls_bar_for_all_panels(monkeypatch, tmp_path):
    d_all = pd.DataFrame(
        {
            "uncertainty": [0.1, 0.2],
            "correct": [1, 0],
            "aha_words": [1, 0],
            "aha_gpt": [0, 1],
            "aha_formal": [0, 1],
        }
    )
    edges = np.linspace(-1, 1, 5)

    axes = [_DummyAxis() for _ in range(4)]

    def fake_subplots(*_args, **_kwargs):
        return _DummyFig(), axes

    monkeypatch.setattr(fig2.plt, "subplots", fake_subplots)
    monkeypatch.setattr(fig2, "save_figure_outputs", lambda *a, **k: None)
    monkeypatch.setattr(fig2, "compute_correct_hist", lambda *a, **k: np.array([1, 2, 3, 4]))

    cfg = fig2.FourHistConfig(
        out_png=str(tmp_path / "four.png"),
        out_pdf=str(tmp_path / "four.pdf"),
        title_suffix="demo",
        a4_pdf=False,
        a4_orientation="portrait",
        edges=edges,
    )

    fig2.plot_four_correct_hists(d_all, cfg)

    assert all(ax.bar_called for ax in axes)
    assert all(ax.grid_called for ax in axes)
    assert any(ax.titles for ax in axes)
