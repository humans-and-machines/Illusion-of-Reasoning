import types
from pathlib import Path

import numpy as np
import pandas as pd

import src.analysis.h3_uncertainty.plotting as plotting


class _AxisStub:
    def __init__(self):
        self.calls = []

    def bar(self, *args, **kwargs):
        self.calls.append(("bar", args, kwargs))

    def errorbar(self, *args, **kwargs):
        self.calls.append(("errorbar", args, kwargs))

    def set_xticks(self, *args, **kwargs):
        self.calls.append(("xticks", args, kwargs))

    def set_xticklabels(self, *args, **kwargs):
        self.calls.append(("xticklabels", args, kwargs))

    def set_ylabel(self, *args, **kwargs):
        self.calls.append(("ylabel", args, kwargs))

    def set_xlabel(self, *args, **kwargs):
        self.calls.append(("xlabel", args, kwargs))

    def set_title(self, *args, **kwargs):
        self.calls.append(("title", args, kwargs))

    def set_ylim(self, *args, **kwargs):
        self.calls.append(("ylim", args, kwargs))

    def legend(self, *args, **kwargs):
        self.calls.append(("legend", args, kwargs))


def _fig_stub(axis: _AxisStub, save_cb=None):
    def _savefig(path, *args, **kwargs):
        if save_cb:
            save_cb(path)

    return types.SimpleNamespace(
        add_subplot=lambda *a, **k: axis,
        tight_layout=lambda *a, **k: None,
        savefig=_savefig,
    )


def test_plot_bars_with_ci_handles_nan_and_sets_labels():
    axis = _AxisStub()
    cfg = plotting.PairedBarPlot(
        axis=axis,
        labels=["a", "b"],
        series=[
            plotting.BarSeries(values=[0.1, np.nan], lower=[0.0, np.nan], upper=[0.2, np.nan], label="s1"),
            plotting.BarSeries(values=[np.nan, 0.4], lower=[np.nan, 0.3], upper=[np.nan, 0.5], label="s2"),
        ],
        title="T",
        ylabel="Y",
    )
    plotting._plot_bars_with_ci(cfg)
    assert any(call[0] == "bar" for call in axis.calls)
    assert ("ylabel", ("Y",), {}) in axis.calls
    assert ("title", ("T",), {}) in axis.calls
    assert any(call[0] == "legend" for call in axis.calls)


def _patch_matplotlib(monkeypatch, axis, save_path: Path):
    fig = _fig_stub(axis, save_cb=lambda path: Path(path).touch())
    monkeypatch.setattr(plotting, "apply_paper_font_style", lambda: None)
    monkeypatch.setattr(plotting.plt, "figure", lambda *a, **k: fig)
    monkeypatch.setattr(plotting.plt, "close", lambda *a, **k: None)
    return fig


def test_plot_question_overall_ci_saves(monkeypatch, tmp_path):
    axis = _AxisStub()
    _patch_matplotlib(monkeypatch, axis, tmp_path / "out.png")
    df = pd.DataFrame(
        {
            "group": ["g1", "g2"],
            "any_pass1": [0.1, 0.2],
            "any_pass1_lo": [0.0, 0.1],
            "any_pass1_hi": [0.2, 0.3],
            "any_pass2": [0.2, 0.3],
            "any_pass2_lo": [0.1, 0.2],
            "any_pass2_hi": [0.3, 0.4],
        }
    )
    out = tmp_path / "overall.png"
    plotting.plot_question_overall_ci(df, out_png=str(out), also_pdf=True)
    assert out.exists()
    assert (tmp_path / "overall.pdf").exists()


def test_plot_question_overall_ci_noop_on_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(
        plotting, "apply_paper_font_style", lambda: (_ for _ in ()).throw(AssertionError("should not style"))
    )
    monkeypatch.setattr(
        plotting.plt, "figure", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not figure"))
    )
    plotting.plot_question_overall_ci(pd.DataFrame(), out_png=str(tmp_path / "noop.png"), also_pdf=True)


def test_plot_question_by_step_and_bucket(monkeypatch, tmp_path):
    axis = _AxisStub()
    _patch_matplotlib(monkeypatch, axis, tmp_path / "out.png")
    df = pd.DataFrame(
        {
            "group": ["g1", "g1", "g2", "g2"],
            "step": [1, 2, 1, 2],
            "perplexity_bucket": [0, 1, 0, 1],
            "any_pass1": [0.1, 0.2, 0.3, 0.4],
            "any_pass1_lo": [0.0, 0.1, 0.2, 0.3],
            "any_pass1_hi": [0.2, 0.3, 0.4, 0.5],
            "any_pass2": [0.2, 0.3, 0.4, 0.5],
            "any_pass2_lo": [0.1, 0.2, 0.3, 0.4],
            "any_pass2_hi": [0.3, 0.4, 0.5, 0.6],
        }
    )
    out1 = tmp_path / "by_step.png"
    plotting.plot_question_by_step_ci(df, out_png=str(out1), also_pdf=False)
    assert out1.exists()

    out2 = tmp_path / "by_bucket.png"
    plotting.plot_question_by_bucket_ci(df, out_png=str(out2), also_pdf=False)
    assert out2.exists()
    assert any(call[0] == "errorbar" for call in axis.calls)


def test_plot_question_by_step_and_bucket_pdf(monkeypatch, tmp_path):
    axis = _AxisStub()
    _patch_matplotlib(monkeypatch, axis, tmp_path / "out.png")
    df = pd.DataFrame(
        {
            "group": ["g1"],
            "step": [1],
            "perplexity_bucket": [0],
            "any_pass1": [0.1],
            "any_pass1_lo": [0.0],
            "any_pass1_hi": [0.2],
            "any_pass2": [0.2],
            "any_pass2_lo": [0.1],
            "any_pass2_hi": [0.3],
        }
    )
    out_step = tmp_path / "by_step_pdf.png"
    plotting.plot_question_by_step_ci(df, out_png=str(out_step), also_pdf=True)
    assert (tmp_path / "by_step_pdf.pdf").exists()

    out_bucket = tmp_path / "by_bucket_pdf.png"
    plotting.plot_question_by_bucket_ci(df, out_png=str(out_bucket), also_pdf=True)
    assert (tmp_path / "by_bucket_pdf.pdf").exists()


def test_plot_prompt_overall_and_step_and_bucket(monkeypatch, tmp_path):
    axis = _AxisStub()
    _patch_matplotlib(monkeypatch, axis, tmp_path / "out.png")
    df = pd.DataFrame(
        {
            "group": ["g1", "g2"],
            "acc_pass1": [0.1, 0.2],
            "acc_pass1_lo": [0.0, 0.1],
            "acc_pass1_hi": [0.2, 0.3],
            "acc_pass2": [0.2, 0.3],
            "acc_pass2_lo": [0.1, 0.2],
            "acc_pass2_hi": [0.3, 0.4],
            "step": [1, 2],
            "perplexity_bucket": [0, 1],
        }
    )
    out = tmp_path / "prompt_overall.png"
    plotting.plot_prompt_overall_ci(df, out_png=str(out), also_pdf=False)
    assert out.exists()

    out_step = tmp_path / "prompt_step.png"
    plotting.plot_prompt_by_step_ci(df, out_png=str(out_step), also_pdf=False)
    assert out_step.exists()

    out_bucket = tmp_path / "prompt_bucket.png"
    plotting.plot_prompt_by_bucket_ci(df, out_png=str(out_bucket), also_pdf=False)
    assert out_bucket.exists()


def test_plot_prompt_overall_and_step_empty_short_circuit(monkeypatch, tmp_path):
    monkeypatch.setattr(plotting, "apply_paper_font_style", lambda: (_ for _ in ()).throw(AssertionError("no style")))
    monkeypatch.setattr(plotting.plt, "figure", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no figure")))
    plotting.plot_prompt_overall_ci(pd.DataFrame(), out_png=str(tmp_path / "noop.png"), also_pdf=True)
    plotting.plot_prompt_by_step_ci(pd.DataFrame(), out_png=str(tmp_path / "noop_step.png"), also_pdf=True)
    plotting.plot_prompt_by_bucket_ci(pd.DataFrame(), out_png=str(tmp_path / "noop_bucket.png"), also_pdf=True)


def test_plot_prompt_by_step_and_bucket_pdf(monkeypatch, tmp_path):
    axis = _AxisStub()
    _patch_matplotlib(monkeypatch, axis, tmp_path / "out.png")
    df = pd.DataFrame(
        {
            "group": ["g1"],
            "step": [1],
            "perplexity_bucket": [0],
            "acc_pass1": [0.1],
            "acc_pass1_lo": [0.0],
            "acc_pass1_hi": [0.2],
            "acc_pass2": [0.2],
            "acc_pass2_lo": [0.1],
            "acc_pass2_hi": [0.3],
        }
    )
    out_step = tmp_path / "prompt_step_pdf.png"
    plotting.plot_prompt_by_step_ci(df, out_png=str(out_step), also_pdf=True)
    assert (tmp_path / "prompt_step_pdf.pdf").exists()

    out_bucket = tmp_path / "prompt_bucket_pdf.png"
    plotting.plot_prompt_by_bucket_ci(df, out_png=str(out_bucket), also_pdf=True)
    assert (tmp_path / "prompt_bucket_pdf.pdf").exists()


def test_plot_prompt_level_deltas_grouped_and_plain(monkeypatch, tmp_path):
    axis = _AxisStub()
    _patch_matplotlib(monkeypatch, axis, tmp_path / "out.png")

    df_grouped = pd.DataFrame({"delta": [1, 0, -1], "forced_insight": [1, 1, 0]})
    out_grouped = tmp_path / "deltas_grouped.png"
    plotting.plot_prompt_level_deltas(df_grouped, out_png=str(out_grouped), by_forced=True, also_pdf=True)
    assert out_grouped.exists()
    assert (tmp_path / "deltas_grouped.pdf").exists()

    axis.calls.clear()
    df_plain = pd.DataFrame({"delta": [1, -1, -1]})
    out_plain = tmp_path / "deltas_plain.png"
    plotting.plot_prompt_level_deltas(df_plain, out_png=str(out_plain), by_forced=False, also_pdf=False)
    assert out_plain.exists()
    assert any(call[0] == "bar" for call in axis.calls)


def test_plot_prompt_level_deltas_empty_and_forced_none(monkeypatch, tmp_path):
    # Empty -> early return
    monkeypatch.setattr(plotting.plt, "figure", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no plot")))
    plotting.plot_prompt_level_deltas(pd.DataFrame(), out_png=str(tmp_path / "none.png"))

    # Grouped counts empty triggers xtick branch
    class AxisWithTicks(_AxisStub):
        def __init__(self):
            super().__init__()
            self.xticks_called = False
            self.xticklabels_called = False

        def set_xticks(self, *args, **kwargs):
            self.xticks_called = True
            super().set_xticks(*args, **kwargs)

        def set_xticklabels(self, *args, **kwargs):
            self.xticklabels_called = True
            super().set_xticklabels(*args, **kwargs)

    axis = AxisWithTicks()
    plotting._plot_forced_delta_bars(axis, grouped_counts=[])
    assert axis.xticks_called and axis.xticklabels_called
