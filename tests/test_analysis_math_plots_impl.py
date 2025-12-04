import importlib
import os
import sys

import numpy as np
import pandas as pd
import pytest

import src.analysis.math_plots_impl as impl


def test_coerce_pct_series_handles_strings_and_numbers():
    series = pd.Series(["10%", " 20.5 %", "-", "3.3"])
    coerced = impl._coerce_pct_series(series)
    assert coerced[0] == 10.0 and coerced[1] == 20.5
    assert np.isnan(coerced[2]) and coerced[3] == 3.3

    nums = pd.Series([1, 2, 3], dtype=int)
    coerced_nums = impl._coerce_pct_series(nums)
    assert coerced_nums.dtype.kind == "f" and coerced_nums.tolist() == [1.0, 2.0, 3.0]


def test_load_summary_coerces_pct_and_sorts(tmp_path):
    csv_path = tmp_path / "summary.csv"
    df = pd.DataFrame(
        {
            "step": [2, 1],
            "acc1S_pct": ["50%", "75%"],
            "acc2S_pct": ["60%", "80%"],
            "impS_pct": ["10%", "5%"],
        }
    )
    df.to_csv(csv_path, index=False)

    loaded = impl._load_summary(csv_path)
    assert loaded["step"].tolist() == [1, 2]  # sorted
    assert loaded["acc1S_pct"].tolist() == [75.0, 50.0]
    assert loaded["impS_pct"].tolist() == [5.0, 10.0]


def test_build_phase_entropy_frame_expands_columns():
    summary_df = pd.DataFrame({"step": [0, 1], "t1": [1.0, 2.0], "a1": [3.0, 4.0], "t2": [5.0, 6.0], "a2": [7.0, 8.0]})
    out = impl._build_phase_entropy_frame(summary_df)
    # Should have four rows per input row (2 passes × 2 phases)
    assert len(out) == 8
    assert set(out["pass"]) == {"Pass 1", "Pass 2"}
    think_rows = out[out["phase"] == "Think"]
    assert think_rows.iloc[0]["entropy"] == 1.0 and think_rows.iloc[-1]["entropy"] == 6.0


class _FakeAxes:
    def __init__(self):
        self.labels = {}
        self.title = None

    def set_xlabel(self, x):
        self.labels["x"] = x

    def set_ylabel(self, y):
        self.labels["y"] = y

    def set_title(self, t):
        self.title = t

    def legend(self, *args, **kwargs):
        return None

    def grid(self, *args, **kwargs):
        return None

    def axhline(self, *args, **kwargs):
        return None

    def get_legend_handles_labels(self):
        return [], []


class _FakeFigure:
    def __init__(self, out_paths):
        self.out_paths = out_paths
        self.axis = _FakeAxes()

    def tight_layout(self):
        return None

    def savefig(self, path, **_kwargs):
        self.out_paths.append(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("stub")


class _FakePlt:
    def __init__(self, out_paths):
        self.out_paths = out_paths

    def subplots(self, *args, **kwargs):
        fig = _FakeFigure(self.out_paths)
        return fig, fig.axis

    def close(self, *_args, **_kwargs):
        return None


class _FakeSeaborn:
    def __init__(self):
        self.calls = []

    def set_theme(self, **kwargs):
        self.calls.append(("theme", kwargs))

    def set_palette(self, *args, **kwargs):
        self.calls.append(("palette", args, kwargs))

    def lineplot(self, *args, **kwargs):
        self.calls.append(("lineplot", args, kwargs))


class _FakePdfPages:
    def __init__(self, path):
        self.path = path
        self.saved = 0

    def savefig(self, _fig):
        self.saved += 1

    def close(self):
        return None


def test_make_plots_runs_all_paths(monkeypatch, tmp_path):
    out_paths = []
    fake_seaborn = _FakeSeaborn()
    fake_pdf = _FakePdfPages(tmp_path / "plots" / "combined.pdf")

    monkeypatch.setattr(impl, "_get_seaborn", lambda: fake_seaborn)
    monkeypatch.setattr(impl, "plt", _FakePlt(out_paths))
    monkeypatch.setattr(impl, "PdfPages", lambda path: fake_pdf)

    summary_df = pd.DataFrame(
        {
            "step": [1, 2],
            "acc1S_pct": [50.0, 60.0],
            "acc2S_pct": [55.0, 65.0],
            "acc1E_pct": [45.0, 55.0],
            "acc2E_pct": [50.0, 60.0],
            "ent1": [1.0, 0.9],
            "ent2": [0.8, 0.7],
            "t1": [1.1, 1.0],
            "a1": [0.9, 0.8],
            "t2": [0.7, 0.6],
            "a2": [0.5, 0.4],
            "impS_pct": [5.0, 6.0],
            "impE_pct": [4.0, 5.0],
            "tag1_pct": [90.0, 91.0],
            "tag2_pct": [92.0, 93.0],
            "n1S": [100, 110],
            "n2S": [120, 130],
        }
    )

    impl.make_plots(summary_df, outdir=str(tmp_path / "plots"), pdf_path=str(tmp_path / "plots" / "combined.pdf"))

    # Verify figures were saved and PDF collected pages
    assert len(out_paths) >= 1
    assert fake_pdf.saved >= 1
    # Seaborn helpers invoked (theme + several lineplot calls)
    assert any(call[0] == "theme" for call in fake_seaborn.calls)
    assert any(call[0] == "lineplot" for call in fake_seaborn.calls)


def test_get_seaborn_raises_runtime_on_import_error(monkeypatch):
    monkeypatch.setattr(importlib, "import_module", lambda name: (_ for _ in ()).throw(ImportError("no seaborn")))
    with pytest.raises(RuntimeError):
        impl._get_seaborn()


def test_plot_functions_short_circuit_when_columns_missing(monkeypatch):
    df = pd.DataFrame({"step": [1]})
    called = []

    def fail_line(*args, **kwargs):
        raise AssertionError("line should not be called when columns missing")

    monkeypatch.setattr(impl, "_line", fail_line)

    def save_page(_fig, name):
        called.append(name)

    impl._plot_sample_accuracy(df, save_page)
    impl._plot_example_accuracy(df, save_page)
    impl._plot_delta_accuracy(pd.DataFrame({}), save_page)
    impl._plot_entropy_overall(df, save_page)
    impl._plot_entropy_phase(df, save_page)
    impl._plot_improvement_rates(df, save_page)
    impl._plot_tag_validity(df, save_page)
    impl._plot_sample_counts(df, save_page)
    assert called == []


def test_main_with_pdf_prints(monkeypatch, tmp_path, capsys):
    csv_path = tmp_path / "summary.csv"
    pd.DataFrame({"step": [0]}).to_csv(csv_path, index=False)
    called = {}

    def fake_make_plots(df, outdir, pdf_path=None):
        called["outdir"] = outdir
        called["pdf"] = pdf_path

    monkeypatch.setattr(impl, "make_plots", fake_make_plots)
    monkeypatch.setattr(sys, "argv", ["prog", str(csv_path), "--pdf"])
    impl.main()
    out = capsys.readouterr().out
    assert "Saved plots" in out and "Wrote combined PDF" in out
    assert called["pdf"].endswith("summary_plots.pdf")
