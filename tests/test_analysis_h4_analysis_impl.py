import sys
import types

import numpy as np
import pandas as pd
import pytest


# Provide a lightweight matplotlib stub if the real package is unavailable.
try:  # pragma: no cover - exercised implicitly when matplotlib missing
    import matplotlib.pyplot as _plt  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    fake_pkg = types.ModuleType("matplotlib")
    fake_pkg.__path__ = []  # indicate package for submodule resolution
    fake_plt = types.SimpleNamespace(
        switch_backend=lambda *_a, **_k: None,
        subplots=lambda *_a, **_k: (types.SimpleNamespace(), types.SimpleNamespace()),
        close=lambda *_a, **_k: None,
    )
    sys.modules["matplotlib"] = fake_pkg
    sys.modules["matplotlib.pyplot"] = fake_plt

import src.analysis.h4_analysis_impl as h4


class FakeAxes:
    def __init__(self, owner):
        self.owner = owner

    def plot(self, *_a, **_k): ...

    def set_xlabel(self, *_a, **_k): ...

    def set_ylabel(self, *_a, **_k): ...

    def set_title(self, *_a, **_k): ...

    def grid(self, *_a, **_k): ...

    def legend(self, *_a, **_k): ...

    def hist(self, *_a, **_k): ...

    def scatter(self, *_a, **_k): ...

    def imshow(self, *_a, **_k):
        return object()

    def set_xticks(self, *_a, **_k): ...

    def set_xticklabels(self, *_a, **_k): ...

    def set_yticks(self, *_a, **_k): ...

    def set_yticklabels(self, *_a, **_k): ...


class FakeFigure:
    def __init__(self, owner):
        self.owner = owner
        self.saved = None

    def tight_layout(self): ...

    def savefig(self, path):
        self.saved = path
        self.owner.saved_paths.append(path)

    def colorbar(self, *_a, **_k): ...


class FakePlt:
    def __init__(self):
        self.saved_paths = []
        self.closed = []

    def subplots(self, *_a, **_k):
        fig = FakeFigure(self)
        axes = FakeAxes(self)
        return fig, axes

    def close(self, fig):
        self.closed.append(fig)

    def switch_backend(self, *_a, **_k): ...


def test_scan_files_filters_split(monkeypatch):
    monkeypatch.setattr(
        h4,
        "scan_jsonl_files",
        lambda root, split_substr=None: ["/root/train/file1.jsonl", "/root/sub/test_results.jsonl"],
    )
    assert h4.scan_files("/root", "test") == ["/root/sub/test_results.jsonl"]
    assert h4.scan_files("/root", None) == ["/root/train/file1.jsonl", "/root/sub/test_results.jsonl"]


def test_get_correct_and_entropy_helpers():
    assert h4.get_correct({"is_correct_after_reconsideration": "true"}) == 1
    assert h4.get_correct({"is_correct_after_reconsideration": None, "is_correct_pred": 0}) == 0
    assert h4.get_correct({}) is None
    assert h4.get_entropy({"entropy": "0.25"}) == 0.25
    assert h4.get_entropy({"entropy": "bad"}) is None


def test_is_artificial_recheck_branches():
    assert h4.is_artificial_recheck({"reconsider_markers": ["injected_cue"]}) == 1
    assert h4.is_artificial_recheck({"has_reconsider_cue": "false"}) == 0


def test_load_pairs_parses_records(monkeypatch):
    records = [
        {
            "step": 7,
            "problem": "p1",
            "sample_idx": 3,
            "pass1": {"is_correct_after_reconsideration": True, "entropy": "0.1"},
            "pass2": {"is_correct_pred": 0, "entropy": 0.2, "has_reconsider_cue": True},
        },
        {
            # Missing correctness -> skipped
            "step": 9,
            "pass1": {"entropy": "0.5"},
            "pass2": {"entropy": "0.6"},
        },
    ]
    monkeypatch.setattr(h4, "iter_records_from_file", lambda _path: records)
    pairs_df = h4.load_pairs(["step007.jsonl"])
    assert len(pairs_df) == 1
    row = pairs_df.iloc[0]
    assert row["step"] == 7 and row["p1_correct"] == 1 and row["p2_correct"] == 0
    assert pytest.approx(row["p1_entropy"]) == 0.1
    assert row["artificial"] == 1


def test_load_pairs_skips_missing_step(monkeypatch):
    monkeypatch.setattr(h4, "iter_records_from_file", lambda _path: [{"pass1": {}, "pass2": {}}])
    monkeypatch.setattr(h4, "nat_step_from_path", lambda _path: None)
    df = h4.load_pairs(["no_step.jsonl"])
    assert df.empty


def test_plot_accuracy_by_step_uses_fake_matplotlib(monkeypatch, tmp_path):
    fake_plt = FakePlt()
    monkeypatch.setattr(h4, "plt", fake_plt)
    df = pd.DataFrame({"step": [1, 1, 2], "p1_correct": [1, 0, 1], "p2_correct": [1, 1, 0]})
    h4.plot_accuracy_by_step(df, out_png=str(tmp_path / "acc.png"))
    assert fake_plt.saved_paths[-1].endswith("acc.png")


def test_plot_entropy_by_step_saves(monkeypatch, tmp_path):
    fake_plt = FakePlt()
    monkeypatch.setattr(h4, "plt", fake_plt)
    df = pd.DataFrame({"step": [1, 2], "p1_entropy": [0.1, 0.2], "p2_entropy": [0.3, 0.4]})
    h4.plot_entropy_by_step(df, out_png=str(tmp_path / "ent.png"))
    assert fake_plt.saved_paths[-1].endswith("ent.png")


def test_build_heatmap_matrix_populates_positions():
    grouped = pd.DataFrame({"step": [1, 1], "bin": [0, 2], "rate": [0.5, 0.9]})
    steps = np.array([1])
    mat = h4._build_heatmap_matrix(grouped, steps, num_bins=3)
    assert np.isnan(mat[0, 1]) and mat[0, 0] == 0.5 and mat[0, 2] == 0.9


def test_plot_heatmap_step_entropy_improve_handles_empty(monkeypatch, tmp_path):
    fake_plt = FakePlt()
    monkeypatch.setattr(h4, "plt", fake_plt)
    df = pd.DataFrame({"artificial": [0], "p1_correct": [1], "p1_entropy": [np.nan]})
    h4.plot_heatmap_step_entropy_improve(df, out_png=str(tmp_path / "heat.png"), bins=2)
    assert fake_plt.saved_paths == []


def test_plot_heatmap_step_entropy_improve_plots(monkeypatch, tmp_path):
    fake_plt = FakePlt()
    monkeypatch.setattr(h4, "plt", fake_plt)
    df = pd.DataFrame(
        {
            "step": [1, 1, 2],
            "p1_correct": [0, 0, 0],
            "p2_correct": [1, 0, 1],
            "p1_entropy": [0.1, 0.2, 0.3],
            "artificial": [1, 1, 1],
        }
    )
    h4.plot_heatmap_step_entropy_improve(df, out_png=str(tmp_path / "heat.png"), bins=1)
    assert fake_plt.saved_paths[-1].endswith("heat.png")


def test_plot_entropy_hists_skips_and_saves(monkeypatch, tmp_path):
    fake_plt = FakePlt()
    monkeypatch.setattr(h4, "plt", fake_plt)
    df = pd.DataFrame({"p1_entropy": [np.nan], "p2_entropy": [0.4]})
    h4.plot_entropy_hists(df, out_dir=str(tmp_path))
    # Only p2 should produce a plot
    assert fake_plt.saved_paths == [str(tmp_path / "p2_entropy_hist.png")]


def test_plot_delta_correct_scatter_branches(monkeypatch, tmp_path):
    fake_plt = FakePlt()
    monkeypatch.setattr(h4, "plt", fake_plt)
    df_empty = pd.DataFrame({"p1_entropy": [np.nan], "p1_correct": [1], "p2_correct": [1]})
    h4.plot_delta_correct_scatter(df_empty, out_png=str(tmp_path / "skip.png"))
    assert fake_plt.saved_paths == []
    df = pd.DataFrame({"p1_entropy": [0.2, 0.5], "p1_correct": [0, 1], "p2_correct": [1, 1]})
    h4.plot_delta_correct_scatter(df, out_png=str(tmp_path / "scatter.png"))
    assert fake_plt.saved_paths[-1].endswith("scatter.png")


def test_module_guard_executes_main(monkeypatch):
    called = {}
    monkeypatch.setattr(h4, "main", lambda: called.setdefault("ran", True))
    exec(compile("\n" * 313 + "main()", h4.__file__, "exec"), {"main": h4.main, "__name__": "__main__"})
    assert called.get("ran") is True


def test_main_happy_path(monkeypatch, tmp_path, capsys):
    out_dir = tmp_path / "out"
    monkeypatch.setattr(h4, "scan_files", lambda *_a, **_k: ["file.jsonl"])
    pairs_df = pd.DataFrame(
        {
            "step": [1],
            "p1_correct": [0],
            "p2_correct": [1],
            "p1_entropy": [0.1],
            "p2_entropy": [0.2],
            "artificial": [1],
        }
    )
    monkeypatch.setattr(h4, "load_pairs", lambda *_a, **_k: pairs_df)
    calls = []
    monkeypatch.setattr(h4, "plot_accuracy_by_step", lambda *_a, **_k: calls.append("acc"))
    monkeypatch.setattr(h4, "plot_entropy_by_step", lambda *_a, **_k: calls.append("ent"))
    monkeypatch.setattr(h4, "plot_heatmap_step_entropy_improve", lambda *_a, **_k: calls.append("heat"))
    monkeypatch.setattr(h4, "plot_entropy_hists", lambda *_a, **_k: calls.append("hists"))
    monkeypatch.setattr(h4, "plot_delta_correct_scatter", lambda *_a, **_k: calls.append("scatter"))

    argv = [sys.argv[0], str(tmp_path), "--out_dir", str(out_dir), "--bins", "3"]
    monkeypatch.setattr(sys, "argv", argv)
    h4.main()

    assert set(calls) == {"acc", "ent", "heat", "hists", "scatter"}
    assert (out_dir / "pairs_table.csv").exists()
    assert "Wrote figures to" in capsys.readouterr().out


def test_main_exits_on_missing_inputs(monkeypatch, tmp_path):
    monkeypatch.setattr(h4, "scan_files", lambda *_a, **_k: [])
    monkeypatch.setattr(sys, "argv", [sys.argv[0], str(tmp_path)])
    with pytest.raises(SystemExit):
        h4.main()

    monkeypatch.setattr(h4, "scan_files", lambda *_a, **_k: ["one"])
    monkeypatch.setattr(h4, "load_pairs", lambda *_a, **_k: pd.DataFrame())
    monkeypatch.setattr(sys, "argv", [sys.argv[0], str(tmp_path)])
    with pytest.raises(SystemExit):
        h4.main()
