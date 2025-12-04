import types

import numpy as np
import pytest

import src.analysis.graph_3_stacked as g3s


def test_detect_aha_pass1_uses_aha_gpt(monkeypatch):
    monkeypatch.setattr(g3s, "aha_gpt", lambda pass1, rec, mode, gate_by_words: 1)
    rec = {"pass1": {"flag": True}}
    assert g3s.detect_aha_pass1(rec, mode="broad") is True
    assert g3s.detect_aha_pass1({"pass1": "bad"}, mode="broad") is False


def test_extract_entropy_pass1_handles_token_entropy_and_invalid():
    rec = {"pass1": {"answer_token_entropies": [1, 2, 3]}}
    assert g3s.extract_entropy_pass1(rec) == pytest.approx(2.0)
    rec_bad = {"pass1": {"answer_token_entropies": ["x"]}}
    assert g3s.extract_entropy_pass1(rec_bad) is None


def test_compute_binned_counts_normalize_and_uniform():
    ent = np.array([0.0, 1.0, 2.0])
    aha = np.array([0, 1, 0])
    centers, no_aha, aha_counts, bin_width, ylabel = g3s._compute_binned_counts(
        ent,
        aha,
        num_bins=2,
        binning="uniform",
        normalize=True,
    )
    assert len(centers) == 2 and bin_width > 0
    assert np.isclose(no_aha.sum() + aha_counts.sum(), 2.0)  # two bins normalized
    assert ylabel.startswith("Proportion")


def test_iter_jsonl_files_handles_missing_root(tmp_path):
    assert list(g3s.iter_jsonl_files("")) == []
    assert list(g3s.iter_jsonl_files(str(tmp_path / "missing"))) == []


def test_plot_stacked_histogram_saves(monkeypatch, tmp_path):
    saved = {}

    class FakeAxes:
        def bar(self, *a, **k): ...
        def set_xlabel(self, *_): ...
        def set_ylabel(self, *_): ...
        def set_title(self, *_): ...
        def set_ylim(self, *_): ...
        def legend(self, *_, **__): ...
        def grid(self, *_, **__): ...

    def fake_subplots(*args, **kwargs):
        return types.SimpleNamespace(), FakeAxes()

    monkeypatch.setattr(
        g3s,
        "plt",
        types.SimpleNamespace(subplots=fake_subplots, savefig=lambda path, dpi=None: saved.update({"path": path})),
    )

    args = types.SimpleNamespace(
        width_in=1,
        height_in=1,
        title="t",
        normalize=True,
        outdir=str(tmp_path),
        outfile_tag="tag",
        dpi=100,
    )
    hist = g3s.BinnedHistogram(
        centers=np.array([0.5]),
        counts_no_aha=np.array([1.0]),
        counts_aha=np.array([0.0]),
        bin_width=0.1,
        ylabel="y",
    )
    g3s._plot_stacked_histogram(args, hist)
    assert "path" in saved and saved["path"].endswith(".png")


def test_main_exits_when_no_rows(monkeypatch, tmp_path):
    args = types.SimpleNamespace(
        outdir=str(tmp_path),
        outfile_tag=None,
        normalize=False,
        bins=2,
        binning="uniform",
        split=None,
        min_step=0,
        max_step=10,
        gpt_mode="canonical",
        root_carpark=None,
        root_crossword=None,
        root_math=None,
        title="t",
        dpi=100,
        width_in=1.0,
        height_in=1.0,
    )
    monkeypatch.setattr(g3s, "parse_args", lambda: args)
    monkeypatch.setattr(g3s, "load_pass1_entropy_and_aha", lambda *a, **k: [])
    with pytest.raises(SystemExit):
        g3s.main()
