#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import src.analysis.graph_3_stacked as g3s


def test_detect_aha_and_extract_step(monkeypatch):
    monkeypatch.setattr(g3s, "aha_gpt", lambda pass1, rec, mode, gate_by_words: True)
    rec = {"pass1": {}}
    assert g3s.detect_aha_pass1(rec, "canonical") is True
    assert g3s.detect_aha_pass1({"pass1": "bad"}, "canonical") is False

    assert g3s.extract_step({"step": 5}, "p") == 5
    assert g3s.extract_step({}, "/path/step-42/file.jsonl") == 42
    assert g3s.extract_step({}, "/path/global_step123/file.jsonl") == 123
    assert g3s.extract_step({}, "no-step") == 0


def test_extract_entropy_pass1_and_iter_json(monkeypatch, tmp_path):
    rec = {"pass1": {"entropy_answer": 0.5}}
    assert g3s.extract_entropy_pass1(rec) == 0.5
    rec2 = {"pass1": {"answer_token_entropies": [0.0, 1.0]}}
    assert g3s.extract_entropy_pass1(rec2) == 0.5
    assert g3s.extract_entropy_pass1({"pass1": {}}) is None

    # iter_jsonl_files should ignore missing roots and return paths for existing dir
    sub = tmp_path / "a"
    sub.mkdir()
    file = sub / "f.jsonl"
    file.write_text("{}", encoding="utf-8")
    paths = list(g3s.iter_jsonl_files(str(sub)))
    assert str(file) in paths
    assert list(g3s.iter_jsonl_files("missing")) == []


def test_load_pass1_entropy_and_aha(monkeypatch, tmp_path):
    records = [
        {"split": "train", "pass1": {"entropy_answer": 0.1, "is_correct_pred": True}},
        {"split": "other", "pass1": {"entropy_answer": 0.2, "is_correct_pred": False}},  # filtered by split
    ]
    monkeypatch.setattr(g3s, "iter_jsonl_files", lambda root: ["dummy.jsonl"])
    monkeypatch.setattr(g3s, "iter_records_from_file", lambda path: records)
    monkeypatch.setattr(g3s, "detect_aha_pass1", lambda rec, mode: bool(rec["pass1"]["is_correct_pred"]))
    rows = g3s.load_pass1_entropy_and_aha(
        root="r",
        split="train",
        min_step=0,
        max_step=1000,
        gpt_mode="canonical",
    )
    assert rows == [(0.1, 1)]


def test_load_pass1_entropy_skips_missing_entropy(monkeypatch):
    records = [
        {"split": "train", "step": 1, "pass1": {}},  # triggers entropy None branch
        {"split": "train", "step": 1, "pass1": {"answer_entropy": 0.4}},
    ]
    monkeypatch.setattr(g3s, "iter_jsonl_files", lambda root: ["step-1.jsonl"])
    monkeypatch.setattr(g3s, "iter_records_from_file", lambda path: records)
    monkeypatch.setattr(g3s, "detect_aha_pass1", lambda rec, mode: False)
    rows = g3s.load_pass1_entropy_and_aha(
        root="r",
        split="train",
        min_step=0,
        max_step=10,
        gpt_mode="canonical",
    )
    assert rows == [(0.4, 0)]


def test_rows_to_arrays_and_binning():
    rows = [(0.1, 0), (0.9, 1)]
    ent, aha = g3s._rows_to_arrays(rows)
    assert ent.tolist() == [0.1, 0.9]
    centers, no_aha, aha_counts, width, ylabel = g3s._compute_binned_counts(
        ent,
        aha,
        num_bins=2,
        binning="fixed",
        normalize=False,
    )
    assert len(centers) == len(no_aha) == len(aha_counts) == 2
    assert ylabel == "Count"
    # At least one bin populated, totals do not exceed sample count.
    assert (no_aha + aha_counts).sum() <= len(rows)

    centers2, no_aha_norm, aha_norm, _, ylabel2 = g3s._compute_binned_counts(
        ent,
        aha,
        num_bins=2,
        binning="fixed",
        normalize=True,
    )
    assert ylabel2 == "Proportion (per bin)"
    totals = no_aha_norm + aha_norm
    assert totals[0] == pytest.approx(1.0)
    assert np.allclose(centers, centers2)


def test_compute_binned_from_rows_and_plot(monkeypatch, tmp_path):
    rows = [(0.1, 0), (0.2, 1), (0.3, 0)]
    args = SimpleNamespace(
        bins=2,
        binning="fixed",
        normalize=False,
        width_in=1,
        height_in=1,
        title="t",
        dpi=10,
        outdir=str(tmp_path),
        outfile_tag=None,
    )
    hist = g3s._compute_binned_from_rows(rows, args)
    assert isinstance(hist, g3s.BinnedHistogram)

    # Monkeypatch plt to avoid writing a real file
    class FakeAxes:
        def __init__(self):
            self.bar_calls = []

        def bar(self, *a, **k):
            self.bar_calls.append((a, k))

        def set_xlabel(self, *a, **k):
            pass

        def set_ylabel(self, *a, **k):
            pass

        def set_title(self, *a, **k):
            pass

        def set_ylim(self, *a, **k):
            pass

        def legend(self, *a, **k):
            pass

        def grid(self, *a, **k):
            pass

    class FakeFig:
        pass

    class FakePlt:
        def subplots(self, **kwargs):
            return FakeFig(), FakeAxes()

        def savefig(self, path, dpi=None):
            self.saved = path

    fake_plt = FakePlt()
    monkeypatch.setattr(g3s, "plt", fake_plt)
    g3s._plot_stacked_histogram(args, hist)
    assert hasattr(fake_plt, "saved")


def test_parse_args_defaults_and_normalize(monkeypatch):
    # Avoid pulling in real helper behavior; just add minimal placeholders.
    monkeypatch.setattr(
        g3s, "add_split_and_gpt_mode_args", lambda parser: parser.add_argument("--split", default="train")
    )
    monkeypatch.setattr(g3s, "add_binning_argument", lambda parser: parser.add_argument("--binning", default="fixed"))
    monkeypatch.setattr(sys, "argv", ["prog", "--split", "dev", "--normalize"])
    args = g3s.parse_args()
    assert args.split == "dev"
    assert args.normalize is True
    assert args.bins == 20 and args.outdir == "graphs"


def test_extract_entropy_pass1_non_dict_and_load_empty():
    assert g3s.extract_entropy_pass1({"pass1": "bad"}) is None
    assert g3s.load_pass1_entropy_and_aha("", split="train", min_step=0, max_step=1, gpt_mode="canon") == []


def test_compute_binned_counts_quantile_fallback():
    ent = np.array([1.0, 1.0, 1.0])
    aha = np.array([0, 1, 0])
    centers, no_aha, aha_counts, bin_width, ylabel = g3s._compute_binned_counts(
        ent,
        aha,
        num_bins=2,
        binning="quantile",
        normalize=False,
    )
    assert len(centers) == len(no_aha) == len(aha_counts)
    assert bin_width >= 0
    assert ylabel == "Count"


def test_plot_stacked_histogram_sets_ylim_when_normalized(monkeypatch, tmp_path):
    class AxisStub:
        def __init__(self):
            self.calls = {"set_ylim": 0}

        def bar(self, *a, **k):
            return None

        def set_xlabel(self, *a, **k):
            return None

        def set_ylabel(self, *a, **k):
            return None

        def set_title(self, *a, **k):
            return None

        def set_ylim(self, *a, **k):
            self.calls["set_ylim"] += 1

        def legend(self, *a, **k):
            return None

        def grid(self, *a, **k):
            return None

    class FakePlt:
        def __init__(self):
            self.axis = None

        def subplots(self, **kwargs):
            self.axis = AxisStub()
            return None, self.axis

        def savefig(self, *a, **k):
            return None

    hist = g3s.BinnedHistogram(
        centers=np.array([0.1, 0.2]),
        counts_no_aha=np.array([0.5, 0.5]),
        counts_aha=np.array([0.5, 0.5]),
        bin_width=0.1,
        ylabel="y",
    )
    args = SimpleNamespace(
        width_in=1.0,
        height_in=1.0,
        normalize=True,
        title="t",
        outdir=str(tmp_path),
        outfile_tag="tag",
        dpi=72,
    )
    fake_plt = FakePlt()
    monkeypatch.setattr(g3s, "plt", fake_plt)
    g3s._plot_stacked_histogram(args, hist)
    # Axis created inside FakePlt; ensure set_ylim was invoked.
    assert fake_plt.axis.calls["set_ylim"] >= 1


def test_main_exits_on_no_rows(monkeypatch, tmp_path):
    args = SimpleNamespace(
        root_carpark="c",
        root_crossword="x",
        root_math="m",
        split="dev",
        min_step=0,
        max_step=1,
        gpt_mode="canon",
        bins=2,
        binning="fixed",
        normalize=False,
        width_in=1.0,
        height_in=1.0,
        title="t",
        dpi=50,
        outdir=str(tmp_path),
        outfile_tag=None,
    )
    monkeypatch.setattr(g3s, "parse_args", lambda: args)
    monkeypatch.setattr(g3s, "load_pass1_entropy_and_aha", lambda *a, **k: [])
    with pytest.raises(SystemExit):
        g3s.main()


def test_main_calls_pipeline(monkeypatch, tmp_path):
    args = SimpleNamespace(
        root_carpark="c",
        root_crossword="x",
        root_math="m",
        split="dev",
        min_step=0,
        max_step=1,
        gpt_mode="canon",
        bins=2,
        binning="fixed",
        normalize=False,
        width_in=1.0,
        height_in=1.0,
        title="t",
        dpi=50,
        outdir=str(tmp_path),
        outfile_tag=None,
    )
    monkeypatch.setattr(g3s, "parse_args", lambda: args)
    returns = {(args.root_carpark): [(0.1, 0)], (args.root_crossword): [], (args.root_math): []}
    monkeypatch.setattr(
        g3s,
        "load_pass1_entropy_and_aha",
        lambda root, *a, **k: returns.get(root, []),
    )
    called = {}

    def fake_compute(rows, arg_obj):
        called["rows"] = rows
        return g3s.BinnedHistogram(
            centers=np.array([0.1]),
            counts_no_aha=np.array([1.0]),
            counts_aha=np.array([0.0]),
            bin_width=0.1,
            ylabel="Count",
        )

    monkeypatch.setattr(g3s, "_compute_binned_from_rows", fake_compute)

    def fake_plot(args_obj, hist_obj):
        called["plotted"] = hist_obj

    monkeypatch.setattr(g3s, "_plot_stacked_histogram", fake_plot)
    g3s.main()
    assert called["rows"] == [(0.1, 0)]
    assert isinstance(called["plotted"], g3s.BinnedHistogram)


def test_main_guard_runs_via_runpy(monkeypatch, capsys, tmp_path):
    # Run the module as a script to hit the __main__ guard and exit path.
    monkeypatch.delitem(sys.modules, "src.analysis.graph_3_stacked", raising=False)
    monkeypatch.setattr(sys, "argv", ["prog", "--outdir", str(tmp_path)])
    with pytest.raises(SystemExit):
        runpy.run_module("src.analysis.graph_3_stacked", run_name="__main__")
    assert "No PASS1 records" in capsys.readouterr().err
