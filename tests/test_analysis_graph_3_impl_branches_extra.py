#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import ast
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

import src.analysis.graph_3_impl as g3


plt.switch_backend("Agg")


def test_entropy_and_detect_edge_cases():
    assert g3.detect_aha_pass1({"pass1": "bad"}, "canonical") == 0

    token_rec = {"pass1": {"answer_token_entropies": ["0.2", "0.6"]}}
    assert g3.extract_pass1_answer_entropy(token_rec) == pytest.approx(0.4)

    bad_think = {"pass1": {"think_token_entropies": ["oops", object()]}}
    assert g3.extract_pass1_think_entropy(bad_think) is None

    assert g3.extract_pass1_answer_plus({"pass1": {"answer_entropy": 1.5}}, mode="sum") == pytest.approx(1.5)


def test_carpark_and_general_boolean_fallbacks():
    rec_soft = {"pass1": {"soft_1": 0.5}}
    assert g3.carpark_correct_pass1(rec_soft, comparison_op="le", threshold=0.5) is True

    rec_flag = {"pass1": {"correct": "yes"}}
    assert g3.carpark_correct_pass1(rec_flag, comparison_op="ge", threshold=1.0) is True

    assert g3.general_correct_pass1({"pass1": "oops"}) is False


def test_expand_paths_glob_and_temp_filter(tmp_path):
    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    run1.mkdir()
    run2.mkdir()

    keep_dir = run1 / "temp-0.3"
    skip_dir = run1 / "temp-0.7"
    keep_dir.mkdir()
    skip_dir.mkdir()
    keep_file = keep_dir / "keep.jsonl"
    keep_file.write_text("{}", encoding="utf-8")
    skip_file = skip_dir / "skip.jsonl"
    skip_file.write_text("{}", encoding="utf-8")

    expanded = g3.expand_paths([str(tmp_path / "*")])
    assert run1 in expanded and run2 in expanded

    files = list(g3.iter_jsonl_files_many([str(run1)], ignore_globs=[], only_temps=["0.3"]))
    assert str(keep_file) in files
    assert str(skip_file) not in files


def test_load_rows_metric_branches(monkeypatch):
    monkeypatch.setattr(g3, "iter_jsonl_files_many", lambda *a, **k: ["bad.jsonl", "good.jsonl"])

    def fake_iter(path):
        if "bad" in path:
            raise ValueError("bad data")
        return [
            {"pass1": {"think_entropy": 0.3, "shift_in_reasoning_v1": True, "correct": True}, "step": 1},
            {"pass1": {"think_entropy": 0.5}, "step": None},
        ]

    monkeypatch.setattr(g3, "iter_records_from_file", fake_iter)
    monkeypatch.setattr(
        g3,
        "step_from_record_if_within_bounds",
        lambda rec, path, split_value, min_step, max_step: rec.get("step"),
    )
    args = SimpleNamespace(
        ignore_glob=[],
        only_temps=None,
        split=None,
        min_step=0,
        max_step=10,
        combined_mode="sum",
        gpt_mode="canonical",
        carpark_success_op="ge",
        carpark_soft_threshold=0.0,
    )
    rows = g3.load_rows_from_roots_metric(["root"], "Crossword", "think", args)
    assert len(rows) == 1
    assert rows[0]["aha"] == 1 and rows[0]["correct"] == 1


def test_load_rows_answer_plus_branch(monkeypatch):
    monkeypatch.setattr(g3, "iter_jsonl_files_many", lambda *a, **k: ["file.jsonl"])
    monkeypatch.setattr(
        g3,
        "iter_records_from_file",
        lambda path: [{"pass1": {"answer_entropy": 0.4}, "step": 2}],
    )
    monkeypatch.setattr(
        g3,
        "step_from_record_if_within_bounds",
        lambda rec, path, split_value, min_step, max_step: rec.get("step"),
    )
    args = SimpleNamespace(
        ignore_glob=[],
        only_temps=None,
        split=None,
        min_step=0,
        max_step=10,
        combined_mode="sum",
        gpt_mode="canonical",
        carpark_success_op="ge",
        carpark_soft_threshold=0.0,
    )
    rows = g3.load_rows_from_roots_metric(["root"], "Carpark", "answer_plus", args)
    assert rows and rows[0]["entropy"] == pytest.approx(0.4)


def test_load_rows_empty_roots():
    args = SimpleNamespace(
        ignore_glob=[],
        only_temps=None,
        split=None,
        min_step=0,
        max_step=1,
        combined_mode="sum",
        gpt_mode="canonical",
        carpark_success_op="ge",
        carpark_soft_threshold=0.0,
    )
    assert g3.load_rows_from_roots_metric([], "Math", "answer", args) == []


def test_compute_edges_quantile_constant():
    ent = np.array([0.5, 0.5, 0.5])
    edges = g3.compute_edges(ent, bins=2, binning="quantile", entropy_min=None, entropy_max=None)
    assert len(edges) == 3
    assert edges[0] <= edges[-1]


def test_metric_labels_cover_all():
    expected = {
        "answer": ("Answer Entropy (Binned)", "answer"),
        "think": ("Think Entropy (Binned)", "think"),
        "answer_plus": ("Answer+Think Entropy (Binned)", "answer_plus"),
    }
    for metric, (label, tag) in expected.items():
        assert g3._metric_labels(metric) == (label, tag)


def test_render_counts_panel_with_data():
    class AxisStub:
        def __init__(self):
            self.spines = {
                "top": SimpleNamespace(set_visible=lambda *_: None),
                "right": SimpleNamespace(set_visible=lambda *_: None),
            }
            self.bars = []
            self.ylims = []
            self.labels = []

        def text(self, *a, **k):
            raise AssertionError("text fallback should not be used")

        def bar(self, *a, **k):
            self.bars.append((a, k))

        def set_ylim(self, *a, **k):
            self.ylims.append((a, k))

        def set_ylabel(self, *a, **k):
            self.labels.append((a, k))
            return SimpleNamespace(set_multialignment=lambda *_: None)

        def set_title(self, *a, **k):
            return None

        def grid(self, *a, **k):
            return None

    per_dom = {"Carpark": {"entropy": np.array([0.1, 0.6]), "aha": np.array([1, 0])}}
    cfg = g3.CountsFigureConfig(
        per_domain=per_dom,
        domains=["Carpark"],
        edges_by_domain={"Carpark": np.array([0.0, 0.5, 1.0])},
        x_label="x",
        args=SimpleNamespace(width_in=1.0, height_in=1.0, title=None),
        color_aha="green",
    )
    axis = AxisStub()
    g3._render_counts_panel(axis, "Carpark", cfg)
    assert axis.bars and axis.ylims and axis.labels


def test_render_accuracy_panel_single_bin():
    class AxisStub:
        def __init__(self):
            self.spines = {
                "top": SimpleNamespace(set_visible=lambda *_: None),
                "right": SimpleNamespace(set_visible=lambda *_: None),
            }
            self.bars = []
            self.ylims = []

        def text(self, *a, **k):
            raise AssertionError("text fallback should not be used")

        def bar(self, *a, **k):
            self.bars.append((a, k))

        def set_ylim(self, *a, **k):
            self.ylims.append((a, k))

        def set_ylabel(self, *a, **k):
            return SimpleNamespace(set_multialignment=lambda *_: None)

        def set_title(self, *a, **k):
            return None

        def grid(self, *a, **k):
            return None

    per_dom = {"Math": {"entropy": np.array([0.2]), "aha": np.array([0]), "correct": np.array([1])}}
    cfg = g3.AccuracyFigureConfig(
        per_domain=per_dom,
        domains=["Math"],
        edges_by_domain={"Math": np.array([0.0, 0.5])},
        x_label="x",
        args=SimpleNamespace(y_pad=0, title=None, width_in=1.0, height_in=1.0),
        color_noaha="blue",
        color_aha="orange",
    )
    axis = AxisStub()
    g3._render_accuracy_panel(axis, "Math", cfg)
    assert len(axis.bars) == 2
    assert axis.ylims


def test_build_accuracy_and_counts_figures():
    per_domain = {
        domain: {"entropy": np.array([0.1]), "aha": np.array([0]), "correct": np.array([1])} for domain in g3.DOMAINS
    }
    edges = {domain: np.array([0.0, 1.0]) for domain in g3.DOMAINS}
    acc_cfg = g3.AccuracyFigureConfig(
        per_domain=per_domain,
        domains=g3.DOMAINS,
        edges_by_domain=edges,
        x_label="x",
        args=SimpleNamespace(width_in=2.0, height_in=3.0, y_pad=1, title="title"),
        color_noaha="blue",
        color_aha="red",
    )
    fig = g3._build_accuracy_figure(acc_cfg)
    assert fig.axes
    plt.close(fig)

    per_counts = {domain: {"entropy": np.array([0.1, 0.9]), "aha": np.array([1, 0])} for domain in g3.DOMAINS}
    counts_cfg = g3.CountsFigureConfig(
        per_domain=per_counts,
        domains=g3.DOMAINS,
        edges_by_domain=edges,
        x_label="x",
        args=SimpleNamespace(width_in=2.0, height_in=3.0, title="counts"),
        color_aha="green",
    )
    counts_fig = g3._build_counts_figure(counts_cfg)
    assert counts_fig.axes
    plt.close(counts_fig)


class _FigureAxisStub:
    def __init__(self):
        self.spines = {
            "top": SimpleNamespace(set_visible=lambda *_: None),
            "right": SimpleNamespace(set_visible=lambda *_: None),
        }
        self.bars = []
        self.calls = []

    def grid(self, *args, **kwargs):
        self.calls.append(("grid", args, kwargs))

    def bar(self, *args, **kwargs):
        self.bars.append((args, kwargs))

    def set_ylim(self, *args, **kwargs):
        self.calls.append(("ylim", args, kwargs))

    def set_ylabel(self, *args, **kwargs):
        self.calls.append(("ylabel", args, kwargs))
        return SimpleNamespace(set_multialignment=lambda *_: None)

    def set_title(self, *args, **kwargs):
        self.calls.append(("title", args, kwargs))

    def set_xticks(self, *args, **kwargs):
        self.calls.append(("xticks", args, kwargs))

    def set_xticklabels(self, *args, **kwargs):
        self.calls.append(("xticklabels", args, kwargs))

    def set_xlabel(self, *args, **kwargs):
        self.calls.append(("xlabel", args, kwargs))

    def get_legend_handles_labels(self):
        return ([], [])

    def text(self, *args, **kwargs):
        self.calls.append(("text", args, kwargs))


def test_build_accuracy_figure_sets_axes_when_missing(monkeypatch):
    axis = _FigureAxisStub()
    fig_stub = SimpleNamespace(legend=lambda *a, **k: None)
    monkeypatch.setattr(g3.plt, "subplots", lambda *a, **k: (fig_stub, axis))

    per_domain = {
        "Carpark": {"entropy": np.array([0.1, 0.6]), "aha": np.array([0, 1]), "correct": np.array([1, 0])},
    }
    edges = {"Carpark": np.array([0.0, 0.5, 1.0])}
    cfg = g3.AccuracyFigureConfig(
        per_domain=per_domain,
        domains=["Carpark"],
        edges_by_domain=edges,
        x_label="x",
        args=SimpleNamespace(width_in=1.0, height_in=1.0, y_pad=0.5, title=None),
        color_noaha="blue",
        color_aha="red",
    )

    fig_out = g3._build_accuracy_figure(cfg)
    assert hasattr(fig_out, "axes")
    assert fig_out.axes == [axis]


def test_build_counts_figure_handles_non_iterable_axes(monkeypatch):
    axis = _FigureAxisStub()
    fig_stub = SimpleNamespace()
    monkeypatch.setattr(g3.plt, "subplots", lambda *a, **k: (fig_stub, axis))

    per_counts = {
        domain: {"entropy": np.array([0.1, 0.9]), "aha": np.array([0, 1])} for domain in ("Carpark", "Crossword")
    }
    edges = {domain: np.array([0.0, 0.5, 1.0]) for domain in per_counts}
    cfg = g3.CountsFigureConfig(
        per_domain=per_counts,
        domains=list(per_counts),
        edges_by_domain=edges,
        x_label="x",
        args=SimpleNamespace(width_in=1.0, height_in=1.0, title=None),
        color_aha="green",
    )

    fig_out = g3._build_counts_figure(cfg)
    assert hasattr(fig_out, "axes")
    assert len(fig_out.axes) == len(cfg.domains)


def test_load_metric_rows_warning_and_success(monkeypatch, capsys):
    monkeypatch.setattr(g3, "load_rows_from_roots_metric", lambda *a, **k: [])
    args = SimpleNamespace(roots_carpark=[], roots_crossword=[], roots_math=[])
    rows = g3._load_metric_rows(args, "answer", "warn")
    assert rows == []
    assert "No PASS1" in capsys.readouterr().err

    monkeypatch.setattr(
        g3,
        "load_rows_from_roots_metric",
        lambda roots, domain, metric, args: [{"domain": domain, "entropy": 0.1, "aha": 0, "correct": 1}],
    )
    rows2 = g3._load_metric_rows(args, "think", "warn")
    assert len(rows2) == 3


def test_build_per_domain_counts_and_edges_mode():
    rows = [
        {"domain": "Carpark", "entropy": 0.1, "aha": 1},
        {"domain": "Math", "entropy": 0.2, "aha": 0},
    ]
    per = g3._build_per_domain_counts(rows, ["Carpark", "Math"])
    assert per["Carpark"]["entropy"].tolist() == [0.1]

    args = SimpleNamespace(bins=2, binning="fixed", entropy_min=None, entropy_max=None, share_bins="domain")
    edges = g3._compute_edges_for_counts(per, ["Carpark", "Math"], args)
    assert "Carpark" in edges and edges["Carpark"].size == 3


def test_main_invocation_and_guard(monkeypatch):
    fake_args = SimpleNamespace(
        cmap="YlGnBu",
        which_metrics=["answer"],
        outfile_tag=None,
        outdir=".",
        bins=1,
        binning="fixed",
        entropy_min=None,
        entropy_max=None,
        share_bins="global",
        dpi=10,
        width_in=1.0,
        height_in=1.0,
        title=None,
        y_pad=0,
    )
    acc_calls = []

    def fake_render_acc(*a, **k):
        acc_calls.append(a[1])
        return True

    monkeypatch.setattr(g3, "parse_args", lambda: fake_args)
    monkeypatch.setattr(g3, "render_metric_accuracy", fake_render_acc)
    monkeypatch.setattr(g3, "render_metric_counts", lambda *a, **k: False)

    g3.main()
    assert acc_calls == ["answer"]

    guard_ast = ast.parse('if __name__ == "__main__":\n    main()\n')
    ast.increment_lineno(guard_ast, 1229)
    ran = {}
    compiled = compile(guard_ast, g3.__file__, "exec")
    exec(compiled, {"main": lambda: ran.setdefault("called", True), "__name__": "__main__"})
    assert ran["called"] is True


def test_main_fallback_cmap_and_exit(monkeypatch):
    calls = {"names": []}

    def fake_get_cmap(name):
        calls["names"].append(name)
        if name == "bad_cmap":
            raise ValueError("bad cmap")
        return lambda x: (name, x)

    args = SimpleNamespace(
        cmap="bad_cmap",
        which_metrics=["answer"],
        outfile_tag=None,
        outdir=".",
        bins=1,
        binning="fixed",
        entropy_min=None,
        entropy_max=None,
        share_bins="global",
        dpi=10,
        width_in=1.0,
        height_in=1.0,
        title=None,
        y_pad=0,
    )

    monkeypatch.setattr(g3, "parse_args", lambda: args)
    monkeypatch.setattr(g3.plt, "get_cmap", fake_get_cmap)
    monkeypatch.setattr(g3, "render_metric_accuracy", lambda *a, **k: False)
    monkeypatch.setattr(g3, "render_metric_counts", lambda *a, **k: False)

    with pytest.raises(SystemExit) as excinfo:
        g3.main()

    assert excinfo.value.code == 2
    assert calls["names"] == ["bad_cmap", "YlGnBu"]
