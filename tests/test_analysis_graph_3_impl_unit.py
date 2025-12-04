#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import src.analysis.graph_3_impl as g3


def test_detect_aha_pass1_and_extract_step():
    rec = {"pass1": {"shift_in_reasoning_v1": True}}
    assert g3.detect_aha_pass1(rec, "canonical") == 1
    assert g3.detect_aha_pass1({}, "canonical") == 0
    # explicit step wins over path
    assert g3.extract_step({"step": 7}, "path/step123.jsonl") == 7
    assert g3.extract_step({}, "path/global_step-42.jsonl") == 42
    assert g3.extract_step({}, "no-step") == 0


def test_entropy_extractors_and_combined_mode():
    rec = {"pass1": {"entropy_answer": 0.5, "think_token_entropies": [1, 3]}}
    assert g3.extract_pass1_answer_entropy(rec) == 0.5
    rec2 = {"pass1": {"think_token_entropies": [1.0, 3.0]}}
    assert g3.extract_pass1_think_entropy(rec2) == 2.0
    rec3 = {"pass1": {"entropy_answer": 1.0, "entropy_think": 2.0}}
    assert g3.extract_pass1_answer_plus(rec3, mode="sum") == 3.0
    assert g3.extract_pass1_answer_plus(rec3, mode="mean") == 1.5
    assert g3.extract_pass1_answer_plus({"pass1": {}}, mode="sum") is None


def test_correctness_helpers_carpark_and_general():
    rec = {"pass1": {"soft_reward": 0.5}}
    assert g3.carpark_correct_pass1(rec, comparison_op="gt", threshold=0.4) is True
    assert g3.carpark_correct_pass1(rec, comparison_op="lt", threshold=0.4) is False
    # Fallback to boolean flags
    rec2 = {"pass1": {"is_correct_pred": True}}
    assert g3.general_correct_pass1(rec2) is True
    assert g3.general_correct_pass1({"pass1": {}}) is False


def test_temp_match_and_expand_paths(tmp_path):
    assert g3.temp_match("a/temp-0.30/run", ["0.3"]) is True
    assert g3.temp_match("a/temp-0.50/run", ["0.3"]) is False
    # expand_paths should ignore files and keep dirs
    dir1 = tmp_path / "d1"
    dir1.mkdir()
    file1 = tmp_path / "f.txt"
    file1.write_text("x")
    paths = g3.expand_paths([str(dir1), str(file1)])
    assert dir1 in paths
    assert file1 not in paths


def test_compute_edges_quantile_and_fixed():
    ent = np.array([0.1, 0.2, 0.3])
    edges = g3.compute_edges(ent, bins=2, binning="quantile", entropy_min=None, entropy_max=None)
    assert len(edges) == 3
    fixed = g3.compute_edges(ent, bins=2, binning="fixed", entropy_min=0.0, entropy_max=1.0)
    assert np.allclose(fixed, np.array([0.0, 0.5, 1.0]))


def test_binned_accuracy_and_counts():
    ent = np.array([0.1, 0.2, 0.9, 1.0])
    aha = np.array([0, 1, 0, 1])
    corr = np.array([1, 0, 1, 0])
    edges = np.array([0.0, 0.5, 1.0])
    centers, acc = g3.binned_accuracy(ent, aha, corr, edges, aha_flag=0)
    assert len(centers) == len(acc) == 2
    assert acc[0] == 1.0  # bin1 aha=0 correct=1
    centers2, counts = g3.binned_aha_counts(ent, aha, edges)
    assert counts.sum() == 2
    assert np.array_equal(centers, centers2)


def test_compute_bin_table_with_totals():
    bin_inputs = g3.BinInputs(
        entropies=np.array([0.1, 0.6]),
        aha_flags=np.array([0, 1]),
        correctness=np.array([1.0, 0.0]),
        edges=np.array([0.0, 0.5, 1.0]),
    )
    table = g3.compute_bin_table("metric", "scope", "domain", bin_inputs)
    # Two bins plus total row
    assert len(table) == 3
    total = table.iloc[-1]
    assert total["n_total"] == 2
    assert total["n_aha"] == 1
    assert total["acc_noshift"] == 1.0
    assert table.iloc[0]["bin_label"].startswith("[")


def test_parse_args_defaults_and_num_helpers(monkeypatch):
    argv_backup = sys.argv
    sys.argv = ["prog"]
    args = g3.parse_args()
    sys.argv = argv_backup
    assert args.ignore_glob == ["*compare-1shot*"]
    assert args.which_metrics == ["answer", "think", "answer_plus"]

    assert g3._num("not-a-dict", ["a"]) is None
    rec = {"pass1": {"answer_token_entropies": ["bad"]}}
    assert g3.extract_pass1_answer_entropy(rec) is None
    rec2 = {"pass1": {"think_entropy": 1.2}}
    assert g3.extract_pass1_answer_plus(rec2, mode="sum") == 1.2


def test_carpark_correct_pass1_defaults_and_general_flags():
    rec = {"pass1": {"soft_1": 0.2}}
    # Unknown op falls back to "gt"
    assert g3.carpark_correct_pass1(rec, comparison_op="unknown", threshold=0.1) is True
    assert g3.carpark_correct_pass1({"pass1": "bad"}, comparison_op="gt", threshold=0.0) is False
    rec2 = {"pass1": {"correct": "yes"}}
    assert g3.general_correct_pass1(rec2) is True


def test_iter_jsonl_files_many_filters(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    keep = root / "temp-0.3" / "keep.jsonl"
    skip = root / "temp-0.5" / "skip.jsonl"
    keep.parent.mkdir(parents=True, exist_ok=True)
    skip.parent.mkdir(parents=True, exist_ok=True)
    keep.write_text("{}", encoding="utf-8")
    skip.write_text("{}", encoding="utf-8")

    files = list(
        g3.iter_jsonl_files_many(
            [str(root)],
            ignore_globs=["*skip*"],
            only_temps=["0.3"],
        ),
    )
    assert str(keep) in files
    assert all("skip" not in f for f in files)


def test_load_rows_from_roots_metric(monkeypatch, tmp_path):
    data_path = tmp_path / "data"
    data_path.mkdir()
    jsonl = data_path / "file.jsonl"
    jsonl.write_text("{}", encoding="utf-8")

    records = [
        {"pass1": {"answer_entropy": 0.5, "shift_in_reasoning_v1": True, "soft_reward": 0.2}},
        {"pass1": {"answer_entropy": None}},  # skipped
    ]
    monkeypatch.setattr(g3, "iter_records_from_file", lambda path: records)
    monkeypatch.setattr(
        g3,
        "step_from_record_if_within_bounds",
        lambda rec, path, split_value, min_step, max_step: 1,
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
        carpark_soft_threshold=0.1,
    )
    rows = g3.load_rows_from_roots_metric([str(data_path)], "Carpark", "answer", args)
    assert len(rows) == 1
    assert rows[0]["aha"] == 1 and rows[0]["correct"] == 1


def test_compute_edges_range_and_empty():
    ent = np.array([0.1, 0.2])
    edges = g3.compute_edges(ent, bins=2, binning="fixed", entropy_min=0.0, entropy_max=0.2)
    assert edges[0] == 0.0 and edges[-1] == 0.2
    empty_edges = g3.compute_edges(np.array([]), bins=2, binning="quantile", entropy_min=None, entropy_max=None)
    assert np.allclose(empty_edges, np.array([0.0, 1.0]))
    with pytest.raises(ValueError):
        g3.compute_edges(ent, bins=2, binning="fixed", entropy_min=1.0, entropy_max=0.5)


def test_compute_bin_table_empty_and_edges_helpers(tmp_path):
    bin_inputs = g3.BinInputs(
        entropies=np.array([]),
        aha_flags=np.array([]),
        correctness=np.array([]),
        edges=np.array([0.0, 1.0]),
    )
    empty = g3.compute_bin_table("m", "s", "d", bin_inputs)
    assert empty.empty

    per_domain = {
        "Carpark": {"entropy": np.array([0.1]), "aha": np.array([1]), "correct": np.array([1])},
        "Crossword": {"entropy": np.array([0.2, 0.3]), "aha": np.array([0, 1]), "correct": np.array([1, 0])},
    }
    args = SimpleNamespace(bins=2, binning="fixed", entropy_min=None, entropy_max=None, share_bins="domain")
    edges_by, overall = g3._compute_edges_for_accuracy(per_domain, list(per_domain.keys()), args)
    assert edges_by["Carpark"].shape[0] == 3
    assert overall.shape[0] == 3

    counts_edges = g3._compute_edges_for_counts(
        per_domain,
        list(per_domain.keys()),
        SimpleNamespace(bins=2, binning="fixed", entropy_min=None, entropy_max=None, share_bins="global"),
    )
    assert all(np.array_equal(edges, list(counts_edges.values())[0]) for edges in counts_edges.values())


def test_write_accuracy_tables_outputs_csv(tmp_path):
    per_domain = {
        "Carpark": {"entropy": np.array([0.1]), "aha": np.array([1]), "correct": np.array([1])},
    }
    edges = {"Carpark": np.array([0.0, 0.5])}
    args = SimpleNamespace(outdir=str(tmp_path), outfile_tag=None)
    config = g3.AccuracyTablesConfig(
        per_domain=per_domain,
        domains=["Carpark"],
        edges_by_domain=edges,
        overall_edges=np.array([0.0, 0.5]),
        metric="answer",
        file_tag="answer",
        args=args,
    )
    g3._write_accuracy_tables(config)
    tables_dir = Path(args.outdir) / "tables"
    assert list(tables_dir.glob("graph_3_pass1_table_answer_combined__per_domain.csv"))
    assert list(tables_dir.glob("graph_3_pass1_table_answer_combined__overall.csv"))


def test_render_accuracy_panel_no_data():
    class AxisStub:
        def __init__(self):
            self.texts = []
            self.spines = {
                "top": SimpleNamespace(set_visible=lambda flag: None),
                "right": SimpleNamespace(set_visible=lambda flag: None),
            }

        def text(self, *a, **k):
            self.texts.append((a, k))

        def grid(self, *a, **k):
            return None

        def set_ylabel(self, *a, **k):
            return SimpleNamespace(set_multialignment=lambda *a, **k: None)

        def set_title(self, *a, **k):
            return None

    cfg = g3.AccuracyFigureConfig(
        per_domain={"Math": {"entropy": np.array([]), "aha": np.array([]), "correct": np.array([])}},
        domains=["Math"],
        edges_by_domain={"Math": np.array([0.0, 1.0])},
        x_label="x",
        args=SimpleNamespace(y_pad=1, title=None, width_in=1.0, height_in=1.0),
        color_noaha="a",
        color_aha="b",
    )
    axis = AxisStub()
    g3._render_accuracy_panel(axis, "Math", cfg)
    assert axis.texts


def test_render_metric_counts_returns_false(monkeypatch):
    monkeypatch.setattr(g3, "_load_metric_rows", lambda *a, **k: [])
    args = SimpleNamespace(
        outdir=".",
        outfile_tag=None,
        bins=2,
        binning="fixed",
        entropy_min=None,
        entropy_max=None,
        share_bins="global",
        dpi=10,
        width_in=1.0,
        height_in=1.0,
        title=None,
    )
    assert g3.render_metric_counts(args, "answer", "c") is False
