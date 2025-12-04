import argparse
import csv
import json
import runpy
import sys
import types
from pathlib import Path

import pytest

from src.analysis import figure_1, figure_2
from src.analysis.export_cue_variants import DEFAULT_PASSES, export_cue_variants, iter_flat_rows


def _write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec) + "\n")


def test_iter_flat_rows_flattens_passes_and_serializes(tmp_path):
    record = {
        "problem": "prob-1",
        "problem_id": "prob-1",
        "step": 3,
        "split": "test",
        "sample_idx": 7,
        "pass1": {
            "is_correct_pred": 1,
            "has_reconsider_cue": 0,
            "reconsider_markers": ["a", "b"],
        },
        "pass2": {
            "is_correct_pred": 0,
            "is_correct_after_reconsideration": 1,
            "entropy": "0.5",
            "entropy_reconsider_full": 0.9,
            "reconsider_excerpt": "snippet",
            "shift_in_reasoning_v1": True,
            "shift_markers_v1": ["m1"],
            "tokens_total": 42,
            "valid_tag_structure": "true",
        },
    }
    file_path = tmp_path / "step0003.jsonl"
    _write_jsonl(file_path, [record])

    rows = list(iter_flat_rows([str(file_path)], passes=["pass1", "pass2"]))
    assert {row["variant"] for row in rows} == {"pass1", "pass2"}
    base = next(row for row in rows if row["variant"] == "pass1")
    reconsider = next(row for row in rows if row["variant"] == "pass2")

    assert base["is_baseline"] == 1 and base["is_reconsider_variant"] == 0
    assert reconsider["is_baseline"] == 0 and reconsider["is_reconsider_variant"] == 1
    assert base["reconsider_markers"] == json.dumps(["a", "b"])
    assert reconsider["entropy"] == pytest.approx(0.5)
    assert reconsider["entropy_reconsider_full"] == pytest.approx(0.9)
    assert reconsider["shift_in_reasoning_v1"] == 1
    assert reconsider["valid_tag_structure"] == 1
    assert reconsider["problem_id"] == "prob-1"
    assert reconsider["sample_idx"] == 7


def test_export_cue_variants_writes_csv(tmp_path):
    # Build a small JSONL with two passes to export.
    record = {
        "problem": "prob-2",
        "sample_idx": 1,
        "pass1": {"is_correct_pred": 1},
        "pass2": {"is_correct_pred": 0},
    }
    results_root = tmp_path / "results"
    file_path = results_root / "step0001.jsonl"
    _write_jsonl(file_path, [record])
    out_csv = tmp_path / "out.csv"

    export_cue_variants(str(results_root), split_substr=None, out_csv=str(out_csv), passes=DEFAULT_PASSES[:2])

    with out_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert len(rows) == 2
    assert {row["variant"] for row in rows} == {"pass1", "pass2"}
    assert rows[0]["path"].endswith("step0001.jsonl")


def test_export_cue_variants_errors_when_missing(tmp_path):
    with pytest.raises(SystemExit):
        export_cue_variants(
            str(tmp_path / "empty"), split_substr=None, out_csv=str(tmp_path / "out.csv"), passes=DEFAULT_PASSES
        )

    file_path = tmp_path / "step0001.jsonl"
    _write_jsonl(file_path, [{"problem": "prob-3"}])  # no pass entries
    with pytest.raises(SystemExit):
        export_cue_variants(str(tmp_path), split_substr=None, out_csv=str(tmp_path / "out.csv"), passes=DEFAULT_PASSES)


def test_figure1_main_delegates(monkeypatch):
    called = []

    def _fake_main():
        called.append(True)

    monkeypatch.setattr(figure_1, "_figure1_main", _fake_main)
    figure_1.main()
    assert called == [True]


def test_figure2_main_parses_and_runs(monkeypatch):
    captured = {}

    class _FakeParser:
        def parse_args(self):
            return argparse.Namespace(sentinal=True)

    def _fake_build_parser():
        return _FakeParser()

    def _fake_run(args):
        captured["args"] = args

    monkeypatch.setattr(figure_2, "build_arg_parser", _fake_build_parser)
    monkeypatch.setattr(figure_2, "run_uncertainty_figures", _fake_run)

    figure_2.main()
    assert captured["args"].sentinal is True


def test_figure2_runs_when_executed_as_script(monkeypatch):
    captured = {}

    class _FakeParser:
        def parse_args(self):
            captured["parsed"] = True
            return argparse.Namespace(sentinal=True)

    fake_helpers = types.ModuleType("src.analysis.figure_2_helpers")
    fake_helpers.build_arg_parser = lambda: _FakeParser()
    fake_helpers.run_uncertainty_figures = lambda args: captured.update({"args": args})

    monkeypatch.setitem(sys.modules, "src.analysis.figure_2_helpers", fake_helpers)
    monkeypatch.delitem(sys.modules, "src.analysis.figure_2", raising=False)

    runpy.run_module("src.analysis.figure_2", run_name="__main__")
    assert captured["parsed"] is True
    assert captured["args"].sentinal is True
