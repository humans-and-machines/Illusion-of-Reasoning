#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import pytest

import src.analysis.export_cue_variants as ecv


def test_as_json_handles_primitives_and_structures():
    assert ecv._as_json(None) == ""
    assert ecv._as_json(1) == "1"
    assert ecv._as_json(True) == "True"
    assert ecv._as_json({"a": 1}) == json.dumps({"a": 1}, ensure_ascii=False)


def test_iter_flat_rows_filters_and_coerces(monkeypatch):
    recs = [
        {
            "problem": "p1",
            "step": 1,
            "split": "test",
            "sample_idx": 0,
            "pass1": {"is_correct_pred": "1", "entropy": "0.1", "reconsider_markers": ["x"]},
            "pass2": {"is_correct_pred": 0, "entropy_think": 0.2},
            "other": {"ignored": True},
        },
        ["not_a_dict"],
    ]
    monkeypatch.setattr(ecv, "iter_records_from_file", lambda path: recs)
    monkeypatch.setattr(ecv, "nat_step_from_path", lambda path: 9)

    rows = list(ecv.iter_flat_rows(["file.jsonl"], passes=["pass1", "pass2"]))
    assert len(rows) == 2
    baseline = rows[0]
    assert baseline["variant"] == "pass1"
    assert baseline["is_baseline"] == 1
    assert baseline["entropy"] == pytest.approx(0.1)
    assert baseline["reconsider_markers"].startswith("[")

    reconsider = rows[1]
    assert reconsider["variant"] == "pass2"
    assert reconsider["is_reconsider_variant"] == 1
    assert reconsider["entropy_think"] == pytest.approx(0.2)


def test_export_cue_variants_writes_csv(tmp_path):
    path = tmp_path / "input.jsonl"
    path.write_text(json.dumps({"problem": "p", "pass1": {"is_correct_pred": 1}}), encoding="utf-8")
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ecv, "scan_jsonl_files", lambda root, split_substr=None: [str(path)])
    monkeypatch.setattr(ecv, "iter_records_from_file", lambda p: [{"problem": "p", "pass1": {"is_correct_pred": 1}}])
    monkeypatch.setattr(ecv, "nat_step_from_path", lambda p: 0)

    out_csv = tmp_path / "out.csv"
    ecv.export_cue_variants(str(tmp_path), None, str(out_csv), passes=["pass1"])

    with out_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["variant"] == "pass1"
    monkeypatch.undo()

    # Missing files should raise
    with pytest.raises(SystemExit):
        ecv.export_cue_variants(str(tmp_path / "missing"), None, str(out_csv), passes=["pass1"])


def test_build_argparser_defaults_and_main(monkeypatch, tmp_path):
    parser = ecv.build_argparser()
    args = parser.parse_args(["root"])
    assert args.results_root == "root"
    assert args.out_csv is None

    fake_args = SimpleNamespace(results_root=str(tmp_path), split=None, out_csv=None, passes="pass1,pass2")
    monkeypatch.setattr(ecv, "build_argparser", lambda: ecv.argparse.ArgumentParser())
    monkeypatch.setattr(ecv.argparse.ArgumentParser, "parse_args", lambda self: fake_args)
    monkeypatch.setattr(ecv, "export_cue_variants", lambda **kwargs: kwargs.setdefault("called", True))

    ecv.main()


def test_main_entrypoint_runs(monkeypatch, tmp_path):
    fake_args = SimpleNamespace(
        results_root=str(tmp_path),
        split=None,
        out_csv=str(tmp_path / "out.csv"),
        passes="pass1",
    )
    called = {}
    monkeypatch.setattr(ecv, "build_argparser", lambda: ecv.argparse.ArgumentParser())
    monkeypatch.setattr(ecv.argparse.ArgumentParser, "parse_args", lambda self: fake_args)
    monkeypatch.setattr(ecv, "export_cue_variants", lambda **kwargs: called.setdefault("args", kwargs))
    ecv.main()
    assert called["args"]["out_csv"] == str(tmp_path / "out.csv")
