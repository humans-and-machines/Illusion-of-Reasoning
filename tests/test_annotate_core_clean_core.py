#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json

import pytest

import src.annotate.core.clean_core as clean_core


def test_should_clear_detects_bad_rationale():
    bad = {"shift_rationale_gpt": "Model call failed; defaulting to FALSE. Reason..."}
    good = {"shift_rationale_gpt": "All good"}
    assert clean_core._should_clear(bad) is True
    assert clean_core._should_clear(good) is False
    assert clean_core._should_clear({"shift_rationale_gpt": None}) is False


def test_clean_file_removes_shift_fields(tmp_path):
    path = tmp_path / "data.jsonl"
    recs = [
        {"pass1": {"shift_rationale_gpt": "Model call failed; defaulting to FALSE.", "shift_in_reasoning_v1": True}},
        {"pass1": {"shift_rationale_gpt": "ok", "shift_in_reasoning_v1": True}},
        {"other": 1},
    ]
    path.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")

    total, cleared = clean_core.clean_file(str(path))
    assert total == 3
    assert cleared == 1

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert "shift_in_reasoning_v1" not in rows[0]["pass1"]
    assert "shift_in_reasoning_v1" in rows[1]["pass1"]


def test_clean_root_walks_directory(tmp_path, capsys):
    subdir = tmp_path / "sub"
    subdir.mkdir()
    f1 = subdir / "a.jsonl"
    f1.write_text(
        json.dumps({"pass1": {"shift_rationale_gpt": "Model call failed; defaulting to FALSE."}}), encoding="utf-8"
    )
    f2 = subdir / "b.txt"
    f2.write_text("skip", encoding="utf-8")

    files, records, cleared = clean_core.clean_root(str(tmp_path))
    assert files == 1
    assert records == 1
    assert cleared == 1

    out = capsys.readouterr().out
    assert "clean_shift_fallbacks" in out

    # Non-directory should raise
    with pytest.raises(SystemExit):
        clean_core.clean_root(str(tmp_path / "missing"))


def test_clean_file_preserves_blank_and_bad_json_lines(tmp_path):
    path = tmp_path / "messy.jsonl"
    valid = {"pass1": {"shift_rationale_gpt": "Unparseable response; default FALSE.", "shift_in_reasoning_v1": True}}
    raw = "\nnot json\n" + json.dumps(valid) + "\n[]\n"
    path.write_text(raw, encoding="utf-8")

    total, cleared = clean_core.clean_file(str(path))
    assert total == 1
    assert cleared == 1

    lines = path.read_text().splitlines()
    assert lines[0] == ""  # blank line preserved
    assert lines[1] == "not json"
    cleaned = json.loads(lines[2])
    assert cleaned["pass1"] == {}
    assert lines[3] == "[]"
