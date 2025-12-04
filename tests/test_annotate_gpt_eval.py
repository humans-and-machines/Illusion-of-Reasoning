#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json

import pytest


annotate_mod = pytest.importorskip("src.annotate.core.shift_core")


def _make_opts(**overrides):
    """Build a minimal AnnotateOpts instance for tests."""
    base = dict(
        seed=0,
        max_calls=None,
        dry_run=False,
        jitter=0.0,
        force_relabel=False,
        client_cfg={"deployment": "dummy"},
        passes=["pass1"],
    )
    base.update(overrides)
    return annotate_mod.AnnotateOpts(**base)


def test_annotate_file_dry_run_prefilters_without_llm(tmp_path):
    path = tmp_path / "results.jsonl"
    rows = [
        # No pass1 section at all → prefilter should create one and mark FALSE.
        {"problem": "p0"},
        # pass1 with empty output → conservative FALSE, no LLM call.
        {"problem": "p1", "pass1": {"output": ""}},
        # pass1 with non-empty output → prefilter stamps markers but dry_run skips LLM.
        {"problem": "p2", "pass1": {"output": "<think>wait, reconsider</think>"}},
    ]
    with path.open("w", encoding="utf-8") as file_handle:
        for row in rows:
            json.dump(row, file_handle, ensure_ascii=False)
            file_handle.write("\n")

    opts = _make_opts(dry_run=True)
    annotate_mod.annotate_file(str(path), opts)

    with path.open("r", encoding="utf-8") as file_handle:
        new_rows = [json.loads(line) for line in file_handle]

    # Row 0: pass1 should be created and marked as no-shift (FALSE).
    p0_pass1 = new_rows[0]["pass1"]
    assert p0_pass1["shift_in_reasoning_v1"] is False

    # Row 1: already had empty output; should be marked FALSE as well.
    p1_pass1 = new_rows[1]["pass1"]
    assert p1_pass1["shift_in_reasoning_v1"] is False

    # Row 2: prefilter should have populated internal markers but,
    # with dry_run=True, no final shift_in_reasoning_v1 field is added.
    p2_pass1 = new_rows[2]["pass1"]
    assert "_shift_prefilter_markers" in p2_pass1
    assert "_shift_prefilter_pos" in p2_pass1
    assert "shift_in_reasoning_v1" not in p2_pass1


def test_annotate_file_respects_max_calls(tmp_path, monkeypatch):
    path = tmp_path / "results.jsonl"
    rows = [
        {"problem": "p0", "pass1": {"output": "first"}},
        {"problem": "p1", "pass1": {"output": "second"}},
        {"problem": "p2", "pass1": {"output": "third"}},
    ]
    with path.open("w", encoding="utf-8") as file_handle:
        for row in rows:
            json.dump(row, file_handle, ensure_ascii=False)
            file_handle.write("\n")

    call_counter = {"count": 0}

    def fake_annotate(rec, pass_key, opts, ctx):
        call_counter["count"] += 1
        section = rec.get(pass_key) or {}
        section["annotated"] = True
        rec[pass_key] = section
        ctx.mark_dirty()
        return True

    monkeypatch.setattr(annotate_mod, "_annotate_record_for_pass", fake_annotate)

    opts = _make_opts(max_calls=1, dry_run=False)
    annotate_mod.annotate_file(str(path), opts, hook=fake_annotate)

    with path.open("r", encoding="utf-8") as file_handle:
        new_rows = [json.loads(line) for line in file_handle]

    # Only one LLM-style annotation should have been applied.
    assert call_counter["count"] == 1
    annotated_flags = [bool(row.get("pass1", {}).get("annotated", False)) for row in new_rows]
    assert sum(annotated_flags) == 1


def test_annotate_file_flushes_after_each_call(tmp_path, monkeypatch):
    path = tmp_path / "results.jsonl"
    rows = [
        {"problem": "p0", "pass1": {"output": "first"}},
        {"problem": "p1", "pass1": {"output": "second"}},
    ]
    with path.open("w", encoding="utf-8") as file_handle:
        for row in rows:
            json.dump(row, file_handle, ensure_ascii=False)
            file_handle.write("\n")

    write_calls: list[int] = []
    original_write = annotate_mod._write_records_to_disk

    def spy_write(path_arg, records_arg, dirty_idxs=None):
        write_calls.append(len(records_arg))
        original_write(path_arg, records_arg, dirty_idxs)

    monkeypatch.setattr(annotate_mod, "_write_records_to_disk", spy_write)

    def fake_annotate(rec, pass_key, opts, ctx):
        section = rec.get(pass_key) or {}
        section["annotated"] = True
        rec[pass_key] = section
        ctx.mark_dirty()
        return True

    opts = _make_opts(max_calls=2, dry_run=False)
    annotate_mod.annotate_file(str(path), opts, hook=fake_annotate)

    # Two writes for the two fake LLM calls, plus one final flush.
    assert len(write_calls) == 3


def test_sanitize_jsonish_strips_llm_math_escapes():
    raw = r'{"text": "\\(x+1\\) and \\[y\\]"}'
    cleaned = annotate_mod._sanitize_jsonish(raw)
    assert "\\(" not in cleaned and "\\)" not in cleaned
    assert "\\[" not in cleaned and "\\]" not in cleaned
    obj = json.loads(cleaned)
    assert obj["text"] == "(x+1) and [y]"


def test_json_from_text_parses_plain_and_embedded_objects():
    plain = '{"a": 1, "b": 2}'
    assert annotate_mod._json_from_text(plain) == {"a": 1, "b": 2}

    embedded = 'noise before {"c": 3} noise after'
    assert annotate_mod._json_from_text(embedded) == {"c": 3}


def test_json_from_text_returns_none_on_invalid_json():
    invalid = "not json at all"
    assert annotate_mod._json_from_text(invalid) is None


def test_nat_step_from_path_extracts_integer_or_none():
    from src.annotate.core.shift_core import nat_step_from_path

    assert nat_step_from_path("step00050_test.jsonl") == 50
    assert nat_step_from_path("no_step_here.jsonl") is None


def test_scan_jsonl_sorts_by_step_and_filters_split(tmp_path):
    from src.annotate.core.shift_core import scan_jsonl

    root = tmp_path / "results"
    (root / "sub").mkdir(parents=True)
    paths = [
        root / "step00010_test.jsonl",
        root / "step00005_train.jsonl",
        root / "nostep_test.jsonl",
        root / "sub" / "step00020_test.jsonl",
    ]
    for path in paths:
        path.write_text("{}", encoding="utf-8")

    # No split filter: ordered by step desc, then name.
    all_files = scan_jsonl(str(root), split=None)
    assert all_files[0].endswith("step00020_test.jsonl")
    assert all_files[1].endswith("step00010_test.jsonl")

    # With split filter: only filenames containing 'test'.
    test_files = scan_jsonl(str(root), split="test")
    assert all(p.endswith("test.jsonl") for p in test_files)
