#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import runpy
import sys

import src.annotate.tasks.math_cue_variants as mcv


def test_percentile_and_entropy_helpers():
    assert mcv._percentile([], 0.5) is None
    values = [1, 2, 3, 4]
    assert mcv._percentile(values, -0.1) == 1
    assert mcv._percentile(values, 1.2) == 4
    assert mcv._percentile(values, 0.5) == 3  # round to nearest index

    records = [{"pass1": {"entropy": 0.1}}, {"pass1": {"entropy": 0.5}}, {"pass1": {"entropy": 0.9}}]
    q1, q2, q3 = mcv._compute_entropy_quartile_boundaries(records)
    assert (q1, q2, q3) == (0.1, 0.5, 0.9)

    assert mcv._entropy_quartile(None, q1, q2, q3) is None
    assert mcv._entropy_quartile(0.1, q1, q2, q3) == 1
    assert mcv._entropy_quartile(0.5, q1, q2, q3) == 2
    assert mcv._entropy_quartile(0.9, q1, q2, q3) == 3
    assert mcv._entropy_quartile(1.0, q1, q2, q3) == 4


def test_compute_entropy_quartile_boundaries_handles_empty():
    q1, q2, q3 = mcv._compute_entropy_quartile_boundaries([{"pass1": {"entropy": "bad"}}])
    assert (q1, q2, q3) == (None, None, None)


def test_build_flattened_row_includes_weights_and_shift():
    ctx = mcv._RowContext(
        base_meta={"problem": "p", "sample_idx": 1},
        baseline_correct=True,
        quartiles=(0.25, 0.5, 0.75),
    )
    section = {
        "is_correct_pred": False,
        "entropy": 0.6,
        "shift_in_reasoning_v1": True,
        "tokens_total": 10,
        "output": "answer",
    }
    row = mcv._build_flattened_row(ctx, section, "pass2a", "C1", 0.6)
    assert row["cue_variant"] == "C1"
    assert row["baseline_correct"] is True
    assert row["intervention_correct"] is False
    assert row["entropy_quartile"] == 3
    assert row["shift_in_reasoning_v1"] is True
    assert row["output_len"] == len("answer")
    assert row["tokens_total"] == 10


def test_extract_shift_fields_and_weights_nonstring_output():
    assert mcv._extract_shift_fields(None) == {}
    section = {"output": [1, 2, 3], "tokens_answer": 2}
    weights = mcv._section_weights(section)
    assert weights["output_len"] == len(str([1, 2, 3]))
    assert weights["tokens_answer"] == 2


def test_infer_output_path_prefers_override(tmp_path):
    input_path = tmp_path / "step0001.jsonl"
    override = tmp_path / "custom.jsonl"
    assert mcv._infer_output_path(str(input_path), None).endswith("_flat_cues.jsonl")
    assert mcv._infer_output_path(str(input_path), str(override)) == str(override)


def test_infer_output_path_appends_ext_when_missing(tmp_path):
    input_path = tmp_path / "step0001"
    inferred = mcv._infer_output_path(str(input_path), None)
    assert inferred.endswith("step0001_flat_cues.jsonl")


def test_flatten_math_cue_variants_writes_rows(tmp_path):
    # Prepare a single record with baseline + one cue section
    record = {
        "problem": "p1",
        "gold_answer": "g",
        "gold_answer_canon": "g",
        "step": 1,
        "split": "test",
        "sample_idx": 0,
        "pass1": {"is_correct_pred": True, "entropy": 0.2, "output": "base"},
        "pass2a": {"is_correct_pred": False, "entropy": 0.3, "output": "c1"},
    }
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    mcv.flatten_math_cue_variants(str(input_path))
    lines = [json.loads(line) for line in input_path.with_name("input_flat_cues.jsonl").read_text().splitlines()]

    assert len(lines) == 2  # baseline + one cue variant
    baseline_row = lines[0]
    cue_row = lines[1]
    assert baseline_row["cue_variant"] == "baseline"
    assert cue_row["cue_variant"] == "C1"
    assert cue_row["pass"] == "pass2a"
    assert cue_row["entropy_quartile"] == 4  # higher than quartile_3 == 0.2


def test_main_guard_runs_via_runpy(monkeypatch, tmp_path, capsys):
    record = {"pass1": {"is_correct_pred": True, "entropy": 0.1, "output": "o"}}
    input_path = tmp_path / "in.jsonl"
    input_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["math_cue_variants.py", str(input_path)])
    monkeypatch.delitem(sys.modules, "src.annotate.tasks.math_cue_variants", raising=False)
    runpy.run_module("src.annotate.tasks.math_cue_variants", run_name="__main__", alter_sys=True)

    out = capsys.readouterr().out
    assert "Wrote flattened cue file" in out


def test_load_records_filters_non_dict(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"a": 1}\n42\n', encoding="utf-8")
    records = mcv._load_records(str(path))
    assert records == [{"a": 1}]


def test_build_argparser_parses_defaults():
    parser = mcv._build_argparser()
    args = parser.parse_args(["/tmp/in.jsonl"])
    assert args.input_path.endswith("in.jsonl")
