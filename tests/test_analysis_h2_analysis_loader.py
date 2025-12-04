#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import re

import pandas as pd
import pytest

import src.analysis.h2_analysis_loader as h2


def test_maybe_temp_from_path_and_uncertainty_choice(monkeypatch):
    assert h2.maybe_temp_from_path("...temp-0.3/file.jsonl") == 0.3
    assert h2.maybe_temp_from_path("no-temp") is None
    # Regex capture that cannot be converted should safely fall through
    monkeypatch.setattr(h2, "TEMP_PATS", [re.compile(r"temp-(bad)", re.I)])
    assert h2.maybe_temp_from_path("foo/temp-bad.jsonl") is None

    # Preference ordering
    pass1 = {"entropy_answer": 1.0, "entropy": 2.0, "entropy_think": 3.0}
    assert h2._choose_uncertainty(pass1, preference="answer") == 1.0
    assert h2._choose_uncertainty(pass1, preference="overall") == 2.0
    assert h2._choose_uncertainty(pass1, preference="think") == 3.0
    assert h2._choose_uncertainty({}, preference="unknown") is None


def test_get_aha_native_handles_injected_cue_and_bool():
    pass1 = {"has_reconsider_cue": "1", "reconsider_markers": ["injected_cue"]}
    assert h2._get_aha_native(pass1) == 0  # injected cue => force 0
    pass1b = {"has_reconsider_cue": True, "reconsider_markers": []}
    assert h2._get_aha_native(pass1b) == 1


def test_build_pass1_row_and_missing_fields(monkeypatch):
    context = h2.Pass1RowContext(
        path="p.jsonl",
        step_from_name=5,
        temp_from_path=0.1,
        unc_field="answer",
        aha_source="native",
    )
    # Patch GPT flag to a known value
    monkeypatch.setattr(h2, "get_aha_gpt_flag", lambda pass1, rec: 1)

    rec = {
        "problem": "prob",
        "step": 3,
        "sample_idx": 0,
        "pass1": {"is_correct_pred": True, "has_reconsider_cue": False, "entropy_answer": 0.2},
    }
    row = h2._build_pass1_row(rec, context)
    assert row["problem"] == "prob"
    assert row["aha"] in (0, 1)  # derived from native/gpt flags
    assert row["uncertainty"] == 0.2
    assert row["temperature"] == 0.1

    # Missing pass1 -> None
    assert h2._build_pass1_row({"problem": "p"}, context) is None
    # Missing step falls back to step_from_name in context
    rec2 = {"problem": "p", "pass1": {"is_correct_pred": True, "entropy_answer": 0.1}}
    row2 = h2._build_pass1_row(rec2, context)
    assert row2 is not None and row2["step"] == 5
    # Missing correctness -> None
    rec3 = {"problem": "p", "step": 1, "pass1": {"entropy_answer": 0.1}}
    assert h2._build_pass1_row(rec3, context) is None
    # Missing uncertainty -> None
    rec4 = {"problem": "p", "step": 1, "pass1": {"is_correct_pred": True}}
    assert h2._build_pass1_row(rec4, context) is None


def test_build_pass1_row_problem_fallbacks(monkeypatch):
    context = h2.Pass1RowContext(
        path="p.jsonl",
        step_from_name=None,
        temp_from_path=None,
        unc_field="answer",
        aha_source="native",
    )
    monkeypatch.setattr(h2, "get_aha_gpt_flag", lambda pass1, rec: None)

    rec = {
        "dataset_index": 3,
        "step": 2,
        "pass1": {"has_reconsider_cue": True, "is_correct_pred": True, "entropy_answer": 0.4},
    }
    row = h2._build_pass1_row(rec, context)
    assert row["problem"] == "idx:3"

    # Missing step and both Aha sources -> None
    rec_missing_step = {
        "dataset_index": 1,
        "pass1": {"has_reconsider_cue": None, "is_correct_pred": True, "entropy_answer": 0.1},
    }
    assert h2._build_pass1_row(rec_missing_step, context) is None

    rec_missing_aha = {
        "problem": "p",
        "step": 1,
        "pass1": {"has_reconsider_cue": None, "is_correct_pred": True, "entropy_answer": 0.1},
    }
    monkeypatch.setattr(h2, "_get_aha_native", lambda payload: None)
    assert h2._build_pass1_row(rec_missing_aha, context) is None


def test_load_pass1_rows_reads_file(tmp_path, monkeypatch):
    monkeypatch.setattr(h2, "get_aha_gpt_flag", lambda pass1, rec: 1)
    content = [
        {
            "problem": "p",
            "step": 1,
            "sample_idx": 0,
            "temperature": 0.5,
            "pass1": {"is_correct_pred": True, "entropy": 0.3, "has_reconsider_cue": True},
        },
        {"invalid": "line"},
    ]
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(json.dumps(c) for c in content), encoding="utf-8")

    df = h2.load_pass1_rows([str(path)], unc_field="overall", aha_source="gpt")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["uncertainty"] == 0.3
    assert df.iloc[0]["temperature"] == 0.5

    # No usable rows -> RuntimeError
    path2 = tmp_path / "empty.jsonl"
    path2.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError):
        h2.load_pass1_rows([str(path2)], unc_field="answer", aha_source="native")


def test_load_pass1_rows_skips_blank_and_bad_json(tmp_path, monkeypatch):
    monkeypatch.setattr(h2, "get_aha_gpt_flag", lambda pass1, rec: 0)
    path = tmp_path / "temp-0.7.jsonl"
    lines = [
        "",  # blank line ignored
        "{not json}",  # bad json ignored
        json.dumps(
            {
                "row_key": "rk",
                "step": 4,
                "pass1": {"is_correct_pred": True, "has_reconsider_cue": True, "entropy": 0.5},
            },
        ),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")

    df = h2.load_pass1_rows([str(path)], unc_field="overall", aha_source="gpt")
    assert len(df) == 1
    assert df.iloc[0]["temperature"] == 0.7
