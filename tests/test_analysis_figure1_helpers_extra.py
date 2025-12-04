import json
import os

import numpy as np
import pandas as pd

import src.analysis.core.figure_1_helpers as f1


def test_mark_formal_pairs_adds_flags_and_respects_domains():
    df = pd.DataFrame(
        {
            "domain": ["D1", "D1", "D2"],
            "problem": ["p1", "p1", "p2"],
            "step": [1, 2, 1],
            "freq_correct": [0.0, 0.5, 0.6],
            "aha_rate_gpt": [0.0, 0.6, 0.7],
            "aha_any_gpt": [0, 1, 1],
            "p_correct_given_shift": [0.0, 0.8, 0.9],
        }
    )
    out = f1.mark_formal_pairs(df, delta1=0.1, delta2=0.1, min_prior_steps=1, delta3=0.05)
    assert "aha_formal" in out.columns
    # Only rows with shift==1 and sufficient prior/gain should be flagged
    assert out.loc[out["domain"] == "D1", "aha_formal"].tolist() == [0, 1]
    # Not enough prior for D2 -> remains unflagged
    assert out.loc[out["domain"] == "D2", "aha_formal"].tolist() == [0]


def test_export_formal_aha_json_with_text_truncates_and_writes(tmp_path, monkeypatch):
    problem_step_df = pd.DataFrame(
        {
            "domain": ["Math"],
            "problem": ["p1"],
            "step": [1],
            "freq_correct": [0.2],
            "aha_rate_gpt": [0.5],
            "aha_any_gpt": [1],
            "p_correct_given_shift": [0.7],
            "aha_formal": [1],
            "n_samples": [3],
        }
    )
    cfg = f1.FormalAhaExportConfig(
        meta=f1.FormalExportMeta(dataset="d", model="m"),
        thresholds=f1.make_formal_thresholds(delta1=0.1, delta2=0.1, min_prior_steps=1, delta3=None),
        gpt_keys=["shift_in_reasoning_v1"],
        gpt_subset_native=True,
        out_dir=str(tmp_path),
        slug="test",
        max_chars=5,
    )
    # Stub dependencies
    monkeypatch.setattr(f1, "nat_step_from_path", lambda path: 1)
    monkeypatch.setattr(
        f1, "iter_records_from_file", lambda path: [{"pass1": {"answer": "long-answer"}, "problem": "p1", "step": 1}]
    )
    monkeypatch.setattr(f1, "aha_gpt_for_rec", lambda *a, **k: 1)
    monkeypatch.setattr(f1, "extract_pass_answer", lambda p1: "long-answer")

    json_path, jsonl_path, count = f1.export_formal_aha_json_with_text(
        problem_step_df,
        files_by_domain={"Math": ["step0001.jsonl"]},
        config=cfg,
    )

    assert count == 1
    with open(json_path, "r", encoding="utf-8") as jf:
        data = json.load(jf)
    assert data[0]["answer"].endswith("…[truncated]")
    assert os.path.exists(jsonl_path)


def test_build_positive_delta_flags():
    df = pd.DataFrame(
        {
            "domain": ["A", "A", "B"],
            "step": [1, 1, 2],
            "p_correct_given_shift": [0.5, 0.8, np.nan],
            "freq_correct": [0.2, 0.3, 0.4],
            "aha_any_gpt": [1, 1, 0],
        }
    )
    flags = f1.build_positive_delta_flags(df)
    assert flags["A"][1] is True  # mean gain > 0.12
    assert flags["B"][2] is False


def test_truncate_answer_text_handles_limits():
    assert f1._truncate_answer_text("short", max_chars=10) == "short"
    assert f1._truncate_answer_text("longanswer", max_chars=0) == "longanswer"
    assert f1._truncate_answer_text("toolong", max_chars=3) == "too …[truncated]"


def test_maybe_build_event_for_record_skips_missing_remaining(monkeypatch):
    cfg = f1.FormalAhaExportConfig(
        meta=f1.FormalExportMeta(dataset="d", model="m"),
        thresholds=f1.make_formal_thresholds(delta1=0.1, delta2=0.1, min_prior_steps=1, delta3=None),
        gpt_keys=["k"],
        gpt_subset_native=True,
        out_dir=".",
        slug="s",
    )
    context = f1._FormalExportContext(config=cfg, remaining=set(), index_map={})
    rec = {"problem": "p", "step": 1, "pass1": {}}
    assert f1._maybe_build_event_for_record("Math", rec, 1, context) is None


def test_collect_events_for_domain_stops_when_no_remaining(monkeypatch):
    cfg = f1.FormalAhaExportConfig(
        meta=f1.FormalExportMeta(dataset="d", model="m"),
        thresholds=f1.make_formal_thresholds(delta1=0.1, delta2=0.1, min_prior_steps=1, delta3=None),
        gpt_keys=["k"],
        gpt_subset_native=True,
        out_dir=".",
        slug="s",
    )
    key = ("Math", "p1", 1)
    context = f1._FormalExportContext(
        config=cfg,
        remaining={key},
        index_map={
            key: pd.Series({"p_correct_given_shift": 0.5, "freq_correct": 0.4, "aha_rate_gpt": 0.3, "n_samples": 1})
        },
    )
    monkeypatch.setattr(f1, "nat_step_from_path", lambda path: 1)
    monkeypatch.setattr(f1, "iter_records_from_file", lambda path: [{"problem": "p1", "pass1": {}, "step": 1}])
    monkeypatch.setattr(f1, "aha_gpt_for_rec", lambda *a, **k: 1)
    monkeypatch.setattr(f1, "extract_pass_answer", lambda *_a, **_k: "ans")

    events = f1._collect_events_for_domain("Math", ["step0001.jsonl"], context)
    assert events and context.remaining == set()
