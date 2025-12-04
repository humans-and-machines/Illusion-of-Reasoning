import json
import os

import numpy as np
import pandas as pd
import pytest

from src.analysis.core import figure_1_helpers as f1h


def test_mark_formal_pairs_and_missing_columns():
    base_df = pd.DataFrame(
        {
            "problem": ["p1", "p1"],
            "step": [1, 2],
            "freq_correct": [0.0, 0.1],
            "aha_rate_gpt": [0.0, 0.1],
            "aha_any_gpt": [0, 1],
            "p_correct_given_shift": [np.nan, 0.5],
        }
    )
    # Missing required column
    with pytest.raises(ValueError):
        f1h.mark_formal_pairs(base_df.drop(columns=["freq_correct"]))

    marked = f1h.mark_formal_pairs(base_df, delta1=0.1, delta2=0.2, min_prior_steps=1, delta3=0.2)
    assert marked["aha_formal"].tolist() == [0, 1]


def test_bootstrap_problem_ratio_edge_cases():
    df = pd.DataFrame({"step": [1], "flag": [1]})
    with pytest.raises(KeyError):
        f1h.bootstrap_problem_ratio(df, "missing")

    ratios = f1h.bootstrap_problem_ratio(df, "flag", num_bootstrap_samples=0)
    assert ratios.loc[0, "k"] == 1 and np.isnan(ratios.loc[0, "lo"])


def test_export_index_and_truncate_and_problem_id():
    df = pd.DataFrame(
        {
            "problem": ["p1"],
            "step": [1],
            "aha_formal": [1],
            "n_samples": [1],
            "freq_correct": [0.5],
            "aha_rate_gpt": [0.5],
            "p_correct_given_shift": [0.7],
        }
    )
    targets, index_map = f1h._build_export_index(df)
    assert targets == {("All", "p1", 1)}
    assert ("All", "p1", 1) in index_map

    assert f1h._truncate_answer_text("abc", max_chars=2) == "ab …[truncated]"
    assert f1h._truncate_answer_text("abc", max_chars=5) == "abc"

    rec = {"problem": "p2"}
    assert f1h._problem_id_from_record(rec) == "p2"
    assert f1h._problem_id_from_record({"dataset_index": 3}) == "idx:3"


def test_maybe_build_event_for_record(monkeypatch):
    cfg = f1h.FormalAhaExportConfig(
        meta=f1h.FormalExportMeta(dataset="ds", model="m"),
        thresholds=f1h.make_formal_thresholds(0.1, 0.2, 1, None),
        gpt_keys=["shift"],
        gpt_subset_native=False,
        out_dir=".",
        slug="s",
        max_chars=10,
    )
    key = ("Math", "prob", 1)
    row = pd.Series(
        {
            "n_samples": 1,
            "freq_correct": 0.4,
            "aha_rate_gpt": 0.5,
            "p_correct_given_shift": 0.6,
        }
    )
    context = f1h._FormalExportContext(
        config=cfg,
        remaining={key},
        index_map={key: row},
    )
    monkeypatch.setattr(f1h, "aha_gpt_for_rec", lambda *args, **kwargs: 1)
    monkeypatch.setattr(f1h, "extract_pass_answer", lambda payload: "ans")

    record = {"problem": "prob", "step": 1, "pass1": {"shift": True}}
    event = f1h._maybe_build_event_for_record("Math", record, step_from_name=None, context=context)
    assert event["problem"] == "prob" and event["answer"] == "ans"
    assert key not in context.remaining


def test_collect_events_and_export(tmp_path, monkeypatch):
    cfg = f1h.FormalAhaExportConfig(
        meta=f1h.FormalExportMeta(dataset="ds", model="m"),
        thresholds=f1h.make_formal_thresholds(0.1, 0.2, 1, None),
        gpt_keys=["shift"],
        gpt_subset_native=False,
        out_dir=str(tmp_path),
        slug="demo",
        max_chars=10,
    )
    problem_step_df = pd.DataFrame(
        {
            "domain": ["Math"],
            "problem": ["prob"],
            "step": [1],
            "aha_formal": [1],
            "aha_any_gpt": [1],
            "n_samples": [1],
            "freq_correct": [0.5],
            "aha_rate_gpt": [0.5],
            "p_correct_given_shift": [0.6],
        }
    )
    data_path = tmp_path / "step1.jsonl"
    data_path.write_text(json.dumps({"problem": "prob", "step": 1, "pass1": {"shift": True}}) + "\n", encoding="utf-8")

    monkeypatch.setattr(f1h, "nat_step_from_path", lambda p: 1)
    monkeypatch.setattr(f1h, "iter_records_from_file", lambda path: [json.loads(data_path.read_text())])
    monkeypatch.setattr(f1h, "aha_gpt_for_rec", lambda *args, **kwargs: 1)
    monkeypatch.setattr(f1h, "extract_pass_answer", lambda payload: "ans")

    json_path, jsonl_path, count = f1h.export_formal_aha_json_with_text(
        problem_step_df,
        {"Math": [str(data_path)]},
        cfg,
    )
    assert count == 1
    assert os.path.exists(json_path) and os.path.exists(jsonl_path)
    saved = json.loads(open(json_path, encoding="utf-8").read())
    assert saved[0]["question"] == "prob"

    # Missing targets path -> placeholder event
    count_json, count_jsonl, count_total = f1h.export_formal_aha_json_with_text(
        problem_step_df.assign(aha_formal=[0]),
        {"Math": []},
        cfg,
    )
    assert count_total == 0


def test_write_empty_export_and_positive_delta(tmp_path):
    cfg = f1h.FormalAhaExportConfig(
        meta=f1h.FormalExportMeta(dataset="ds", model="m"),
        thresholds=f1h.make_formal_thresholds(0.1, 0.2, 1, None),
        gpt_keys=[],
        gpt_subset_native=False,
        out_dir=str(tmp_path),
        slug="demo",
    )
    json_path, jsonl_path, count = f1h._write_empty_export(cfg)
    assert count == 0 and os.path.exists(json_path) and os.path.exists(jsonl_path)

    frame = pd.DataFrame(
        {
            "problem": ["p1", "p1"],
            "step": [1, 1],
            "freq_correct": [0.1, 0.2],
            "aha_any_gpt": [1, 1],
            "p_correct_given_shift": [0.4, 0.5],
        }
    )
    flags = f1h.build_positive_delta_flags(frame)
    assert flags["All"][1] is True
