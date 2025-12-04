import json
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.core.figure_1_export import (
    ExportDestinations,
    FormalExportConfig,
    FormalThresholds,
    GptFilterConfig,
    _build_event_dict,
    _build_target_index,
    _event_from_pass_record,
    _truncate_answer,
    export_formal_aha_json_with_text,
    nat_step_from_path,
)


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _export_config(tmp_path) -> FormalExportConfig:
    return FormalExportConfig(
        dataset="DS",
        model="Model",
        thresholds=FormalThresholds(delta1=0.1, delta2=0.2, delta3=None, min_prior_steps=1),
        gpt_filter=GptFilterConfig(keys=["shift_in_reasoning_v1"], subset_native=False),
        destinations=ExportDestinations(out_dir=str(tmp_path), slug="slug"),
        max_chars=5,
    )


def test_build_target_index_handles_domain_and_nodomain():
    df_domain = pd.DataFrame({"domain": ["Math"], "problem": ["p1"], "step": [1], "aha_formal": [1]})
    targets, index = _build_target_index(df_domain)
    assert ("Math", "p1", 1) in targets and ("Math", "p1", 1) in index

    df_no_domain = pd.DataFrame({"problem": ["p2"], "step": [2], "aha_formal": [1]})
    targets, index = _build_target_index(df_no_domain)
    assert ("All", "p2", 2) in targets and ("All", "p2", 2) in index


def test_event_from_pass_record_filters_and_builds(monkeypatch, tmp_path):
    problem_step_df = pd.DataFrame(
        {
            "domain": ["Math"],
            "problem": ["p1"],
            "step": [1],
            "aha_formal": [1],
            "n_samples": [2],
            "freq_correct": [0.5],
            "aha_rate_gpt": [0.5],
            "p_correct_given_shift": [0.7],
        }
    )
    targets, index = _build_target_index(problem_step_df)
    config = _export_config(tmp_path)
    # Sanity use of nat_step_from_path to keep import exercised.
    assert nat_step_from_path("step0001_run.jsonl") == 1

    # Non-dict pass1 should return None
    rec_payload = {"pass1": "not a dict"}
    assert (
        _event_from_pass_record(
            type("PR", (), {"payload": rec_payload, "domain": "Math", "problem": "p1", "step": 1}),
            index,
            config,
        )
        is None
    )

    # Valid pass1 yields event
    pass1_payload = {"shift_in_reasoning_v1": 1, "has_reconsider_cue": 1, "final_answer": "ABCDE"}
    # Use small max_chars to trigger truncation for the answer
    config = FormalExportConfig(
        dataset="DS",
        model="Model",
        thresholds=config.thresholds,
        gpt_filter=config.gpt_filter,
        destinations=config.destinations,
        max_chars=2,
    )
    pr = type(
        "PR",
        (),
        {"payload": {"pass1": pass1_payload}, "domain": "Math", "problem": "p1", "step": 1},
    )
    key, event = _event_from_pass_record(pr, index, config)
    assert key in targets
    assert event["question"] == "p1"
    assert event["answer"].endswith("…[truncated]")


def test_truncate_answer_respects_max_chars():
    assert _truncate_answer("abcdef", max_chars=3) == "abc …[truncated]"
    assert _truncate_answer(None, max_chars=3) is None


def test_build_event_dict_computes_delta(monkeypatch, tmp_path):
    row = pd.Series(
        {
            "n_samples": 2,
            "freq_correct": 0.2,
            "aha_rate_gpt": 0.5,
            "p_correct_given_shift": 0.7,
        }
    )
    config = _export_config(tmp_path)
    record = type(
        "PR",
        (),
        {"domain": "Math", "problem": "p1", "step": 1},
    )
    event = _build_event_dict(record, row, config, question="q", answer="a")
    assert event["delta_gain_at_shift"] == pytest.approx(0.5)
    assert event["thresholds"]["delta1"] == pytest.approx(0.1)


def test_export_formal_aha_json_with_text_with_missing(monkeypatch, tmp_path):
    problem_step_df = pd.DataFrame(
        {
            "domain": ["Math", "Math"],
            "problem": ["p1", "p2"],
            "step": [1, 2],
            "aha_formal": [1, 1],
            "n_samples": [1, 1],
            "freq_correct": [0.1, 0.2],
            "aha_rate_gpt": [0.5, 0.6],
            "p_correct_given_shift": [0.4, 0.5],
        }
    )
    file_path = tmp_path / "step0001.jsonl"
    _write_jsonl(
        file_path,
        [
            {
                "step": 1,
                "problem": "p1",
                "pass1": {"shift_in_reasoning_v1": 1, "has_reconsider_cue": 1, "answer": "ok"},
            }
        ],
    )
    config = _export_config(tmp_path)
    json_path, jsonl_path, count = export_formal_aha_json_with_text(
        problem_step_df, {"Math": [str(file_path)]}, config
    )
    assert count == 2
    events = json.loads(Path(json_path).read_text())
    assert {event["problem"] for event in events} == {"p1", "p2"}
    assert Path(jsonl_path).exists()


def test_collect_events_breaks_when_remaining_empty(monkeypatch):
    from src.analysis.core import figure_1_export as f1e

    # Start with no targets to force early break path inside the loop.
    config = _export_config(Path("."))
    produced = []

    def fake_iter(domain, files):
        produced.append((domain, tuple(files)))
        yield f1e.PassRecord(domain="Math", problem="p1", step=1, payload={})

    monkeypatch.setattr(f1e, "_iter_domain_records", fake_iter)
    events, remaining = f1e._collect_events_for_targets({"Math": ["a.jsonl"]}, {}, config, set())
    assert events == []
    assert remaining == set()
    # Ensure generator was touched once (break triggered on first record)
    assert produced == [("Math", ("a.jsonl",))]
