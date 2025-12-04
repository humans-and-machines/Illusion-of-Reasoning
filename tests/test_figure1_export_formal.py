import json
from pathlib import Path

import pandas as pd

import src.analysis.core.figure_1_export as exp


def _make_config(tmp_path, max_chars=10):
    return exp.FormalExportConfig(
        dataset="d",
        model="m",
        thresholds=exp.FormalThresholds(delta1=0.1, delta2=0.2, delta3=None, min_prior_steps=1),
        gpt_filter=exp.GptFilterConfig(keys=["k"], subset_native=False),
        destinations=exp.ExportDestinations(out_dir=str(tmp_path), slug="slug"),
        max_chars=max_chars,
    )


def test_export_formal_empty_writes_files(tmp_path):
    cfg = _make_config(tmp_path)
    df = pd.DataFrame([{"problem": "p1", "step": 1, "aha_formal": 0}])
    json_path, jsonl_path, count = exp.export_formal_aha_json_with_text(df, {}, cfg)
    assert count == 0
    assert json.loads(Path(json_path).read_text()) == []
    assert Path(jsonl_path).read_text() == ""


def test_iter_domain_records_handles_missing_fields(tmp_path):
    path = tmp_path / "dom.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps("not a dict"),
                json.dumps({"step": None, "pass1": {}}),
                json.dumps({"step": 3, "dataset_index": 1, "pass1": {}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    records = list(exp._iter_domain_records("Dom", [str(path)]))
    # First two entries skipped; third yields fallback problem id.
    assert len(records) == 1
    assert records[0].problem == "idx:1"
    assert records[0].step == 3


def test_event_from_pass_record_filters_and_truncates(monkeypatch, tmp_path):
    cfg = _make_config(tmp_path, max_chars=5)
    record_bad = exp.PassRecord(domain="D", problem="p", step=1, payload={"pass1": "bad"})
    assert exp._event_from_pass_record(record_bad, {}, cfg) is None

    record = exp.PassRecord(domain="D", problem="p", step=1, payload={"pass1": {}, "other": True})
    monkeypatch.setattr(exp, "aha_gpt_for_rec", lambda *a, **k: 0)
    assert exp._event_from_pass_record(record, {}, cfg) is None

    monkeypatch.setattr(exp, "aha_gpt_for_rec", lambda *a, **k: 1)
    record.payload["pass1"] = {"is_correct_pred": True}
    assert exp._event_from_pass_record(record, {}, cfg) is None  # missing row in index

    # Successful path with truncation
    problem_index = {
        ("D", "p", 1): pd.Series(
            {
                "p_correct_given_shift": 0.6,
                "freq_correct": 0.2,
                "aha_rate_gpt": 0.4,
                "n_samples": 10,
            }
        )
    }
    monkeypatch.setattr(exp, "extract_pass_answer", lambda payload: "longanswer")
    event_key, event = exp._event_from_pass_record(record, problem_index, cfg)
    assert event_key == ("D", "p", 1)
    assert event["answer"].endswith("…[truncated]")


def test_collect_events_for_targets_hits_continue_paths(monkeypatch, tmp_path):
    cfg = _make_config(tmp_path)
    target_key = ("D", "p", 1)
    problem_index = {
        target_key: pd.Series({"p_correct_given_shift": 0.5, "freq_correct": 0.1, "aha_rate_gpt": 0.3, "n_samples": 5})
    }
    calls = {"count": 0}

    def fake_iter(domain, files):
        yield exp.PassRecord(domain="D", problem="p", step=1, payload={})
        yield exp.PassRecord(domain="D", problem="p", step=1, payload={})
        yield exp.PassRecord(domain="D", problem="p", step=1, payload={})

    def fake_event(record, index, config):
        calls["count"] += 1
        if calls["count"] == 1:
            return None  # triggers continue at result is None
        if calls["count"] == 2:
            return ("X", {"skip": True})  # key not in remaining
        return (target_key, {"ok": True})

    monkeypatch.setattr(exp, "_iter_domain_records", fake_iter)
    monkeypatch.setattr(exp, "_event_from_pass_record", fake_event)

    events, remaining = exp._collect_events_for_targets(
        {"D": ["hasfiles"], "E": []},  # empty list triggers continue branch
        problem_index,
        cfg,
        {target_key},
    )
    assert events == [{"ok": True}]
    assert remaining == set()


def test_write_event_outputs_round_trip(tmp_path):
    cfg = _make_config(tmp_path)
    events = [{"domain": "D", "problem": "p", "step": 1}]
    json_path, jsonl_path, count = exp._write_event_outputs(events, cfg)
    assert count == 1
    assert json.loads(Path(json_path).read_text(encoding="utf-8")) == events
    assert json.loads(Path(jsonl_path).read_text(encoding="utf-8").splitlines()[0]) == events[0]
