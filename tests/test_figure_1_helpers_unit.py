import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.analysis.core.figure_1_helpers as f1


def _make_config(out_dir: Path) -> f1.FormalAhaExportConfig:
    meta = f1.FormalExportMeta(dataset="ds", model="m")
    thresholds = f1.make_formal_thresholds(delta1=0.1, delta2=0.2, min_prior_steps=1, delta3=None)
    return f1.FormalAhaExportConfig(
        meta=meta,
        thresholds=thresholds,
        gpt_keys=["shift_in_reasoning_v1"],
        gpt_subset_native=False,
        out_dir=str(out_dir),
        slug="unit",
    )


def _make_row():
    return pd.Series(
        {
            "freq_correct": 0.5,
            "aha_rate_gpt": 0.25,
            "p_correct_given_shift": 0.9,
            "n_samples": 4,
        }
    )


def test_bootstrap_ratio_branches_and_truncate():
    rng = np.random.default_rng(0)
    empty = f1._bootstrap_ratio_for_values(np.array([]), num_bootstrap_samples=10, rng=rng)
    assert empty[:2] == (0, 0)
    assert np.isnan(empty[2])

    indicator = np.array([1, 0, 1, 1])
    res = f1._bootstrap_ratio_for_values(indicator, num_bootstrap_samples=20, rng=np.random.default_rng(1))
    assert res[0] == 3 and res[1] == 4
    assert res[2] == pytest.approx(0.75)
    assert not np.isnan(res[3]) and not np.isnan(res[4])

    long_answer = "abcde12345"
    truncated = f1._truncate_answer_text(long_answer, max_chars=5)
    assert truncated.endswith("…[truncated]") and truncated.startswith("abcde")


def test_maybe_build_event_filters(monkeypatch, tmp_path):
    config = _make_config(tmp_path)
    key = ("dom", "prob", 1)
    record_base = {"step": 1, "problem": "prob"}

    # step missing -> None
    ctx = f1._FormalExportContext(config=config, remaining={key}, index_map={key: _make_row()})
    assert f1._maybe_build_event_for_record("dom", {}, None, ctx) is None

    # pass1 not dict
    ctx = f1._FormalExportContext(config=config, remaining={key}, index_map={key: _make_row()})
    assert f1._maybe_build_event_for_record("dom", {**record_base, "pass1": "bad"}, None, ctx) is None

    # aha flag != 1
    monkeypatch.setattr(f1, "aha_gpt_for_rec", lambda *_a, **_k: 0)
    ctx = f1._FormalExportContext(config=config, remaining={key}, index_map={key: _make_row()})
    assert f1._maybe_build_event_for_record("dom", {**record_base, "pass1": {"answer": "x"}}, None, ctx) is None

    # row missing
    monkeypatch.setattr(f1, "aha_gpt_for_rec", lambda *_a, **_k: 1)
    ctx = f1._FormalExportContext(config=config, remaining={key}, index_map={})
    assert f1._maybe_build_event_for_record("dom", {**record_base, "pass1": {"answer": "x"}}, None, ctx) is None

    # success path consumes remaining
    ctx = f1._FormalExportContext(config=config, remaining={key}, index_map={key: _make_row()})
    event = f1._maybe_build_event_for_record(
        "dom",
        {**record_base, "pass1": {"answer": "final", "is_correct_pred": 1}},
        None,
        ctx,
    )
    assert event is not None
    assert ctx.remaining == set()
    assert event["problem"] == "prob" and event["answer"].startswith("final")


def test_collect_events_and_export_short_circuits(monkeypatch, tmp_path):
    config = _make_config(tmp_path)
    key = ("d1", "prob", 1)

    # Remaining empty -> early return
    empty_ctx = f1._FormalExportContext(config=config, remaining=set(), index_map={})
    assert f1._collect_events_for_domain("d1", ["f1"], empty_ctx) == []

    calls = []

    def fake_iter_records(path):
        calls.append(path)
        if path == "file1":
            return [
                {"step": 1, "problem": "prob", "pass1": {"answer": "gold", "is_correct_pred": 1}},
            ]
        raise AssertionError("should not be called")

    monkeypatch.setattr(f1, "iter_records_from_file", fake_iter_records)
    monkeypatch.setattr(f1, "nat_step_from_path", lambda path: 1)
    monkeypatch.setattr(f1, "aha_gpt_for_rec", lambda *_a, **_k: 1)
    monkeypatch.setattr(f1, "extract_pass_answer", lambda pass1: pass1.get("answer"))

    df = pd.DataFrame(
        [
            {
                "domain": "d1",
                "problem": "prob",
                "step": 1,
                "freq_correct": 0.1,
                "aha_rate_gpt": 0.2,
                "aha_any_gpt": 1,
                "p_correct_given_shift": 0.5,
                "aha_formal": 1,
                "n_samples": 3,
            }
        ]
    )

    paths = {"d1": ["file1"], "d2": ["file2"]}
    json_path, jsonl_path, count = f1.export_formal_aha_json_with_text(df, paths, config)
    assert count == 1
    assert calls == ["file1"]  # second domain skipped after remaining emptied
    data = json.loads(Path(json_path).read_text())
    assert data[0]["dataset"] == "ds" and data[0]["model"] == "m"
    assert Path(jsonl_path).exists()

    # Ensure _collect_events_for_domain stops iterating records once remaining is empty.
    ctx = f1._FormalExportContext(config=config, remaining={key}, index_map={key: _make_row()})
    record_calls = []

    def fake_iter(path):
        record_calls.append(path)
        # First record will consume the only target.
        yield {"step": 1, "problem": "prob", "pass1": {"answer": "x", "is_correct_pred": 1}}
        # This record should never be seen because remaining will be empty.
        record_calls.append("should_not_iterate")
        yield {"step": 1, "problem": "prob", "pass1": {"answer": "y", "is_correct_pred": 1}}

    monkeypatch.setattr(f1, "iter_records_from_file", fake_iter)
    monkeypatch.setattr(f1, "nat_step_from_path", lambda path: 1)
    monkeypatch.setattr(f1, "aha_gpt_for_rec", lambda *_a, **_k: 1)
    monkeypatch.setattr(f1, "extract_pass_answer", lambda pass1: pass1.get("answer"))
    events = f1._collect_events_for_domain("d1", ["p1"], ctx)
    assert len(events) == 1
    # Generator advanced only until remaining emptied; second record yield not used for events.
    assert record_calls[0] == "p1"


def test_export_requires_columns():
    df = pd.DataFrame([{"problem": "prob", "step": 1}])
    config = _make_config(Path("."))
    with pytest.raises(ValueError):
        f1.export_formal_aha_json_with_text(df, {}, config)


def test_collect_events_breaks_once_targets_consumed(monkeypatch, tmp_path):
    config = _make_config(tmp_path)
    key = ("d", "p", 1)
    ctx = f1._FormalExportContext(
        config=config,
        remaining={key},
        index_map={key: _make_row()},
    )

    def fake_iter(path):
        if path == "f2":
            raise AssertionError("should not reach second file when remaining empty")
        yield {"step": 1, "problem": "p", "pass1": {"answer": "x", "is_correct_pred": 1}}

    monkeypatch.setattr(f1, "iter_records_from_file", fake_iter)
    monkeypatch.setattr(f1, "nat_step_from_path", lambda path: 1)
    monkeypatch.setattr(f1, "aha_gpt_for_rec", lambda *_a, **_k: 1)
    monkeypatch.setattr(f1, "extract_pass_answer", lambda pass1: pass1.get("answer"))

    events = f1._collect_events_for_domain("d", ["f1", "f2"], ctx)
    assert len(events) == 1
    assert ctx.remaining == set()
