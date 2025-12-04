#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import types

import pytest

import src.annotate.core.shift_core as sc


@pytest.fixture(autouse=True)
def reset_client_state():
    sc._CLIENT_STATE["client"] = None
    sc._CLIENT_STATE["uses_v1"] = False
    yield
    sc._CLIENT_STATE["client"] = None
    sc._CLIENT_STATE["uses_v1"] = False


def test_clamp_and_dump_filtered_and_sanitize(monkeypatch, tmp_path):
    assert sc._clamp("a" * 10, lim=5) == "a" * 5
    assert sc._clamp(None, lim=5) == ""

    # Redirect dump to tmp_path to avoid polluting CWD.
    def fake_open(filename, mode="r", encoding=None):
        return (tmp_path / os.path.basename(filename)).open(mode, encoding=encoding)

    monkeypatch.setattr(sc, "open", fake_open, raising=False)
    messages = []
    monkeypatch.setattr(sc.logging, "warning", lambda msg, *a, **k: messages.append(msg))
    monkeypatch.setattr(sc.time, "strftime", lambda *_a, **_k: "TS")
    sc._dump_filtered("PROMPT")
    dumped = list(tmp_path.glob("filtered_prompt_TS_*"))
    assert dumped and dumped[0].read_text() == "PROMPT"
    assert messages and "filtered" in messages[0]

    # Fallback path
    assert sc._sanitize_jsonish("\\(bad\\)[\\]") == "(bad)[]"
    # Parsed path with nested list/dict replacements
    nested = '{"a":"\\\\(x\\\\)","b":["\\\\[y\\\\]"],"c":{"d":"\\\\)"} }'
    cleaned = sc._sanitize_jsonish(nested)
    assert "\\(" not in cleaned and "\\]" not in cleaned


def test_json_from_text_invalid_and_partial():
    assert sc._json_from_text("{bad}") is None  # top-level parse failure
    assert sc._json_from_text("xx{bad}yy") is None  # substring parse failure
    assert sc._json_from_text("") is None


def test_client_lazy_errors_and_success(monkeypatch):
    sc._CLIENT_STATE["client"] = None
    with pytest.raises(RuntimeError):
        sc._client_lazy({"backend": "portkey"})

    sc._CLIENT_STATE["client"] = None
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        sc._client_lazy({"backend": "azure", "endpoint": "ep", "api_version": "v1", "use_v1": 0})

    sc._CLIENT_STATE["client"] = None
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "key")
    dummy_client = types.SimpleNamespace()
    monkeypatch.setattr(
        sc,
        "build_preferred_client",
        lambda endpoint, api_key, api_version, use_v1: (dummy_client, True),
    )
    sc._client_lazy({"backend": "azure", "endpoint": "ep", "api_version": "v1", "use_v1": 1})
    assert sc._CLIENT_STATE["client"] is dummy_client
    assert sc._CLIENT_STATE["uses_v1"] is True


def test_llm_judge_shift_error_and_no_cue(monkeypatch):
    sc._CLIENT_STATE["client"] = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(
                create=lambda **kwargs: (_ for _ in ()).throw(sc.OpenAIError("boom")),
            ),
        ),
        uses_v1=False,
    )
    sc._CLIENT_STATE["uses_v1"] = False
    monkeypatch.setattr(sc, "_client_lazy", lambda cfg: None)
    captured = []
    monkeypatch.setattr(sc, "_dump_filtered", lambda prompt: captured.append(prompt))
    result = sc.llm_judge_shift({"deployment": "dep"}, "dep", {"problem": "p", "think": "t", "cues": [], "pos": 5})
    assert result["shift_in_reasoning"] is False
    assert captured, "expected prompt to be dumped on error"

    # V1 path with no explicit cues should force conservative FALSE
    sc._CLIENT_STATE["client"] = types.SimpleNamespace(
        responses=types.SimpleNamespace(
            create=lambda **kwargs: types.SimpleNamespace(
                output_text=json.dumps(
                    {
                        "shift_in_reasoning": True,
                        "markers_found": [],
                        "first_marker_index": 3,
                        "before_excerpt": "b",
                        "after_excerpt": "a",
                        "explanation_short": "e",
                    },
                ),
            ),
        ),
    )
    sc._CLIENT_STATE["uses_v1"] = True
    out = sc.llm_judge_shift({"deployment": "dep"}, "dep", {"problem": "p", "think": "t", "cues": [], "pos": None})
    assert out["shift_in_reasoning"] is False
    assert out["confidence"] == "low"


def test_load_records_handles_raw_and_write_merge(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"a":1}\nraw_line\n', encoding="utf-8")
    records = sc._load_records(str(path))
    assert records[1]["__raw__"] == "raw_line\n"
    records[0]["a"] = 2
    sc._write_records_to_disk(str(path), records, dirty_idxs={0})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["a"] == 2
    assert lines[1] == "raw_line"

    new_path = tmp_path / "new.jsonl"
    sc._write_records_to_disk(str(new_path), [{"b": 1}], dirty_idxs=None)
    assert json.loads(new_path.read_text(encoding="utf-8")).get("b") == 1


def test_prefilter_skips_non_dict_and_existing(monkeypatch):
    records = [
        {"__raw__": "bad"},
        {"pass1": "not-a-dict"},
        {"pass1": {"shift_in_reasoning_v1": True, "output": "t"}},
    ]
    todo = sc._prefilter_records_for_pass(records, "pass1", force_relabel=False, dirty_idxs=set())
    assert todo == []


def test_annotate_record_non_dict_and_jitter(monkeypatch):
    opts = sc.AnnotateOpts(
        seed=0,
        max_calls=None,
        dry_run=False,
        jitter=0.1,
        force_relabel=False,
        client_cfg={"deployment": "dep"},
        passes=["p"],
    )
    ctx = sc.PassRunContext(rng=None, record_index=0, dirty_idxs=None)
    assert sc._annotate_record_for_pass({"p": "notdict"}, "p", opts, ctx) is False

    sleep_called = {}
    monkeypatch.setattr(sc.time, "sleep", lambda t: sleep_called.setdefault("t", t))
    rng = types.SimpleNamespace(random=lambda: 0.0)
    ctx2 = sc.PassRunContext(rng=rng, record_index=0, dirty_idxs=set())
    monkeypatch.setattr(sc, "llm_judge_shift", lambda *a, **k: {})
    rec = {"p": {"output": "think text"}}
    opts.jitter = 0.05
    assert sc._annotate_record_for_pass(rec, "p", opts, ctx2) is True
    assert "t" in sleep_called and sleep_called["t"] == 0.0
