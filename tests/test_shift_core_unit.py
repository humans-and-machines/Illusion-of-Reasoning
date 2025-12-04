import json

import pytest

import src.annotate.core.shift_core as sc


def test_sanitize_jsonish_and_json_from_text():
    raw = '{"msg": "hello \\\\(world\\\\)"}'
    cleaned = sc._sanitize_jsonish(raw)
    assert "\\(" not in cleaned and "\\)" not in cleaned

    assert sc._sanitize_jsonish("") == ""
    assert sc._sanitize_jsonish(None) is None

    text = 'prefix {"a":1, "b":2} suffix'
    obj = sc._json_from_text(text)
    assert obj == {"a": 1, "b": 2}
    assert sc._json_from_text("not json") is None


def test_nat_step_and_scan_sort(monkeypatch):
    assert sc.nat_step_from_path("step0010_test.jsonl") == 10

    files = ["a/step0002.jsonl", "a/step10.jsonl", "a/step0005.jsonl"]
    monkeypatch.setattr(sc, "scan_jsonl_files", lambda root, split_substr=None: files)
    sorted_files = sc.scan_jsonl("root", split=None)
    assert sorted_files[0].endswith("step10.jsonl") and sorted_files[-1].endswith("step0002.jsonl")


def test_prefilter_records_for_pass_marks_empty_and_pending(monkeypatch):
    monkeypatch.setattr(sc, "_find_shift_cues", lambda text: (["cue"], 0))
    monkeypatch.setattr(sc, "_extract_think", lambda text: text.split("</think>")[0] if "</think>" in text else text)
    records = [
        {"pass1": {"output": ""}},
        {"pass1": {"output": "<think>abc</think>rest"}},
    ]
    dirty = set()
    todo = sc._prefilter_records_for_pass(records, "pass1", force_relabel=False, dirty_idxs=dirty)
    assert 0 not in todo and records[0]["pass1"]["shift_in_reasoning_v1"] is False
    assert 1 in todo and records[1]["pass1"]["_shift_prefilter_markers"] == ["cue"]
    assert dirty == {0, 1}


def test_annotate_record_paths(monkeypatch):
    # No think text -> conservative false
    rec = {"problem": "p", "pass1": {"output": ""}}
    opts = sc.AnnotateOpts(
        seed=0,
        max_calls=None,
        dry_run=False,
        jitter=0.0,
        force_relabel=False,
        client_cfg={"deployment": "dep"},
        passes=["pass1"],
    )
    ctx = sc.PassRunContext(rng=None, record_index=0, dirty_idxs=set())
    assert sc._annotate_record_for_pass(rec, "pass1", opts, ctx) is False
    assert rec["pass1"]["shift_in_reasoning_v1"] is False

    # Dry run should skip call
    rec2 = {"problem": "p", "pass1": {"output": "text"}}
    opts.dry_run = True
    ctx2 = sc.PassRunContext(rng=None, record_index=0, dirty_idxs=set())
    assert sc._annotate_record_for_pass(rec2, "pass1", opts, ctx2) is False

    # Actual call path with mocked llm_judge_shift
    monkeypatch.setattr(
        sc,
        "llm_judge_shift",
        lambda cfg, dep, ex: {
            "shift_in_reasoning": True,
            "markers_found": ["m"],
            "first_marker_index": 1,
            "before_excerpt": "b",
            "after_excerpt": "a",
            "explanation_short": "ok",
        },
    )
    opts.dry_run = False
    opts.jitter = 0.0
    rec3 = {"problem": "p", "pass1": {"output": "think"}}
    ctx3 = sc.PassRunContext(rng=None, record_index=0, dirty_idxs=set())
    assert sc._annotate_record_for_pass(rec3, "pass1", opts, ctx3) is True
    assert rec3["pass1"]["shift_in_reasoning_v1"] is True
    assert rec3["pass1"]["shift_markers_v1"] == ["m"]


def test_client_lazy_cached_and_portkey(monkeypatch):
    monkeypatch.setattr(sc, "_CLIENT_STATE", {"client": "cached", "uses_v1": True})
    # Should no-op when a client already exists.
    monkeypatch.setattr(
        sc, "build_preferred_client", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("should not build"))
    )
    sc._client_lazy({"endpoint": "x", "api_version": "v", "use_v1": 0})
    assert sc._CLIENT_STATE["client"] == "cached"

    # Portkey backend error when API key missing.
    class DummyPortkey:
        def __init__(self, api_key):
            self.api_key = api_key

    monkeypatch.setattr(sc, "Portkey", DummyPortkey)
    monkeypatch.setattr(sc, "_CLIENT_STATE", {"client": None, "uses_v1": False})
    monkeypatch.delenv("AI_SANDBOX_KEY", raising=False)
    monkeypatch.delenv("PORTKEY_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        sc._client_lazy({"backend": "portkey"})
    assert sc._CLIENT_STATE["client"] is None

    # Successful Portkey path populates the cache.
    monkeypatch.setenv("AI_SANDBOX_KEY", "secret")
    sc._client_lazy({"backend": "portkey"})
    assert isinstance(sc._CLIENT_STATE["client"], DummyPortkey)
    assert sc._CLIENT_STATE["client"].api_key == "secret"
    assert sc._CLIENT_STATE["uses_v1"] is False


def test_llm_judge_shift_unparseable(monkeypatch):
    # Pre-populate client state to avoid hitting real lazy init.
    class FakeResp:
        def __init__(self, content):
            self.choices = [type("Choice", (), {"message": type("Msg", (), {"content": content})()})]

    class FakeClient:
        def __init__(self, resp):
            self.chat = type("Chat", (), {"completions": type("Comp", (), {"create": lambda self=None, **kw: resp})()})

    resp = FakeResp("not json at all")
    monkeypatch.setattr(sc, "_CLIENT_STATE", {"client": FakeClient(resp), "uses_v1": False})
    captured = {}
    monkeypatch.setattr(sc, "_dump_filtered", lambda prompt: captured.setdefault("msg", prompt))

    out = sc.llm_judge_shift(
        {"backend": "azure"},
        "deployment",
        {"problem": "p", "think": "t", "cues": [], "pos": 7},
    )
    assert out["shift_in_reasoning"] is False
    assert out["first_marker_index"] == 7
    assert "[UNPARSEABLE]" in captured["msg"]


def test_annotate_file_respects_max_calls(tmp_path):
    path = tmp_path / "data.jsonl"
    records = [
        {"problem": "p1", "pass1": {"output": "t1"}},
        {"problem": "p2", "pass1": {"output": "t2"}},
    ]
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    called = []

    def fake_hook(rec, pass_key, opts, ctx):
        called.append(rec.get("problem"))
        rec[pass_key]["shift_in_reasoning_v1"] = True
        ctx.mark_dirty()
        return True

    opts = sc.AnnotateOpts(
        seed=0,
        max_calls=1,
        dry_run=False,
        jitter=0.0,
        force_relabel=False,
        client_cfg={"deployment": "dep"},
        passes=["pass1"],
    )
    sc.annotate_file(str(path), opts, hook=fake_hook)
    saved = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    # Only one record should have been annotated due to max_calls=1
    assert len([r for r in saved if r["pass1"].get("shift_in_reasoning_v1")]) == 1
    assert len(called) == 1


def test_annotate_file_breaks_when_calls_reduced_mid_loop(tmp_path):
    path = tmp_path / "data.jsonl"
    records = [
        {"problem": "p1", "pass1": {"output": "t1"}},
        {"problem": "p2", "pass1": {"output": "t2"}},
    ]
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    calls = []

    def hook(rec, pass_key, opts, ctx):
        calls.append(rec.get("problem"))
        rec[pass_key]["shift_in_reasoning_v1"] = True
        ctx.mark_dirty()
        # Force the next iteration to hit the guard break.
        opts.max_calls = 1
        return True

    opts = sc.AnnotateOpts(
        seed=0,
        max_calls=2,
        dry_run=False,
        jitter=0.0,
        force_relabel=False,
        client_cfg={"deployment": "dep"},
        passes=["pass1"],
    )
    sc.annotate_file(str(path), opts, hook=hook)
    saved = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert calls == ["p1"]
    assert saved[0]["pass1"]["shift_in_reasoning_v1"] is True
    assert saved[1]["pass1"].get("shift_in_reasoning_v1") is None
