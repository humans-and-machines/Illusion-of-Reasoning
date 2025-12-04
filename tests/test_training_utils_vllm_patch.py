from dataclasses import replace
from types import SimpleNamespace

import pytest

import src.training.utils.vllm_patch as vp


def test_parse_nonstream_json_variants_and_token_ids():
    assert vp._parse_nonstream_json({"choices": [{"text": "a"}]}) == [["a"]]
    assert vp._parse_nonstream_json({"results": [{"text": "b"}]}) == [["b"]]
    assert vp._parse_nonstream_json({"text": ["c", "d"]}) == [["c"], ["d"]]

    tokenizer = SimpleNamespace(decode=lambda ids, skip_special_tokens: f"decoded-{ids}")
    assert vp._parse_nonstream_json({"completion_ids": [[1, 2]]}, tokenizer) == [["decoded-[1, 2]"]]
    with pytest.raises(RuntimeError):
        vp._parse_nonstream_json({"completion_ids": [[1]]}, tokenizer=None)

    with pytest.raises(RuntimeError):
        vp._parse_nonstream_json({"unexpected": 1})


def test_parse_streaming_response_groups_by_prompt_index(monkeypatch):
    lines = [
        b"",
        b'{"prompt_index": 0, "text": "Hello "}',
        b'{"prompt_index": 1, "text": "World"}',
        b'{"prompt_index": 0, "text": "there"}',
    ]

    class FakeResponse:
        def iter_lines(self):
            return lines

    result = vp._parse_streaming_response(FakeResponse(), num_prompts=2)
    assert result == [["Hello there"], ["World"]]


def test_generate_with_retries_handles_errors(monkeypatch):
    prompts = ["hi"]
    payload = {"prompts": prompts}

    class FakeResponse:
        def __init__(self, status, json_data=None, text="err"):
            self.status_code = status
            self._json = json_data or {}
            self.text = text

        def json(self):
            return self._json

    calls = {"count": 0}

    def fake_post(url, json=None, timeout=None, stream=False):
        calls["count"] += 1
        if calls["count"] < 2:
            raise vp.requests.ConnectionError("fail")
        return FakeResponse(200, {"choices": [{"text": "ok"}]})

    monkeypatch.setattr(vp.requests, "post", fake_post)
    config = vp.VLLMGenerateConfig(
        url="http://x",
        params=vp.VLLMGenerationParams(stream=False),
        retry=vp.VLLMRetryConfig(max_retries=3, backoff=0.0, timeout=1.0),
    )
    out = vp._generate_with_retries(
        prompts=prompts,
        payload=payload,
        config=config,
    )
    assert out == [["ok"]]

    # Now force HTTP error
    def fake_post_error(url, json=None, timeout=None, stream=False):
        return FakeResponse(500, text="boom")

    monkeypatch.setattr(vp.requests, "post", fake_post_error)
    with pytest.raises(RuntimeError):
        error_config = replace(
            config,
            retry=replace(config.retry, max_retries=1),
        )
        vp._generate_with_retries(
            prompts=prompts,
            payload=payload,
            config=error_config,
        )


def test_safe_generate_builds_payload_and_delegates(monkeypatch):
    called = {}

    def fake_generate(**kwargs):
        called.update(kwargs)
        return [["ok"]]

    monkeypatch.setattr(vp, "_generate_with_retries", fake_generate)
    out = vp.safe_generate(prompts=["a"], max_tokens=5, temperature=0.1, top_p=0.2, n=2, stream=True, max_retries=2)
    assert out == [["ok"]]
    assert called["payload"]["max_tokens"] == 5
    assert called["payload"]["stream"] is True
    assert isinstance(called["config"], vp.VLLMGenerateConfig)
    assert called["config"].retry.max_retries == 2


def test_safe_request_retries_and_final_error(monkeypatch):
    calls = {"count": 0}

    def fake_get(url, timeout):
        calls["count"] += 1
        if calls["count"] < 2:
            raise vp.requests.ConnectionError("fail")
        return SimpleNamespace(status_code=200, json=lambda: {"ok": True}, text="")

    monkeypatch.setattr(vp.requests, "get", fake_get)
    assert vp.safe_request("http://x", max_retries=3, backoff=0.0) == {"ok": True}

    # No attempts (max_retries=0) -> runtime error
    with pytest.raises(RuntimeError):
        vp.safe_request("http://x", max_retries=0)


def test_generate_with_retries_zero_attempts(monkeypatch):
    with pytest.raises(RuntimeError):
        zero_attempts = vp.VLLMGenerateConfig(
            retry=vp.VLLMRetryConfig(
                max_retries=0,
                backoff=0.0,
                timeout=1.0,
            ),
        )
        vp._generate_with_retries(
            prompts=["a"],
            payload={},
            config=zero_attempts,
        )


def test_resolve_config_rejects_unexpected_keys():
    with pytest.raises(TypeError) as excinfo:
        vp._resolve_config(None, {"foo": 1, "bar": 2})
    assert "safe_generate got unexpected config keys: ['bar', 'foo']" in str(excinfo.value)
