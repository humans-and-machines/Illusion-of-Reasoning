import json
from types import SimpleNamespace

import pytest

import src.training.utils.vllm_patch as vllm


class _DummyTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return f"decoded-{len(ids)}"


def test_safe_request_success_and_http_error(monkeypatch):
    class _Resp:
        status_code = 200

        def json(self):
            return {"ok": True}

    monkeypatch.setattr(vllm.requests, "get", lambda url, timeout: _Resp())
    assert vllm.safe_request("http://example") == {"ok": True}

    class _RespFail:
        status_code = 500
        text = "bad"

        def json(self):
            return {}

    monkeypatch.setattr(vllm.requests, "get", lambda url, timeout: _RespFail())
    with pytest.raises(RuntimeError):
        vllm.safe_request("http://example")


def test_parse_nonstream_json_branches():
    choices = {"choices": [{"text": "a"}, {"text": "b"}]}
    assert vllm._parse_nonstream_json(choices) == [["a", "b"]]

    results = {"results": [{"text": "x"}]}
    assert vllm._parse_nonstream_json(results) == [["x"]]

    text_list = {"text": ["t1", "t2"]}
    assert vllm._parse_nonstream_json(text_list) == [["t1"], ["t2"]]

    ids_payload = {"completion_ids": [[1, 2], [3]]}
    assert vllm._parse_nonstream_json(ids_payload, tokenizer=_DummyTokenizer()) == [
        ["decoded-2"],
        ["decoded-1"],
    ]

    with pytest.raises(RuntimeError):
        vllm._parse_nonstream_json({}, tokenizer=None)


def test_parse_streaming_response_combines(monkeypatch):
    class _Resp:
        def iter_lines(self):
            payloads = [
                {"prompt_index": 0, "text": "hello "},
                {"prompt_index": 0, "text": "world"},
                {"prompt_index": 1, "text": "foo"},
            ]
            for p in payloads:
                yield json.dumps(p).encode()

    combined = vllm._parse_streaming_response(_Resp(), num_prompts=2)
    assert combined == [["hello world"], ["foo"]]


def test_generate_with_retries_stream_and_nonstream(monkeypatch):
    monkeypatch.setattr(vllm.time, "sleep", lambda *_: None)

    # Streaming path
    resp_stream = SimpleNamespace(
        status_code=200,
        iter_lines=lambda: [json.dumps({"prompt_index": 0, "text": "hi"}).encode()],
    )
    monkeypatch.setattr(
        vllm.requests,
        "post",
        lambda url, json, timeout, stream: resp_stream,
    )
    stream_config = vllm.VLLMGenerateConfig(
        url="u",
        params=vllm.VLLMGenerationParams(stream=True),
        retry=vllm.VLLMRetryConfig(max_retries=1, backoff=0, timeout=1),
    )
    out = vllm._generate_with_retries(
        prompts=["p"],
        payload={},
        config=stream_config,
    )
    assert out == [["hi"]]

    # Non-stream path with retry on first failure.
    calls = {"count": 0}

    def _post(url, json, timeout, stream):
        calls["count"] += 1
        if calls["count"] == 1:
            raise vllm.requests.Timeout()
        return SimpleNamespace(status_code=200, json=lambda: {"results": [{"text": "ok"}]})

    monkeypatch.setattr(vllm.requests, "post", _post)
    nonstream_config = vllm.VLLMGenerateConfig(
        url="u",
        params=vllm.VLLMGenerationParams(stream=False),
        retry=vllm.VLLMRetryConfig(max_retries=2, backoff=0, timeout=1),
    )
    out = vllm._generate_with_retries(
        prompts=["p"],
        payload={},
        config=nonstream_config,
    )
    assert out == [["ok"]]


def test_generate_with_retries_raises(monkeypatch):
    monkeypatch.setattr(vllm.time, "sleep", lambda *_: None)

    def _post(url, json, timeout, stream):
        return SimpleNamespace(status_code=500, text="fail")

    monkeypatch.setattr(vllm.requests, "post", _post)
    with pytest.raises(RuntimeError):
        config = vllm.VLLMGenerateConfig(
            url="u",
            params=vllm.VLLMGenerationParams(stream=False),
            retry=vllm.VLLMRetryConfig(max_retries=1, backoff=0, timeout=1),
        )
        vllm._generate_with_retries(
            prompts=["p"],
            payload={},
            config=config,
        )


def test_safe_generate_wrapper(monkeypatch):
    monkeypatch.setattr(
        vllm,
        "_generate_with_retries",
        lambda **kwargs: [["wrapped"]],
    )
    out = vllm.safe_generate(prompts=["p1"])
    assert out == [["wrapped"]]
