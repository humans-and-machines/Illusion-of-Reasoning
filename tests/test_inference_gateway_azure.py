from pathlib import Path
from types import SimpleNamespace

import pytest

import src.inference.gateways.providers.azure as azure


def _stub_args(**overrides):
    base = dict(
        endpoint="https://example.azure.com",
        deployment="ds",
        api_version="2024-02-01",
        api_key="key",
        use_v1=1,
        split="test",
        output_dir="/tmp/out",
        step=0,
        max_output_tokens=16,
        request_timeout=5,
        temperature=0.1,
        top_p=0.9,
        num_samples=1,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_make_client_raises_without_key(monkeypatch):
    monkeypatch.setattr(azure, "load_azure_config", lambda: {"endpoint": "e", "deployment": "d", "api_version": "v"})
    args = _stub_args(api_key=None)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "")
    with pytest.raises(RuntimeError):
        azure._make_client(args)


def test_call_model_responses_and_chat(monkeypatch):
    params = azure.AzureCallParams(temperature=0.1, top_p=0.9, max_output_tokens=10, request_timeout=5)

    class FakeMessage:
        def __init__(self, content):
            self.content = content

    class FakeChoice:
        def __init__(self, message=None, finish_reason=None):
            self.message = message
            self.finish_reason = finish_reason

    class FakeOutput:
        def __init__(self):
            self.choices = [FakeChoice(FakeMessage("resp-text"), finish_reason="stop")]

    class FakeResponses:
        def __init__(self):
            self.output = FakeOutput()
            self.usage = {"input_tokens": 1}

        def create(self, **kwargs):
            return self

    class FakeChat:
        def __init__(self):
            self.choices = [FakeChoice(FakeMessage("chat-text"), finish_reason="length")]
            self.usage = {"completion_tokens": 2}

    class FakeClient:
        def __init__(self):
            self.responses = FakeResponses()
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **k: FakeChat()))

    client = FakeClient()
    text, finish, usage = azure._call_model(client, uses_v1=True, deployment="dep", problem="p", params=params)
    assert text == "resp-text" and finish == "stop" and usage == client.responses.usage

    # Chat path
    no_v1_client = FakeClient()
    text2, finish2, usage2 = azure._call_model(
        no_v1_client, uses_v1=False, deployment="dep", problem="p", params=params
    )
    assert text2 == "chat-text" and finish2 == "length" and usage2 == no_v1_client.chat.completions.create().usage


def test_prepare_dataset_uses_helper(monkeypatch):
    called = {}

    def fake_prepare_math_gateway_dataset_from_args(
        *, args, outpath, logger, load_math500_fn, load_remote_dataset_fn, cache_dir
    ):
        called.update({"outpath": outpath, "cache_dir": cache_dir})
        return "ds", {"p": {0}}, None

    monkeypatch.setattr(azure, "prepare_math_gateway_dataset_from_args", fake_prepare_math_gateway_dataset_from_args)
    ds, existing = azure._prepare_dataset(_stub_args(), "out.jsonl")
    assert ds == "ds" and existing == {"p": {0}}
    assert called["outpath"] == "out.jsonl" and "hf_cache" in called["cache_dir"]


def test_build_result_row_and_usage(monkeypatch):
    args = _stub_args(endpoint="e", deployment="d", api_version="v")
    params = azure.AzureCallParams(temperature=0.1, top_p=0.2, max_output_tokens=10, request_timeout=5)
    monkeypatch.setattr(azure, "_canon_math", lambda x: f"canon:{x}")
    monkeypatch.setattr(azure, "_extract_blocks", lambda text: ("think", "ans"))
    monkeypatch.setattr(azure, "_valid_tag_structure", lambda text: text.startswith("<tag>"))

    usage_obj = SimpleNamespace(total_tokens=3)
    row = azure._build_result_row(
        problem="prob",
        gold_answer="gold",
        sample_idx=1,
        text="<tag> output",
        finish_reason="stop",
        usage=usage_obj,
        args=args,
        call_params=params,
    )
    assert row["pass1"]["pred_answer"] == "ans"
    assert row["pass1"]["valid_tag_structure"] is True
    assert row["usage"]["total_tokens"] == 3


def test_generate_samples_writes_and_increments(monkeypatch, tmp_path):
    args = _stub_args(output_dir=str(tmp_path), deployment="dep", num_samples=1)
    args.split = "test"
    args.step = 1
    call_params = azure.AzureCallParams(temperature=0.1, top_p=0.9, max_output_tokens=8, request_timeout=5)

    dataset = [("prob", "ans", 0)]
    monkeypatch.setattr(azure, "_prepare_dataset", lambda a, p: (dataset, {}))
    monkeypatch.setattr(azure, "iter_math_gateway_samples", lambda ds, n, ex: ds)
    monkeypatch.setattr(azure, "call_with_gateway_retries", lambda fn, **k: ("out", "stop", {"input_tokens": 1}))
    monkeypatch.setattr(azure, "append_jsonl_row", lambda path, row: Path(path).write_text("{}\n"))

    out_file = tmp_path / "step0001_test.jsonl"
    azure._generate_samples(client=None, uses_v1=False, args=args, call_params=call_params, output_path=str(out_file))
    assert out_file.exists()


def test_main_uses_parse_args(monkeypatch, tmp_path):
    args = _stub_args(output_dir=str(tmp_path), deployment="dep", api_version="v", endpoint="e", use_v1=0)
    args.split = "test"
    args.step = 2
    args.max_output_tokens = 8
    args.request_timeout = 5
    args.seed = 0

    monkeypatch.setattr(azure, "_parse_args", lambda: args)
    monkeypatch.setattr(azure, "_make_client", lambda a: ("client", False, "e2", "dep2", "v2"))
    monkeypatch.setattr(azure, "_generate_samples", lambda **kwargs: kwargs)

    azure.main()
    # verify args normalized and handed through
    assert args.endpoint == "e2"
    assert args.deployment == "dep2"
    assert args.api_version == "v2"
