import argparse
import sys
from types import SimpleNamespace

import pytest

from src.inference.gateways.providers import azure, openrouter, portkey


# ---------------- Azure ----------------
def test_azure_make_client_requires_api_key(monkeypatch):
    monkeypatch.setattr(azure, "load_azure_config", lambda: {"endpoint": "e", "deployment": "d", "api_version": "v"})
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    args = SimpleNamespace(endpoint=None, deployment=None, api_version=None, api_key=None, use_v1=False)
    with pytest.raises(RuntimeError):
        azure._make_client(args)


def test_azure_call_model_responses_and_chat(monkeypatch):
    params = azure.AzureCallParams(temperature=0.1, top_p=0.9, max_output_tokens=8, request_timeout=30)

    class _Choice:
        def __init__(self):
            self.finish_reason = "stop"
            self.message = SimpleNamespace(content="resp text")

    class _Output:
        def __init__(self):
            self.choices = [_Choice()]

    resp_obj = SimpleNamespace(output=_Output(), usage={"in": 1})

    class _Responses:
        def create(self, **_kwargs):
            return resp_obj

    client_v1 = SimpleNamespace(responses=_Responses())
    text, finish_reason, usage = azure._call_model(client_v1, True, "deploy", "problem", params)
    assert text == "resp text"
    assert finish_reason == "stop"
    assert usage == resp_obj.usage

    # Chat completions fallback
    chat_resp = SimpleNamespace(
        choices=[SimpleNamespace(finish_reason="len", message=SimpleNamespace(content="chat text"))],
        usage={"out": 2},
    )

    class _ChatClient:
        class chat:
            class completions:
                @staticmethod
                def create(**_kwargs):
                    return chat_resp

    text, finish_reason, usage = azure._call_model(_ChatClient(), False, "deploy", "prob", params)
    assert text == "chat text"
    assert finish_reason == "len"
    assert usage == chat_resp.usage


def test_azure_build_result_row_sets_correct_and_usage(monkeypatch):
    args = SimpleNamespace(split="test", step=1, endpoint="ep", deployment="dep", api_version="v1")
    params = azure.AzureCallParams(temperature=0.2, top_p=0.8, max_output_tokens=5, request_timeout=10)
    usage_obj = {"prompt_tokens": 1}
    monkeypatch.setattr(azure, "build_usage_dict", lambda usage: usage)
    row = azure._build_result_row(
        problem="What is 2+2?",
        gold_answer="4",
        sample_idx=0,
        text="<answer>4</answer>",
        finish_reason="stop",
        usage=usage_obj,
        args=args,
        call_params=params,
    )
    assert row["pass1"]["is_correct_pred"] is True
    assert row["usage"]["prompt_tokens"] == 1
    assert row["endpoint"] == "ep"


def test_azure_load_math500_delegates(monkeypatch):
    called = {}

    def fake_load(*args):
        called["args"] = args
        return "out"

    monkeypatch.setattr(azure, "_load_math500_core", fake_load)
    out = azure.load_math500(cache_dir="cache", split="test", seed=123, dataset_path="local.json")
    assert out == "out"
    assert called["args"] == ("cache", "test", 123, "local.json")


def test_azure_call_model_chat_missing_choices():
    params = azure.AzureCallParams(temperature=0.1, top_p=0.9, max_output_tokens=8, request_timeout=30)

    class _ChatResp:
        def __init__(self):
            self.choices = []
            self.usage = {"tok": 1}

    class _ChatClient:
        class chat:
            class completions:
                @staticmethod
                def create(**_kwargs):
                    return _ChatResp()

    text, finish_reason, usage = azure._call_model(_ChatClient(), False, "deploy", "prob", params)
    assert text == ""
    assert finish_reason is None
    assert usage == {"tok": 1}


def test_azure_parse_args_uses_defaults(monkeypatch):
    monkeypatch.setattr(
        azure,
        "load_azure_config",
        lambda: {"endpoint": "cfg_ep", "deployment": "cfg_dep", "api_version": "cfg_v", "use_v1": 0},
    )

    def fake_parser_builder(default_temperature, description):
        parser = argparse.ArgumentParser()
        # Base args expected by downstream code but unused in this test
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--output_dir", default="out")
        parser.add_argument("--step", type=int, default=0)
        parser.add_argument("--split", default="train")
        parser.add_argument("--temperature", type=float, default=0.7)
        parser.add_argument("--top_p", type=float, default=0.9)
        parser.add_argument("--max_output_tokens", type=int, default=5)
        parser.add_argument("--request_timeout", type=int, default=30)
        parser.add_argument("--num_samples", type=int, default=1)
        return parser

    monkeypatch.setattr(azure, "build_math_gateway_arg_parser", fake_parser_builder)
    monkeypatch.setattr(sys, "argv", ["prog"])
    args = azure._parse_args()
    assert args.endpoint == "cfg_ep"
    assert args.api_version == "cfg_v"
    assert args.use_v1 == 0
    assert args.max_retries == 5


def test_azure_main_invokes_generate(monkeypatch, tmp_path):
    called = {}
    fake_args = SimpleNamespace(
        seed=0,
        output_dir=str(tmp_path),
        step=1,
        split="train",
        temperature=0.1,
        top_p=0.9,
        max_output_tokens=5,
        request_timeout=30,
        endpoint="ep",
        deployment="dep",
        api_version="v",
        api_key=None,
        use_v1=1,
        num_samples=1,
        max_retries=1,
        retry_backoff=0.1,
    )
    monkeypatch.setattr(azure, "_parse_args", lambda: fake_args)
    monkeypatch.setattr(
        azure,
        "_make_client",
        lambda args: ("client", True, "resolved_ep", "resolved_dep", "resolved_v"),
    )

    def fake_generate(**kwargs):
        called["kwargs"] = kwargs

    monkeypatch.setattr(azure, "_generate_samples", fake_generate)
    azure.main()
    assert called["kwargs"]["client"] == "client"
    assert fake_args.endpoint == "resolved_ep"
    assert fake_args.deployment == "resolved_dep"


def test_azure_module_main_guard(monkeypatch, tmp_path):
    # Simulate the __main__ guard by calling main after stubbing heavy deps.
    called = {}
    monkeypatch.setattr(
        azure,
        "_parse_args",
        lambda: SimpleNamespace(
            seed=0,
            output_dir=str(tmp_path),
            step=0,
            split="train",
            temperature=0.1,
            top_p=0.9,
            max_output_tokens=5,
            request_timeout=30,
            endpoint="e",
            deployment="d",
            api_version="v",
            api_key=None,
            use_v1=1,
            num_samples=1,
            max_retries=1,
            retry_backoff=0.1,
        ),
    )
    monkeypatch.setattr(azure, "_make_client", lambda args: ("client", False, "e", "d", "v"))
    monkeypatch.setattr(azure, "_generate_samples", lambda **kwargs: called.setdefault("hit", True))
    azure.main()
    assert called.get("hit") is True


# ---------------- OpenRouter ----------------
def test_openrouter_make_client_requires_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        openrouter._make_client()


def test_openrouter_call_model_uses_parse(monkeypatch):
    called = {}

    def fake_parse(resp):
        called["resp"] = resp
        return ("text", "stop", {"usage": 1})

    monkeypatch.setattr(openrouter, "parse_openai_chat_response", fake_parse)

    class _Chat:
        class completions:
            @staticmethod
            def create(**kwargs):
                return kwargs

    client = SimpleNamespace(chat=_Chat())
    args = SimpleNamespace(model="m", temperature=0.1, top_p=0.2, max_output_tokens=5, request_timeout=10)
    out = openrouter._call_model(client, "prob", args)
    assert out[0] == "text"
    assert "messages" in called["resp"]


# ---------------- Portkey ----------------
def test_portkey_make_client_requires_key(monkeypatch):
    monkeypatch.delenv("AI_SANDBOX_KEY", raising=False)
    fake_mod = SimpleNamespace(Portkey=lambda api_key: SimpleNamespace(api_key=api_key))
    monkeypatch.setattr(portkey, "import_module", lambda name: fake_mod)
    with pytest.raises(RuntimeError):
        portkey._make_client()


def test_build_portkey_row_computes_correct(monkeypatch):
    example = portkey.ExampleContext(problem="p", gold_answer="4", canon_gold="4", sample_idx=1)
    result = portkey.PortkeyCallResult(text="Final Answer: 4", answer="4", finish_reason="stop", usage={"u": 1})
    monkeypatch.setattr(portkey, "build_usage_dict", lambda usage: usage)
    cfg = portkey.PortkeyRunConfig(
        output_path="out.jsonl",
        split_name="test",
        model_name="m",
        num_samples=1,
        params=portkey.PortkeyCallParams(temperature=0.1, top_p=0.9, max_output_tokens=5, request_timeout=10),
        seed=0,
        step=1,
    )
    row = portkey._build_portkey_row(example, result, cfg)
    assert row["pass1"]["is_correct_pred"] is True
    assert row["usage"]["u"] == 1


def test_run_portkey_math_inference_respects_existing(monkeypatch, tmp_path):
    calls = []

    def fake_call_model(client, model, problem, params):
        return ("Answer: 1", "stop", {"tok": 1})

    monkeypatch.setattr(portkey, "_call_model", fake_call_model)
    monkeypatch.setattr(portkey, "append_jsonl_row", lambda path, row: calls.append((path, row)))

    dataset = [
        {"problem": "p1", "answer": "1"},
        {"problem": "p2", "answer": "2"},
    ]
    existing = {"p1": {0}}
    cfg = portkey.PortkeyRunConfig(
        output_path=str(tmp_path / "out.jsonl"),
        split_name="test",
        model_name="m",
        num_samples=1,
        params=portkey.PortkeyCallParams(temperature=0.1, top_p=0.9, max_output_tokens=5, request_timeout=10),
        seed=0,
        step=1,
    )
    portkey.run_portkey_math_inference(SimpleNamespace(), dataset, existing, cfg)
    # Should only generate for p2
    assert len(calls) == 1
    assert calls[0][1]["problem"] == "p2"
