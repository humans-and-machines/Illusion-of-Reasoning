import argparse
import sys
import types
from types import SimpleNamespace

import pytest

import src.inference.gateways.providers.openrouter as orouter


def test_make_client_requires_env(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        orouter._make_client()


def test_make_client_import_error(monkeypatch, capsys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "key")
    monkeypatch.setattr(orouter, "import_module", lambda name: (_ for _ in ()).throw(ImportError("missing")))
    with pytest.raises(ImportError):
        orouter._make_client()
    assert "openai>=1.x is required" in capsys.readouterr().err


def test_call_model_uses_parse(monkeypatch):
    called = {}

    class FakeChat:
        def __init__(self):
            self.completions = types.SimpleNamespace(create=self.create)

        def create(self, **kwargs):
            called["kwargs"] = kwargs
            return "resp"

    class FakeClient:
        def __init__(self):
            self.chat = FakeChat()

    monkeypatch.setattr(
        orouter, "build_math_gateway_messages", lambda sys_prompt, prob: [{"role": "user", "content": prob}]
    )
    monkeypatch.setattr(
        orouter, "parse_openai_chat_response", lambda resp: ("txt", "finish", {"input_tokens": 1, "output_tokens": 2})
    )
    args = SimpleNamespace(model="m", temperature=0.1, top_p=0.9, max_output_tokens=50, request_timeout=30)
    text, finish, usage = orouter._call_model(FakeClient(), "prob", args)
    assert called["kwargs"]["model"] == "m"
    assert text == "txt" and finish == "finish" and usage == {"input_tokens": 1, "output_tokens": 2}


def test_prepare_dataset_passthrough(monkeypatch):
    monkeypatch.setattr(
        orouter,
        "prepare_math_gateway_dataset_from_args",
        lambda args, outpath, logger, load_math500_fn, load_remote_dataset_fn: ("ds", {"p": {0}}, "name"),
    )
    args = SimpleNamespace()
    dataset, existing, name = orouter._prepare_dataset(args, "out")
    assert dataset == "ds" and existing == {"p": {0}} and name == "name"


def test_parse_args_uses_gateway_parser(monkeypatch):
    parser_called = {}

    def fake_parser(**kwargs):
        parser_called["kwargs"] = kwargs
        return argparse.ArgumentParser()

    monkeypatch.setattr(orouter, "build_math_gateway_arg_parser", fake_parser)
    monkeypatch.setattr(sys, "argv", ["prog"])
    args = orouter._parse_args()
    assert parser_called["kwargs"]["default_temperature"] == 0.05
    assert args.model == "deepseek/deepseek-r1"
    assert args.retry_backoff == 10.0 and args.max_retries == 15


def test_generate_samples_builds_and_appends(monkeypatch, tmp_path):
    args = SimpleNamespace(
        num_samples=1,
        split="test",
        step=1,
        model="m",
        temperature=0.1,
        top_p=0.9,
        max_output_tokens=10,
        request_timeout=10,
        seed=0,
    )
    monkeypatch.setattr(orouter, "_prepare_dataset", lambda args, outpath: ("ds", {}, "name"))
    monkeypatch.setattr(
        orouter, "iter_math_gateway_samples", lambda dataset, num_samples, existing: [("prob", "gold", 0)]
    )
    monkeypatch.setattr(
        orouter,
        "call_with_gateway_retries",
        lambda fn, args, logger, sample_idx, problem_snippet, min_sleep: (
            "<answer>gold</answer>",
            "stop",
            {"input_tokens": 1, "output_tokens": 1},
        ),
    )
    monkeypatch.setattr(orouter, "_canon_math", lambda ans: ans.strip())
    monkeypatch.setattr(orouter, "_extract_blocks", lambda text: ("think", "gold"))
    monkeypatch.setattr(orouter, "_valid_tag_structure", lambda text: True)
    monkeypatch.setattr(
        orouter, "build_math_gateway_row_base", lambda **kwargs: {"gold_answer_canon": kwargs["gold_answer_canon"]}
    )
    captured = {}
    monkeypatch.setattr(orouter, "append_jsonl_row", lambda outpath, row: captured.setdefault("row", row))
    orouter._generate_samples("client", args, str(tmp_path / "out.jsonl"))
    row = captured["row"]
    assert row["pass1"]["pred_answer_canon"] == "gold"
    assert row["pass1"]["is_correct_pred"] is True
    assert row["endpoint"] == "openrouter"


def test_generate_samples_without_usage(monkeypatch, tmp_path):
    args = SimpleNamespace(
        num_samples=1,
        split="test",
        step=1,
        model="m",
        temperature=0.1,
        top_p=0.9,
        max_output_tokens=10,
        request_timeout=10,
        seed=0,
    )
    monkeypatch.setattr(orouter, "_prepare_dataset", lambda args, outpath: ("ds", {}, "name"))
    monkeypatch.setattr(
        orouter, "iter_math_gateway_samples", lambda dataset, num_samples, existing: [("prob", "gold", 0)]
    )
    monkeypatch.setattr(
        orouter,
        "call_with_gateway_retries",
        lambda fn, args, logger, sample_idx, problem_snippet, min_sleep: ("<answer>ans</answer>", "stop", None),
    )
    monkeypatch.setattr(orouter, "_canon_math", lambda ans: ans)
    monkeypatch.setattr(orouter, "_extract_blocks", lambda text: ("think", "ans"))
    monkeypatch.setattr(orouter, "_valid_tag_structure", lambda text: True)
    monkeypatch.setattr(
        orouter, "build_math_gateway_row_base", lambda **kwargs: {"gold_answer_canon": kwargs["gold_answer_canon"]}
    )
    captured = {}
    monkeypatch.setattr(orouter, "append_jsonl_row", lambda outpath, row: captured.setdefault("row", row))
    orouter._generate_samples("client", args, str(tmp_path / "out.jsonl"))
    assert "usage" not in captured["row"]


def test_main_entrypoint(monkeypatch, tmp_path):
    called = {}
    fake_args = SimpleNamespace(
        seed=0,
        output_dir=str(tmp_path),
        step=1,
        split="test",
        model="m",
        num_samples=1,
        temperature=0.1,
        top_p=0.9,
        max_output_tokens=10,
        request_timeout=10,
    )
    monkeypatch.setattr(orouter, "_parse_args", lambda: fake_args)
    monkeypatch.setattr(orouter, "_make_client", lambda: "client")

    def fake_generate(client, args, outpath):
        called["client"] = client
        called["outpath"] = outpath

    monkeypatch.setattr(orouter, "_generate_samples", fake_generate)
    orouter.main()
    assert called["client"] == "client"
    assert called["outpath"].endswith("step0001_test.jsonl")
