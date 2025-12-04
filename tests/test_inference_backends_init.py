#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np

import src.inference.backends as backends


def test_load_torch_and_transformers_stub(monkeypatch):
    calls = {}

    @contextmanager
    def _no_grad():
        calls["no_grad"] = True
        yield

    torch_stub = SimpleNamespace(no_grad=_no_grad)

    def fake_import_module(name):
        if name == "torch":
            return torch_stub
        if name == "transformers":
            raise ImportError("missing transformers")
        raise AssertionError(f"unexpected import {name}")

    monkeypatch.setattr(backends, "import_module", fake_import_module)
    torch_mod, tok_cls, model_cls, stop_cls = backends._load_torch_and_transformers(require_transformers=False)
    assert tok_cls is None and model_cls is None
    assert hasattr(torch_mod, "inference_mode")
    stub_list = stop_cls([])
    assert isinstance(stub_list([1]), stop_cls)


def test_hfbackend_from_pretrained_uses_tokenizer_path(monkeypatch):
    captured = {}

    class TokCls:
        @classmethod
        def from_pretrained(cls, src, **kwargs):
            captured["tok_src"] = src
            return f"tok-{src}"

    class ModelCls:
        @classmethod
        def from_pretrained(cls, model_name_or_path, **kwargs):
            captured["model_src"] = model_name_or_path
            captured["dtype"] = kwargs["torch_dtype"]
            return SimpleNamespace(eval=lambda: "model-eval")

    torch_stub = SimpleNamespace(bfloat16="bf16", float16="f16")
    env = (torch_stub, TokCls, ModelCls, "StoppingCriteriaList")
    monkeypatch.setattr(backends, "_load_torch_and_transformers", lambda: env)
    monkeypatch.setattr(backends, "_load_hf_tokenizer", lambda *args, **kwargs: f"tok-{args[1]}")
    monkeypatch.setattr(backends, "_load_hf_model", lambda *args, **kwargs: f"model-{args[2]}")

    backend = backends.HFBackend.from_pretrained("model-id", tokenizer_path="tok-id", dtype="bfloat16")
    assert backend.tokenizer == "tok-tok-id"
    assert backend.model == "model-model-id"


def test_decode_and_classify_handles_no_eos(monkeypatch):
    monkeypatch.setattr(
        backends,
        "decode_generated_row",
        lambda tokenizer, sequences, input_lengths, row_i, skip_special_tokens=True: (
            np.array([1, 2]),
            f"text{row_i}_stop",
            None,
        ),
    )
    monkeypatch.setattr(backends, "classify_stop_reason", lambda found, eos, hit: f"{found}-{eos}-{hit}")
    ctx = backends.HFDecodeContext(
        tokenizer=None,
        model=SimpleNamespace(config=SimpleNamespace(eos_token_id=None)),
        sequences=SimpleNamespace(shape=(1, 2)),
        input_lengths=[2],
        stop_strings=["stop"],
        max_new_tokens=2,
    )
    texts, reasons = backends.HFBackend._decode_and_classify(ctx)
    assert texts == ["text0_stop"]
    assert reasons == ["True-False-True"]


def test_azure_call_selects_chat_completions(monkeypatch):
    class ChatResp:
        def __init__(self):
            self.choices = [SimpleNamespace(message=SimpleNamespace(content="hi"), finish_reason="stop")]

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: ChatResp())))
    backend = backends.AzureBackend(client=client, deployment="d", uses_v1=False)
    text, finish, raw = backend._call([{"role": "user", "content": "hi"}])
    assert text == "hi" and finish == "stop" and raw.choices


def test_azure_responses_content_path(monkeypatch):
    class Part:
        def __init__(self, text):
            self.text = text

    class Output:
        def __init__(self):
            self.content = [Part("A"), Part("B")]
            self.finish_reason = "end"

    class Resp:
        def __init__(self):
            self.output_text = None
            self.output = Output()
            self.finish_reason = None

    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: Resp()))
    backend = backends.AzureBackend(client=client, deployment="d", uses_v1=True)
    text, finish, raw = backend._call_responses_api([], temperature=0.0, top_p=None, max_output_tokens=1)
    assert text == "AB"
    assert finish == "end"
    assert isinstance(raw, Resp)
