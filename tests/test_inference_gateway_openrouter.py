#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib
import types

import src.inference.gateways.providers.openrouter as openrouter


def test_make_client_uses_env(monkeypatch):
    fake_client_calls = {}

    class FakeClient:
        def __init__(self, base_url=None, api_key=None):
            fake_client_calls["base_url"] = base_url
            fake_client_calls["api_key"] = api_key

    fake_openai_mod = types.SimpleNamespace(OpenAI=FakeClient)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("OPENROUTER_API_BASE", "http://base")
    monkeypatch.setattr(
        openrouter,
        "import_module",
        lambda name: fake_openai_mod if name == "openai" else importlib.import_module(name),
    )

    client = openrouter._make_client()
    assert isinstance(client, FakeClient)
    assert fake_client_calls == {"base_url": "http://base", "api_key": "k"}


def test_main_invokes_generate_with_client(monkeypatch, tmp_path):
    called = {}

    args = argparse.Namespace(
        seed=123,
        output_dir=str(tmp_path),
        step=1,
        split="train",
        model="m",
    )
    monkeypatch.setattr(openrouter, "_parse_args", lambda: args)
    monkeypatch.setattr(openrouter, "_make_client", lambda: "CLIENT")
    monkeypatch.setattr(
        openrouter,
        "_generate_samples",
        lambda client, parsed_args, outpath: called.update(
            {"client": client, "args": parsed_args, "outpath": outpath}
        ),
    )

    openrouter.main()

    assert called["client"] == "CLIENT"
    assert called["args"] is args
    assert called["outpath"].endswith("step0001_train.jsonl")
