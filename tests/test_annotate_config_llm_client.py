#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import types

import pytest


config_mod = pytest.importorskip("src.annotate.config")
llm_client_mod = pytest.importorskip("src.annotate.llm_client")


def test_load_azure_config_uses_yaml_when_no_env(monkeypatch, tmp_path):
    yaml_path = tmp_path / "azure.yml"
    yaml_path.write_text(
        "\n".join(
            [
                "endpoint: https://yaml-endpoint.example.com/",
                "api_key: yaml-key",
                "deployment: yaml-deploy",
                "api_version: 2025-01-01-preview",
                "use_v1: true",
            ]
        ),
        encoding="utf-8",
    )

    for key in (
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_DEPLOYMENT",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_USE_V1",
    ):
        monkeypatch.delenv(key, raising=False)

    cfg = config_mod.load_azure_config(str(yaml_path))
    assert cfg["endpoint"] == "https://yaml-endpoint.example.com"
    assert cfg["api_key"] == "yaml-key"
    assert cfg["deployment"] == "yaml-deploy"
    assert cfg["api_version"] == "2025-01-01-preview"
    assert cfg["use_v1"] is True


def test_load_azure_config_env_overrides_yaml(monkeypatch, tmp_path):
    yaml_path = tmp_path / "azure.yml"
    yaml_path.write_text(
        "\n".join(
            [
                "endpoint: https://yaml-endpoint.example.com/",
                "api_key: yaml-key",
                "deployment: yaml-deploy",
                "api_version: 2025-01-01-preview",
                "use_v1: false",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://env-endpoint.example.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "env-deploy")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2023-12-01")
    monkeypatch.setenv("AZURE_OPENAI_USE_V1", "1")

    cfg = config_mod.load_azure_config(str(yaml_path))
    assert cfg["endpoint"] == "https://env-endpoint.example.com"
    assert cfg["api_key"] == "env-key"
    assert cfg["deployment"] == "env-deploy"
    assert cfg["api_version"] == "2023-12-01"
    assert cfg["use_v1"] is True


def test_build_preferred_client_prefers_v1_when_available(monkeypatch):
    calls = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            calls["openai_kwargs"] = kwargs

    class FakeAzure:
        def __init__(self, **kwargs):
            calls["azure_kwargs"] = kwargs

    class FakeError(Exception):
        pass

    monkeypatch.setattr(llm_client_mod, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(llm_client_mod, "AzureOpenAI", FakeAzure)
    monkeypatch.setattr(llm_client_mod, "OpenAIError", FakeError)

    client, uses_v1 = llm_client_mod.build_preferred_client(
        endpoint="https://api.example.com/",
        api_key="KEY",
        api_version="2024-01-01",
        use_v1=True,
    )

    assert uses_v1 is True
    assert isinstance(client, FakeOpenAI)
    assert calls["openai_kwargs"]["base_url"] == "https://api.example.com/openai/v1/"
    assert calls["openai_kwargs"]["api_key"] == "KEY"


def test_build_preferred_client_falls_back_to_azure_on_v1_failure(monkeypatch):
    calls = {}

    class FakeError(Exception):
        pass

    class FailingOpenAI:
        def __init__(self, **_kwargs):
            raise FakeError("boom")

    class FakeAzure:
        def __init__(self, **kwargs):
            calls["azure_kwargs"] = kwargs

    monkeypatch.setattr(llm_client_mod, "OpenAI", FailingOpenAI)
    monkeypatch.setattr(llm_client_mod, "AzureOpenAI", FakeAzure)
    monkeypatch.setattr(llm_client_mod, "OpenAIError", FakeError)

    client, uses_v1 = llm_client_mod.build_preferred_client(
        endpoint="https://api.example.com/",
        api_key="KEY",
        api_version="2024-01-01",
        use_v1=True,
    )

    assert uses_v1 is False
    assert isinstance(client, FakeAzure)
    assert calls["azure_kwargs"]["azure_endpoint"] == "https://api.example.com"
    assert calls["azure_kwargs"]["api_key"] == "KEY"
    assert calls["azure_kwargs"]["api_version"] == "2024-01-01"


def test_build_preferred_client_uses_azure_when_v1_disabled(monkeypatch):
    calls = {}

    class FakeAzure:
        def __init__(self, **kwargs):
            calls["azure_kwargs"] = kwargs

    monkeypatch.setattr(llm_client_mod, "OpenAI", None)
    monkeypatch.setattr(llm_client_mod, "AzureOpenAI", FakeAzure)

    client, uses_v1 = llm_client_mod.build_preferred_client(
        endpoint="https://api.example.com/",
        api_key="KEY",
        api_version="2024-01-01",
        use_v1=False,
    )

    assert uses_v1 is False
    assert isinstance(client, FakeAzure)
    assert calls["azure_kwargs"]["azure_endpoint"] == "https://api.example.com"


def test_build_chat_client_wraps_azure_client(monkeypatch):
    calls = {}

    class FakeAzure:
        def __init__(self, **kwargs):
            calls["kwargs"] = kwargs

    monkeypatch.setattr(llm_client_mod, "AzureOpenAI", FakeAzure)

    client = llm_client_mod.build_chat_client(
        endpoint="https://api.example.com/",
        api_key="KEY",
        api_version="2024-01-01",
    )
    assert isinstance(client, FakeAzure)
    assert calls["kwargs"]["azure_endpoint"] == "https://api.example.com"
    assert calls["kwargs"]["api_key"] == "KEY"
    assert calls["kwargs"]["api_version"] == "2024-01-01"

