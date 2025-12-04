import pytest

import src.annotate.infra.llm_client as llm


def test_build_preferred_client_prefers_v1(monkeypatch):
    class FakeClient:
        def __init__(self, base_url=None, api_key=None):
            self.base_url = base_url
            self.api_key = api_key

    monkeypatch.setattr(llm, "OpenAI", FakeClient)
    client, uses_v1 = llm.build_preferred_client("https://endpoint", "key", "v", use_v1=True)
    assert uses_v1 is True
    assert client.base_url.endswith("/openai/v1/")


def test_build_preferred_client_falls_back_to_azure(monkeypatch):
    class FakeError(Exception):
        pass

    class FakeOpenAI:
        def __init__(self, *a, **k):
            raise FakeError("fail")

    class FakeAzure:
        def __init__(self, api_key, azure_endpoint, api_version):
            self.azure_endpoint = azure_endpoint
            self.api_version = api_version

    monkeypatch.setattr(llm, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(llm, "AzureOpenAI", FakeAzure)
    monkeypatch.setattr(llm, "OpenAIError", FakeError)
    client, uses_v1 = llm.build_preferred_client("https://endpoint/", "k", "2024-01-01")
    assert uses_v1 is False
    assert client.azure_endpoint == "https://endpoint"


def test_build_preferred_client_raises_when_missing(monkeypatch):
    monkeypatch.setattr(llm, "OpenAI", None)
    monkeypatch.setattr(llm, "AzureOpenAI", None)
    with pytest.raises(RuntimeError):
        llm.build_preferred_client("e", "k", "v")


def test_build_preferred_client_azure_error(monkeypatch):
    class FakeError(Exception):
        pass

    class FakeAzure:
        def __init__(self, *a, **k):
            raise FakeError("boom")

    monkeypatch.setattr(llm, "OpenAI", None)
    monkeypatch.setattr(llm, "AzureOpenAI", FakeAzure)
    monkeypatch.setattr(llm, "OpenAIError", FakeError)

    with pytest.raises(RuntimeError):
        llm.build_preferred_client("https://endpoint", "k", "v", use_v1=False)


def test_build_chat_client(monkeypatch):
    class FakeAzure:
        def __init__(self, api_key, azure_endpoint, api_version):
            self.azure_endpoint = azure_endpoint
            self.api_version = api_version

    monkeypatch.setattr(llm, "AzureOpenAI", FakeAzure)
    client = llm.build_chat_client("ep/", "k", "v")
    assert client.azure_endpoint == "ep"


def test_build_chat_client_requires_azure(monkeypatch):
    monkeypatch.setattr(llm, "AzureOpenAI", None)
    with pytest.raises(RuntimeError):
        llm.build_chat_client("ep", "k", "v")
