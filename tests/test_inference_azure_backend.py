import importlib
import types

import pytest


def _get_backends_module():
    try:
        return importlib.import_module("src.inference.backends")
    except RuntimeError as exc:
        pytest.skip(f"Azure/HF backends optional dependencies missing: {exc}")


def test_azure_backend_from_env_uses_config_and_client(monkeypatch):
    backends_mod = _get_backends_module()
    AzureBackend = backends_mod.AzureBackend

    cfg = {
        "endpoint": "https://example.azure.com",
        "deployment": "test-deployment",
        "api_key": "KEY",
        "api_version": "2024-01-01",
        "use_v1": True,
    }

    def fake_load_azure_config():
        return cfg

    def fake_build_preferred_client(**kwargs):
        assert kwargs["endpoint"] == cfg["endpoint"]
        assert kwargs["api_key"] == cfg["api_key"]
        assert kwargs["api_version"] == cfg["api_version"]
        assert kwargs["use_v1"] == cfg["use_v1"]
        return object(), True

    monkeypatch.setattr(backends_mod, "load_azure_config", fake_load_azure_config)
    monkeypatch.setattr(backends_mod, "build_preferred_client", fake_build_preferred_client)

    backend = AzureBackend.from_env()
    assert backend.deployment == cfg["deployment"]
    assert backend.uses_v1 is True
    assert backend.client is not None


def test_azure_backend_from_env_missing_helpers_raises(monkeypatch):
    backends_mod = _get_backends_module()
    AzureBackend = backends_mod.AzureBackend

    monkeypatch.setattr(backends_mod, "load_azure_config", None)
    monkeypatch.setattr(backends_mod, "build_preferred_client", None)

    with pytest.raises(RuntimeError):
        AzureBackend.from_env()


def test_azure_backend_call_chat_completions_happy_path():
    messages = [{"role": "user", "content": "hi"}]

    class DummyChoiceMessage:
        def __init__(self, content: str):
            self.content = content

    class DummyChoice:
        def __init__(self, content: str, finish_reason: str = "stop"):
            self.message = DummyChoiceMessage(content)
            self.finish_reason = finish_reason

    class DummyResponse:
        def __init__(self, content: str):
            self.choices = [DummyChoice(content)]

    class DummyCompletions:
        def __init__(self):
            self.last_kwargs = None

        def create(self, **kwargs):
            self.last_kwargs = kwargs
            return DummyResponse("ok")

    class DummyChat:
        def __init__(self):
            self.completions = DummyCompletions()

    client = types.SimpleNamespace(chat=DummyChat())
    backends_mod = _get_backends_module()
    AzureBackend = backends_mod.AzureBackend
    backend = AzureBackend(client=client, deployment="dep", uses_v1=False)

    text, finish, resp = backend._call_chat_completions(
        messages,
        temperature=0.5,
        top_p=0.9,
        max_output_tokens=42,
    )

    assert text == "ok"
    assert finish == "stop"
    assert isinstance(resp, DummyResponse)
    assert client.chat.completions.last_kwargs["model"] == "dep"
    assert client.chat.completions.last_kwargs["messages"] == messages


def test_azure_backend_call_chat_completions_missing_attr_raises():
    client = types.SimpleNamespace()  # no 'chat' attribute
    backends_mod = _get_backends_module()
    AzureBackend = backends_mod.AzureBackend
    backend = AzureBackend(client=client, deployment="dep", uses_v1=False)

    with pytest.raises(RuntimeError):
        backend._call_chat_completions(
            [{"role": "user", "content": "hi"}],
            temperature=0.0,
            top_p=None,
            max_output_tokens=16,
        )
