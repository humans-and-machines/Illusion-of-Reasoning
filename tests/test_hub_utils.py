# Use delayed import to avoid typing-only errors on older Python versions
import importlib
import types

import pytest


hub = importlib.import_module("src.training.utils.hub")


class DummyConfig:
    def __init__(self, heads):
        self.num_attention_heads = heads


def test_check_hub_revision_exists_errors_when_readme_present(monkeypatch):
    training_args = types.SimpleNamespace(
        hub_model_id="repo",
        hub_model_revision="rev",
        push_to_hub_revision=True,
        overwrite_hub_revision=False,
    )
    monkeypatch.setattr(hub, "repo_exists", lambda repo_id: True)
    monkeypatch.setattr(
        hub,
        "list_repo_refs",
        lambda repo_id: types.SimpleNamespace(branches=[types.SimpleNamespace(name="rev")]),
    )
    monkeypatch.setattr(
        hub,
        "list_repo_files",
        lambda repo_id, revision: ["README.md", "other.txt"],
    )
    with pytest.raises(ValueError):
        hub.check_hub_revision_exists(training_args)


def test_get_param_count_from_repo_id_parses_patterns(monkeypatch):
    # Safetensors metadata path
    monkeypatch.setattr(
        hub,
        "get_safetensors_metadata",
        lambda repo_id: types.SimpleNamespace(parameter_count={"a": 123}),
    )
    assert hub.get_param_count_from_repo_id("anything") == 123

    # Pattern fallbacks
    monkeypatch.setattr(
        hub,
        "get_safetensors_metadata",
        lambda repo_id: (_ for _ in ()).throw(OSError("no meta")),
    )
    assert hub.get_param_count_from_repo_id("my-8x7b-model") == 56_000_000_000
    assert hub.get_param_count_from_repo_id("tiny-42m") == 42_000_000
    assert hub.get_param_count_from_repo_id("nomatch") == -1


def test_get_gpu_count_for_vllm_reduces(monkeypatch):
    monkeypatch.setattr(hub, "AutoConfig", types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyConfig(16)))
    count = hub.get_gpu_count_for_vllm("model", num_gpus=8)
    assert count == 8  # 16 and 64 divisible by 8

    monkeypatch.setattr(hub, "AutoConfig", types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyConfig(10)))
    count2 = hub.get_gpu_count_for_vllm("model", num_gpus=8)
    assert count2 == 2


def test_get_gpu_count_for_vllm_requires_autoconfig(monkeypatch):
    monkeypatch.setattr(hub, "AutoConfig", None)
    with pytest.raises(RuntimeError, match="transformers.AutoConfig"):
        hub.get_gpu_count_for_vllm("repo")
