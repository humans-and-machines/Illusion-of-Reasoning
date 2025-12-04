#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from types import SimpleNamespace

import pytest


model_utils = pytest.importorskip("src.training.utils.model_utils")
configs_mod = pytest.importorskip("src.training.configs")


def _import_hub_utils():
    import importlib

    try:
        return importlib.import_module("src.training.utils.hub")
    except TypeError:
        pytest.skip("src.training.utils.hub not importable on this Python version")


def _import_eval_utils():
    import importlib

    try:
        return importlib.import_module("src.training.utils.evaluation")
    except TypeError:
        pytest.skip("src.training.utils.evaluation not importable on this Python version")


def test_import_hub_utils_skips_on_typeerror(monkeypatch):
    import importlib

    def raise_typeerror(_name):
        raise TypeError("boom")

    monkeypatch.setattr(importlib, "import_module", raise_typeerror)
    with pytest.raises(pytest.skip.Exception):
        _import_hub_utils()


def test_import_eval_utils_skips_on_typeerror(monkeypatch):
    import importlib

    def raise_typeerror(_name):
        raise TypeError("boom")

    monkeypatch.setattr(importlib, "import_module", raise_typeerror)
    with pytest.raises(pytest.skip.Exception):
        _import_eval_utils()


def test_get_tokenizer_uses_auto_tokenizer_and_chat_template(monkeypatch):
    import sys
    import types

    class FakeAutoTokenizer:
        def __init__(self):
            self.chat_template = None
            self.called_with = None

        @classmethod
        def from_pretrained(cls, model_name_or_path, revision=None, trust_remote_code=None):
            inst = cls()
            inst.called_with = (model_name_or_path, revision, trust_remote_code)
            return inst

    fake_transformers = types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    model_args = SimpleNamespace(
        model_name_or_path="dummy-model",
        model_revision="rev",
        trust_remote_code=True,
    )
    training_args = configs_mod.SFTConfig()
    training_args.chat_template = "CHAT-TEMPLATE"

    tok = model_utils.get_tokenizer(model_args, training_args)
    assert isinstance(tok, FakeAutoTokenizer)
    assert tok.chat_template == "CHAT-TEMPLATE"
    assert tok.called_with == ("dummy-model", "rev", True)


def test_resolve_torch_dtype_and_build_model_kwargs(monkeypatch):
    import sys

    class FakeTorch:
        float16 = "float16-dtype"
        bfloat16 = "bfloat16-dtype"

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)

    class FakeTrl:
        @staticmethod
        def get_quantization_config(model_args):
            return {"q": model_args.torch_dtype}

        @staticmethod
        def get_kbit_device_map():
            return {"device": "map"}

    monkeypatch.setitem(sys.modules, "trl", FakeTrl)

    # auto / None paths first
    auto_args = SimpleNamespace(torch_dtype="auto")
    none_args = SimpleNamespace(torch_dtype=None)
    assert model_utils._resolve_torch_dtype(auto_args) == "auto"
    assert model_utils._resolve_torch_dtype(none_args) is None

    # non-auto path should map through FakeTorch
    model_args = SimpleNamespace(
        torch_dtype="float16",
        model_revision="rev",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    training_args = configs_mod.SFTConfig()
    training_args.gradient_checkpointing = False

    kwargs = model_utils._build_model_kwargs(model_args, training_args)
    assert kwargs["torch_dtype"] == FakeTorch.float16
    assert kwargs["use_cache"] is True
    assert kwargs["quantization_config"] == {"q": "float16"}
    assert kwargs["device_map"] == {"device": "map"}


def test_get_model_uses_auto_model_and_build_kwargs(monkeypatch):
    import sys
    import types

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, model_name_or_path, **kwargs):
            return SimpleNamespace(name=model_name_or_path, kwargs=kwargs)

    fake_transformers = types.SimpleNamespace(AutoModelForCausalLM=FakeAutoModel)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    class FakeTrl:
        @staticmethod
        def get_quantization_config(_model_args):
            return None

        @staticmethod
        def get_kbit_device_map():
            return None

    monkeypatch.setitem(sys.modules, "trl", FakeTrl)

    model_args = SimpleNamespace(
        model_name_or_path="dummy-model",
        model_revision="rev",
        trust_remote_code=False,
        attn_implementation="sdpa",
        torch_dtype="auto",
    )
    training_args = configs_mod.GRPOConfig()
    training_args.gradient_checkpointing = True

    model = model_utils.get_model(model_args, training_args)
    assert model.name == "dummy-model"
    # verify a couple of important kwargs were forwarded
    assert model.kwargs["revision"] == "rev"
    assert model.kwargs["attn_implementation"] == "sdpa"
    assert model.kwargs["use_cache"] is False


def test_get_model_populates_device_map_when_quantized(monkeypatch):
    import sys
    import types

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, model_name_or_path, **kwargs):
            return SimpleNamespace(name=model_name_or_path, kwargs=kwargs)

    fake_transformers = types.SimpleNamespace(AutoModelForCausalLM=FakeAutoModel)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    class FakeTrl:
        @staticmethod
        def get_quantization_config(_model_args):
            return {"bits": 4}

        @staticmethod
        def get_kbit_device_map():
            return {"auto": True}

    monkeypatch.setitem(sys.modules, "trl", FakeTrl)

    model_args = SimpleNamespace(
        model_name_or_path="quant-model",
        model_revision=None,
        trust_remote_code=False,
        attn_implementation="sdpa",
        torch_dtype="auto",
    )
    training_args = configs_mod.SFTConfig()
    training_args.gradient_checkpointing = False

    model = model_utils.get_model(model_args, training_args)
    assert model.kwargs["quantization_config"] == {"bits": 4}
    assert model.kwargs["device_map"] == {"auto": True}


def test_push_to_hub_revision_calls_hub_functions(monkeypatch, tmp_path):
    hub_utils = _import_hub_utils()
    calls = {}

    def fake_create_repo(repo_id, private=True, exist_ok=True):
        calls["create_repo"] = (repo_id, private, exist_ok)
        return f"https://hub/{repo_id}"

    def fake_list_repo_commits(repo_id):
        calls["list_repo_commits"] = repo_id
        Commit = SimpleNamespace(commit_id="abc123")
        return [Commit]

    def fake_create_branch(repo_id, branch, revision, exist_ok=True):
        calls["create_branch"] = (repo_id, branch, revision, exist_ok)

    def fake_upload_folder(
        repo_id,
        folder_path,
        revision,
        commit_message,
        ignore_patterns,
        run_as_future,
    ):
        calls["upload_folder"] = (
            repo_id,
            folder_path,
            revision,
            commit_message,
            tuple(ignore_patterns),
            run_as_future,
        )

        class DummyFuture:
            def add_done_callback(self, *_args, **_kwargs):
                return None

        return DummyFuture()

    monkeypatch.setattr(hub_utils, "create_repo", fake_create_repo)
    monkeypatch.setattr(hub_utils, "list_repo_commits", fake_list_repo_commits)
    monkeypatch.setattr(hub_utils, "create_branch", fake_create_branch)
    monkeypatch.setattr(hub_utils, "upload_folder", fake_upload_folder)

    training_args = SimpleNamespace(
        hub_model_id="org/model",
        hub_model_revision="main",
        output_dir=str(tmp_path),
    )

    fut = hub_utils.push_to_hub_revision(training_args, extra_ignore_patterns=["*.tmp"])
    assert isinstance(fut, object)
    fut.add_done_callback(lambda *_args, **_kwargs: None)

    assert calls["create_repo"][0] == "org/model"
    assert calls["list_repo_commits"] == "org/model"
    assert calls["create_branch"][0] == "org/model"

    upload = calls["upload_folder"]
    assert upload[0] == "org/model"
    assert upload[1] == training_args.output_dir
    assert upload[2] == "main"
    ignore_patterns = upload[4]
    assert "checkpoint-*" in ignore_patterns
    assert "*.tmp" in ignore_patterns
    assert upload[5] is True


def test_check_hub_revision_exists_raises_when_readme_present(monkeypatch):
    hub_utils = _import_hub_utils()

    def fake_repo_exists(_repo_id):
        return True

    def fake_list_refs(_repo_id):
        return SimpleNamespace(branches=[SimpleNamespace(name="main")])

    def fake_list_files(_repo_id, _revision):
        return ["README.md", "other"]

    monkeypatch.setattr(hub_utils, "repo_exists", fake_repo_exists)
    monkeypatch.setattr(hub_utils, "list_repo_refs", fake_list_refs)
    monkeypatch.setattr(hub_utils, "list_repo_files", fake_list_files)

    args = SimpleNamespace(
        hub_model_id="org/model",
        hub_model_revision="main",
        push_to_hub_revision=True,
        overwrite_hub_revision=False,
    )
    with pytest.raises(ValueError):
        hub_utils.check_hub_revision_exists(args)


def test_check_hub_revision_exists_no_error_when_no_readme(monkeypatch):
    hub_utils = _import_hub_utils()

    def fake_repo_exists(_repo_id):
        return True

    def fake_list_refs(_repo_id):
        return SimpleNamespace(branches=[SimpleNamespace(name="dev")])

    def fake_list_files(_repo_id, _revision):
        return ["NOT_README"]

    monkeypatch.setattr(hub_utils, "repo_exists", fake_repo_exists)
    monkeypatch.setattr(hub_utils, "list_repo_refs", fake_list_refs)
    monkeypatch.setattr(hub_utils, "list_repo_files", fake_list_files)

    args = SimpleNamespace(
        hub_model_id="org/model",
        hub_model_revision="main",
        push_to_hub_revision=True,
        overwrite_hub_revision=False,
    )
    # Should not raise
    hub_utils.check_hub_revision_exists(args)


def test_check_hub_revision_exists_calls_list_files_when_revision_present(monkeypatch):
    hub_utils = _import_hub_utils()
    calls = {}

    def fake_repo_exists(_repo_id):
        return True

    def fake_list_refs(_repo_id):
        return SimpleNamespace(branches=[SimpleNamespace(name="main")])

    def fake_list_files(repo_id, revision):
        calls["list_repo_files"] = (repo_id, revision)
        return ["NOT_README"]

    monkeypatch.setattr(hub_utils, "repo_exists", fake_repo_exists)
    monkeypatch.setattr(hub_utils, "list_repo_refs", fake_list_refs)
    monkeypatch.setattr(hub_utils, "list_repo_files", fake_list_files)

    args = SimpleNamespace(
        hub_model_id="org/model",
        hub_model_revision="main",
        push_to_hub_revision=True,
        overwrite_hub_revision=False,
    )
    hub_utils.check_hub_revision_exists(args)
    assert calls["list_repo_files"] == ("org/model", "main")


def test_get_param_count_from_repo_id_pattern_fallback(monkeypatch):
    hub_utils = _import_hub_utils()

    def failing_metadata(_repo_id):
        raise OSError("no metadata")

    monkeypatch.setattr(hub_utils, "get_safetensors_metadata", failing_metadata)

    assert hub_utils.get_param_count_from_repo_id("model-42m") == 42_000_000
    assert hub_utils.get_param_count_from_repo_id("big-1.5b") == 1_500_000_000
    assert hub_utils.get_param_count_from_repo_id("mix-8x7b") == 56_000_000_000
    assert hub_utils.get_param_count_from_repo_id("no-size-here") == -1


def test_get_gpu_count_for_vllm_uses_autoconfig(monkeypatch):
    hub_utils = _import_hub_utils()

    class FakeConfig:
        def __init__(self, num_attention_heads):
            self.num_attention_heads = num_attention_heads

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_name, revision=None, trust_remote_code=True):
            assert model_name == "repo"
            return FakeConfig(num_attention_heads=16)

    monkeypatch.setattr(hub_utils, "AutoConfig", FakeAutoConfig)
    num_gpus = hub_utils.get_gpu_count_for_vllm("repo", revision="main", num_gpus=8)
    assert num_gpus == 8  # 16 and 64 both divisible by 8


def test_run_benchmark_jobs_dispatches_to_run_lighteval_job(monkeypatch):
    eval_utils = _import_eval_utils()
    called = []

    def fake_run_lighteval_job(benchmark, training_args, model_args):
        called.append((benchmark, training_args, model_args))

    monkeypatch.setattr(eval_utils, "run_lighteval_job", fake_run_lighteval_job)

    # Explicit benchmark list
    train_args = SimpleNamespace(benchmarks=["math_500"])
    model_args = SimpleNamespace()
    eval_utils.run_benchmark_jobs(train_args, model_args)
    assert called[0][0] == "math_500"

    # "all" expands to every registered task
    called.clear()
    train_args_all = SimpleNamespace(benchmarks=["all"])
    eval_utils.run_benchmark_jobs(train_args_all, model_args)
    expected = eval_utils.get_lighteval_tasks()
    assert [bench for bench, _, _ in called] == expected
