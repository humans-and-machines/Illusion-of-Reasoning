#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest


# Stub runtime.main before importing env to avoid heavy deps.
stub_main = ModuleType("src.training.runtime.main")
stub_main.main = lambda *a, **k: None
sys.modules["src.training.runtime.main"] = stub_main

from src.training.runtime import env  # noqa: E402


def _clear_cache():
    env._EXPORT_CACHE.clear()


def test_register_and_getattr_cache(monkeypatch):
    _clear_cache()
    env._register_exports({"foo": 1})
    assert env._EXPORT_CACHE["foo"] == 1
    assert env.foo == 1  # via globals update
    _clear_cache()
    env._EXPORT_CACHE["bar"] = 2
    assert env.__getattr__("bar") == 2
    with pytest.raises(AttributeError):
        env.__getattr__("missing")


def test_load_transformers_bundle_success(monkeypatch):
    _clear_cache()

    class FakeTFM:
        TrainerCallback = object

        @staticmethod
        def set_seed(_seed=None):
            return "seeded"

    def fake_import(name):
        if name == "transformers":
            return FakeTFM
        if name == "transformers.trainer_utils":
            return SimpleNamespace(get_last_checkpoint=lambda *_a, **_k: "chk")
        raise ImportError

    monkeypatch.setattr(env.importlib, "import_module", fake_import)
    exports = env._load_transformers_bundle()
    assert exports["transformers"] is FakeTFM
    assert exports["TrainerCallback"] is FakeTFM.TrainerCallback
    assert exports["get_last_checkpoint"]() == "chk"


def test_load_transformers_bundle_fallback(monkeypatch):
    _clear_cache()
    monkeypatch.setattr(env.importlib, "import_module", lambda name: (_ for _ in ()).throw(ImportError("missing")))
    exports = env._load_transformers_bundle()
    assert exports["transformers"] is None
    assert exports["TrainerCallback"] is env._StubTrainerCallback
    assert exports["set_seed"] is env._stub_set_seed
    assert exports["get_last_checkpoint"] is env._stub_get_last_checkpoint


def test_load_torch_bundle_cached_and_importerror(monkeypatch):
    _clear_cache()
    env._EXPORT_CACHE.update(
        {"torch": "t", "dist": "d", "torch_serialization": "s", "DataLoader": "dl", "RandomSampler": "rs"}
    )
    cached = env._load_torch_bundle()
    assert cached["torch"] == "t"
    _clear_cache()
    monkeypatch.setattr(env.importlib, "import_module", lambda name: (_ for _ in ()).throw(ImportError("no torch")))
    none_bundle = env._load_torch_bundle()
    assert none_bundle["torch"] is None and none_bundle["RandomSampler"] is None


def test_load_torch_bundle_success(monkeypatch):
    _clear_cache()
    calls = {}

    class FakeTorch:
        def __init__(self):
            self.load = lambda *a, **k: "loaded"

    class FakeSerialization:
        def __init__(self):
            self.safe_globals = None

        def add_safe_globals(self, mapping):
            calls["safe_globals"] = mapping

    class FakeData:
        DataLoader = object
        RandomSampler = object

    class FakeZeroCfg:
        ZeroStageEnum = "ZSE"

    class FakeZeroPartition:
        ZeroParamStatus = "ZPS"

    def fake_import(name):
        if name == "torch":
            return FakeTorch()
        if name == "torch.distributed":
            return "dist"
        if name == "torch.serialization":
            return FakeSerialization()
        if name == "torch.utils.data":
            return FakeData
        if name == "deepspeed.runtime.zero.config":
            return FakeZeroCfg
        if name == "deepspeed.runtime.zero.partition_parameters":
            return FakeZeroPartition
        raise ImportError(name)

    monkeypatch.setattr(env.importlib, "import_module", fake_import)
    bundle = env._load_torch_bundle()
    assert bundle["torch"] is not None
    assert bundle["dist"] == "dist"
    assert calls["safe_globals"][("deepspeed.runtime.zero.config", "ZeroStageEnum")] == "ZSE"
    # patched load should be functools.partial
    assert hasattr(bundle["torch"], "load")


def test_load_deepspeed_and_trl(monkeypatch):
    _clear_cache()
    # cached branch
    env._EXPORT_CACHE.update({"ZeroStageEnum": "ZSE", "ZeroParamStatus": "ZPS"})
    cached = env._load_deepspeed_bundle()
    assert cached["ZeroStageEnum"] == "ZSE"
    _clear_cache()

    def fake_import(name):
        if name == "deepspeed.runtime.zero.config":
            return SimpleNamespace(ZeroStageEnum="A")
        if name == "deepspeed.runtime.zero.partition_parameters":
            return SimpleNamespace(ZeroParamStatus="B")
        if name == "trl":
            return SimpleNamespace(get_peft_config=lambda *_: "peft")
        if name == "trl.trainer.grpo_trainer":
            return SimpleNamespace(GRPOTrainer="grpo")
        raise ImportError(name)

    monkeypatch.setattr(env.importlib, "import_module", fake_import)
    ds_bundle = env._load_deepspeed_bundle()
    assert ds_bundle["ZeroStageEnum"] == "A"
    trl_bundle = env._load_trl_bundle()
    assert trl_bundle["get_peft_config"]() == "peft"
    assert trl_bundle["GRPOTrainer"] == "grpo"

    # trl fallback
    monkeypatch.setattr(env.importlib, "import_module", lambda name: (_ for _ in ()).throw(ImportError(name)))
    fb_trl = env._load_trl_bundle()
    assert fb_trl["get_peft_config"] is env._stub_get_peft_config
    assert fb_trl["GRPOTrainer"] is env._StubGRPOTrainer


def test_dir_lists_exports():
    assert set(env.__dir__()) == set(env.__all__)
