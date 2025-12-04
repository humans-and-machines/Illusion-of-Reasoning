#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib
import sys
import types

import pytest


# Stub runtime.main to avoid pulling heavy imports during package init.
stub_main = types.ModuleType("src.training.runtime.main")
stub_main.main = lambda *a, **k: None
sys_modules = sys.modules  # type: ignore[name-defined]
sys_modules["src.training.runtime.main"] = stub_main

import src.training.runtime.env as env  # noqa: E402


def test_stub_constructors_and_dataset_fallback(monkeypatch):
    # Instantiate stubs to exercise __init__ branches.
    assert env._StubTrainerCallback() is not None
    assert env._StubGRPOTrainer() is not None

    # Make datasets import fail to hit fallback path.
    monkeypatch.setattr(env, "importlib", importlib)
    monkeypatch.setattr(
        env.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError()) if name == "datasets" else None,
    )
    assert env._load_datasets()["datasets"] is None


def test_load_accelerate_cache_hit_and_transformers_cache(monkeypatch):
    cache_before = env._EXPORT_CACHE.copy()
    try:
        env._EXPORT_CACHE.clear()
        env._EXPORT_CACHE["AcceleratorState"] = "cached_accel"
        assert env._load_accelerate()["AcceleratorState"] == "cached_accel"

        env._EXPORT_CACHE.update(
            {
                "transformers": "tmod",
                "TrainerCallback": "cb",
                "set_seed": "seed",
                "get_last_checkpoint": "chk",
            }
        )
        out = env._load_transformers_bundle()
        assert out["transformers"] == "tmod"
    finally:
        env._EXPORT_CACHE.clear()
        env._EXPORT_CACHE.update(cache_before)


def test_load_accelerate_import_success(monkeypatch):
    cache_before = env._EXPORT_CACHE.copy()
    try:
        env._EXPORT_CACHE.clear()
        fake_accel = types.SimpleNamespace(AcceleratorState="accel_cls")
        monkeypatch.setattr(env, "importlib", importlib)
        monkeypatch.setattr(
            env.importlib,
            "import_module",
            lambda name: fake_accel if name == "accelerate.state" else (_ for _ in ()).throw(AssertionError(name)),
        )
        out = env._load_accelerate()
        assert out["AcceleratorState"] == "accel_cls"
        assert env._EXPORT_CACHE == {}  # loader should not mutate cache directly
    finally:
        env._EXPORT_CACHE.clear()
        env._EXPORT_CACHE.update(cache_before)


def test_load_accelerate_import_failure(monkeypatch):
    cache_before = env._EXPORT_CACHE.copy()
    try:
        env._EXPORT_CACHE.clear()
        monkeypatch.setattr(env, "importlib", importlib)
        monkeypatch.setattr(
            env.importlib,
            "import_module",
            lambda name: (_ for _ in ()).throw(ImportError()) if name == "accelerate.state" else None,
        )
        out = env._load_accelerate()
        assert out["AcceleratorState"] is None
    finally:
        env._EXPORT_CACHE.clear()
        env._EXPORT_CACHE.update(cache_before)


def test_load_trl_bundle_cache(monkeypatch):
    cache_before = env._EXPORT_CACHE.copy()
    try:
        env._EXPORT_CACHE.clear()
        env._EXPORT_CACHE["get_peft_config"] = "peft"
        env._EXPORT_CACHE["GRPOTrainer"] = "trainer"
        out = env._load_trl_bundle()
        assert out == {"get_peft_config": "peft", "GRPOTrainer": "trainer"}
    finally:
        env._EXPORT_CACHE.clear()
        env._EXPORT_CACHE.update(cache_before)


def test_patch_torch_serialization_applies(monkeypatch):
    class TorchSer:
        def __init__(self):
            self.added = None

        def add_safe_globals(self, mapping):
            self.added = mapping

    class TorchMod:
        def __init__(self):
            self.load_calls = []

        def load(self, *args, **kwargs):
            self.load_calls.append((args, kwargs))
            return "loaded"

    ser = TorchSer()
    torch_mod = TorchMod()
    env._patch_torch_serialization(torch_mod, ser, "zero_enum", "zero_status")
    assert ser._default_weights_only is False  # type: ignore[attr-defined]
    assert ("deepspeed.runtime.zero.config", "ZeroStageEnum") in ser.added
    torch_mod.load("file")
    assert torch_mod.load_calls[0][1]["weights_only"] is False


def test_getattr_and_dir(monkeypatch):
    # Temporary loader that records calls.
    called = {}

    def loader():
        called["ran"] = True
        return {"dummy": 123}

    monkeypatch.setitem(env._LAZY_LOADERS, "dummy", loader)
    env._EXPORT_CACHE.clear()
    assert getattr(env, "dummy") == 123
    assert called["ran"] is True
    with pytest.raises(AttributeError):
        getattr(env, "missing_attr")
    assert "dummy" not in env.__dir__()  # __all__ does not include ad-hoc keys
