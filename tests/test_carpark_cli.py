#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import runpy
import sys
import types
from importlib import import_module
from types import SimpleNamespace


def test_load_carpark_module_uses_import_module(monkeypatch):
    called = {}

    def fake_import(name):
        called["name"] = name
        return "carpark_module"

    monkeypatch.setattr("src.inference.domains.carpark.carpark_cli.import_module", fake_import)
    mod = import_module("src.inference.domains.carpark.carpark_cli")
    assert mod._load_carpark_module() == "carpark_module"
    assert called["name"].endswith("carpark.carpark_core")


def test_main_invokes_runner(monkeypatch):
    calls = {}

    def fake_run(load_module, backend_cls, argv=None):
        calls["load_module"] = load_module
        calls["backend_cls"] = backend_cls
        calls["argv"] = argv

    monkeypatch.setattr("src.inference.domains.carpark.carpark_cli.run_carpark_main", fake_run)
    mod = import_module("src.inference.domains.carpark.carpark_cli")
    mod.main(argv=["--help"])
    assert calls["load_module"] is mod._load_carpark_module
    assert calls["backend_cls"].__name__ == "HFBackend"
    assert calls["argv"] == ["--help"]


def test_carpark_core_main_delegates(monkeypatch):
    # Stub heavy dependencies before importing carpark_core.
    stub_data = SimpleNamespace(
        _canon_rush_generic=None,
        _canon_rush_gold=None,
        load_rush_dataset=lambda *_a, **_k: None,
    )
    stub_solver = SimpleNamespace(
        SYSTEM_PROMPT="prompt",
        CarparkInferenceConfig=object,
        InferenceContext=object,
        run_inference_on_split=lambda *_a, **_k: None,
    )
    monkeypatch.setitem(sys.modules, "src.inference.domains.carpark.carpark_data", stub_data)
    monkeypatch.setitem(sys.modules, "src.inference.domains.carpark.carpark_solver", stub_solver)
    monkeypatch.delitem(sys.modules, "src.inference.domains.carpark.carpark_core", raising=False)

    carpark_core = import_module("src.inference.domains.carpark.carpark_core")

    calls = {}

    def fake_import(name):
        calls["name"] = name
        return SimpleNamespace(main=lambda argv=None: calls.setdefault("argv", argv))

    monkeypatch.setattr(carpark_core, "import_module", fake_import)
    carpark_core.main(["--x"])

    assert calls["name"].endswith("carpark_cli")
    assert calls["argv"] == ["--x"]


def test_main_guard_executes_run_carpark(monkeypatch):
    calls = {}
    # Lightweight backend and runner stubs to avoid heavy imports.
    stub_backend_mod = types.SimpleNamespace(HFBackend=type("HFBackend", (), {}))
    monkeypatch.setitem(sys.modules, "src.inference.backends", stub_backend_mod)
    monkeypatch.setattr(
        "src.inference.runners.unified_runner_base.run_carpark_main",
        lambda **kwargs: calls.setdefault("kwargs", kwargs),
    )
    # Reload module as __main__ to hit the guard.
    monkeypatch.delitem(sys.modules, "src.inference.domains.carpark.carpark_cli", raising=False)
    runpy.run_module("src.inference.domains.carpark.carpark_cli", run_name="__main__")
    assert "kwargs" in calls
    assert calls["kwargs"]["backend_cls"].__name__ == "HFBackend"
