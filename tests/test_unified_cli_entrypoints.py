#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import runpy
import sys
import types


def _with_stub_modules(stubs):
    """Context manager to temporarily install stub modules into sys.modules."""

    class _Ctx:
        def __enter__(self):
            self._orig = {name: sys.modules.get(name) for name in stubs}
            sys.modules.update(stubs)
            return self

        def __exit__(self, exc_type, exc, tb):
            for name, mod in self._orig.items():
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod
            return False

    return _Ctx()


def test_unified_carpark_main_guard_invokes_run(monkeypatch):
    called = {}
    runner_base = types.ModuleType("src.inference.runners.unified_runner_base")
    runner_base.run_carpark_main = lambda **kwargs: called.setdefault("kwargs", kwargs)

    carpark_runner = types.ModuleType("src.inference.runners.unified_carpark_runner")
    carpark_runner.load_carpark_module = lambda: "carpark_mod"

    backends = types.ModuleType("src.inference.backends")
    backends.HFBackend = type("HFBackend", (), {})

    stubs = {
        "src.inference.runners.unified_runner_base": runner_base,
        "src.inference.runners.unified_carpark_runner": carpark_runner,
        "src.inference.backends": backends,
    }

    sys.modules.pop("src.inference.cli.unified_carpark", None)
    with _with_stub_modules(stubs):
        runpy.run_module("src.inference.cli.unified_carpark", run_name="__main__", alter_sys=True)

    kwargs = called["kwargs"]
    assert kwargs["load_module"]() == "carpark_mod"
    assert kwargs["backend_cls"] is backends.HFBackend


def test_unified_crossword_main_guard_invokes_run(monkeypatch):
    called = {}
    runner_base = types.ModuleType("src.inference.runners.unified_runner_base")
    runner_base.run_crossword_main = lambda **kwargs: called.setdefault("kwargs", kwargs)

    crossword_runner = types.ModuleType("src.inference.runners.unified_crossword_runner")
    crossword_runner.load_crossword_module = lambda: "cross_mod"

    backends = types.ModuleType("src.inference.backends")
    backends.HFBackend = type("HFBackend", (), {})

    stubs = {
        "src.inference.runners.unified_runner_base": runner_base,
        "src.inference.runners.unified_crossword_runner": crossword_runner,
        "src.inference.backends": backends,
    }

    sys.modules.pop("src.inference.cli.unified_crossword", None)
    with _with_stub_modules(stubs):
        runpy.run_module("src.inference.cli.unified_crossword", run_name="__main__", alter_sys=True)

    kwargs = called["kwargs"]
    assert kwargs["load_module"]() == "cross_mod"
    assert kwargs["backend_cls"] is backends.HFBackend


def test_unified_math_main_guard_invokes_run(monkeypatch):
    called = {}
    runner_base = types.ModuleType("src.inference.runners.unified_runner_base")
    runner_base.run_math_main = lambda **kwargs: called.setdefault("kwargs", kwargs)

    backends = types.ModuleType("src.inference.backends")
    backends.HFBackend = type("HFBackend", (), {})

    stubs = {
        "src.inference.runners.unified_runner_base": runner_base,
        "src.inference.backends": backends,
    }

    sys.modules.pop("src.inference.cli.unified_math", None)
    with _with_stub_modules(stubs):
        runpy.run_module("src.inference.cli.unified_math", run_name="__main__", alter_sys=True)

    kwargs = called["kwargs"]
    assert kwargs["backend_cls"] is backends.HFBackend
