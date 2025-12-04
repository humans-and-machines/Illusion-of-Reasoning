#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest


def _clear_math_modules(monkeypatch):
    for name in list(sys.modules):
        if name == "src.inference.domains.math" or name.startswith("src.inference.domains.math."):
            monkeypatch.delitem(sys.modules, name, raising=False)


def test_getattr_loads_math_submodules(monkeypatch):
    _clear_math_modules(monkeypatch)

    calls = []
    real_import = importlib.import_module

    def fake_import(name):
        if name == "src.inference.domains.math":
            return real_import(name)
        if name.startswith("src.inference.domains.math."):
            calls.append(name)
            module = SimpleNamespace(name=name)
            monkeypatch.setitem(sys.modules, name, module)
            pkg_name, _, attr = name.rpartition(".")
            pkg = sys.modules.get(pkg_name)
            if pkg is not None:
                setattr(pkg, attr, module)
            return module
        return real_import(name)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    math_pkg = importlib.import_module("src.inference.domains.math")

    assert math_pkg.math_core.name == "src.inference.domains.math.math_core"
    assert math_pkg.math_core_runner.name == "src.inference.domains.math.math_core_runner"
    assert math_pkg.math_llama_core.name == "src.inference.domains.math.math_llama_core"
    assert calls == [
        "src.inference.domains.math.math_core",
        "src.inference.domains.math.math_core_runner",
        "src.inference.domains.math.math_llama_core",
    ]

    calls.clear()
    _ = math_pkg.math_core
    assert calls == []


def test_getattr_raises_for_unknown(monkeypatch):
    _clear_math_modules(monkeypatch)
    math_pkg = importlib.import_module("src.inference.domains.math")
    with pytest.raises(AttributeError):
        _ = math_pkg.missing_submodule
