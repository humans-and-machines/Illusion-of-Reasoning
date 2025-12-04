#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import builtins
import importlib
import sys
import types


def test_training_init_pops_partial_runtime(monkeypatch):
    # Force a fresh import path and ensure any pre-existing entries are removed.
    sys.modules.pop("src.training", None)
    sys.modules["src.training.runtime"] = types.SimpleNamespace()

    real_import_module = importlib.import_module

    def fake_import_module(name):
        if name == "src.training.runtime":
            raise ImportError("missing runtime")
        return real_import_module(name)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    module = importlib.import_module("src.training")

    assert module is sys.modules["src.training"]
    assert "src.training.runtime" not in sys.modules


def test_training_init_handles_utils_import_error(monkeypatch):
    sys.modules.pop("src.training", None)
    sys.modules.pop("src.training.utils", None)

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "src.training.utils":
            raise ImportError("boom")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    module = importlib.import_module("src.training")
    assert module is sys.modules["src.training"]

    sys.modules.pop("src.training", None)
    sys.modules.pop("src.training.utils", None)
