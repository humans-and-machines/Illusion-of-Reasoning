#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib
import sys

import pytest


def test_getattr_loads_submodules(monkeypatch):
    calls = {}
    real_import = importlib.import_module

    # Stub importlib.import_module to observe imports and return sentinel objects.
    def fake_import(name):
        if name == "src.analysis":
            return real_import(name)
        calls[name] = True
        return f"module:{name}"

    monkeypatch.setattr(importlib, "import_module", fake_import)

    # Ensure a fresh import of the package under test so __getattr__ runs.
    sys.modules.pop("src.analysis", None)
    mod = importlib.import_module("src.analysis")

    # Access lazy submodules and confirm memoization.
    assert mod.rq1_analysis == "module:src.analysis.rq1_analysis"
    assert mod.rq2_analysis == "module:src.analysis.rq2_analysis"
    # Second access should reuse cached globals without re-importing.
    calls.clear()
    _ = mod.rq1_analysis
    assert calls == {}


def test_getattr_loads_common_wrappers(monkeypatch):
    common = type("Common", (), {"io": "io_mod", "labels": "labels_mod", "metrics": "metrics_mod"})
    monkeypatch.setitem(sys.modules, "src.analysis.common", common)

    real_import = importlib.import_module

    def fake_import(name):
        if name == "src.analysis":
            return real_import(name)
        if name == "src.analysis.common":
            return common
        pytest.fail(f"Unexpected import {name}")

    monkeypatch.setattr(importlib, "import_module", fake_import)

    sys.modules.pop("src.analysis", None)
    mod = importlib.import_module("src.analysis")

    assert mod.io == "io_mod"
    assert mod.labels == "labels_mod"
    assert mod.metrics == "metrics_mod"


def test_getattr_raises_for_unknown(monkeypatch):
    sys.modules.pop("src.analysis", None)
    mod = importlib.import_module("src.analysis")
    with pytest.raises(AttributeError):
        _ = mod.missing_attr
