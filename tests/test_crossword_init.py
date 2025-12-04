#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib
import sys
import types


def _install_crossword_stub(monkeypatch):
    stub_core = types.ModuleType("src.inference.domains.crossword.crossword_core")
    stub_core.SENTINEL = object()

    monkeypatch.setitem(sys.modules, stub_core.__name__, stub_core)
    monkeypatch.delitem(sys.modules, "src.inference.domains.crossword", raising=False)
    return stub_core


def test_crossword_init_exposes_crossword_core(monkeypatch):
    stub_core = _install_crossword_stub(monkeypatch)
    mod = importlib.import_module("src.inference.domains.crossword")

    assert mod.crossword_core is stub_core
    assert getattr(mod, "__all__", []) == ["crossword_core"]


def test_crossword_init_star_import_uses_dunder_all(monkeypatch):
    stub_core = _install_crossword_stub(monkeypatch)

    namespace = {}
    exec("from src.inference.domains.crossword import *", {}, namespace)

    assert namespace["crossword_core"] is stub_core
    assert getattr(namespace["crossword_core"], "SENTINEL") is stub_core.SENTINEL


def test_crossword_init_reload_picks_up_new_core(monkeypatch):
    stub_core = _install_crossword_stub(monkeypatch)
    mod = importlib.import_module("src.inference.domains.crossword")
    assert mod.crossword_core is stub_core

    fresh_core = types.ModuleType("src.inference.domains.crossword.crossword_core")
    fresh_core.MARKER = "fresh"
    monkeypatch.setitem(sys.modules, fresh_core.__name__, fresh_core)
    monkeypatch.delitem(sys.modules, "src.inference.domains.crossword", raising=False)

    reimported = importlib.import_module("src.inference.domains.crossword")
    assert reimported.crossword_core is fresh_core
    assert getattr(reimported.crossword_core, "MARKER") == "fresh"
