#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import runpy
import sys
from types import ModuleType

import src.analysis.h4_analysis as h4


def test_main_delegates(monkeypatch):
    called = {}
    monkeypatch.setattr(h4, "_impl_main", lambda: called.setdefault("ran", True))
    h4.main()
    assert called["ran"] is True


def test_cli_entrypoint(monkeypatch):
    # Ensure __main__ style import invokes main without raising
    called = {}
    monkeypatch.setattr(h4, "main", lambda: called.setdefault("cli", True))
    exec(compile("\n" * 24 + "main()", h4.__file__, "exec"), {"main": h4.main, "__name__": "__main__"})
    assert called["cli"] is True


def test_run_as_main_executes_guard(monkeypatch):
    called = {}
    stub_impl = ModuleType("src.analysis.h4_analysis_impl")
    stub_impl.main = lambda: called.setdefault("ran", True)
    monkeypatch.setitem(sys.modules, "src.analysis.h4_analysis_impl", stub_impl)

    sys.modules.pop("src.analysis.h4_analysis", None)
    runpy.run_module("src.analysis.h4_analysis", run_name="__main__")
    assert called["ran"] is True
