#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import runpy
import sys
import types


def test_backcompat_clean_failed_shift_labels_shim(monkeypatch):
    from src.annotate.backcompat import clean_failed_shift_labels as shim

    called = {}

    def fake_main():
        called["ran"] = True

    monkeypatch.setattr(shim, "main", fake_main)
    shim.main()
    assert called.get("ran") is True


def test_backcompat_clean_failed_shift_labels_main_guard(monkeypatch):
    called = {}

    stub_clean_cli = types.ModuleType("src.annotate.cli.clean_cli")
    stub_clean_cli.main = lambda: called.setdefault("ran", True)
    monkeypatch.setitem(sys.modules, "src.annotate.cli.clean_cli", stub_clean_cli)
    monkeypatch.delitem(sys.modules, "src.annotate.backcompat.clean_failed_shift_labels", raising=False)

    runpy.run_module("src.annotate.backcompat.clean_failed_shift_labels", run_name="__main__", alter_sys=True)

    assert called.get("ran") is True
