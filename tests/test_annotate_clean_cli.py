#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import runpy
import sys
from types import SimpleNamespace


def test_clean_cli_main_calls_clean_root(monkeypatch):
    import src.annotate.cli.clean_cli as clean_cli

    called = {}

    def fake_clean_root(root):
        called["root"] = root

    monkeypatch.setattr(clean_cli, "clean_root", fake_clean_root)
    monkeypatch.setattr(
        clean_cli.argparse.ArgumentParser, "parse_args", lambda self: SimpleNamespace(results_root="ROOT")
    )

    clean_cli.main()
    assert called["root"] == "ROOT"


def test_clean_cli_guard_runs(monkeypatch):
    called = {}

    def fake_clean_root(root):
        called["root"] = root

    monkeypatch.setitem(sys.modules, "src.annotate.core.clean_core", SimpleNamespace(clean_root=fake_clean_root))
    monkeypatch.delitem(sys.modules, "src.annotate.cli.clean_cli", raising=False)
    monkeypatch.setattr(sys, "argv", ["prog", "ROOTDIR"])

    runpy.run_module("src.annotate.cli.clean_cli", run_name="__main__")
    assert called["root"] == "ROOTDIR"
