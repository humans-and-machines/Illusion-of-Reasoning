#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import src.inference.cli.unified_crossword as cli


def test_main_forwards_to_runner(monkeypatch):
    called = {}

    class DummyBackend:
        pass

    def fake_run_crossword_main(**kwargs):
        called.update(kwargs)

    def fake_loader():
        called["loader_called"] = True
        return "module"

    monkeypatch.setattr(cli, "run_crossword_main", fake_run_crossword_main)
    monkeypatch.setattr(cli, "HFBackend", DummyBackend)
    monkeypatch.setattr(cli, "_load_crossword_module", fake_loader)

    cli.main(argv=["--model", "m"])

    assert called["load_module"]() == "module"
    assert called["backend_cls"] is DummyBackend
    assert "argv" in called and called["argv"] == ["--model", "m"]


def test_load_crossword_module_wrapper(monkeypatch):
    monkeypatch.setattr(cli, "load_crossword_module", lambda: "loaded")
    assert cli._load_crossword_module() == "loaded"
