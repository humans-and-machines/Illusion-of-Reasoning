import sys

import pytest

import src.analysis.graph_3 as graph3


def test_main_delegates(monkeypatch):
    called = {}

    def fake_main():
        called["ran"] = True

    monkeypatch.setattr("src.analysis.graph_3._impl_main", fake_main)
    graph3.main()
    assert called.get("ran")


def test_module_dunder_main_runs(monkeypatch):
    called = {}

    def fake_main():
        called["ran"] = True

    monkeypatch.setattr("src.analysis.graph_3_impl.main", fake_main)
    # Ensure backward-compatible aliases exist
    assert getattr(graph3, "graph_3") is graph3.main
    assert getattr(graph3, "graph_3_impl") is graph3.main

    monkeypatch.setattr("sys.argv", ["graph_3"])
    import runpy

    sys.modules.pop("src.analysis.graph_3", None)
    runpy.run_module("src.analysis.graph_3", run_name="__main__")
    assert called.get("ran")


def test_graph3_alias_invokes_impl(monkeypatch):
    called = {}

    def fake_main():
        called["via_alias"] = True

    monkeypatch.setattr("src.analysis.graph_3._impl_main", fake_main)
    graph3.graph_3_impl()
    assert called.get("via_alias") is True


def test_graph3_impl_dunder_main_exits_without_data(monkeypatch):
    import runpy
    import sys

    argv_before = sys.argv
    sys.argv = ["graph_3_impl"]
    try:
        sys.modules.pop("src.analysis.graph_3_impl", None)
        with pytest.raises(SystemExit) as excinfo:
            runpy.run_module("src.analysis.graph_3_impl", run_name="__main__")
        assert excinfo.value.code == 2
    finally:
        sys.argv = argv_before
