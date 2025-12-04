import importlib
import runpy
import sys
import types


def _stub_wrapper_deps(monkeypatch, calls):
    """Install stubbed dependencies before importing the wrapper."""
    stub_utils = types.SimpleNamespace(ensure_script_context=lambda: calls.append("ctx"))
    stub_impl = types.SimpleNamespace(main=lambda: calls.append("main"))
    monkeypatch.setitem(sys.modules, "src.analysis.script_utils", stub_utils)
    monkeypatch.setitem(sys.modules, "src.analysis.forced_aha_effect_impl", stub_impl)
    monkeypatch.delitem(sys.modules, "src.analysis.forced_aha_effect", raising=False)


def test_import_calls_ensure_context(monkeypatch):
    calls = []
    _stub_wrapper_deps(monkeypatch, calls)

    mod = importlib.import_module("src.analysis.forced_aha_effect")

    assert calls == ["ctx"]  # main not called on import
    assert mod.main is sys.modules["src.analysis.forced_aha_effect_impl"].main


def test_dunder_main_invokes_main(monkeypatch):
    calls = []
    _stub_wrapper_deps(monkeypatch, calls)

    runpy.run_module("src.analysis.forced_aha_effect", run_name="__main__")

    assert calls == ["ctx", "main"]
