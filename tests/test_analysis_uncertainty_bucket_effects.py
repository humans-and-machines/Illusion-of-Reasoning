import importlib
import runpy
import sys
from types import SimpleNamespace


def _fresh_import(monkeypatch, run_as_main: bool = False):
    calls = []

    monkeypatch.setitem(
        sys.modules,
        "src.analysis.script_utils",
        SimpleNamespace(ensure_script_context=lambda: calls.append("ensure")),
    )
    monkeypatch.setitem(
        sys.modules,
        "src.analysis.uncertainty_bucket_effects_impl",
        SimpleNamespace(main=lambda: calls.append("impl_main")),
    )
    # Force a fresh import of the wrapper under test.
    monkeypatch.delitem(sys.modules, "src.analysis.uncertainty_bucket_effects", raising=False)

    if run_as_main:
        runpy.run_module("src.analysis.uncertainty_bucket_effects", run_name="__main__")
    else:
        importlib.import_module("src.analysis.uncertainty_bucket_effects")

    return calls


def test_import_invokes_ensure_script_context_only(monkeypatch):
    calls = _fresh_import(monkeypatch, run_as_main=False)
    assert calls and "impl_main" not in calls
    assert all(entry == "ensure" for entry in calls)


def test_run_as_script_forwards_to_impl_main(monkeypatch):
    calls = _fresh_import(monkeypatch, run_as_main=True)
    assert calls[-1] == "impl_main"
    # ensure_script_context should run before delegating to impl.main (possibly more than once)
    assert "ensure" in calls
    assert calls.index("ensure") < calls.index("impl_main")
