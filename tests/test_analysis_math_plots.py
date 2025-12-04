import importlib
import runpy
import sys
import types


def _install_stub(monkeypatch):
    """Install a stub implementation module so math_plots delegates safely."""
    called = {"count": 0}
    stub_impl = types.ModuleType("src.analysis.math_plots_impl")

    def stub_main():
        called["count"] += 1

    stub_impl.main = stub_main

    monkeypatch.setitem(sys.modules, "src.analysis.math_plots_impl", stub_impl)
    # Ensure a fresh import of the wrapper uses the stub
    monkeypatch.delitem(sys.modules, "src.analysis.math_plots", raising=False)
    return called


def test_main_delegates_to_impl(monkeypatch):
    called = _install_stub(monkeypatch)
    wrapper = importlib.import_module("src.analysis.math_plots")
    importlib.reload(wrapper)

    wrapper.main()
    assert called["count"] == 1


def test_cli_entrypoint_runs_impl(monkeypatch):
    called = _install_stub(monkeypatch)
    runpy.run_module("src.analysis.math_plots", run_name="__main__")
    assert called["count"] == 1
