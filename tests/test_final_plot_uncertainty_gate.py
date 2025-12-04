import runpy
import sys

from src.analysis import final_plot_uncertainty_gate


def test_final_plot_uncertainty_gate_main_delegates(monkeypatch):
    called = {}

    def fake_main():
        called["ran"] = True

    monkeypatch.setattr(final_plot_uncertainty_gate.final_plot, "main", fake_main)
    final_plot_uncertainty_gate.main()
    assert called["ran"] is True


def test_final_plot_uncertainty_gate_runs_under_dunder_main(monkeypatch):
    called = {}

    def fake_main():
        called["ran"] = True

    monkeypatch.setattr(final_plot_uncertainty_gate.final_plot, "main", fake_main)
    monkeypatch.delitem(sys.modules, "src.analysis.final_plot_uncertainty_gate", raising=False)
    runpy.run_module("src.analysis.final_plot_uncertainty_gate", run_name="__main__")
    assert called["ran"] is True
