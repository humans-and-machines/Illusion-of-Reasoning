import importlib
import sys
import types


def test_main_delegates_to_components(monkeypatch):
    stub = types.SimpleNamespace(called=0)

    def fake_main():
        stub.called += 1

    # Provide stubbed core component module
    mod_name = "src.analysis.core.figure_1_components"
    monkeypatch.setitem(sys.modules, mod_name, types.SimpleNamespace(main=fake_main))

    # Reload wrapper so it picks up the stub
    module = importlib.reload(importlib.import_module("src.analysis.figure_1"))

    module.main()
    assert stub.called == 1
