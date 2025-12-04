import importlib
import runpy
import sys


def test_main_delegates_to_clean_cli(monkeypatch):
    import src.annotate.cli.clean_cli as clean_cli

    called = {"count": 0}

    def fake_main():
        called["count"] += 1
        return "ok"

    monkeypatch.setattr(clean_cli, "main", fake_main)
    module = importlib.reload(importlib.import_module("src.annotate.cli.clean_failed_shift_labels"))

    module.main()

    assert called["count"] == 1
    assert "main" in module.__all__


def test_module_guard_invokes_main(monkeypatch):
    # Reload to ensure we patch the module used by runpy.
    importlib.import_module("src.annotate.cli.clean_failed_shift_labels")
    called = {}
    stub_clean_cli = type("StubCleanCli", (), {"main": staticmethod(lambda: called.setdefault("ran", True))})
    monkeypatch.setitem(sys.modules, "src.annotate.cli.clean_cli", stub_clean_cli)
    monkeypatch.setattr(sys, "argv", ["prog"])
    monkeypatch.delitem(sys.modules, "src.annotate.cli.clean_failed_shift_labels", raising=False)

    runpy.run_module("src.annotate.cli.clean_failed_shift_labels", run_name="__main__", alter_sys=True)
    assert called.get("ran") is True
