import importlib
import runpy
import sys


def test_import_reexports_main_and_argparser(monkeypatch):
    calls = []

    def fake_main():
        calls.append("main")

    def fake_build():
        calls.append("build")

    monkeypatch.setitem(
        sys.modules, "src.annotate.cli.shift_cli", type("m", (), {"main": fake_main, "build_argparser": fake_build})
    )
    mod = importlib.import_module("src.annotate.backcompat.shift_cli")
    assert mod.main is fake_main
    assert mod.build_argparser is fake_build
    # Accessing __all__ should reflect the shimmed functions.
    assert set(mod.__all__) == {"build_argparser", "main"}


def test_exec_as_script_calls_main(monkeypatch):
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "src.annotate.cli.shift_cli",
        type("m", (), {"main": lambda: calls.append("main"), "build_argparser": lambda: None}),
    )
    # Ensure fresh import for __main__ execution.
    sys.modules.pop("src.annotate.backcompat.shift_cli", None)
    runpy.run_module("src.annotate.backcompat.shift_cli", run_name="__main__")
    assert "main" in calls
