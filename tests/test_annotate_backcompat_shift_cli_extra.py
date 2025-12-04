import importlib
import sys

import pytest


def test_getattr_delegates_fresh_impl(monkeypatch):
    calls = []

    class Stub1:
        def __init__(self):
            self.main = lambda: calls.append("main1")
            self.build_argparser = lambda: calls.append("arg1")

    class Stub2:
        def __init__(self):
            self.main = lambda: calls.append("main2")
            self.build_argparser = lambda: calls.append("arg2")

    # First import uses Stub1
    monkeypatch.setitem(sys.modules, "src.annotate.cli.shift_cli", Stub1())
    sys.modules.pop("src.annotate.backcompat.shift_cli", None)
    mod = importlib.import_module("src.annotate.backcompat.shift_cli")
    assert mod.main is sys.modules["src.annotate.cli.shift_cli"].main
    assert mod.build_argparser is sys.modules["src.annotate.cli.shift_cli"].build_argparser

    # Remove cached attributes to force __getattr__ and swap underlying impl.
    monkeypatch.setitem(sys.modules, "src.annotate.cli.shift_cli", Stub2())
    assert callable(mod.main)
    mod.main()
    mod.build_argparser()
    assert calls == ["main2", "arg2"]


def test_getattr_raises_for_unknown(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "src.annotate.cli.shift_cli",
        type("Stub", (), {"main": lambda: None, "build_argparser": lambda: None}),
    )
    sys.modules.pop("src.annotate.backcompat.shift_cli", None)
    mod = importlib.import_module("src.annotate.backcompat.shift_cli")
    with pytest.raises(AttributeError):
        getattr(mod, "missing_attr")
