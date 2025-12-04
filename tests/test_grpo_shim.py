import importlib
import runpy
import sys
from types import SimpleNamespace


def test_main_delegates_with_merge(monkeypatch):
    called = {}
    stub_cli = SimpleNamespace(main=lambda merge_fn=None: called.setdefault("merge", merge_fn))
    monkeypatch.setitem(sys.modules, "src.training.cli.grpo", stub_cli)
    # Force fresh import to pick up stub.
    sys.modules.pop("src.training.grpo", None)
    mod = importlib.import_module("src.training.grpo")
    mod.main()
    assert called["merge"] is mod.merge_dataclass_attributes


def test_module_entrypoint_executes(monkeypatch):
    called = {}
    stub_cli = SimpleNamespace(main=lambda merge_fn=None: called.setdefault("ran", True))
    monkeypatch.setitem(sys.modules, "src.training.cli.grpo", stub_cli)
    sys.modules.pop("src.training.grpo", None)
    runpy.run_module("src.training.grpo", run_name="__main__")
    assert called.get("ran") is True
