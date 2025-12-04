import sys
import types

import pytest

from src.inference.utils.task_registry import DatasetSpec, _resolve_callable


def test_resolve_callable_valid_and_invalid(monkeypatch):
    # Valid dotted path loads from sys.modules without real import.
    dummy = types.ModuleType("dummy_module")
    dummy.example = lambda: "ok"
    monkeypatch.setitem(sys.modules, "dummy_module", dummy)
    assert _resolve_callable("dummy_module:example")() == "ok"

    # Missing colon should raise.
    with pytest.raises(ValueError):
        _resolve_callable("dummy_module.example")


def test_dataset_spec_loader_fn_requires_resolvable():
    spec = DatasetSpec(loader="")  # falsy loader yields None from _resolve_callable
    with pytest.raises(ValueError, match="Could not resolve dataset loader"):
        spec.loader_fn()
