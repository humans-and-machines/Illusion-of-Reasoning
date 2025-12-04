from types import SimpleNamespace

import pytest

from src.inference.gateways import __all__ as gateways_all
from src.inference.gateways import base, providers
from src.inference.utils.task_registry import TaskSpec


def test_gateways_init_exports_list_exists():
    # gateways.__all__ is intentionally empty but should be defined.
    assert isinstance(gateways_all, list)


def test_get_task_spec_returns_and_raises(monkeypatch):
    dummy_spec = TaskSpec(name="dummy", config={})
    monkeypatch.setattr(base, "TASK_REGISTRY", {"dummy": dummy_spec})
    assert base.get_task_spec("dummy") is dummy_spec
    with pytest.raises(KeyError):
        base.get_task_spec("missing")


def test_setup_gateway_logger_delegates(monkeypatch):
    captured = {}

    def fake_setup(name):
        captured["name"] = name
        return SimpleNamespace(name=name)

    monkeypatch.setattr(base, "setup_script_logger", fake_setup)
    logger = base.setup_gateway_logger(None)
    assert logger.name == base.__name__
    logger2 = base.setup_gateway_logger("custom")
    assert logger2.name == "custom"


def test_providers_import_no_errors():
    # Module should import cleanly; nothing else to assert.
    assert providers.__name__.endswith("providers")
