import importlib
import sys
import types

import pytest


def _install_runtime_stubs(monkeypatch):
    real_import = importlib.import_module
    stub_main = types.ModuleType("src.training.runtime.main")
    stub_main.main = lambda *a, **k: None
    stub_runtime = types.ModuleType("src.training.runtime")
    stub_runtime.main = stub_main.main
    monkeypatch.setitem(sys.modules, "src.training.runtime.main", stub_main)
    monkeypatch.setitem(sys.modules, "src.training.runtime", stub_runtime)
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name, package=None: stub_runtime if name == "src.training.runtime" else real_import(name, package),
    )
    return stub_main


def test_grpo_impl_main_delegates(monkeypatch):
    stub_main = _install_runtime_stubs(monkeypatch)
    called = {}
    stub_grpo_runtime = types.ModuleType("src.training.grpo_runtime")
    stub_grpo_runtime.main = lambda *a, **k: called.setdefault("ran", True)
    monkeypatch.setitem(sys.modules, "src.training.grpo_runtime", stub_grpo_runtime)

    import src.training.grpo_impl as grpo_impl

    importlib.reload(grpo_impl)
    grpo_impl.main()
    assert called.get("ran") is True
    assert stub_main.main() is None  # stub callable


def test_grpo_runtime_alias(monkeypatch):
    stub_main = _install_runtime_stubs(monkeypatch)
    called = {}
    stub_main.main = lambda *a, **k: called.setdefault("ran", True)

    import src.training.grpo_runtime as grpo_runtime

    importlib.reload(grpo_runtime)
    grpo_runtime.main()
    assert called.get("ran") is True


def test_training_init_handles_missing_runtime(monkeypatch):
    real_import = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name, package=None: (_ for _ in ()).throw(ImportError("missing"))
        if name == "src.training.runtime"
        else real_import(name, package),
    )
    import src.training as training

    importlib.reload(training)
    assert "src.training.runtime" not in sys.modules or sys.modules["src.training.runtime"] is None


def test_training_utils_init_reload(monkeypatch):
    _install_runtime_stubs(monkeypatch)
    stub_trainer = types.ModuleType("src.training.utils.hierarchical_grpo_trainer")
    monkeypatch.setitem(sys.modules, "src.training.utils.hierarchical_grpo_trainer", stub_trainer)

    import src.training.utils as utils_mod

    importlib.reload(utils_mod)
    assert hasattr(utils_mod, "hierarchical_grpo_trainer")


def test_callbacks_import_utils_module_reports_last_exc(monkeypatch):
    _install_runtime_stubs(monkeypatch)
    import src.training.utils.callbacks as callbacks

    calls = []

    def fake_import(name):
        calls.append(name)
        raise ModuleNotFoundError(f"missing {name}")

    monkeypatch.setattr(callbacks, "import_module", fake_import)
    with pytest.raises(ModuleNotFoundError):
        callbacks._import_utils_module("not_there")
    assert calls == ["training.utils.not_there", "src.training.utils.not_there"]
