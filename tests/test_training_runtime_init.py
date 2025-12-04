import importlib
import sys
import types


def test_runtime_init_exports_main(monkeypatch):
    # Stub heavy runtime dependencies to allow import
    FakeTensor = type("FakeTensor", (), {})
    fake_torch = types.SimpleNamespace(
        no_grad=lambda: types.SimpleNamespace(
            __enter__=lambda s: None, __exit__=lambda *a: None, __call__=lambda self, fn=None: fn
        ),
        Tensor=FakeTensor,
        SymFloat=FakeTensor,  # satisfy attribute lookups in optional deps
        SymBool=FakeTensor,
        ones=lambda *a, **k: None,
        long="long",
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "pydantic", types.SimpleNamespace(BaseModel=object, Field=lambda *a, **k: None))
    monkeypatch.setitem(sys.modules, "pydantic_settings", types.SimpleNamespace(BaseSettings=object))

    # Short-circuit runtime.main import to avoid pulling full trainer stack.
    stub_main_mod = types.ModuleType("src.training.runtime.main")
    stub_main_mod.main = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "src.training.runtime.main", stub_main_mod)

    mod = importlib.import_module("src.training.runtime")
    assert hasattr(mod, "main")
    assert "main" in mod.__all__


def test_runtime_init_runpy_executes(monkeypatch):
    stub_main_mod = types.ModuleType("src.training.runtime.main")
    stub_main_mod.main = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "src.training.runtime.main", stub_main_mod)
    monkeypatch.delitem(sys.modules, "src.training.runtime", raising=False)

    mod = importlib.import_module("src.training.runtime")
    assert mod.main is stub_main_mod.main
    assert "main" in getattr(mod, "__all__", [])
