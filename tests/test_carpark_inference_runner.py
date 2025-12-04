from __future__ import annotations

import runpy
import sys
import types


def test_carpark_runner_delegates_to_core(monkeypatch):
    # Ensure we import a fresh runner wired to a stubbed carpark_core.main.
    sys.modules.pop("src.inference.runners.carpark_inference_runner", None)
    sys.modules.pop("src.inference.domains.carpark.carpark_core", None)

    called = {}

    def fake_main(argv=None):
        called["argv"] = argv

    stub_core = types.ModuleType("src.inference.domains.carpark.carpark_core")
    stub_core.main = fake_main
    monkeypatch.setitem(sys.modules, "src.inference.domains.carpark.carpark_core", stub_core)

    # Running as __main__ should trigger the stubbed core main.
    runpy.run_module("src.inference.runners.carpark_inference_runner", run_name="__main__")

    assert called.get("argv") is None
