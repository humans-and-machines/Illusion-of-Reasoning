#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import runpy
import sys
import types
from pathlib import Path


class _StubLogger:
    def info(self, *args, **kwargs):
        return None


def _with_stub_modules(stubs):
    """Temporarily install stub modules into sys.modules."""

    class _Ctx:
        def __enter__(self):
            self._orig = {name: sys.modules.get(name) for name in stubs}
            sys.modules.update(stubs)
            return self

        def __exit__(self, exc_type, exc, tb):
            for name, mod in self._orig.items():
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod
            return False

    return _Ctx()


def _make_common_stub(calls):
    common = types.ModuleType("src.inference.utils.common")

    def build_math_gateway_arg_parser(default_temperature=0.0, description=""):
        parser = argparse.ArgumentParser()
        parser.add_argument("--output_dir", default="/tmp/out")
        parser.add_argument("--split", default="train")
        parser.add_argument("--step", type=int, default=1)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--temperature", type=float, default=default_temperature)
        parser.add_argument("--top_p", type=float, default=0.9)
        parser.add_argument("--max_output_tokens", type=int, default=10)
        parser.add_argument("--request_timeout", type=int, default=5)
        parser.add_argument("--num_samples", type=int, default=1)
        return parser

    def prepare_math_gateway_dataset_from_args(*args, **kwargs):
        calls["prepare"] = True
        return [], {}, "name"

    common.build_math_gateway_arg_parser = build_math_gateway_arg_parser
    common.prepare_math_gateway_dataset_from_args = prepare_math_gateway_dataset_from_args
    common.append_jsonl_row = lambda *a, **k: None
    common.build_math_gateway_row_base = lambda **kwargs: kwargs
    common.build_usage_dict = lambda usage: {"usage": usage}
    common.call_with_gateway_retries = lambda *a, **k: ("", None, None)
    common.canon_math = lambda text: text
    common.extract_blocks = lambda text: ("", text)
    common.iter_math_gateway_samples = lambda dataset, num_samples, existing: []
    common.load_remote_dataset_default = lambda *a, **k: ("ds", "name")
    common.valid_tag_structure = lambda text: True
    return common


def test_azure_main_guard_runs_with_stubs(monkeypatch, tmp_path):
    calls = {}
    common_stub = _make_common_stub(calls)

    repo_root = Path(__file__).resolve().parents[1]
    inference_pkg = types.ModuleType("src.inference")
    inference_pkg.__path__ = [str(repo_root / "src" / "inference")]
    gateways_pkg = types.ModuleType("src.inference.gateways")
    gateways_pkg.__path__ = [str(repo_root / "src" / "inference" / "gateways")]
    providers_pkg = types.ModuleType("src.inference.gateways.providers")
    providers_pkg.__path__ = [str(repo_root / "src" / "inference" / "gateways" / "providers")]
    utils_pkg = types.ModuleType("src.inference.utils")
    utils_pkg.__path__ = [str(repo_root / "src" / "inference" / "utils")]
    domains_pkg = types.ModuleType("src.inference.domains")
    domains_pkg.__path__ = [str(repo_root / "src" / "inference" / "domains")]
    domains_math_pkg = types.ModuleType("src.inference.domains.math")
    domains_math_pkg.__path__ = [str(repo_root / "src" / "inference" / "domains" / "math")]

    annotate = types.ModuleType("src.annotate")
    annotate.load_azure_config = lambda: {
        "endpoint": "http://endpoint",
        "deployment": "dep",
        "api_version": "2024-01-01",
        "api_key": "key",
        "use_v1": 1,
    }
    annotate.build_preferred_client = lambda endpoint, api_key, api_version, use_v1: (
        calls.setdefault("client_args", (endpoint, api_key, api_version, use_v1)) or ("CLIENT", True)
    ) and ("CLIENT", True)

    base = types.ModuleType("src.inference.gateways.base")
    base.get_task_spec = lambda *_: "task"
    base.setup_gateway_logger = lambda *_: _StubLogger()

    task_registry = types.ModuleType("src.inference.utils.task_registry")
    task_registry.MATH_SYSTEM_PROMPT = "prompt"

    math_core = types.ModuleType("src.inference.domains.math.math_core")
    math_core.load_math500 = lambda *a, **k: "math500"

    stubs = {
        "src.annotate": annotate,
        "src.inference.gateways.base": base,
        "src.inference.utils.common": common_stub,
        "src.inference.utils.task_registry": task_registry,
        "src.inference.domains.math.math_core": math_core,
        "src.inference": inference_pkg,
        "src.inference.gateways": gateways_pkg,
        "src.inference.gateways.providers": providers_pkg,
        "src.inference.utils": utils_pkg,
        "src.inference.domains": domains_pkg,
        "src.inference.domains.math": domains_math_pkg,
    }

    sys.modules.pop("src.inference.gateways.providers.azure", None)
    with _with_stub_modules(stubs):
        sys.argv = ["azure.py"]
        runpy.run_module("src.inference.gateways.providers.azure", run_name="__main__", alter_sys=True)

    assert calls.get("client_args") == ("http://endpoint", "key", "2024-01-01", True)
    assert calls.get("prepare") is True


def test_openrouter_main_guard_runs_with_stubs(monkeypatch, tmp_path):
    calls = {}
    common_stub = _make_common_stub(calls)
    monkeypatch.setenv("OPENROUTER_API_KEY", "key")

    repo_root = Path(__file__).resolve().parents[1]
    inference_pkg = types.ModuleType("src.inference")
    inference_pkg.__path__ = [str(repo_root / "src" / "inference")]
    gateways_pkg = types.ModuleType("src.inference.gateways")
    gateways_pkg.__path__ = [str(repo_root / "src" / "inference" / "gateways")]
    providers_pkg = types.ModuleType("src.inference.gateways.providers")
    providers_pkg.__path__ = [str(repo_root / "src" / "inference" / "gateways" / "providers")]
    utils_pkg = types.ModuleType("src.inference.utils")
    utils_pkg.__path__ = [str(repo_root / "src" / "inference" / "utils")]
    domains_pkg = types.ModuleType("src.inference.domains")
    domains_pkg.__path__ = [str(repo_root / "src" / "inference" / "domains")]
    domains_math_pkg = types.ModuleType("src.inference.domains.math")
    domains_math_pkg.__path__ = [str(repo_root / "src" / "inference" / "domains" / "math")]

    base = types.ModuleType("src.inference.gateways.base")
    base.setup_gateway_logger = lambda *_: _StubLogger()

    gateway_utils = types.ModuleType("src.inference.utils.gateway_utils")
    gateway_utils.append_jsonl_row = lambda *a, **k: None
    gateway_utils.build_math_gateway_arg_parser = common_stub.build_math_gateway_arg_parser
    gateway_utils.build_math_gateway_messages = lambda *a, **k: []
    gateway_utils.build_math_gateway_row_base = lambda **kwargs: kwargs
    gateway_utils.build_usage_dict = lambda usage: {"usage": usage}
    gateway_utils.call_with_gateway_retries = lambda *a, **k: ("", None, None)
    gateway_utils.iter_math_gateway_samples = lambda dataset, num_samples, existing: []
    gateway_utils.parse_openai_chat_response = lambda resp: ("", None, None)

    def _prepare(**kwargs):
        calls["prepare"] = True
        return [], {}, "name"

    gateway_utils.prepare_math_gateway_dataset_from_args = _prepare
    gateway_utils.require_datasets = lambda: (object, lambda *a, **k: None)

    math_utils = types.ModuleType("src.inference.utils.math_pass_utils")
    math_utils.canon_math = lambda text: text
    math_utils.extract_blocks = lambda text: ("", text)
    math_utils.valid_tag_structure = lambda text: True

    math_core = types.ModuleType("src.inference.domains.math.math_core")
    math_core.load_math500 = lambda *a, **k: "math500"

    openai_mod = types.ModuleType("openai")
    openai_mod.OpenAI = (
        lambda base_url=None, api_key=None: calls.setdefault("client_args", (base_url, api_key)) or "CLIENT"
    )

    task_registry = types.ModuleType("src.inference.utils.task_registry")
    task_registry.MATH_SYSTEM_PROMPT = "prompt"

    stubs = {
        "src.inference.gateways.base": base,
        "src.inference.utils.gateway_utils": gateway_utils,
        "src.inference.utils.math_pass_utils": math_utils,
        "src.inference.domains.math.math_core": math_core,
        "src.inference.utils.task_registry": task_registry,
        "src.inference": inference_pkg,
        "src.inference.gateways": gateways_pkg,
        "src.inference.gateways.providers": providers_pkg,
        "src.inference.utils": utils_pkg,
        "src.inference.domains": domains_pkg,
        "src.inference.domains.math": domains_math_pkg,
        "openai": openai_mod,
    }

    sys.modules.pop("src.inference.gateways.providers.openrouter", None)
    with _with_stub_modules(stubs):
        sys.argv = ["openrouter.py"]
        runpy.run_module("src.inference.gateways.providers.openrouter", run_name="__main__", alter_sys=True)

    assert calls.get("prepare") is True
