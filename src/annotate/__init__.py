"""
Public annotation utilities API.

Downstream code should generally import from :mod:`src.annotate` rather than
reaching into submodules directly, e.g.:

    from src.annotate import AnnotateOpts, annotate_file
"""

from __future__ import annotations

import sys
from importlib import import_module


_PKG_ROOT = __name__

_backcompat = import_module(f"{_PKG_ROOT}.backcompat")
_config_mod = import_module(f"{_PKG_ROOT}.infra.config")
_llm_mod = import_module(f"{_PKG_ROOT}.infra.llm_client")
_clean_mod = import_module(f"{_PKG_ROOT}.core.clean_core")
_shift_mod = import_module(f"{_PKG_ROOT}.core.shift_core")
_prompts_mod = import_module(f"{_PKG_ROOT}.core.prompts")

load_azure_config = _config_mod.load_azure_config
load_sandbox_config = _config_mod.load_sandbox_config
build_preferred_client = _llm_mod.build_preferred_client
clean_file = _clean_mod.clean_file
clean_root = _clean_mod.clean_root
SHIFT_JUDGE_SYSTEM_PROMPT = _prompts_mod.SHIFT_JUDGE_SYSTEM_PROMPT
SHIFT_JUDGE_USER_TEMPLATE = _prompts_mod.SHIFT_JUDGE_USER_TEMPLATE

AnnotateOpts = _shift_mod.AnnotateOpts
DEFAULT_API_VERSION = _shift_mod.DEFAULT_API_VERSION
DEFAULT_DEPLOYMENT = _shift_mod.DEFAULT_DEPLOYMENT
DEFAULT_ENDPOINT = _shift_mod.DEFAULT_ENDPOINT
DEFAULT_USE_V1 = _shift_mod.DEFAULT_USE_V1
annotate_file = _shift_mod.annotate_file
llm_judge_shift = _shift_mod.llm_judge_shift
scan_jsonl = _shift_mod.scan_jsonl

_ANNOTATION_PUBLIC_API = [
    "AnnotateOpts",
    "annotate_file",
    "scan_jsonl",
    "llm_judge_shift",
]

__all__ = [
    *_ANNOTATION_PUBLIC_API,
    "DEFAULT_ENDPOINT",
    "DEFAULT_DEPLOYMENT",
    "DEFAULT_API_VERSION",
    "DEFAULT_USE_V1",
    "clean_file",
    "clean_root",
    "load_azure_config",
    "load_sandbox_config",
    "build_preferred_client",
    "SHIFT_JUDGE_SYSTEM_PROMPT",
    "SHIFT_JUDGE_USER_TEMPLATE",
]

_BACKCOMPAT_SUBMODULES = list(getattr(_backcompat, "__all__", []))

for _name in _BACKCOMPAT_SUBMODULES:
    _module = getattr(_backcompat, _name)
    globals()[_name] = _module
    sys.modules[f"{__name__}.{_name}"] = _module

__all__.extend(_BACKCOMPAT_SUBMODULES)
