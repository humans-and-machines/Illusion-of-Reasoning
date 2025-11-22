"""
Public annotation utilities API.

Downstream code should generally import from :mod:`src.annotate` rather than
reaching into submodules directly, e.g.:

    from src.annotate import AnnotateOpts, annotate_file
"""

import sys

from . import backcompat as _backcompat
from .infra.config import load_azure_config, load_sandbox_config
from .infra.llm_client import build_preferred_client
from .core.prompts import SHIFT_JUDGE_SYSTEM_PROMPT, SHIFT_JUDGE_USER_TEMPLATE
from .core.clean_core import clean_file, clean_root
from .core.shift_core import (  # noqa: F401
    AnnotateOpts,
    DEFAULT_API_VERSION,
    DEFAULT_DEPLOYMENT,
    DEFAULT_ENDPOINT,
    DEFAULT_USE_V1,
    annotate_file,
    llm_judge_shift,
    scan_jsonl,
)

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
