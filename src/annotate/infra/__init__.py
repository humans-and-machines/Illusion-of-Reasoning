"""
Infrastructure helpers for annotation (config loading, client construction).

Most callers should import via :mod:`src.annotate` instead of this subpackage.
"""

from .config import load_azure_config, load_sandbox_config
from .llm_client import build_preferred_client


__all__ = ["load_azure_config", "load_sandbox_config", "build_preferred_client"]
