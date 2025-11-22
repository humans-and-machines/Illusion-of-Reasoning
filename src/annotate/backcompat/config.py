"""
Backwards-compatible shim for configuration helpers.

Canonical implementation now lives in :mod:`src.annotate.infra.config`.
New code should import from :mod:`src.annotate` or
``src.annotate.infra.config`` instead of this module.
"""

from ..infra.config import load_azure_config, load_sandbox_config

__all__ = ["load_azure_config", "load_sandbox_config"]
