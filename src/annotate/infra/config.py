"""Lightweight config loader for Azure/OpenAI settings.

Loads from environment first, then optional YAML (configs/azure.yml or user-specified).
Intended to avoid hardcoding keys in scripts.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional


try:
    import yaml as _yaml  # type: ignore
except ImportError:  # pragma: no cover - yaml optional
    _yaml = None  # type: ignore


def load_yaml(path: Path) -> Dict[str, Any]:
    """
    Best-effort YAML loader.

    :param path: Path to a YAML file on disk.
    :returns: Parsed YAML mapping, or an empty dict if the file is missing or
        the optional ``yaml`` dependency is not installed.
    """
    if not _yaml or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file_handle:
        return _yaml.safe_load(file_handle) or {}


def load_azure_config(yaml_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load Azure OpenAI settings from environment variables and optional YAML.

    Precedence: environment variables override YAML, YAML overrides built-in
    defaults. The resulting mapping is suitable for constructing an Azure
    client or passed into helpers like :func:`build_preferred_client`.

    :param yaml_path: Optional path to a YAML config file. When omitted,
        ``configs/azure.yml`` is used if present.
    :returns: Mapping with keys ``endpoint``, ``api_key``, ``deployment``,
        ``api_version``, and ``use_v1``.
    """
    defaults = {
        # Local default: your Azure OpenAI resource for math experiments.
        # NOTE: api_key intentionally left blank here; provide it via env
        # (AZURE_OPENAI_API_KEY) or configs/azure.yml, both of which are gitignored.
        "endpoint": "https://cos-wiki-ai.openai.azure.com/",
        "api_key": "",
        "deployment": "gpt-4o",
        "api_version": "2024-12-01-preview",
        # Default to legacy Chat Completions-style client; can be overridden
        # via AZURE_OPENAI_USE_V1=1 if you want to prefer the v1 Responses API.
        "use_v1": False,
    }
    ypath = Path(yaml_path) if yaml_path else Path("configs/azure.yml")
    ycfg = load_yaml(ypath)

    cfg = {
        "endpoint": os.getenv(
            "AZURE_OPENAI_ENDPOINT",
            ycfg.get("endpoint", defaults["endpoint"]),
        ).rstrip("/")
        or defaults["endpoint"],
        "api_key": os.getenv(
            "AZURE_OPENAI_API_KEY",
            ycfg.get("api_key", defaults["api_key"]),
        ),
        "deployment": os.getenv(
            "AZURE_OPENAI_DEPLOYMENT",
            ycfg.get("deployment", defaults["deployment"]),
        ),
        "api_version": os.getenv(
            "AZURE_OPENAI_API_VERSION",
            ycfg.get("api_version", defaults["api_version"]),
        ),
        "use_v1": bool(
            int(
                os.getenv(
                    "AZURE_OPENAI_USE_V1",
                    str(int(ycfg.get("use_v1", defaults["use_v1"]))),
                )
            )
        ),
    }
    return cfg


def load_sandbox_config() -> Dict[str, Any]:
    """
    Load configuration for the Princeton AI Sandbox (Portkey) endpoint.

    :returns: Mapping with keys ``endpoint``, ``api_key``, ``api_version``,
        and ``deployment`` derived from environment variables.
    """
    return {
        "endpoint": os.getenv("SANDBOX_ENDPOINT", "https://api-ai-sandbox.princeton.edu/"),
        "api_key": os.getenv("SANDBOX_API_KEY", ""),
        "api_version": os.getenv("SANDBOX_API_VER", "2025-03-01-preview"),
        "deployment": os.getenv("SANDBOX_DEPLOYMENT", "gpt-4o"),
    }


__all__ = ["load_azure_config", "load_sandbox_config"]
