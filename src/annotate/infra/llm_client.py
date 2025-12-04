"""
Lightweight helpers for constructing Azure/OpenAI clients.

Notes:
- We keep behavior identical to existing scripts: prefer the v1 Responses API
  when available, otherwise fall back to legacy chat.completions.
- Callers are responsible for providing api_key/endpoint/api_version via args or env.
"""

import logging
from typing import Tuple


try:
    from openai import AzureOpenAI as _AzureOpenAI
    from openai import OpenAI as _OpenAI
    from openai import OpenAIError as _OpenAIError
except ImportError:  # pragma: no cover - optional dependency
    _OpenAI = None
    _AzureOpenAI = None

    class _OpenAIError(Exception):
        """Fallback when openai is not installed."""


OpenAI = _OpenAI
AzureOpenAI = _AzureOpenAI
OpenAIError = _OpenAIError


def build_preferred_client(
    endpoint: str,
    api_key: str,
    api_version: str,
    use_v1: bool = True,
) -> Tuple[object, bool]:
    """
    Attempt to create a client that prefers the v1 Responses API.

    The function first tries an ``OpenAI`` client pointed at the
    ``/openai/v1/`` base URL (Responses API). If that fails or ``use_v1`` is
    ``False``, it falls back to an :class:`AzureOpenAI` client for legacy Chat
    Completions.

    :param endpoint: Azure OpenAI resource endpoint (without trailing ``/openai/v1/``).
    :param api_key: API key for the resource.
    :param api_version: API version string for the Azure client.
    :param use_v1: Whether to prefer the v1 Responses API when available.
    :returns: Tuple ``(client, uses_v1_flag)`` where ``uses_v1_flag`` indicates
        whether the returned client is configured for Responses (v1) or legacy chat.
    :raises RuntimeError: If no suitable client can be constructed.
    """
    endpoint = endpoint.rstrip("/")

    # Try v1 base-url client first if allowed
    if use_v1 and OpenAI is not None:
        base_url = f"{endpoint}/openai/v1/"
        try:
            client = OpenAI(base_url=base_url, api_key=api_key)
            logging.info("[client] Using Responses API (v1) at %s", base_url)
            return client, True
        except OpenAIError as exc:
            logging.warning(
                "Failed to init v1 Responses client (%s). Falling back. %r",
                base_url,
                exc,
            )

    if AzureOpenAI is None:
        raise RuntimeError("openai>=1.x with AzureOpenAI or v1 OpenAI client is required. pip install -U openai")

    try:
        client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
        logging.info(
            "[client] Using legacy Chat Completions at %s (api_version=%s)",
            endpoint,
            api_version,
        )
        return client, False
    except OpenAIError as exc:
        raise RuntimeError(f"Failed to initialize AzureOpenAI client: {exc}") from exc


def build_chat_client(endpoint: str, api_key: str, api_version: str):
    """
    Construct a simple Azure Chat Completions client (no Responses probing).

    :param endpoint: Azure OpenAI resource endpoint.
    :param api_key: API key for the resource.
    :param api_version: API version string for the Azure client.
    :returns: :class:`AzureOpenAI` client configured for Chat Completions.
    :raises RuntimeError: If the ``openai`` package with ``AzureOpenAI`` is not available.
    """
    if AzureOpenAI is None:
        raise RuntimeError("openai>=1.x with AzureOpenAI is required. pip install -U openai")
    return AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint.rstrip("/"),
        api_version=api_version,
    )
