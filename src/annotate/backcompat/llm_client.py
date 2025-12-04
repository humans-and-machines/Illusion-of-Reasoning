"""
Backwards-compatible shim for LLM client helpers.

Canonical implementation now lives in :mod:`src.annotate.infra.llm_client`.
New code should prefer importing from :mod:`src.annotate.infra.llm_client`,
but tests and legacy callers often monkeypatch the symbols on this module.
To keep that behavior working, we delegate to the infra implementation while
syncing any patched symbols into it at call time.
"""

from __future__ import annotations

from typing import Tuple

from ..infra import llm_client as _infra


# Re-export client classes / error type so callers can patch them on this module.
OpenAI = _infra.OpenAI
AzureOpenAI = _infra.AzureOpenAI
OpenAIError = _infra.OpenAIError


def build_preferred_client(
    endpoint: str,
    api_key: str,
    api_version: str,
    use_v1: bool = True,
) -> Tuple[object, bool]:
    """
    Backwards-compatible wrapper that delegates to :mod:`infra.llm_client`.

    Tests expect that monkeypatching ``src.annotate.llm_client.OpenAI`` or
    ``AzureOpenAI`` affects the client returned here. To honor that contract,
    we first propagate this module's symbols into the infra module and then
    call its ``build_preferred_client``.
    """
    _infra.OpenAI = OpenAI
    _infra.AzureOpenAI = AzureOpenAI
    _infra.OpenAIError = OpenAIError
    return _infra.build_preferred_client(
        endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        use_v1=use_v1,
    )


def build_chat_client(endpoint: str, api_key: str, api_version: str):
    """
    Backwards-compatible wrapper that delegates to :mod:`infra.llm_client`.

    As with :func:`build_preferred_client`, we propagate any monkeypatched
    Azure client from this module into the infra module before constructing
    the chat client.
    """
    _infra.AzureOpenAI = AzureOpenAI
    _infra.OpenAIError = OpenAIError
    return _infra.build_chat_client(
        endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )


__all__ = [
    "AzureOpenAI",
    "OpenAI",
    "OpenAIError",
    "build_chat_client",
    "build_preferred_client",
]
