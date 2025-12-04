"""
Helpers for talking to an external `trl vllm-serve` instance.

vLLM ≤ 0.8.5 returns *token IDs* by default:
    {"completion_ids": [[ids...]], "prompt_ids": [[ids...]]}

This helper now detects that schema and decodes it if a tokenizer is passed.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from typing import Any, List


try:
    import requests
except ImportError:  # pragma: no cover - optional in minimal test environments
    from types import SimpleNamespace

    def _missing_requests(*_args, **_kwargs):
        msg = "The 'requests' package is required for vLLM helpers."
        raise ImportError(msg)

    requests = SimpleNamespace(
        ConnectionError=RuntimeError,
        Timeout=RuntimeError,
        get=_missing_requests,
        post=_missing_requests,
    )


@dataclass
class VLLMGenerationParams:
    """Payload parameters for a vLLM `/generate` request."""

    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    n: int = 1
    stream: bool = False


@dataclass
class VLLMRetryConfig:
    """Retry/backoff configuration for a vLLM request."""

    max_retries: int = 3
    backoff: float = 1.0
    timeout: float = 30.0


@dataclass
class VLLMGenerateConfig:
    """
    Configuration for a vLLM `/generate` request.

    The defaults mirror the legacy keyword arguments that were previously
    exposed on :func:`safe_generate`.
    """

    url: str = "http://localhost:8000/generate"
    params: VLLMGenerationParams = field(default_factory=VLLMGenerationParams)
    retry: VLLMRetryConfig = field(default_factory=VLLMRetryConfig)
    tokenizer: Any | None = None

    def to_payload(self, prompts: List[str]) -> dict:
        """Build the JSON payload expected by vLLM."""
        return {
            "prompts": prompts,
            "temperature": self.params.temperature,
            "top_p": self.params.top_p,
            "n": self.params.n,
            "max_tokens": self.params.max_tokens,
            "stream": self.params.stream,
        }


# ─────────────────── generic GET helper ──────────────────────────────────────
def safe_request(
    url: str,
    max_retries: int = 3,
    backoff: float = 1.0,
    timeout: float = 10.0,
) -> dict:
    """Perform a GET request with simple retry and exponential backoff."""
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                return response.json()
            last_error = RuntimeError(f"HTTP {response.status_code}: {response.text[:120]}")
            break
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(backoff * (2**attempt))

    if last_error is not None:
        raise last_error
    msg = "safe_request failed without performing any HTTP request."
    raise RuntimeError(msg)


# ─────────────────── helper to parse non-stream JSON ─────────────────────────
def _parse_nonstream_json(
    data: dict,
    tokenizer: Any | None = None,
) -> List[List[str]]:
    # OpenAI route
    if "choices" in data:
        return [[choice["text"] for choice in data["choices"]]]
    # Plain /generate route (newer default)
    if "results" in data:
        return [[result["text"] for result in data["results"]]]
    # vLLM 0.8.x batched output
    if "text" in data and isinstance(data["text"], list):
        return [[text] for text in data["text"]]
    # vLLM 0.8.x token-ID output
    if "completion_ids" in data:
        if tokenizer is None:
            raise RuntimeError("Server returned token IDs but no tokenizer was supplied to safe_generate().")
        return [[tokenizer.decode(ids, skip_special_tokens=True)] for ids in data["completion_ids"]]
    raise RuntimeError(f"Unknown vLLM response format: {data}")


def _parse_streaming_response(
    response: requests.Response,
    num_prompts: int,
) -> List[List[str]]:
    """Convert a streaming vLLM response into a list-of-lists of strings."""
    texts: list[list[str]] = [[] for _ in range(num_prompts)]
    for line_bytes in response.iter_lines():
        if not line_bytes:
            continue
        row = json.loads(line_bytes.decode())
        index = row.get("prompt_index", 0)
        texts[index].append(row["text"])
    return [["".join(parts)] for parts in texts]


def _generate_with_retries(
    *,
    prompts: List[str],
    payload: dict,
    config: VLLMGenerateConfig,
) -> List[List[str]]:
    """Call the vLLM server with retries and decode the response."""
    for attempt in range(config.retry.max_retries):
        try:
            response = requests.post(
                config.url,
                json=payload,
                timeout=config.retry.timeout,
                stream=config.params.stream,
            )
            if response.status_code != 200:
                msg = f"HTTP {response.status_code}: {response.text[:120]}"
                raise RuntimeError(msg)

            if config.params.stream:
                return _parse_streaming_response(response, len(prompts))
            return _parse_nonstream_json(response.json(), config.tokenizer)
        except (requests.ConnectionError, requests.Timeout, RuntimeError) as exc:
            if attempt < config.retry.max_retries - 1:
                time.sleep(config.retry.backoff * (2**attempt))
            else:
                msg = f"safe_generate failed: {exc}"
                raise RuntimeError(msg) from exc

    raise RuntimeError("safe_generate failed without performing any HTTP request.")


def _resolve_config(
    config: VLLMGenerateConfig | None,
    overrides: dict[str, Any],
) -> VLLMGenerateConfig:
    """
    Normalize configuration by applying user overrides to a base dataclass.
    """
    resolved = config or VLLMGenerateConfig()
    if not overrides:
        return resolved

    payload_fields = {"max_tokens", "temperature", "top_p", "n", "stream"}
    retry_fields = {"max_retries", "backoff", "timeout"}
    root_fields = {"url", "tokenizer"}

    unexpected = set(overrides) - payload_fields - retry_fields - root_fields
    if unexpected:
        msg = f"safe_generate got unexpected config keys: {sorted(unexpected)}"
        raise TypeError(msg)

    params_kwargs = {key: overrides[key] for key in payload_fields if key in overrides}
    retry_kwargs = {key: overrides[key] for key in retry_fields if key in overrides}

    new_params = replace(resolved.params, **params_kwargs) if params_kwargs else resolved.params
    new_retry = replace(resolved.retry, **retry_kwargs) if retry_kwargs else resolved.retry
    return replace(
        resolved,
        url=overrides.get("url", resolved.url),
        tokenizer=overrides.get("tokenizer", resolved.tokenizer),
        params=new_params,
        retry=new_retry,
    )


# ─────────────────── POST /generate helper ────────────────────────────────────
def safe_generate(
    prompts: List[str],
    config: VLLMGenerateConfig | None = None,
    **overrides: Any,
) -> List[List[str]]:
    """
    Robust call to /generate with retry + schema-agnostic decoding.

    Use ``config`` to control request behavior; individual fields can be
    overridden via keyword arguments for convenience (e.g., ``top_p=0.8``).
    """
    resolved_config = _resolve_config(config, overrides)
    payload = resolved_config.to_payload(prompts)
    return _generate_with_retries(
        prompts=prompts,
        payload=payload,
        config=resolved_config,
    )
