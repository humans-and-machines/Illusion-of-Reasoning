"""
Helpers for talking to an external `trl vllm-serve` instance.

vLLM ≤ 0.8.5 returns *token IDs* by default:
    {"completion_ids": [[ids...]], "prompt_ids": [[ids...]]}

This helper now detects that schema and decodes it if a tokenizer is passed.
"""

from __future__ import annotations

import json
import time
from typing import Any, List

import requests


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
            last_error = RuntimeError(
                f"HTTP {response.status_code}: {response.text[:120]}"
            )
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
            raise RuntimeError(
                "Server returned token IDs but no tokenizer was supplied to safe_generate()."
            )
        return [
            [tokenizer.decode(ids, skip_special_tokens=True)]
            for ids in data["completion_ids"]
        ]
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
    url: str,
    payload: dict,
    tokenizer: Any | None,
    stream: bool,
    max_retries: int,
    backoff: float,
    timeout: float,
) -> List[List[str]]:
    """Call the vLLM server with retries and decode the response."""
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=timeout,
                stream=stream,
            )
            if response.status_code != 200:
                msg = f"HTTP {response.status_code}: {response.text[:120]}"
                raise RuntimeError(msg)

            if stream:
                return _parse_streaming_response(response, len(prompts))
            return _parse_nonstream_json(response.json(), tokenizer)
        except (requests.ConnectionError, requests.Timeout, RuntimeError) as exc:
            if attempt < max_retries - 1:
                time.sleep(backoff * (2**attempt))
            else:
                msg = f"safe_generate failed: {exc}"
                raise RuntimeError(msg) from exc

    raise RuntimeError("safe_generate failed without performing any HTTP request.")


# ─────────────────── POST /generate helper ────────────────────────────────────
def safe_generate(
    *,
    prompts: List[str],
    url: str = "http://localhost:8000/generate",
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    n: int = 1,
    stream: bool = False,
    tokenizer: Any | None = None,
    max_retries: int = 3,
    backoff: float = 1.0,
    timeout: float = 30.0,
) -> List[List[str]]:
    """Robust call to /generate with retry + schema-agnostic decoding."""
    payload = {
        "prompts": prompts,
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    return _generate_with_retries(
        prompts=prompts,
        url=url,
        payload=payload,
        tokenizer=tokenizer,
        stream=stream,
        max_retries=max_retries,
        backoff=backoff,
        timeout=timeout,
    )
