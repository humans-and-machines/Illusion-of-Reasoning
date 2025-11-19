"""
Helpers for talking to an external `trl vllm-serve` instance.

vLLM ≤ 0.8.5 returns *token IDs* by default:
    {"completion_ids": [[ids...]], "prompt_ids": [[ids...]]}

This helper now detects that schema and decodes it if a tokenizer is passed.
"""

from __future__ import annotations
import json, time
from typing import List, Optional
import requests


# ─────────────────── generic GET helper ──────────────────────────────────────
def safe_request(url: str, max_retries: int = 3, backoff: float = 1.0, timeout: float = 10.0):
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:120]}")
        except (requests.ConnectionError, requests.Timeout):
            if attempt < max_retries - 1:
                time.sleep(backoff * (2**attempt))
            else:
                raise


# ─────────────────── helper to parse non-stream JSON ─────────────────────────
def _parse_nonstream_json(data: dict, tokenizer=None) -> List[List[str]]:
    # OpenAI route
    if "choices" in data:
        return [[c["text"] for c in data["choices"]]]
    # Plain /generate route (newer default)
    if "results" in data:
        return [[r["text"] for r in data["results"]]]
    # vLLM 0.8.x batched output
    if "text" in data and isinstance(data["text"], list):
        return [[t] for t in data["text"]]
    # vLLM 0.8.x token-ID output
    if "completion_ids" in data:
        if tokenizer is None:
            raise RuntimeError(
                "Server returned token IDs but no tokenizer was supplied to safe_generate()."
            )
        return [[tokenizer.decode(ids, skip_special_tokens=True)] for ids in data["completion_ids"]]
    raise RuntimeError(f"Unknown vLLM response format: {data}")


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
    tokenizer=None,                 # ← new optional arg
    max_retries: int = 3,
    backoff: float = 1.0,
    timeout: float = 30.0,
) -> List[List[str]]:
    """Robust call to /generate with retry + schema-agnostic decoding."""
    payload = dict(
        prompts=prompts, temperature=temperature, top_p=top_p,
        n=n, max_tokens=max_tokens, stream=stream,
    )

    for attempt in range(max_retries):
        try:
            r = requests.post(url, json=payload, timeout=timeout, stream=stream)
            if r.status_code == 200:
                if stream:
                    texts = [[] for _ in prompts]
                    for line in r.iter_lines():
                        if line:
                            row = json.loads(line.decode())
                            idx = row.get("prompt_index", 0)
                            texts[idx].append(row["text"])
                    return [["".join(parts)] for parts in texts]
                else:
                    return _parse_nonstream_json(r.json(), tokenizer)
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:120]}")
        except (requests.ConnectionError, requests.Timeout, RuntimeError) as e:
            if attempt < max_retries - 1:
                time.sleep(backoff * (2**attempt))
            else:
                raise RuntimeError(f"safe_generate failed: {e}") from e
