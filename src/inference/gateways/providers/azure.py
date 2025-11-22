#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gateway-style MATH-500 inference for Azure-hosted DeepSeek-R1 (open-source).

Behavior:
- Single-pass generation with the same math system prompt used in GRPO runs.
- Writes JSONL to: {output_dir}/step{step:04d}_{split}.jsonl
- Resumable: if a JSONL already exists, only missing sample_idx entries are
  generated.
- Uses Azure OpenAI (Responses API if available; falls back to Chat
  Completions).

This script is a **specialized / legacy gateway runner**. For canonical
HF-based math evaluation, prefer ``src.inference.cli.unified_math`` using local
checkpoints.

Usage (env-based auth):
  export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com"
  export AZURE_OPENAI_API_KEY="***"
  export AZURE_OPENAI_DEPLOYMENT="deepseek-r1"  # or your deployment name
  python -m src.inference.gateways.providers.azure \
      --output_dir artifacts/results/deepseek-r1
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional, TYPE_CHECKING

from src.annotate import load_azure_config
from src.annotate import build_preferred_client
from src.inference.gateways.base import get_task_spec, setup_gateway_logger
from src.inference.utils.common import (
    append_jsonl_row,
    build_math_gateway_arg_parser,
    build_math_gateway_row_base,
    build_usage_dict,
    call_with_gateway_retries,
    canon_math as _canon_math,
    extract_blocks as _extract_blocks,
    iter_math_gateway_samples,
    load_remote_dataset_default,
    prepare_math_gateway_dataset_from_args,
    valid_tag_structure as _valid_tag_structure,
)
from src.inference.domains.math.math_core import load_math500 as _load_math500_core
from src.inference.utils.task_registry import MATH_SYSTEM_PROMPT

if TYPE_CHECKING:
    from datasets import Dataset


logger = setup_gateway_logger(__name__)

TASK_SPEC = get_task_spec("math-azure")


# ----------------------- Prompt -----------------------
SYSTEM_PROMPT = MATH_SYSTEM_PROMPT


def load_math500(
    cache_dir: str,
    split: str,
    seed: int,
    dataset_path: Optional[str] = None,
) -> "Dataset":
    """
    Load the MATH-500 benchmark via :mod:`datasets` using the shared core logic.

    This is a thin wrapper around :func:`src.inference.domains.math.math_core.load_math500`
    so that the Azure runner reuses a single implementation of the loading and
    normalization logic.

    :param cache_dir: Directory to use as a datasets cache.
    :param split: Dataset split name (for example, ``\"test\"``).
    :param seed: Random seed used when sub-sampling competition-math fallback data.
    :param dataset_path: Optional local JSON file to load instead of remote datasets.
    :returns: A datasets-like object exposing ``map`` and ``select``.
    """
    return _load_math500_core(cache_dir, split, seed, dataset_path)


@dataclass
class AzureCallParams:
    """
    Lightweight container for Azure generation parameters.

    :param temperature: Sampling temperature for the model.
    :param top_p: Nucleus-sampling parameter for the model.
    :param max_output_tokens: Maximum number of tokens to generate.
    :param request_timeout: Per-request timeout in seconds.
    """

    temperature: float
    top_p: float
    max_output_tokens: int
    request_timeout: int


# ----------------------- Azure client + call -----------------------
def _make_client(args: argparse.Namespace):
    """
    Construct an Azure/OpenAI client and resolve endpoint/deployment.

    :param args: Parsed CLI arguments that may override endpoint and deployment.
    :returns: Tuple ``(client, uses_v1, endpoint, deployment, api_version)`` where
        ``uses_v1`` indicates whether the Responses API is preferred.
    :raises RuntimeError: If an API key cannot be resolved from arguments or environment.
    """
    cfg = load_azure_config()
    endpoint = (args.endpoint or cfg["endpoint"]).rstrip("/")
    deployment = args.deployment or cfg["deployment"]
    api_version = args.api_version or cfg["api_version"]
    api_key = args.api_key or os.getenv(
        "AZURE_OPENAI_API_KEY",
        cfg.get("api_key", ""),
    )
    if not api_key:
        raise RuntimeError("AZURE_OPENAI_API_KEY is required (env or --api_key).")
    client, uses_v1 = build_preferred_client(
        endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        use_v1=bool(args.use_v1),
    )
    return client, uses_v1, endpoint, deployment, api_version


def _call_model(
    client,
    uses_v1: bool,
    deployment: str,
    problem: str,
    params: AzureCallParams,
):
    """
    Call the Azure DeepSeek deployment and return text, finish reason, and usage.

    :param client: Azure/OpenAI client created by :func:`_make_client`.
    :param uses_v1: Whether to prefer the Responses API (v1) over Chat Completions.
    :param deployment: Name of the Azure deployment to use.
    :param problem: Raw problem text to send to the model.
    :param params: Generation parameters such as temperature and max tokens.
    :returns: Tuple ``(text, finish_reason, usage)`` describing the model response.
    """
    if uses_v1 and hasattr(client, "responses"):
        resp = client.responses.create(
            model=deployment,
            instructions=SYSTEM_PROMPT,
            input=[{"role": "user", "content": problem}],
            temperature=params.temperature,
            top_p=params.top_p,
            max_output_tokens=params.max_output_tokens,
            timeout=params.request_timeout,
        )
        text = ""
        finish_reason = None
        if getattr(resp, "output", None):
            output = resp.output
            if getattr(output, "choices", None):
                choice = output.choices[0]
                finish_reason = getattr(choice, "finish_reason", None)
                msg = getattr(choice, "message", None)
                text = getattr(msg, "content", "") if msg is not None else ""
        usage = getattr(resp, "usage", None)
        return text, finish_reason, usage

    # Legacy Chat Completions
    resp = client.chat.completions.create(
        model=deployment,
        temperature=params.temperature,
        top_p=params.top_p,
        max_tokens=params.max_output_tokens,
        timeout=params.request_timeout,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ],
    )
    finish_reason = None
    if resp and getattr(resp, "choices", None):
        finish_reason = getattr(resp.choices[0], "finish_reason", None)
        text = resp.choices[0].message.content or ""
    else:
        text = ""
    usage = getattr(resp, "usage", None)
    return text, finish_reason, usage


def _prepare_dataset(args: argparse.Namespace, outpath: str):
    """
    Load and shuffle the dataset, applying resume/fill logic.

    :param args: Parsed CLI arguments controlling dataset choice and sampling.
    :param outpath: Output path used to determine which samples already exist.
    :returns: Tuple ``(dataset, existing)`` where ``existing`` maps problems
        to the set of already-filled sample indices.
    """
    dataset, existing, _ = prepare_math_gateway_dataset_from_args(
        args=args,
        outpath=outpath,
        logger=logger,
        load_math500_fn=load_math500,
        load_remote_dataset_fn=load_remote_dataset_default,
        cache_dir=os.path.abspath("./.hf_cache"),
    )
    return dataset, existing


def _generate_samples(
    client,
    uses_v1: bool,
    args: argparse.Namespace,
    call_params: AzureCallParams,
    output_path: str,
) -> None:
    """
    Main generation loop over the dataset.

    :param client: Azure/OpenAI client created by :func:`_make_client`.
    :param uses_v1: Whether to use the Responses API for generation.
    :param args: Parsed CLI arguments controlling generation behavior.
    :param call_params: Azure generation parameters such as temperature and limits.
    :param output_path: Path to the JSONL file where results are written.
    :returns: ``None``. Samples are written to disk and progress is logged.
    """
    dataset, existing = _prepare_dataset(args, output_path)

    total_new = 0
    sample_iter = iter_math_gateway_samples(dataset, args.num_samples, existing)
    for problem, gold_answer, sample_idx in sample_iter:
        call_fn = partial(
            _call_model,
            client,
            uses_v1,
            args.deployment,
            problem,
            call_params,
        )
        call_result = call_with_gateway_retries(
            call_fn,
            args=args,
            logger=logger,
            sample_idx=sample_idx,
            problem_snippet=problem,
        )

        row = _build_result_row(
            problem=problem,
            gold_answer=gold_answer,
            sample_idx=sample_idx,
            text=call_result[0],
            finish_reason=call_result[1],
            usage=call_result[2],
            args=args,
            call_params=call_params,
        )

        append_jsonl_row(output_path, row)
        total_new += 1
        existing.setdefault(problem, set()).add(sample_idx)

    logger.info("All done. Wrote %d new samples → %s", total_new, output_path)


def _build_result_row(
    *,
    problem: str,
    gold_answer: Any,
    sample_idx: int,
    text: str,
    finish_reason: Any,
    usage: Any,
    args: argparse.Namespace,
    call_params: AzureCallParams,
) -> Dict[str, Any]:
    """
    Build a JSONL row for a single generated sample.

    :param problem: Normalized problem text for the example.
    :param gold_answer: Ground-truth answer associated with the problem.
    :param sample_idx: Sample index for this generation.
    :param text: Raw model output text.
    :param finish_reason: Finish reason reported by the Azure API.
    :param usage: Optional usage object returned by the Azure SDK.
    :param args: Parsed CLI arguments (used for metadata fields).
    :param call_params: Azure generation parameters used for the call.
    :returns: Dictionary representing one JSONL row.
    """
    canon_gold = _canon_math(gold_answer)
    _, ans = _extract_blocks(text)
    pred_canon = _canon_math(ans)
    is_correct = bool(pred_canon and canon_gold and canon_gold in pred_canon)

    row = build_math_gateway_row_base(
        problem=problem,
        gold_answer=gold_answer,
        gold_answer_canon=canon_gold,
        split=args.split,
        step=args.step,
        sample_idx=sample_idx,
    )
    row.update(
        {
            "endpoint": args.endpoint,
            "deployment": args.deployment,
            "api_version": args.api_version,
            "temperature": call_params.temperature,
            "top_p": call_params.top_p,
            "pass1": {
                "output": text.strip(),
                "pred_answer": ans,
                "pred_answer_canon": pred_canon,
                "is_correct_pred": is_correct,
                "valid_tag_structure": _valid_tag_structure(text),
                "finish_reason": finish_reason,
            },
        },
    )

    if usage:
        row["usage"] = build_usage_dict(usage)

    return row


def _parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for the Azure DeepSeek MATH-500 runner.

    :returns: Parsed :class:`argparse.Namespace` with configuration values.
    """
    default_cfg = load_azure_config()
    parser = build_math_gateway_arg_parser(
        default_temperature=0.7,
        description="Azure DeepSeek-R1 MATH-500 runner.",
    )

    # Azure params (env defaults from configs/azure.yml)
    parser.add_argument("--endpoint", default=default_cfg.get("endpoint"))
    parser.add_argument(
        "--deployment",
        default=None,
        help="Azure deployment name (e.g., deepseek-r1).",
    )
    parser.add_argument("--api_version", default=default_cfg.get("api_version"))
    parser.add_argument("--api_key", default=None)
    parser.add_argument(
        "--use_v1",
        type=int,
        default=int(default_cfg.get("use_v1", 1)),
        help="1 → prefer Responses API (v1); 0 → force Chat Completions.",
    )

    # Retry controls
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_backoff", type=float, default=2.0)

    return parser.parse_args()


def main() -> None:
    """
    CLI entrypoint for single-pass Azure DeepSeek MATH-500 inference.

    :returns: ``None``. The function parses arguments and runs the generation loop.
    """
    args = _parse_args()
    random.seed(args.seed)

    output_path = os.path.join(
        args.output_dir,
        f"step{args.step:04d}_{args.split}.jsonl",
    )

    client, uses_v1, endpoint, deployment, api_version = _make_client(args)
    logger.info(
        "Client ready | uses_v1=%s | endpoint=%s | deployment=%s",
        uses_v1,
        endpoint,
        deployment,
    )

    call_params = AzureCallParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.max_output_tokens,
        request_timeout=args.request_timeout,
    )
    # Normalise args with resolved endpoint values for logging and rows.
    args.endpoint = endpoint
    args.deployment = deployment
    args.api_version = api_version
    _generate_samples(
        client=client,
        uses_v1=uses_v1,
        args=args,
        call_params=call_params,
        output_path=output_path,
    )


__all__ = ["load_math500", "main"]


if __name__ == "__main__":
    main()
