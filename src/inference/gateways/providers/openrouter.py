#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gateway-style MATH-500 inference for DeepSeek-R1 via OpenRouter.

Behavior:
- Single-pass generation with the same math system prompt used in GRPO runs.
- Writes JSONL to: {output_dir}/step{step:04d}_{split}.jsonl
- Resumable: if a JSONL already exists, only missing sample_idx entries are generated.

This script is a **specialized / legacy gateway runner** intended for remote
API experiments. For canonical HF-based math evaluation, prefer
``src.inference.cli.unified_math``.

Auth:
- Expects an OpenRouter API key in the `OPENROUTER_API_KEY` env var.

Example usage:
  export OPENROUTER_API_KEY="sk-or-..."
  python -m src.inference.gateways.providers.openrouter \\
    --output_dir artifacts/results/deepseek-r1-openrouter \\
    --model deepseek/deepseek-r1:free
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from importlib import import_module
from functools import partial
from typing import Any, Dict

from src.inference.gateways.base import setup_gateway_logger
from src.inference.utils.gateway_utils import (
    append_jsonl_row,
    build_math_gateway_arg_parser,
    build_math_gateway_messages,
    build_math_gateway_row_base,
    build_usage_dict,
    call_with_gateway_retries,
    iter_math_gateway_samples,
    parse_openai_chat_response,
    prepare_math_gateway_dataset_from_args,
    require_datasets,
)
from src.inference.utils.math_pass_utils import (
    canon_math as _canon_math,
    extract_blocks as _extract_blocks,
    valid_tag_structure as _valid_tag_structure,
)
from src.inference.domains.math.math_core import load_math500
from src.inference.utils.task_registry import MATH_SYSTEM_PROMPT


Dataset, load_dataset = require_datasets()
logger = setup_gateway_logger(__name__)


# ----------------------- Prompt -----------------------
SYSTEM_PROMPT = MATH_SYSTEM_PROMPT


# ----------------------- OpenRouter client + call -----------------------
def _make_client():
    """
    Construct an OpenAI client pointed at the OpenRouter base URL.

    :returns: Configured ``openai.OpenAI`` client instance for OpenRouter.
    :raises RuntimeError: If ``OPENROUTER_API_KEY`` is not set in the environment.
    :raises ImportError: If the ``openai`` package is not installed.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY env var is required for OpenRouter.")
    base_url = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
    try:
        openai_mod = import_module("openai")
    except ImportError as import_exc:  # pragma: no cover - optional dependency
        print(
            "openai>=1.x is required for this script: pip install openai",
            file=sys.stderr,
        )
        raise import_exc

    client_cls = getattr(openai_mod, "OpenAI")
    return client_cls(base_url=base_url, api_key=api_key)


def _call_model(client, problem: str, args: argparse.Namespace):
    """
    Call the OpenRouter model for a single math problem.

    :param client: OpenRouter client returned by :func:`_make_client`.
    :param problem: Raw problem text to send to the model.
    :param args: Parsed CLI arguments containing model and sampling options.
    :returns: Parsed response tuple as returned by :func:`parse_openai_chat_response`.
    """
    # If you want OpenRouter's explicit reasoning traces, uncomment extra_body.
    messages = build_math_gateway_messages(SYSTEM_PROMPT, problem)
    resp = client.chat.completions.create(
        model=args.model,
        messages=messages,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_output_tokens,
        timeout=args.request_timeout,
        # extra_body={"reasoning": True},
    )
    return parse_openai_chat_response(resp)


# ----------------------- Main helpers + loop -----------------------
def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the OpenRouter DeepSeek-R1 runner.

    :returns: Parsed :class:`argparse.Namespace` with configuration values.
    """
    parser = build_math_gateway_arg_parser(
        default_temperature=0.05,
        description="OpenRouter DeepSeek-R1 MATH-500 runner.",
    )
    parser.add_argument(
        "--model",
        default="deepseek/deepseek-r1",
        help="OpenRouter model name (e.g., deepseek/deepseek-r1 or deepseek/deepseek-r1:free).",
    )
    parser.add_argument("--max_retries", type=int, default=15)
    parser.add_argument(
        "--retry_backoff",
        type=float,
        default=10.0,
        help="Base backoff in seconds; actual sleep is max(10, retry_backoff * attempt).",
    )
    return parser.parse_args()


def _prepare_dataset(
    args: argparse.Namespace,
    outpath: str,
) -> tuple[Dataset, Dict[str, set[int]], str]:
    """
    Load, optionally subsample, and shuffle the dataset.

    :param args: Parsed CLI arguments controlling dataset choice and sampling.
    :param outpath: Output path used to determine resume/fill behavior.
    :returns: Tuple ``(dataset, existing, dataset_name_for_log)`` where
        ``existing`` maps problems to the set of already-filled sample indices.
    """
    dataset, existing, dataset_name_for_log = prepare_math_gateway_dataset_from_args(
        args=args,
        outpath=outpath,
        logger=logger,
        load_math500_fn=load_math500,
        load_remote_dataset_fn=load_dataset,
    )
    return dataset, existing, dataset_name_for_log


def _generate_samples(client, args: argparse.Namespace, outpath: str) -> None:
    """
    Main generation loop over the dataset.

    :param client: OpenRouter client used to issue chat completions.
    :param args: Parsed CLI arguments controlling generation behavior.
    :param outpath: Path to the JSONL file where results are written.
    :returns: ``None``. Samples are written to disk and progress is logged.
    """
    dataset, existing, _ = _prepare_dataset(args, outpath)

    total_new = 0
    for problem, gold_answer, sample_idx in iter_math_gateway_samples(
        dataset,
        args.num_samples,
        existing,
    ):
        text, finish_reason, usage = call_with_gateway_retries(
            partial(_call_model, client, problem, args),
            args=args,
            logger=logger,
            sample_idx=sample_idx,
            problem_snippet=problem,
            min_sleep=10.0,
        )

        _, ans = _extract_blocks(text)
        pred_canon = _canon_math(ans)

        row: Dict[str, Any] = build_math_gateway_row_base(
            problem=problem,
            gold_answer=gold_answer,
            gold_answer_canon=_canon_math(gold_answer),
            split=args.split,
            step=args.step,
            sample_idx=sample_idx,
        )
        row.update(
            {
                "endpoint": "openrouter",
                "deployment": args.model,
                "api_version": None,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "pass1": {
                    "output": text.strip(),
                    "pred_answer": ans,
                    "pred_answer_canon": pred_canon,
                    "is_correct_pred": bool(
                        pred_canon
                        and row["gold_answer_canon"]
                        and row["gold_answer_canon"] in pred_canon
                    ),
                    "valid_tag_structure": _valid_tag_structure(text),
                    "finish_reason": finish_reason,
                },
            },
        )

        if usage is not None:
            row["usage"] = build_usage_dict(usage)

        append_jsonl_row(outpath, row)
        total_new += 1
        existing.setdefault(problem, set()).add(sample_idx)

    logger.info("All done. Wrote %d new samples → %s", total_new, outpath)


def main() -> None:
    """
    CLI entry point for generating DeepSeek-R1 samples via OpenRouter.

    :returns: ``None``. The function parses arguments and runs the generation loop.
    """
    args = _parse_args()
    random.seed(args.seed)
    outpath = os.path.join(args.output_dir, f"step{args.step:04d}_{args.split}.jsonl")

    client = _make_client()
    logger.info("OpenRouter client ready | model=%s", args.model)
    _generate_samples(client, args, outpath)


if __name__ == "__main__":
    main()
