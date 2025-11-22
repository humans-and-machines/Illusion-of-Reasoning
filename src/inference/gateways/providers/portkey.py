#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gateway-style MATH-500 inference via the Portkey AI Gateway (e.g., Princeton AI
Sandbox).

Behavior:
- Single-pass generation with the same math system prompt used in GRPO runs.
- Writes JSONL to: {output_dir}/step{step:04d}_{split}.jsonl
- Resumable: if a JSONL already exists, only missing sample_idx entries are
  generated.

This script is a **specialized / legacy gateway runner**. For canonical
HF-based math evaluation, prefer
``src.inference.cli.unified_math``; use this module only when you
need Portkey/sandbox-based remote models.

Auth:
- Expects an API key in the `AI_SANDBOX_KEY` env var (Portkey key or sandbox
  key).

Example usage:
  export AI_SANDBOX_KEY="***"
  python -m src.inference.gateways.providers.portkey \
      --output_dir artifacts/results/gpt4o-math-portkey
"""

from __future__ import annotations
import os
import random
import sys
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Dict

from src.inference.gateways.base import setup_gateway_logger
from src.inference.utils.common import (
    append_jsonl_row,
    build_math_gateway_arg_parser,
    build_math_gateway_messages,
    build_math_gateway_row_base,
    build_usage_dict,
    canon_math as _canon_math,
    extract_blocks as _extract_blocks,
    extract_problem_and_answer,
    parse_openai_chat_response,
    prepare_math_gateway_dataset_from_args,
    require_datasets,
    setup_hf_cache_dir_env,
    valid_tag_structure as _valid_tag_structure,
)
from src.inference.domains.math.math_core import load_math500
from src.inference.utils.task_registry import MATH_SYSTEM_PROMPT


_DatasetType, load_dataset = require_datasets()
logger = setup_gateway_logger(__name__)


# ----------------------- Prompt -----------------------
SYSTEM_PROMPT = MATH_SYSTEM_PROMPT


# ----------------------- Portkey client + call -----------------------
@dataclass
class PortkeyCallParams:
    """
    Lightweight container for Portkey generation parameters.

    :param temperature: Sampling temperature for the model.
    :param top_p: Nucleus-sampling parameter for the model.
    :param max_output_tokens: Maximum number of tokens to generate.
    :param request_timeout: Per-request timeout in seconds.
    """

    temperature: float
    top_p: float
    max_output_tokens: int
    request_timeout: int


@dataclass
class PortkeyRunConfig:
    """
    Configuration for a Portkey MATH-500 pass.

    :param output_path: Path to the JSONL file where results are written.
    :param split_name: Dataset split name (for example, ``\"test\"``).
    :param model_name: Model identifier used via Portkey.
    :param num_samples: Number of samples to generate per problem.
    :param params: Generation parameters such as temperature and limits.
    :param seed: Random seed for sampling and dataset shuffling.
    :param step: Training or checkpoint step identifier for filenames.
    """

    output_path: str
    split_name: str
    model_name: str
    num_samples: int
    params: PortkeyCallParams
    seed: int
    step: int


@dataclass
class ExampleContext:
    """
    Per-example metadata for a MATH-500 row.

    :param problem: Normalized problem text.
    :param gold_answer: Ground-truth answer associated with the problem.
    :param canon_gold: Canonicalized gold answer.
    :param sample_idx: Sample index for this generation.
    """

    problem: str
    gold_answer: Any
    canon_gold: Any
    sample_idx: int


@dataclass
class PortkeyCallResult:
    """
    Result of a single Portkey generation call.

    :param text: Raw model output text.
    :param answer: Extracted answer text from the output.
    :param finish_reason: Finish reason reported by the API.
    :param usage: Optional usage object returned by the SDK.
    """

    text: str
    answer: str
    finish_reason: Any
    usage: Any


def _make_client():
    """
    Construct a Portkey client using ``AI_SANDBOX_KEY`` from the environment.

    :returns: Configured ``portkey_ai.Portkey`` client instance.
    :raises RuntimeError: If ``AI_SANDBOX_KEY`` is not set.
    :raises ImportError: If the ``portkey-ai`` package is not installed.
    """
    try:
        portkey_mod = import_module("portkey_ai")
    except ImportError as import_exc:  # pragma: no cover - optional dependency
        print(
            "portkey-ai is required for this script: pip install portkey-ai",
            file=sys.stderr,
        )
        raise import_exc

    api_key = os.getenv("AI_SANDBOX_KEY")
    if not api_key:
        raise RuntimeError("AI_SANDBOX_KEY env var is required for Portkey.")
    client_cls = getattr(portkey_mod, "Portkey")
    return client_cls(api_key=api_key)


def _call_model(
    client,
    model: str,
    problem: str,
    params: PortkeyCallParams,
):
    """
    Call the Portkey model for a single math problem.

    :param client: Portkey client created by :func:`_make_client`.
    :param model: Model identifier to use via Portkey.
    :param problem: Raw problem text to send to the model.
    :param params: Generation parameters such as temperature and limits.
    :returns: Parsed response tuple as returned by :func:`parse_openai_chat_response`.
    """
    messages = build_math_gateway_messages(SYSTEM_PROMPT, problem)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=params.temperature,
        top_p=params.top_p,
        max_tokens=params.max_output_tokens,
        timeout=params.request_timeout,
    )
    return parse_openai_chat_response(resp)


def _iter_examples(dataset, num_examples: int | None):
    """
    Yield at most ``num_examples`` examples from a dataset (or all if ``None``).

    :param dataset: Dataset object supporting ``select`` and iteration.
    :param num_examples: Maximum number of examples to yield, or ``None`` for all.
    :returns: Iterator over dataset examples.
    """
    if num_examples is not None and num_examples > 0:
        dataset = dataset.select(range(min(num_examples, len(dataset))))
    yield from dataset


def _build_portkey_row(
    example: ExampleContext,
    result: PortkeyCallResult,
    config: PortkeyRunConfig,
) -> Dict[str, Any]:
    """
    Build a single JSONL row for Portkey MATH-500 inference and compute correctness.

    :param example: Per-example metadata describing problem and gold answer.
    :param result: Result from the Portkey model call.
    :param config: Run configuration including split, model, and step.
    :returns: Dictionary representing one JSONL row.
    """
    pred_canon = _canon_math(result.answer)
    is_correct = bool(
        pred_canon
        and example.canon_gold
        and example.canon_gold in pred_canon
    )

    row: Dict[str, Any] = build_math_gateway_row_base(
        problem=example.problem,
        gold_answer=example.gold_answer,
        gold_answer_canon=example.canon_gold,
        split=config.split_name,
        step=config.step,
        sample_idx=example.sample_idx,
    )
    row.update(
        {
            "endpoint": "portkey-ai",
            "deployment": config.model_name,
            "api_version": None,
            "temperature": config.params.temperature,
            "top_p": config.params.top_p,
            "pass1": {
                "output": result.text.strip(),
                "pred_answer": result.answer,
                "pred_answer_canon": pred_canon,
                "is_correct_pred": is_correct,
                "valid_tag_structure": _valid_tag_structure(result.text),
                "finish_reason": result.finish_reason,
            },
        },
    )

    if result.usage is not None:
        row["usage"] = build_usage_dict(result.usage)

    return row


def run_portkey_math_inference(
    client,
    dataset,
    existing: Dict[str, set],
    config: PortkeyRunConfig,
) -> None:
    """
    Run single-pass math inference via Portkey and write JSONL results.

    :param client: Portkey client created by :func:`_make_client`.
    :param dataset: Dataset object containing math problems and answers.
    :param existing: Mapping from problem to already-filled sample indices.
    :param config: Run configuration including output path and sampling options.
    :returns: ``None``. Results are appended to the JSONL file.
    """
    random.seed(config.seed)
    total_new = 0
    for example in _iter_examples(dataset, None):
        problem, gold_answer = extract_problem_and_answer(example)
        if not problem or gold_answer is None:
            continue

        todo_indices = [
            idx
            for idx in range(config.num_samples)
            if idx not in existing.get(problem, set())
        ]
        if not todo_indices:
            continue

        for sample_idx in todo_indices:
            text, finish_reason, usage = _call_model(
                client=client,
                model=config.model_name,
                problem=problem,
                params=config.params,
            )

            _, answer = _extract_blocks(text)
            row = _build_portkey_row(
                ExampleContext(
                    problem=problem,
                    gold_answer=gold_answer,
                    canon_gold=_canon_math(gold_answer),
                    sample_idx=sample_idx,
                ),
                PortkeyCallResult(
                    text=text,
                    answer=answer,
                    finish_reason=finish_reason,
                    usage=usage,
                ),
                config,
            )

            append_jsonl_row(config.output_path, row)
            total_new += 1
            existing.setdefault(problem, set()).add(sample_idx)

    logger.info("All done. Wrote %d new samples → %s", total_new, config.output_path)


def main() -> None:
    """
    Parse arguments, load dataset, and run Portkey-based MATH-500 inference.

    :returns: ``None``. The function parses CLI args and runs the main loop.
    """
    parser = build_math_gateway_arg_parser(
        default_temperature=0.05,
        description="Portkey MATH-500 runner.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Model name to use via Portkey (e.g., gpt-4o, gpt-5, o3-mini).",
    )

    args = parser.parse_args()

    client = _make_client()
    logger.info("Portkey client ready | model=%s", args.model)

    output_path = os.path.join(
        args.output_dir,
        f"step{args.step:04d}_{args.split}.jsonl",
    )
    call_params = PortkeyCallParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.max_output_tokens,
        request_timeout=args.request_timeout,
    )
    config = PortkeyRunConfig(
        output_path=output_path,
        split_name=args.split,
        model_name=args.model,
        num_samples=args.num_samples,
        params=call_params,
        seed=args.seed,
        step=args.step,
    )
    dataset, existing, _ = prepare_math_gateway_dataset_from_args(
        args=args,
        outpath=output_path,
        logger=logger,
        load_math500_fn=load_math500,
        load_remote_dataset_fn=load_dataset,
        cache_dir=setup_hf_cache_dir_env("./.hf_cache"),
    )
    run_portkey_math_inference(
        client=client,
        dataset=dataset,
        existing=existing,
        config=config,
    )


if __name__ == "__main__":
    main()
