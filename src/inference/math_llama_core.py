#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-pass batch inference for Llama-style chat LMs that produce:
  <think> ... </think><answer> ... </answer>

DeepSpeed (ZeRO-3) edition:
- Loads ZeRO-3 shards directly from checkpoint-XXX/global_stepXXXX (no merge).
- Uses deepspeed.initialize + engine.load_checkpoint.
- Generation goes through engine.module.generate(...).

This module adapts the generic math_core two-pass loop so that it can run
with DeepSpeed-based Llama checkpoints, while keeping the CLI surface area
backwards-compatible.
"""

from __future__ import annotations

import argparse
import importlib
import os
from typing import Any, List, Optional, Tuple

from packaging import version

from src.inference.common import (
    build_math_inference_config_kwargs_from_args,
    require_datasets,
    require_torch,
    require_transformers,
    setup_script_logger,
)
from src.inference.math_pass_utils import DEFAULT_SECOND_PASS_PHRASE
from src.inference.gateway_utils import (
    configure_unified_runner_common as _configure_unified_runner_common,
)
from src.inference.math_core import (
    MathInferenceConfig,
    load_math500 as _load_math500_core,
    run_inference_on_split as _run_math_core_inference,
)
from src.inference.math_llama_utils import DSModelWrapper

torch = require_torch("math_llama_core")
transformers_mod = require_transformers("math_llama_core")

AutoConfig = transformers_mod.AutoConfig
AutoTokenizer = transformers_mod.AutoTokenizer
AutoModelForCausalLM = transformers_mod.AutoModelForCausalLM

try:
    deepspeed = importlib.import_module("deepspeed")
except ImportError as exc:  # pragma: no cover - hard dependency
    raise RuntimeError(
        "math_llama_core requires 'deepspeed'; install it to use this script.",
    ) from exc

_, load_dataset = require_datasets()

# ───────────────────────── Logging ─────────────────────────
logger = setup_script_logger(__name__)


# ───── PyTorch 2.6 DeepSpeed unpickle patch (no-op if absent) ─────
try:
    if version.parse(torch.__version__) >= version.parse("2.6.0"):
        torch_serialization = importlib.import_module("torch.serialization")
        zero_config = importlib.import_module("deepspeed.runtime.zero.config")
        loss_scaler_mod = importlib.import_module(
            "deepspeed.runtime.fp16.loss_scaler",
        )
        add_safe_globals = getattr(torch_serialization, "add_safe_globals")
        zero_stage_enum_cls = getattr(zero_config, "ZeroStageEnum")
        loss_scaler_cls = getattr(loss_scaler_mod, "LossScaler")
        add_safe_globals([zero_stage_enum_cls, loss_scaler_cls])
        logger.info("DeepSpeed ZeRO patch enabled")
except (ImportError, AttributeError) as exc:  # pragma: no cover - best-effort only
    logger.warning("DeepSpeed patch disabled (missing deps/attrs): %r", exc)


# ───────────────────── Inference adapter ─────────────────────
def run_inference_on_split(*args: Any, **kwargs: Any) -> None:
    """
    Run two-pass math inference for a dataset split.

    This is a thin adapter around :func:`src.inference.math_core.run_inference_on_split`
    that accepts the legacy keyword-only signature used by tests and the CLI
    while delegating the core loop to :class:`MathInferenceConfig`.

    :param args: Positional arguments are rejected to preserve keyword-only usage.
    :param kwargs: Keyword arguments including ``split_name``, ``examples``,
        ``tokenizer``, ``model``, ``step``, ``outdir``, and optional config
        overrides such as ``batch_size`` or ``num_samples``.
    :returns: ``None``. Results are written to JSONL under ``outdir``.
    :raises TypeError: If positional arguments are provided or required keys are missing.
    """
    if args:
        raise TypeError(
            "run_inference_on_split expects only keyword arguments; "
            f"got positional args: {args!r}",
        )

    required_keys = ["split_name", "examples", "tokenizer", "model", "step", "outdir"]
    missing = [name for name in required_keys if name not in kwargs]
    if missing:
        raise TypeError(f"Missing required arguments: {', '.join(sorted(missing))}")

    split_name = kwargs.pop("split_name")
    examples = kwargs.pop("examples")
    tokenizer = kwargs.pop("tokenizer")
    model = kwargs.pop("model")
    step = kwargs.pop("step")
    outdir = kwargs.pop("outdir")

    config_defaults = {
        "batch_size": 8,
        "num_samples": 1,
        "temperature": 0.0,
        "top_p": 0.95,
        "entropy_mode": "reconsider",
        "eos_ids": None,
        "two_pass": False,
        "second_pass_phrase": DEFAULT_SECOND_PASS_PHRASE,
        "second_pass_use_sample_idx": 0,
        "think_cap": 750,
        "answer_cap": 50,
    }
    config_defaults.update(kwargs)

    config = MathInferenceConfig(
        split_name=split_name,
        output_dir=outdir,
        step=step,
        **config_defaults,
    )
    _run_math_core_inference(
        examples=examples,
        tokenizer=tokenizer,
        model=model,
        config=config,
    )


def load_math500(
    cache_dir: str,
    split: str,
    seed: int,
    dataset_path: Optional[str] = None,
):
    """
    Load the MATH-500 benchmark via :mod:`datasets` using the shared core logic.

    This is a thin wrapper around :func:`src.inference.math_core.load_math500`
    so that the Llama runner reuses a single implementation of the loading
    and normalization logic.

    :param cache_dir: Directory to use as a datasets cache.
    :param split: Dataset split name (for example, ``\"test\"``).
    :param seed: Random seed used when sub-sampling competition-math fallback data.
    :param dataset_path: Optional local JSON file to load instead of remote datasets.
    :returns: A datasets-like object exposing ``map`` and ``select``.
    """
    return _load_math500_core(cache_dir, split, seed, dataset_path)


# ─────────────────────────── CLI helpers ───────────────────────────
def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Create an argument parser for the DeepSpeed-backed Llama MATH runner.

    :returns: Configured :class:`argparse.ArgumentParser` for the CLI.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        required=True,
        help="Path to checkpoint-XXX directory (contains global_stepXXXX/)",
    )
    parser.add_argument("--revision")
    parser.add_argument("--output_dir", required=True)

    # Data selection + common decoding/budget/system flags
    parser.add_argument(
        "--dataset_id",
        default="MATH-500",
        help="Use 'MATH-500' or a HF dataset path",
    )
    _configure_unified_runner_common(parser, default_dtype="bfloat16")

    # DeepSpeed controls
    parser.add_argument(
        "--ds_config",
        required=True,
        help="Path to DeepSpeed ZeRO-3 inference JSON",
    )
    parser.add_argument(
        "--ds_tag",
        default=None,
        help=(
            "Checkpoint tag folder (e.g., 'global_step400'). "
            "Defaults to global_step{--step} if set, else 'latest'."
        ),
    )
    return parser


def _init_tokenizer_and_eos_ids(
    args: argparse.Namespace,
) -> Tuple[AutoTokenizer, Optional[List[int]], str]:
    """
    Initialise the tokenizer and derived EOS ID set.

    :param args: Parsed CLI arguments containing model and tokenizer options.
    :returns: A tuple ``(tokenizer, eos_ids, hf_cache_dir)`` where ``eos_ids``
        is a sorted list of EOS token IDs (or ``None``) and ``hf_cache_dir`` is
        the Hugging Face cache directory path.
    """
    hf_cache_dir = os.path.abspath("./.hf_cache")
    tok_src = args.tokenizer_path or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tok_src,
        revision=args.revision,
        trust_remote_code=True,
        cache_dir=hf_cache_dir,
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.truncation_side = "left"

    eos_id_set = set()
    if tokenizer.eos_token_id is not None:
        eos_id_set.add(int(tokenizer.eos_token_id))
    for tok in ("<|eot_id|>", "<|end_of_text|>"):
        token_id = tokenizer.convert_tokens_to_ids(tok)
        if token_id is not None and token_id != tokenizer.pad_token_id:
            eos_id_set.add(int(token_id))
    eos_ids_sorted: Optional[List[int]] = sorted(eos_id_set) if eos_id_set else None
    return tokenizer, eos_ids_sorted, hf_cache_dir


def _init_model(
    args: argparse.Namespace,
    hf_cache_dir: str,
) -> Tuple[Any, str]:
    """
    Initialise the HF model, wrap it in DeepSpeed, and load checkpoints.

    :param args: Parsed CLI arguments containing model and DeepSpeed settings.
    :param hf_cache_dir: Hugging Face cache directory path.
    :returns: A tuple ``(model, ds_tag)`` where ``model`` is a
        :class:`DSModelWrapper` around the DeepSpeed engine and ``ds_tag`` is
        the resolved checkpoint tag.
    """
    cfg = AutoConfig.from_pretrained(
        args.model_name_or_path,
        revision=args.revision,
        trust_remote_code=True,
        cache_dir=hf_cache_dir,
    )
    try:
        cfg.attn_implementation = args.attn_implementation
    except AttributeError:  # pragma: no cover - older HF configs without this field
        pass

    model_hf = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    try:
        target_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        model_hf.to(target_dtype)
    except (TypeError, RuntimeError, ValueError):  # pragma: no cover - dtype edge cases
        pass

    engine, _, _, _ = deepspeed.initialize(
        model=model_hf,
        config=args.ds_config,
        model_parameters=[
            param for param in model_hf.parameters() if param.requires_grad
        ],
    )
    engine.module.eval()

    ds_tag = args.ds_tag or (f"global_step{args.step}" if args.step else "latest")
    logger.info("Loading ZeRO checkpoint: dir=%s | tag=%s", args.model_name_or_path, ds_tag)
    engine.load_checkpoint(
        args.model_name_or_path,
        tag=ds_tag,
        load_optimizer_states=False,
        load_lr_scheduler_states=False,
        load_module_strict=False,
    )

    model = DSModelWrapper(engine).eval()
    return model, ds_tag


def _load_dataset_for_args(
    args: argparse.Namespace,
    hf_cache_dir: str,
) -> Tuple[Any, str]:
    """
    Load the requested dataset (MATH-500 or an arbitrary HF dataset).

    :param args: Parsed CLI arguments specifying ``dataset_id``, ``split``,
        and optional ``num_examples``.
    :param hf_cache_dir: Hugging Face cache directory path.
    :returns: Tuple ``(dataset, dataset_name_for_log)`` describing the loaded data.
    """
    if args.dataset_id.upper() == "MATH-500":
        dataset = load_math500(hf_cache_dir, args.split, args.seed)
        dataset_name_for_log = "MATH-500"
    else:
        dataset = load_dataset(args.dataset_id, split=args.split, cache_dir=hf_cache_dir)
        dataset_name_for_log = args.dataset_id

    if args.num_examples is not None and args.num_examples > 0:
        dataset = dataset.select(range(min(args.num_examples, len(dataset))))

    return dataset, dataset_name_for_log


# ─────────────────────────── Main ───────────────────────────
def main() -> None:
    """
    CLI entrypoint for the DeepSpeed-backed Llama MATH runner.

    :returns: ``None``. The function parses arguments, runs inference, and logs progress.
    """
    parser = _build_arg_parser()
    args = parser.parse_args()

    tokenizer, eos_ids, hf_cache_dir = _init_tokenizer_and_eos_ids(args)
    model, ds_tag = _init_model(args, hf_cache_dir)
    dataset, dataset_name_for_log = _load_dataset_for_args(args, hf_cache_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(
        "Model: %s | dtype-hint=%s | DS tag=%s",
        args.model_name_or_path,
        args.dtype,
        ds_tag,
    )
    logger.info(
        "Dataset: %s split=%s | N=%d",
        dataset_name_for_log,
        args.split,
        len(dataset),
    )
    logger.info("Output dir: %s", args.output_dir)

    run_inference_on_split(
        split_name=args.split,
        examples=dataset,
        tokenizer=tokenizer,
        model=model,
        step=args.step,
        outdir=args.output_dir,
        **build_math_inference_config_kwargs_from_args(args, eos_ids),
    )
    logger.info("All inference complete.")


if __name__ == "__main__":
    main()
