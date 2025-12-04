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
from contextlib import contextmanager, suppress
from typing import Any, List, Optional, Tuple

from packaging import version

from src.inference.utils.common import (
    build_math_inference_config_kwargs_from_args,
    require_datasets,
    require_torch,
    require_transformers,
    setup_script_logger,
)
from src.inference.utils.gateway_utils import configure_unified_runner_common as _configure_unified_runner_common
from src.inference.utils.gateway_utils import limit_dataset_for_args
from src.inference.utils.math_llama_utils import DSModelWrapper
from src.inference.utils.math_pass_utils import DEFAULT_SECOND_PASS_PHRASE


torch = require_torch("math_llama_core")
_math_core = importlib.import_module("src.inference.domains.math.math_core")


# Some lightweight torch stubs used in tests may lack a usable inference_mode.
# Normalize it to a context manager in those cases.
def _normalize_inference_mode(module):  # pragma: no cover - exercised in stub test envs
    @contextmanager
    def _noop_ctx():
        yield

    def _ctx_factory(candidate):
        """Return a callable that yields a context manager, or None if invalid."""
        if candidate is None:
            return None
        if hasattr(candidate, "__enter__"):

            def _return_candidate(*_args, **_kwargs):
                return candidate

            return _return_candidate
        if callable(candidate):
            try:
                ctx = candidate()
                if hasattr(ctx, "__enter__"):

                    def _call_candidate(*_args, **_kwargs):
                        return candidate()

                    return _call_candidate
            except (TypeError, AttributeError):
                pass
        return None

    ctx_call = _ctx_factory(getattr(module, "inference_mode", None))
    if ctx_call is None:
        ctx_call = _ctx_factory(getattr(module, "no_grad", None))
    if ctx_call is None:

        def _noop_wrapper(*_args, **_kwargs):
            return _noop_ctx()

        ctx_call = _noop_wrapper
    module.inference_mode = ctx_call  # type: ignore[attr-defined]


_normalize_inference_mode(torch)
transformers_mod = require_transformers("math_llama_core")


class _MissingHF:
    """Fallback stub used when transformers classes are unavailable."""

    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # pragma: no cover - defensive
        """Raise an informative error mirroring transformers' API surface."""
        raise ImportError(
            "Required transformers class missing; install a full transformers package.",
        )

    @classmethod
    def missing(cls, *args, **kwargs):  # pragma: no cover - defensive
        """Alias for compatibility with callers expecting a callable fallback."""
        return cls.from_pretrained(*args, **kwargs)


def _safe_get_transformers_attr(name: str):
    """Return a transformers attribute or fall back when stubs lack it."""
    try:
        return getattr(transformers_mod, name)
    except AttributeError:  # pragma: no cover - defensive against stub modules
        return _MissingHF


AutoConfig = _safe_get_transformers_attr("AutoConfig")
AutoTokenizer = _safe_get_transformers_attr("AutoTokenizer")
AutoModelForCausalLM = _safe_get_transformers_attr("AutoModelForCausalLM")

try:
    deepspeed = importlib.import_module("deepspeed")
    _HAS_DEEPSPEED = True
except ImportError:
    _HAS_DEEPSPEED = False

    class _DeepSpeedStub:
        """Minimal stub that raises when DeepSpeed functionality is used."""

        def __getattr__(self, name):
            msg = f"math_llama_core requires 'deepspeed' for model initialization (missing attribute: {name})"
            raise RuntimeError(msg)

        def initialize(self, *_, **__):
            """Mirror deepspeed.initialize and raise with a clear message."""
            raise RuntimeError("DeepSpeed unavailable: install the 'deepspeed' package.")

        def is_available(self) -> bool:
            """Signal to callers that DeepSpeed is missing."""
            return False

    deepspeed = _DeepSpeedStub()

_, load_dataset = require_datasets()

# ───────────────────────── Logging ─────────────────────────
logger = setup_script_logger(__name__)


# ───── PyTorch 2.6 DeepSpeed unpickle patch (no-op if absent) ─────
try:
    if _HAS_DEEPSPEED and hasattr(torch, "__version__") and version.parse(torch.__version__) >= version.parse("2.6.0"):
        torch_serialization = importlib.import_module("torch.serialization")
        zero_config = importlib.import_module("deepspeed.runtime.zero.config")
        loss_scaler_mod = importlib.import_module("deepspeed.runtime.fp16.loss_scaler")
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

    This is a thin adapter around
    :func:`src.inference.domains.math.math_core.run_inference_on_split`
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
            f"run_inference_on_split expects only keyword arguments; got positional args: {args!r}",
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

    math_core_mod = _math_core

    config = math_core_mod.MathInferenceConfig(
        split_name=split_name,
        output_dir=outdir,
        step=step,
        **config_defaults,
    )
    # Best-effort breadcrumb for tests; tolerate exotic stubs that forbid setattr.
    with suppress(AttributeError, TypeError):  # pragma: no cover - test stub convenience
        setattr(math_core_mod, "last_call", {"config": config, "examples": examples})
    math_core_mod.run_inference_on_split(
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

    This is a thin wrapper around
    :func:`src.inference.domains.math.math_core.load_math500`
    so that the Llama runner reuses a single implementation of the loading
    and normalization logic.

    :param cache_dir: Directory to use as a datasets cache.
    :param split: Dataset split name (for example, ``\"test\"``).
    :param seed: Random seed used when sub-sampling competition-math fallback data.
    :param dataset_path: Optional local JSON file to load instead of remote datasets.
    :returns: A datasets-like object exposing ``map`` and ``select``.
    """
    math_core_mod = _math_core
    return math_core_mod.load_math500(cache_dir, split, seed, dataset_path)


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
            "Checkpoint tag folder (e.g., 'global_step400'). Defaults to global_step{--step} if set, else 'latest'."
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
        model_parameters=[param for param in model_hf.parameters() if param.requires_grad],
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

    dataset = limit_dataset_for_args(dataset, args)

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
