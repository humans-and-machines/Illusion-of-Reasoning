#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared inference helpers used across math/carpark/crossword entrypoints.
The goal is to keep canonicalization, tag parsing, and small torch utilities
in one place so the task-specific scripts stay focused on their domain logic.

Note: lightweight, text-only helpers have been moved to ``text_utils`` to keep
this module smaller and more focused on torch/transformers glue code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

from src.inference import gateway_utils as _gateway_utils
from src.inference import math_pass_utils as _math_pass_utils

# Explicitly bind selected gateway_utils helpers so that static analyzers see
# them on this module without treating them as unused imports.
DEFAULT_SECOND_PASS_PHRASE = _math_pass_utils.DEFAULT_SECOND_PASS_PHRASE
build_second_pass_cue_strings = _math_pass_utils.build_second_pass_cue_strings
OPENR1_PROMPT_TEMPLATE = _gateway_utils.OPENR1_PROMPT_TEMPLATE
PassOutputs = _gateway_utils.PassOutputs
append_jsonl_row = _gateway_utils.append_jsonl_row
build_eos_ids_from_tokenizer = _gateway_utils.build_eos_ids_from_tokenizer
build_math_gateway_arg_parser = _gateway_utils.build_math_gateway_arg_parser
build_math_gateway_messages = _gateway_utils.build_math_gateway_messages
build_math_gateway_row_base = _gateway_utils.build_math_gateway_row_base
build_usage_dict = _gateway_utils.build_usage_dict
build_two_pass_row_base = _gateway_utils.build_two_pass_row_base
call_with_gateway_retries = _gateway_utils.call_with_gateway_retries
call_with_retries = _gateway_utils.call_with_retries
configure_tokenizer_and_eos = _gateway_utils.configure_tokenizer_and_eos
configure_unified_runner_common = _gateway_utils.configure_unified_runner_common
extract_problem_and_answer = _gateway_utils.extract_problem_and_answer
iter_jsonl_objects = _gateway_utils.iter_jsonl_objects
iter_math_gateway_samples = _gateway_utils.iter_math_gateway_samples
limit_dataset_examples = _gateway_utils.limit_dataset_examples
load_local_json_dataset = _gateway_utils.load_local_json_dataset
load_remote_dataset_default = _gateway_utils.load_remote_dataset_default
parse_openai_chat_response = _gateway_utils.parse_openai_chat_response
prepare_math_gateway_dataset = _gateway_utils.prepare_math_gateway_dataset
prepare_math_gateway_dataset_from_args = _gateway_utils.prepare_math_gateway_dataset_from_args
require_datasets = _gateway_utils.require_datasets
scan_existing_pass1_results = _gateway_utils.scan_existing_pass1_results
scan_existing_problem_samples = _gateway_utils.scan_existing_problem_samples
setup_hf_cache_dir_env = _gateway_utils.setup_hf_cache_dir_env
setup_script_logger = _gateway_utils.setup_script_logger

# Explicitly bind selected math_pass_utils helpers as well.
RECONSIDER_PATTERNS = _math_pass_utils.RECONSIDER_PATTERNS
add_token_and_tag_fields = _math_pass_utils.add_token_and_tag_fields
build_entropy_pass_base = _math_pass_utils.build_entropy_pass_base
build_math_pass_meta = _math_pass_utils.build_math_pass_meta
canon_math = _math_pass_utils.canon_math
contains_canon = _math_pass_utils.contains_canon
extract_blocks = _math_pass_utils.extract_blocks
finite_mean = _math_pass_utils.finite_mean
pack_math_pass_result = _math_pass_utils.pack_math_pass_result
valid_tag_structure = _math_pass_utils.valid_tag_structure

# Retain dynamic re-exports so that any additions to __all__ in the helper
# modules are still reflected here at runtime.
for _mod in (_math_pass_utils, _gateway_utils):
    for _name in getattr(_mod, "__all__", []):
        globals()[_name] = getattr(_mod, _name)

_CORE_EXPORTS = [
    # Core generation / entropy helpers defined in this module
    "move_inputs_to_device",
    "tokenize_prefixes_for_generate",
    "build_generate_kwargs",
    "make_generate_kwargs_for_cap",
    "build_second_pass_cue_strings",
    "build_second_pass_think_prefixes",
    "build_extra_pass_results_for_cues",
    "empty_pass_outputs",
    "decode_generated_row",
    "decode_and_score_batch",
    "generate_and_score_batch",
    "classify_stop_reason",
    "StopOnSubstrings",
    "first_eos_any",
    "entropy_from_start_index",
    "GenerationLimits",
    "repeat_for_samples",
    "SamplingConfigBase",
    "DecodeBatchConfig",
    "DecodeBatchContext",
    "GenerateBatchParams",
    "GenerateBatchRuntime",
    "run_generate_batch",
    "build_math_inference_config_kwargs",
    "build_math_inference_config_kwargs_from_args",
    "require_torch",
    "require_transformers",
    # Explicitly re-export selected helpers so static analyzers
    # can see them on this module without relying on dynamic globals().
    "DEFAULT_SECOND_PASS_PHRASE",
    "OPENR1_PROMPT_TEMPLATE",
    "extract_problem_and_answer",
    "load_local_json_dataset",
    "require_datasets",
    "scan_existing_pass1_results",
    "setup_script_logger",
    "build_math_pass_meta",
    "canon_math",
    "finite_mean",
    "pack_math_pass_result",
]

_HELPER_EXPORTS = []
for _mod in (_math_pass_utils, _gateway_utils):
    _helper_all = getattr(_mod, "__all__", None)
    if _helper_all:
        _HELPER_EXPORTS.extend(list(_helper_all))

__all__ = _CORE_EXPORTS + _HELPER_EXPORTS

if TYPE_CHECKING:
    import torch
    from transformers import StoppingCriteria
else:  # pragma: no cover - optional dependency at runtime
    try:
        torch = import_module("torch")
        transformers_mod = import_module("transformers")
        StoppingCriteria = getattr(transformers_mod, "StoppingCriteria")
    except ImportError:  # pragma: no cover - type-check / lint environments
        class _TorchStub:
            """Stub object that raises if torch-dependent utilities are used."""

            def __getattr__(self, _name: str) -> Any:
                msg = "torch is required for inference utilities in inference.common."
                raise ImportError(msg)

            def is_available(self) -> bool:
                """Return False to mirror torch.cuda.is_available-style probes."""
                return False

            def device(self) -> str:
                """Placeholder device accessor to satisfy style checks."""
                return "cpu"

        torch = _TorchStub()

        class StoppingCriteria:
            """Stub StoppingCriteria base when transformers is unavailable."""

            def __call__(self, *_: Any, **__: Any) -> bool:
                msg = "transformers is required for StoppingCriteria in inference.common."
                raise ImportError(msg)

            def clone(self) -> "StoppingCriteria":
                """Return self; provided solely to satisfy minimal API expectations."""
                return self

            def has_stops(self) -> bool:
                """Placeholder method for compatibility with StopOnSubstrings."""
                return False


# ---------------------------------------------------------------------------
# Torch helpers
# ---------------------------------------------------------------------------
def move_inputs_to_device(
    inputs: dict,
    device: Optional[torch.device] = None,
) -> tuple[dict, torch.Tensor]:
    """
    Move a HuggingFace-style inputs dict to CUDA (or a provided device).

    :param inputs: Mapping of tensor fields (for example, ``input_ids``, ``attention_mask``).
    :param device: Optional explicit device to move tensors to; if ``None``,
        CUDA is used when available.
    :returns: Tuple ``(inputs_on_device, input_lengths)`` where ``input_lengths``
        is a 1D tensor of attention-mask lengths.
    """
    input_lengths = inputs["attention_mask"].sum(dim=1)
    target_device = device
    if target_device is None and torch.cuda.is_available():
        target_device = torch.device("cuda")
    if target_device is not None:
        for k in inputs:
            inputs[k] = inputs[k].to(target_device)
        input_lengths = input_lengths.to(inputs["input_ids"].device)
    return inputs, input_lengths


def tokenize_prefixes_for_generate(
    tokenizer,
    prefixes: Sequence[str],
    *,
    max_length: int = 4096,
    device: Optional[torch.device] = None,
) -> tuple[dict, torch.Tensor]:
    """
    Tokenize a list of string prefixes for generation and move to device.

    This centralizes the common pattern used across math/carpark/crossword
    inference loops.

    :param tokenizer: Tokenizer used to encode the prefix strings.
    :param prefixes: Sequence of prefix strings to tokenize.
    :param max_length: Maximum sequence length used for truncation.
    :param device: Optional explicit device to move tensors to.
    :returns: Tuple ``(inputs_on_device, input_lengths)`` as in :func:`move_inputs_to_device`.
    """
    inputs = tokenizer(
        prefixes,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return move_inputs_to_device(inputs, device=device)


def build_generate_kwargs(
    *,
    cap: int,
    pad_token_id: int,
    eos_ids,
    entropy_mode: str,
    temperature: Optional[float],
    top_p: Optional[float],
    synced_gpus: bool = False,
) -> Dict[str, Any]:
    """
    Build generate() kwargs for a given token cap and sampling configuration.

    If temperature <= 0 → greedy (do_sample=False) and omit temperature/top_p.
    Else → sampling with provided temperature/top_p.

    :param cap: Maximum number of new tokens to generate.
    :param pad_token_id: Token ID used for padding.
    :param eos_ids: EOS token ID or IDs used to terminate generation.
    :param entropy_mode: Entropy computation mode (for example, ``\"reconsider\"`` or ``\"none\"``).
    :param temperature: Sampling temperature; when ``None`` or non-positive, greedy decoding is used.
    :param top_p: Optional nucleus-sampling parameter.
    :param synced_gpus: Whether to request synchronized generation across GPUs when available.
    :returns: Dictionary of keyword arguments suitable for ``model.generate``.
    """
    do_sample = temperature is not None and float(temperature) > 0.0
    kwargs: Dict[str, Any] = {
        "max_new_tokens": cap,
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_ids,
        "do_sample": do_sample,
        "return_dict_in_generate": True,
        "output_scores": entropy_mode != "none",
        "num_return_sequences": 1,
    }
    if synced_gpus and hasattr(torch, "distributed"):
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                kwargs["synced_gpus"] = True
        except (RuntimeError, AttributeError):  # pragma: no cover - defensive
            # If distributed is misconfigured, fall back to unsynced generate.
            pass
    if do_sample:
        kwargs["temperature"] = float(temperature)
        if top_p is not None:
            kwargs["top_p"] = float(top_p)
    return kwargs


def make_generate_kwargs_for_cap(
    *,
    cap: int,
    tokenizer,
    eos_ids,
    entropy_mode: str,
    temperature: Optional[float],
    top_p: Optional[float],
    synced_gpus: bool = False,
) -> Dict[str, Any]:
    """
    Convenience wrapper that derives ``pad_token_id`` from a tokenizer and
    forwards to :func:`build_generate_kwargs`.

    :param cap: Maximum number of new tokens to generate.
    :param tokenizer: Tokenizer providing ``pad_token_id`` and ``eos_token_id``.
    :param eos_ids: EOS token ID or IDs used to terminate generation.
    :param entropy_mode: Entropy computation mode (for example, ``\"reconsider\"`` or ``\"none\"``).
    :param temperature: Sampling temperature; when ``None`` or non-positive, greedy decoding is used.
    :param top_p: Optional nucleus-sampling parameter.
    :param synced_gpus: Whether to request synchronized generation across GPUs when available.
    :returns: Dictionary of keyword arguments suitable for ``model.generate``.
    """
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    return build_generate_kwargs(
        cap=cap,
        pad_token_id=pad_token_id,
        eos_ids=eos_ids,
        entropy_mode=entropy_mode,
        temperature=temperature,
        top_p=top_p,
        synced_gpus=synced_gpus,
    )


def build_second_pass_think_prefixes(
    *,
    base2_per_ex: Sequence[str],
    pre1_think: Sequence[str],
    row_to_ex_idx: Sequence[int],
    cue_str: str,
) -> List[str]:
    """
    Build pass-2 ``<think>`` prefixes aligned with pass-1 rows.

    This helper is shared by math_core and math_llama_core to avoid duplicating
    the row→example mapping logic.

    :param base2_per_ex: Base second-pass prompts, one per example.
    :param pre1_think: First-pass think-prefix prompts, one per row.
    :param row_to_ex_idx: For each row, index of the corresponding example.
    :param cue_str: Cue string to inject into second-pass reasoning.
    :returns: List of second-pass think-prefix prompts, one per row.
    """
    pre2_think: List[str] = []
    for row_index, _ in enumerate(pre1_think):
        ex_idx = row_to_ex_idx[row_index]
        base2 = base2_per_ex[ex_idx]
        pre2_think.append(base2 + "<think>\n" + cue_str)
    return pre2_think


def build_extra_pass_results_for_cues(
    *,
    two_pass: bool,
    extra_passes: Optional[Sequence[Tuple[str, "PassOutputs"]]],
    pack_result_for_extra: "Callable[[str, PassOutputs], Dict[str, Any]]",
    names: Sequence[str] = ("pass2a", "pass2b", "pass2c"),
) -> Dict[str, Dict[str, Any]]:
    """
    Build result dicts for optional multi-cue reconsideration passes
    (pass2a / pass2b / pass2c) given per-cue PassOutputs.

    :param two_pass: Whether second-pass generation is enabled.
    :param extra_passes: Optional sequence of ``(cue_str, PassOutputs)`` pairs.
    :param pack_result_for_extra: Callback that converts a cue and outputs into a result dict.
    :param names: Keys to use for extra passes (for example, ``(\"pass2a\", \"pass2b\")``).
    :returns: Mapping from pass names to packed result dictionaries.
    """
    extra_pass_results: Dict[str, Dict[str, Any]] = {}
    if two_pass and extra_passes:
        for idx, (cue_str_extra, outputs_extra) in enumerate(extra_passes):
            if idx >= len(names):
                break
            name = names[idx]
            extra_pass_results[name] = pack_result_for_extra(cue_str_extra, outputs_extra)
    return extra_pass_results


def empty_pass_outputs(total_rows: int) -> "PassOutputs":
    """
    Construct an empty ``PassOutputs`` instance for cases where a pass is
    skipped (e.g., disabled second pass).

    :param total_rows: Number of rows to represent in the empty outputs.
    :returns: A :class:`PassOutputs` instance with empty fields.
    """
    return PassOutputs(
        full_texts=[""] * total_rows,
        ent_think=[[] for _ in range(total_rows)],
        ent_answer=[[] for _ in range(total_rows)],
        stop_reason_think=[""] * total_rows,
        stop_reason_answer=[""] * total_rows,
    )


def decode_generated_row(tokenizer, seqs: torch.Tensor, input_lengths: torch.Tensor, row_i: int,
                         *, skip_special_tokens: bool = True) -> Tuple[torch.Tensor, str, int]:
    """
    Given batched generation outputs, return ``(gen_ids, decoded_text, start_tok_idx)``
    for a single row.

    This de-duplicates the common indexing/decoding pattern used when
    post-processing HF generate outputs.

    :param tokenizer: Tokenizer used to decode token IDs into text.
    :param seqs: Tensor of generated token IDs for all rows.
    :param input_lengths: Tensor of input lengths for each row.
    :param row_i: Index of the row to decode.
    :param skip_special_tokens: Whether to skip tokenizer special tokens when decoding.
    :returns: Tuple of generated IDs for the row, decoded text, and the start index.
    """
    start_tok_idx = int(input_lengths[row_i].item())
    gen_ids = seqs[row_i, start_tok_idx:]
    raw_txt = tokenizer.decode(gen_ids, skip_special_tokens=skip_special_tokens)
    return gen_ids, raw_txt, start_tok_idx


def _trim_and_classify(
    gen_ids: torch.Tensor,
    raw_text: str,
    stop_strings: Sequence[str],
    cap: int,
    eos_ids: Optional[Sequence[int]],
) -> Tuple[str, str]:
    """
    Helper to trim on stop strings and classify stop reason for a single row.

    :param gen_ids: Generated token IDs for the row.
    :param raw_text: Raw decoded text for the row.
    :param stop_strings: Collection of stop substrings to look for.
    :param cap: Maximum number of new tokens allowed.
    :param eos_ids: Optional sequence of EOS token IDs.
    :returns: Tuple ``(trimmed_text, stop_reason)``.
    """
    active_stop_strings = list(stop_strings) or []
    found_stop = any(stop_str in raw_text for stop_str in active_stop_strings)
    has_eos = False
    if eos_ids:
        for eos_token_id in eos_ids:
            if (gen_ids == eos_token_id).any():
                has_eos = True
                break
    hit_max = len(gen_ids) >= cap
    stop_reason = classify_stop_reason(found_stop, has_eos, hit_max)

    trimmed = raw_text
    for stop_str in active_stop_strings:
        if stop_str in trimmed:
            trimmed = trimmed.split(stop_str, 1)[0]
            break
    return trimmed.strip(), stop_reason


@dataclass
class _EntropyRowContext:
    """
    Container for row-wise entropy computation inputs.

    :param scores: Sequence of score tensors returned by ``generate``.
    :param sequences: Tensor of generated token IDs.
    :param eos_ids: EOS token IDs used to cap entropy computation, if any.
    :param model: Model instance used to compute entropy fallbacks.
    """

    scores: Any
    sequences: Any
    eos_ids: Optional[Sequence[int]]
    model: Any


@dataclass
class GenerationLimits:
    """
    Shared cap and sampling-count configuration used by multiple runners.

    :param batch_size: Number of examples per batch.
    :param num_samples: Number of samples to generate per problem.
    :param think_cap: Token cap for ``<think>`` generations.
    :param answer_cap: Token cap for ``<answer>`` generations.
    """

    batch_size: int
    num_samples: int
    think_cap: int
    answer_cap: int


def repeat_for_samples(values: Sequence[str], num_samples: int) -> List[str]:
    """
    Repeat each value ``num_samples`` times using row-major expansion.

    :param values: Sequence of base values to repeat.
    :param num_samples: Number of repetitions per value.
    :returns: Expanded list of values repeated for each sample.
    """
    return [value for value in values for _ in range(num_samples)]


def _row_entropy_from_scores(
    ctx: _EntropyRowContext,
    row_index: int,
    start_tok_idx: int,
) -> List[float]:
    """
    Compute per-token entropies for a single row, with fallback to
    entropy_from_start_index on NaNs/Infs.
    """
    gen_ids = ctx.sequences[row_index, start_tok_idx:]
    eos_limit = first_eos_any(gen_ids, ctx.eos_ids) if ctx.eos_ids else gen_ids.shape[0]
    token_entropies: List[float] = []
    bad = False

    for score_index, score_step in enumerate(ctx.scores):
        if score_index >= eos_limit:
            break
        logits = score_step[row_index].float()
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            bad = True
            break
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
            bad = True
            break
        probs = log_probs.exp()
        entropy_val = float(-(probs * log_probs).sum().item())
        if not math.isfinite(entropy_val):
            bad = True
            break
        token_entropies.append(entropy_val)

    if bad or not token_entropies:
        start_index = start_tok_idx - 1
        token_entropies = entropy_from_start_index(
            ctx.model,
            ctx.sequences[row_index : row_index + 1],
            start_index,
        ) or []
    return token_entropies


@dataclass
class DecodeBatchConfig:
    """Configuration for decoding and scoring a batch of generations."""

    stop_strings: Sequence[str]
    cap: int
    eos_ids: Optional[Sequence[int]]
    entropy_mode: str
    model: Any


@dataclass
class DecodeBatchContext:
    """Inputs required to decode and score a batch of generations."""

    tokenizer: Any
    sequences: Any
    scores: Any
    input_lengths: Any
    config: DecodeBatchConfig


def decode_and_score_batch(
    ctx: DecodeBatchContext,
) -> Tuple[List[str], List[List[float]], List[str]]:
    """
    Shared row-wise decode and entropy loop used by math/carpark/crossword helpers.

    :param ctx: Decode context bundling tokenizer, scores, sequences, and config.
    :returns: Tuple ``(decoded_texts, entropy_series, stop_reasons)`` per row.
    """
    total_rows = ctx.sequences.shape[0]
    decoded_texts: List[str] = []
    entropy_series: List[List[float]] = []
    stop_reasons: List[str] = []

    for row_index in range(total_rows):
        start_tok_idx = int(ctx.input_lengths[row_index].item())
        gen_ids = ctx.sequences[row_index, start_tok_idx:]
        raw_text = ctx.tokenizer.decode(gen_ids, skip_special_tokens=True)

        trimmed, stop_reason = _trim_and_classify(
            gen_ids,
            raw_text,
            ctx.config.stop_strings,
            ctx.config.cap,
            ctx.config.eos_ids,
        )
        decoded_texts.append(trimmed)
        stop_reasons.append(stop_reason)

        if ctx.config.entropy_mode == "none":
            entropy_series.append([])
            continue

        entropies = _row_entropy_from_scores(
            _EntropyRowContext(
                scores=ctx.scores,
                sequences=ctx.sequences,
                eos_ids=ctx.config.eos_ids,
                model=ctx.config.model,
            ),
            row_index,
            start_tok_idx,
        )
        entropy_series.append(entropies)

    return decoded_texts, entropy_series, stop_reasons


@dataclass
class GenerateBatchParams:
    """
    Static parameters for a single generate-and-decode call.

    :param prefixes: String prefixes to feed into the model.
    :param cap: Maximum number of new tokens to generate per prefix.
    :param stop_strings: Stop substrings that terminate generation.
    :param config_like: Object exposing ``eos_ids``, ``entropy_mode``, ``temperature``, and ``top_p``.
    :param max_length: Maximum tokenized length for inputs.
    """

    prefixes: Sequence[str]
    cap: int
    stop_strings: Sequence[str]
    config_like: Any
    max_length: int


@dataclass
class GenerateBatchRuntime:
    """
    Runtime dependencies required to run generation.

    :param tokenizer: Tokenizer used for encoding and decoding.
    :param model: Model instance exposing a ``generate`` method.
    :param torch_module: Imported :mod:`torch` module.
    :param stopping_criteria_list_cls: Class used to construct stopping criteria lists.
    """

    tokenizer: Any
    model: Any
    torch_module: Any
    stopping_criteria_list_cls: Any


def run_generate_batch(
    prefixes: Sequence[str],
    cap: int,
    stop_strings: Sequence[str],
    *,
    tokenizer: Any,
    model: Any,
    config_like: Any,
    max_length: int,
    torch_module: Any,
    stopping_criteria_list_cls: Any,
) -> Tuple[List[str], List[List[float]], Any, Any, List[str]]:
    """
    Convenience wrapper that constructs :class:`GenerateBatchParams` and
    :class:`GenerateBatchRuntime` and delegates to :func:`generate_and_score_batch`.

    :param prefixes: String prefixes to feed into the model.
    :param cap: Maximum number of new tokens to generate per prefix.
    :param stop_strings: Stop substrings that terminate generation.
    :param tokenizer: Tokenizer used for encoding and decoding.
    :param model: Model instance exposing a ``generate`` method.
    :param config_like: Object exposing ``eos_ids``, ``entropy_mode``, ``temperature``, and ``top_p``.
    :param max_length: Maximum tokenized length for inputs.
    :param torch_module: Imported :mod:`torch` module.
    :param stopping_criteria_list_cls: Class used to construct stopping criteria lists.
    :returns: Tuple ``(decoded_texts, entropy_series, input_lengths, sequences, stop_reasons)``.
    """
    params = GenerateBatchParams(
        prefixes=prefixes,
        cap=cap,
        stop_strings=stop_strings,
        config_like=config_like,
        max_length=max_length,
    )
    runtime = GenerateBatchRuntime(
        tokenizer=tokenizer,
        model=model,
        torch_module=torch_module,
        stopping_criteria_list_cls=stopping_criteria_list_cls,
    )
    return generate_and_score_batch(params, runtime)


def generate_and_score_batch(
    params: GenerateBatchParams,
    runtime: GenerateBatchRuntime,
) -> Tuple[List[str], List[List[float]], Any, Any, List[str]]:
    """
    Shared generate + decode loop used by math/carpark/crossword cores.

    ``params.config_like`` must expose ``eos_ids``, ``entropy_mode``,
    and ``top_p`` attributes.

    :param params: Static parameters describing prefixes, caps, and config-like object.
    :param runtime: Runtime dependencies including tokenizer, model, and torch module.
    :returns: Tuple ``(decoded_texts, entropy_series, input_lengths, sequences, stop_reasons)``.
    """
    inputs, input_lengths = tokenize_prefixes_for_generate(
        runtime.tokenizer,
        params.prefixes,
        max_length=params.max_length,
    )

    stop = (
        runtime.stopping_criteria_list_cls(
            [StopOnSubstrings(runtime.tokenizer, list(params.stop_strings))],
        )
        if params.stop_strings
        else None
    )

    config_like = params.config_like
    eos_ids = getattr(config_like, "eos_ids", None)
    entropy_mode = getattr(config_like, "entropy_mode")
    temperature = getattr(config_like, "temperature")
    top_p = getattr(config_like, "top_p")

    gen_kwargs = make_generate_kwargs_for_cap(
        cap=params.cap,
        tokenizer=runtime.tokenizer,
        eos_ids=eos_ids,
        entropy_mode=entropy_mode,
        temperature=temperature,
        top_p=top_p,
    )

    with runtime.torch_module.inference_mode():
        out = runtime.model.generate(**inputs, **gen_kwargs, stopping_criteria=stop)

    decoded_texts, entropy_series, stop_reasons = decode_and_score_batch(
        DecodeBatchContext(
            tokenizer=runtime.tokenizer,
            sequences=out.sequences,
            scores=out.scores,
            input_lengths=input_lengths,
            config=DecodeBatchConfig(
                stop_strings=list(params.stop_strings) or [],
                cap=params.cap,
                eos_ids=eos_ids,
                entropy_mode=entropy_mode,
                model=runtime.model,
            ),
        ),
    )

    return decoded_texts, entropy_series, input_lengths, out.sequences, stop_reasons


def classify_stop_reason(found_stop: bool, has_eos: bool, hit_max: bool) -> str:
    """
    Map boolean stop conditions into a standardized ``stop_reason`` string.

    :param found_stop: Whether a configured stop substring was observed.
    :param has_eos: Whether an EOS token was observed in the generated IDs.
    :param hit_max: Whether generation hit the maximum new-token cap.
    :returns: One of ``\"stop_token\"``, ``\"eos\"``, ``\"max_new_tokens\"``, or ``\"other\"``.
    """
    if found_stop:
        return "stop_token"
    if has_eos:
        return "eos"
    if hit_max:
        return "max_new_tokens"
    return "other"


class StopOnSubstrings(StoppingCriteria):
    """Stop generation when any of the provided substrings is seen."""

    def __init__(self, tokenizer, stops: List[str]):
        self.stop_ids = [
            tokenizer.encode(text, add_special_tokens=False) for text in stops
        ]

    @staticmethod
    def _endswith(sequence: torch.Tensor, suffix_ids: List[int]) -> bool:
        return (
            len(sequence) >= len(suffix_ids)
            and sequence[-len(suffix_ids) :].tolist() == suffix_ids
        )

    def __call__(
        self,
        input_ids: torch.LongTensor,
        _: torch.FloatTensor,
        **__: Any,
    ) -> bool:
        for row in input_ids:
            for stop_id_seq in self.stop_ids:
                if stop_id_seq and self._endswith(row, stop_id_seq):
                    return True
        return False

    def has_stops(self) -> bool:
        """
        Indicate whether any stop sequences are configured.

        :returns: ``True`` if at least one stop sequence is present, otherwise ``False``.
        """
        return bool(self.stop_ids)


def first_eos_any(token_ids: torch.Tensor, eos_id_list: Optional[Sequence[int]]) -> int:
    """
    Return the first EOS position in a sequence, or full length if absent.

    :param token_ids: 1D tensor of token IDs for a single sequence.
    :param eos_id_list: Optional sequence of EOS token IDs to search for.
    :returns: Index of the first EOS token, or ``len(token_ids)`` if none is found.
    """
    if not eos_id_list:
        return token_ids.numel()
    hit_positions: List[int] = []
    for eos_token_id in eos_id_list:
        positions = (token_ids == eos_token_id).nonzero(as_tuple=False)
        if positions.numel() > 0:
            hit_positions.append(positions[0].item())
    return min(hit_positions) if hit_positions else token_ids.numel()


def entropy_from_start_index(model, seq_ids: torch.Tensor, start_idx: int) -> List[float]:
    """
    Compute token-wise entropy starting at position start_idx (inclusive).
    Safe for NaNs thanks to re-centering.

    :param model: Autoregressive language model exposing ``forward`` with ``use_cache``.
    :param seq_ids: 2D tensor of token IDs for at least one sequence.
    :param start_idx: Index of the first token at which to begin entropy computation.
    :returns: List of entropy values, one per token from ``start_idx`` onward.
    """
    device = next(model.parameters()).device
    seq_ids = seq_ids.to(device)
    entropies: List[float] = []
    with torch.inference_mode():
        out = model(input_ids=seq_ids[:, : start_idx + 1], use_cache=True)
        past_key_values = out.past_key_values
        sequence_length = seq_ids.shape[1]
        for time_index in range(start_idx, sequence_length - 1):
            out = model(
                input_ids=seq_ids[:, time_index : time_index + 1],
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = out.past_key_values
            logits = out.logits[:, -1, :].float()
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            probabilities = log_probs.exp()
            entropy_value = float(-(probabilities * log_probs).sum().item())
            if not math.isfinite(entropy_value):
                logits = (logits - logits.max()).float()
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                probabilities = log_probs.exp()
                entropy_value = float(-(probabilities * log_probs).sum().item())
            entropies.append(entropy_value)
    return entropies


@dataclass
class SamplingConfigBase:
    """
    Shared sampling defaults used by multiple inference configs.

    This centralizes common fields so that task-specific configs (math,
    crossword, etc.) can extend it without duplicating dataclass bodies.
    """

    temperature: float = 0.0
    top_p: float = 0.95
    entropy_mode: str = "reconsider"
    eos_ids: Optional[List[int]] = None


def build_math_inference_config_kwargs(
    *,
    batch_size: int,
    num_samples: int,
    temperature: float,
    top_p: float,
    entropy_mode: str,
    eos_ids,
    two_pass: bool,
    second_pass_phrase: str,
    second_pass_use_sample_idx: int,
    think_cap: int,
    answer_cap: int,
) -> Dict[str, Any]:
    """
    Build the common kwargs dict for math-style inference configs / loops.

    :param batch_size: Number of examples to process per batch.
    :param num_samples: Number of samples to generate per problem.
    :param temperature: Sampling temperature for generation.
    :param top_p: Nucleus-sampling parameter for generation.
    :param entropy_mode: Entropy computation mode (for example, ``\"reconsider\"``).
    :param eos_ids: EOS token ID or IDs used to terminate generation.
    :param two_pass: Whether to enable the reconsideration second pass.
    :param second_pass_phrase: Cue phrase injected into second-pass prompts.
    :param second_pass_use_sample_idx: Preferred sample index to show in pass 2.
    :param think_cap: Token cap for ``<think>`` generations.
    :param answer_cap: Token cap for ``<answer>`` generations.
    :returns: Keyword-argument dictionary suitable for :class:`MathInferenceConfig`.
    """
    return {
        "batch_size": batch_size,
        "num_samples": num_samples,
        "temperature": temperature,
        "top_p": top_p,
        "entropy_mode": entropy_mode,
        "eos_ids": eos_ids,
        "two_pass": two_pass,
        "second_pass_phrase": second_pass_phrase,
        "second_pass_use_sample_idx": second_pass_use_sample_idx,
        "think_cap": think_cap,
        "answer_cap": answer_cap,
    }


def build_math_inference_config_kwargs_from_args(args, eos_ids):
    """
    Map common CLI args into kwargs for math-style inference loops and configs.

    :param args: Argument namespace exposing common math-inference options.
    :param eos_ids: EOS token ID or IDs derived from the tokenizer.
    :returns: Keyword-argument dictionary suitable for :class:`MathInferenceConfig`.
    """
    return build_math_inference_config_kwargs(
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        entropy_mode=args.entropy_mode,
        eos_ids=eos_ids,
        two_pass=args.two_pass,
        second_pass_phrase=args.second_pass_phrase,
        second_pass_use_sample_idx=args.second_pass_use_sample_idx,
        think_cap=args.think_cap,
        answer_cap=args.answer_cap,
    )


def require_torch(caller: str):
    """
    Import and return the ``torch`` module.

    A user-friendly :class:`RuntimeError` is raised if the dependency is missing.

    :param caller: Human-readable name of the caller used in error messages.
    :returns: Imported :mod:`torch` module.
    :raises RuntimeError: If the ``torch`` package is not available.
    """
    try:
        return import_module("torch")
    except ImportError as exc:  # pragma: no cover - hard dependency
        raise RuntimeError(
            f"{caller} requires 'torch'; install the 'torch' package.",
        ) from exc


def require_transformers(caller: str):
    """
    Import and return the ``transformers`` module.

    A user-friendly :class:`RuntimeError` is raised if the dependency is missing.

    :param caller: Human-readable name of the caller used in error messages.
    :returns: Imported :mod:`transformers` module.
    :raises RuntimeError: If the ``transformers`` package is not available.
    """
    try:
        return import_module("transformers")
    except ImportError as exc:  # pragma: no cover - hard dependency
        raise RuntimeError(
            f"{caller} requires 'transformers'; install it to use this script.",
        ) from exc
