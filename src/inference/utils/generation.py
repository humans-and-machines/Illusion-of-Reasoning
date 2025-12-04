#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Torch/generation helpers shared across inference tasks.

These were split out from ``common`` to keep that module smaller while
preserving the public API via re-exports.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, replace
from importlib import import_module
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.inference.utils._torch_stubs import StoppingCriteriaStub, TorchStub  # type: ignore[import-untyped]
from src.inference.utils.gateway_utils import PassOutputs


if TYPE_CHECKING:
    import torch
    from transformers import StoppingCriteria
else:  # pragma: no cover - optional dependency at runtime
    try:
        torch = import_module("torch")
    except ImportError:  # pragma: no cover - type-check / lint environments
        torch = TorchStub()

    try:
        transformers_mod = import_module("transformers")
        StoppingCriteria = getattr(transformers_mod, "StoppingCriteria")
    except ImportError:  # pragma: no cover - type-check / lint environments
        StoppingCriteria = StoppingCriteriaStub


__all__ = [
    "move_inputs_to_device",
    "tokenize_prefixes_for_generate",
    "build_generate_kwargs",
    "make_generate_kwargs_for_cap",
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
]


# ---------------------------------------------------------------------------
# Lightweight config containers to keep public signatures short
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GenerateKwargsConfig:
    """Container for generate() kwargs to avoid long function signatures."""

    cap: int
    pad_token_id: int
    eos_ids: Any
    entropy_mode: str
    temperature: Optional[float]
    top_p: Optional[float]
    synced_gpus: bool = False

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "GenerateKwargsConfig":
        """Build a :class:`GenerateKwargsConfig` from flat kwargs."""
        required = ["cap", "pad_token_id", "entropy_mode"]
        missing = [name for name in required if name not in kwargs]
        if missing:
            raise KeyError(f"Missing required generate fields: {missing}")
        return cls(
            cap=kwargs["cap"],
            pad_token_id=kwargs["pad_token_id"],
            eos_ids=kwargs.get("eos_ids"),
            entropy_mode=kwargs["entropy_mode"],
            temperature=kwargs.get("temperature"),
            top_p=kwargs.get("top_p"),
            synced_gpus=bool(kwargs.get("synced_gpus", False)),
        )


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
    config: GenerateKwargsConfig | None = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Build generate() kwargs for a given token cap and sampling configuration.

    If temperature <= 0 → greedy (do_sample=False) and omit temperature/top_p.
    Else → sampling with provided temperature/top_p.

    :param config: Optional :class:`GenerateKwargsConfig` container. When
        omitted, ``**kwargs`` is used to build one for backwards compatibility.
    :param kwargs: Legacy flat keyword arguments (cap, pad_token_id, etc.).
    :returns: Dictionary of keyword arguments suitable for ``model.generate``.
    """
    cfg = config or GenerateKwargsConfig.from_kwargs(**kwargs)
    do_sample = cfg.temperature is not None and float(cfg.temperature) > 0.0
    kwargs: Dict[str, Any] = {
        "max_new_tokens": cfg.cap,
        "pad_token_id": cfg.pad_token_id,
        "eos_token_id": cfg.eos_ids,
        "do_sample": do_sample,
        "return_dict_in_generate": True,
        "output_scores": cfg.entropy_mode != "none",
        "num_return_sequences": 1,
    }
    if cfg.synced_gpus and hasattr(torch, "distributed"):
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                kwargs["synced_gpus"] = True
        except (RuntimeError, AttributeError):  # pragma: no cover - defensive
            # If distributed is misconfigured, fall back to unsynced generate.
            pass
    if do_sample:
        kwargs["temperature"] = float(cfg.temperature)
        if cfg.top_p is not None:
            kwargs["top_p"] = float(cfg.top_p)
    return kwargs


def make_generate_kwargs_for_cap(
    tokenizer,
    config: GenerateKwargsConfig | None = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Convenience wrapper that derives ``pad_token_id`` from a tokenizer and
    forwards to :func:`build_generate_kwargs`.

    :param tokenizer: Tokenizer providing ``pad_token_id`` and ``eos_token_id``.
    :param config: Optional :class:`GenerateKwargsConfig` container. When
        omitted, ``**kwargs`` is used for backwards compatibility.
    :param kwargs: Legacy flat keyword arguments (cap, eos_ids, entropy_mode, etc.).
    :returns: Dictionary of keyword arguments suitable for ``model.generate``.
    """
    pad_token_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id")
    cfg_kwargs = dict(kwargs)
    cfg_kwargs.setdefault("pad_token_id", pad_token_id)
    cfg = config or GenerateKwargsConfig.from_kwargs(**cfg_kwargs)
    if cfg.pad_token_id is None:
        cfg = replace(cfg, pad_token_id=pad_token_id)
    return build_generate_kwargs(config=cfg)


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
    extra_passes: Optional[Sequence[Tuple[str, PassOutputs]]],
    pack_result_for_extra: Callable[[str, PassOutputs], Dict[str, Any]],
    names: Sequence[str] = ("pass2a", "pass2b", "pass2c"),
) -> Dict[str, Dict[str, Any]]:
    """
    Build result dicts for optional multi-cue reconsideration passes
    (pass2a / pass2b / pass2c) given per-cue PassOutputs.

    :param two_pass: Whether second-pass generation is enabled.
    :param extra_passes: Optional sequence of ``(cue_str, PassOutputs)`` pairs.
    :param pack_result_for_extra: Callback that converts a cue and outputs into a result dict.
    :param names: Keys to use for extra passes (for example, ``("pass2a", "pass2b")``).
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


def empty_pass_outputs(total_rows: int) -> PassOutputs:
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


def decode_generated_row(
    tokenizer,
    seqs: torch.Tensor,
    input_lengths: torch.Tensor,
    row_i: int,
    *,
    skip_special_tokens: bool = True,
) -> Tuple[torch.Tensor, str, int]:
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
        token_entropies = (
            entropy_from_start_index(
                ctx.model,
                ctx.sequences[row_index : row_index + 1],
                start_index,
            )
            or []
        )
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
    :param config_like: Object exposing ``eos_ids``, ``entropy_mode``,
        ``temperature``, and ``top_p``.
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
    params: GenerateBatchParams | None = None,
    runtime: GenerateBatchRuntime | None = None,
    **kwargs: Any,
) -> Tuple[List[str], List[List[float]], Any, Any, List[str]]:
    """
    Convenience wrapper that constructs :class:`GenerateBatchParams` and
    :class:`GenerateBatchRuntime` and delegates to :func:`generate_and_score_batch`.

    :param params: Optional pre-built :class:`GenerateBatchParams`. When
        omitted, ``**kwargs`` is consumed for backwards compatibility with
        legacy keyword arguments (prefixes, cap, stop_strings, config_like,
        max_length).
    :param runtime: Optional :class:`GenerateBatchRuntime`. When omitted,
        ``**kwargs`` is consumed for backwards compatibility with legacy
        keyword arguments (tokenizer, model, torch_module,
        stopping_criteria_list_cls).
    :param kwargs: Backwards-compatible flat keyword arguments.
    :returns: Tuple ``(decoded_texts, entropy_series, input_lengths,
        sequences, stop_reasons)``.
    """
    if params is None:
        params = GenerateBatchParams(
            prefixes=kwargs.pop("prefixes"),
            cap=kwargs.pop("cap"),
            stop_strings=kwargs.pop("stop_strings"),
            config_like=kwargs.pop("config_like"),
            max_length=kwargs.pop("max_length"),
        )
    if runtime is None:
        runtime = GenerateBatchRuntime(
            tokenizer=kwargs.pop("tokenizer"),
            model=kwargs.pop("model"),
            torch_module=kwargs.pop("torch_module"),
            stopping_criteria_list_cls=kwargs.pop("stopping_criteria_list_cls"),
        )
    if kwargs:
        raise TypeError(f"run_generate_batch() got unexpected keyword arguments: {sorted(kwargs)}")
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

    config_fields = {
        "eos_ids": getattr(params.config_like, "eos_ids", None),
        "entropy_mode": getattr(params.config_like, "entropy_mode"),
        "temperature": getattr(params.config_like, "temperature"),
        "top_p": getattr(params.config_like, "top_p"),
    }

    gen_kwargs = make_generate_kwargs_for_cap(
        cap=params.cap,
        tokenizer=runtime.tokenizer,
        eos_ids=config_fields["eos_ids"],
        entropy_mode=config_fields["entropy_mode"],
        temperature=config_fields["temperature"],
        top_p=config_fields["top_p"],
    )

    inference_mode_fn = getattr(runtime.torch_module, "inference_mode", None)
    try:
        cm_inference = inference_mode_fn() if callable(inference_mode_fn) else contextlib.nullcontext()
    except TypeError:
        cm_inference = contextlib.nullcontext()
    if not hasattr(cm_inference, "__enter__"):
        cm_inference = contextlib.nullcontext()

    with cm_inference:
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
                eos_ids=config_fields["eos_ids"],
                entropy_mode=config_fields["entropy_mode"],
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
    :returns: One of ``"stop_token"``, ``"eos"``, ``"max_new_tokens"``, or ``"other"``.
    """
    if found_stop:
        return "stop_token"
    if has_eos:
        return "eos"
    if hit_max:
        return "max_new_tokens"
    return "other"


class StopOnSubstrings(StoppingCriteria):
    """
    Stop generation when any of the provided substrings is seen.

    The class encodes stop substrings into token ID sequences for efficient
    suffix matching on generated token IDs.
    """

    def __init__(self, tokenizer, stops: List[str]):
        """
        Initialize the stopping criterion with a tokenizer and stop substrings.

        :param tokenizer: Tokenizer providing an ``encode`` method (or equivalent).
        :param stops: List of stop substrings to monitor in generated text.
        """
        try:
            super().__init__()
        except TypeError:
            # Some transformer stubs omit __init__; ignore to remain compatible.
            pass
        self.stop_ids: List[List[int]] = []
        if not stops:
            return

        # Some test tokenizers only implement ``__call__`` and ``decode`` but
        # lack an explicit ``encode`` method. In that case we gracefully
        # disable substring-based stopping rather than raising an error.
        encode_fn = getattr(tokenizer, "encode", None)
        if encode_fn is None:
            return

        for text in stops:
            try:
                ids = encode_fn(text, add_special_tokens=False)
            except TypeError:
                # Fallback when encode signature differs; best-effort only.
                ids = encode_fn(text)
            if isinstance(ids, list):
                self.stop_ids.append(ids)

    @staticmethod
    def _endswith(sequence: torch.Tensor, suffix_ids: List[int]) -> bool:
        """
        Check whether a token sequence ends with a given suffix.

        :param sequence: Tensor of token IDs.
        :param suffix_ids: List of token IDs representing the suffix.
        :returns: ``True`` if ``sequence`` ends with ``suffix_ids``.
        """
        return len(sequence) >= len(suffix_ids) and sequence[-len(suffix_ids) :].tolist() == suffix_ids

    def __call__(
        self,
        input_ids: torch.LongTensor,
        _: torch.FloatTensor,
        **__: Any,
    ) -> bool:
        """
        Evaluate the stopping criterion on generated token IDs.

        :param input_ids: Batched tensor of token IDs produced by the model.
        :param _: Logits tensor (unused but required by the interface).
        :param __: Additional keyword arguments ignored by this criterion.
        :returns: ``True`` if any row ends with one of the stop sequences.
        """
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
