#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-pass batch inference for Llama-8B-style chat LMs that produce:
  <think> ... </think><answer> ... </answer>

DeepSpeed (ZeRO-3) edition:
- Loads ZeRO-3 shards directly from checkpoint-XXX/global_stepXXXX (no merge).
- Uses deepspeed.initialize + engine.load_checkpoint.
- Generation goes through engine.module.generate(...).

Resume/fill:
- If step####_<split>.jsonl exists, append only missing samples per problem
  up to --num_samples (preserves prior rows).

Other behaviors preserved:
- Two-phase per pass (think -> answer) with explicit `</think>` / `</answer>` stops.
- Per-pass caps: think <= 750 new tokens, answer <= 50.
- Pass 2 injects a cue at start of <think>, but that cue is excluded from
  reconsider-cue analytics.
- EOS set includes Llama-3 tokens <|eot_id|>, <|end_of_text|>.
- NaN-safe token entropy; optional "reconsider" slicing.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

from src.inference.common import (
    MathPassMeta as PassMeta,
    StopOnSubstrings,
    build_math_inference_config_kwargs_from_args,
    build_two_pass_row_base,
    canon_math as _canon_math,
    classify_stop_reason,
    configure_unified_runner_common as _configure_unified_runner_common,
    decode_generated_row as _decode_generated_row,
    entropy_from_start_index as _entropy_from_start_index,
    first_eos_any as _first_eos_any,
    move_inputs_to_device,
    pack_math_pass_result as _pack_pass_result,
    require_datasets,
    scan_existing_pass1_results,
    setup_script_logger,
)
from src.inference.math_core import (
    chat_base_for_pass1,
    chat_base_for_pass2,
    load_math500 as _load_math500_core,
)
from src.inference.math_llama_utils import DSModelWrapper, GeneratedBatchView
from src.inference.task_registry import MATH_SYSTEM_PROMPT

from packaging import version

try:
    torch = importlib.import_module("torch")
except ImportError as exc:  # pragma: no cover - hard dependency
    raise RuntimeError(
        "math_llama_core requires PyTorch; install the 'torch' package."
    ) from exc

F = torch.nn.functional

try:
    transformers_mod = importlib.import_module("transformers")
except ImportError as exc:  # pragma: no cover - hard dependency
    raise RuntimeError(
        "math_llama_core requires 'transformers'; install it to use this script."
    ) from exc

AutoConfig = transformers_mod.AutoConfig
AutoTokenizer = transformers_mod.AutoTokenizer
AutoModelForCausalLM = transformers_mod.AutoModelForCausalLM
StoppingCriteria = transformers_mod.StoppingCriteria
StoppingCriteriaList = transformers_mod.StoppingCriteriaList

try:
    deepspeed = importlib.import_module("deepspeed")
except ImportError as exc:  # pragma: no cover - hard dependency
    raise RuntimeError(
        "math_llama_core requires 'deepspeed'; install it to use this script."
    ) from exc

_, load_dataset = require_datasets()

# ───────────────────────── System prompt ─────────────────────────
SYSTEM_PROMPT = MATH_SYSTEM_PROMPT

# ───────────────────────── Utilities ─────────────────────────

# ───────────────────────── Logging ─────────────────────────
logger = setup_script_logger(__name__)

# ───── PyTorch 2.6 DeepSpeed unpickle patch (no-op if absent) ─────
try:
    if version.parse(torch.__version__) >= version.parse("2.6.0"):
        torch_serialization = importlib.import_module("torch.serialization")
        zero_config = importlib.import_module("deepspeed.runtime.zero.config")
        loss_scaler_mod = importlib.import_module(
            "deepspeed.runtime.fp16.loss_scaler"
        )
        add_safe_globals = getattr(torch_serialization, "add_safe_globals")
        zero_stage_enum_cls = getattr(zero_config, "ZeroStageEnum")
        loss_scaler_cls = getattr(loss_scaler_mod, "LossScaler")
        add_safe_globals([zero_stage_enum_cls, loss_scaler_cls])
        logger.info("DeepSpeed ZeRO patch enabled")
except (ImportError, AttributeError) as exc:
    logger.warning("DeepSpeed patch disabled (missing deps/attrs): %r", exc)

def _compute_row_entropies(
    row_index: int,
    start_token_index: int,
    generated_ids: torch.Tensor,
    eos_token_ids: Optional[List[int]],
    batch: GeneratedBatchView,
) -> List[float]:
    """Compute per-token entropies for a single generated row."""
    if not batch.scores:
        return []

    scores_length = len(batch.scores)
    if eos_token_ids:
        first_eos_index = _first_eos_any(generated_ids, eos_token_ids)
    else:
        first_eos_index = generated_ids.shape[0]
    token_limit = min(first_eos_index, scores_length)

    entropies: List[float] = []
    has_bad_values = False

    for score_index in range(token_limit):
        logits = batch.scores[score_index][row_index].float()
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            has_bad_values = True
            break
        log_probs = F.log_softmax(logits, dim=-1)
        if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
            has_bad_values = True
            break
        probs = log_probs.exp()
        entropy_value = float(-(probs * log_probs).sum().item())
        if not math.isfinite(entropy_value):
            has_bad_values = True
            break
        entropies.append(entropy_value)

    if has_bad_values or not entropies:
        start_index = start_token_index - 1
        entropies = _entropy_from_start_index(
            batch.model,
            batch.sequences[row_index : row_index + 1],
            start_index,
        ) or []

    return entropies


@dataclass
class GenerationContext:
    """Context bundle for processing a single generated row."""

    input_lengths: torch.Tensor
    sequences: torch.Tensor
    tokenizer: AutoTokenizer
    stop_strings: Optional[List[str]]
    cap: int
    eos_token_ids: Optional[List[int]]
    scores: Any
    model: Any
    entropy_mode: str


@dataclass
class BatchGenerationConfig:
    """Configuration shared across generation batches within a split."""

    tokenizer: AutoTokenizer
    model: Any
    eos_ids: Optional[List[int]]
    entropy_mode: str
    temperature: float
    top_p: Optional[float]


@dataclass
class FirstPassOutputs:
    """Pack first-pass think/answer texts and entropy/stop metadata."""

    full_rows: List[str]
    think_entropies: List[List[float]]
    think_stop_reasons: List[str]
    answer_entropies: List[List[float]]
    answer_stop_reasons: List[str]


@dataclass
class SecondPassOutputs:
    """Pack second-pass think/answer texts and entropy/stop metadata."""

    full_rows: List[str]
    think_entropies: List[List[float]]
    think_stop_reasons: List[str]
    answer_entropies: List[List[float]]
    answer_stop_reasons: List[str]


def _process_generated_row(
    row_index: int,
    context: GenerationContext,
) -> Tuple[str, List[float], str]:
    """Decode, classify stop reason, and compute entropies for a generated row."""
    generated_ids, raw_text, start_token_index = _decode_generated_row(
        context.tokenizer,
        context.sequences,
        context.input_lengths,
        row_index,
        skip_special_tokens=True,
    )

    active_stop_strings = context.stop_strings or []
    found_stop = any(
        stop_string in raw_text for stop_string in active_stop_strings
    )

    has_eos = False
    if context.eos_token_ids:
        for eos_id in context.eos_token_ids:
            if (generated_ids == eos_id).any():
                has_eos = True
                break

    hit_max_tokens = len(generated_ids) >= context.cap
    stop_reason = classify_stop_reason(found_stop, has_eos, hit_max_tokens)

    trimmed_text = raw_text
    for stop_string in active_stop_strings:
        if stop_string in trimmed_text:
            trimmed_text = trimmed_text.split(stop_string, 1)[0]
            break
    decoded_text = trimmed_text.strip()

    if context.entropy_mode == "none":
        entropy_series: List[float] = []
    else:
        batch_view = GeneratedBatchView(
            sequences=context.sequences,
            scores=context.scores,
            model=context.model,
        )
        entropy_series = _compute_row_entropies(
            row_index=row_index,
            start_token_index=start_token_index,
            generated_ids=generated_ids,
            eos_token_ids=context.eos_token_ids,
            batch=batch_view,
        )

    return decoded_text, entropy_series, stop_reason


def _make_gen_kwargs(
    cap: int,
    config: BatchGenerationConfig,
) -> Dict[str, Any]:
    """Build generate() kwargs for a given token cap and sampling configuration."""
    do_sample = (config.temperature is not None) and (float(config.temperature) > 0.0)
    pad_token_id = config.tokenizer.pad_token_id or config.tokenizer.eos_token_id
    kwargs: Dict[str, Any] = {
        "max_new_tokens": cap,
        "pad_token_id": pad_token_id,
        "eos_token_id": config.eos_ids,
        "do_sample": do_sample,
        "return_dict_in_generate": True,
        "output_scores": config.entropy_mode != "none",
        "num_return_sequences": 1,
    }
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        kwargs["synced_gpus"] = True
    if do_sample:
        kwargs["temperature"] = float(config.temperature)
        if config.top_p is not None:
            kwargs["top_p"] = float(config.top_p)
    return kwargs


def _model_device(model: Any) -> torch.device:
    """Infer the primary device from the model or its DeepSpeed engine."""
    device = getattr(model, "device", None)
    if isinstance(device, torch.device):
        return device
    if hasattr(model, "module"):
        module_device = getattr(model.module, "device", None)
        if isinstance(module_device, torch.device):
            return module_device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _gen_batch(
    prefixes: Sequence[str],
    cap: int,
    stop_strs: Sequence[str],
    config: BatchGenerationConfig,
) -> Tuple[List[str], List[List[float]], torch.Tensor, torch.Tensor, List[str]]:
    """Generate a batch of continuations and collect entropies + stop reasons."""
    inputs = config.tokenizer(
        list(prefixes),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    )
    inputs, input_lengths = move_inputs_to_device(
        inputs,
        device=_model_device(config.model),
    )

    stop = (
        StoppingCriteriaList([StopOnSubstrings(config.tokenizer, list(stop_strs))])
        if stop_strs
        else None
    )
    gen_kwargs = _make_gen_kwargs(cap, config)
    with torch.inference_mode():
        out = config.model.generate(
            **inputs,
            **gen_kwargs,
            stopping_criteria=stop,
        )

    decoded_texts: List[str] = []
    entropy_series: List[List[float]] = []
    stop_reasons: List[str] = []

    for row_index in range(out.sequences.shape[0]):
        decoded_text, row_entropies, stop_reason = _process_generated_row(
            row_index=row_index,
            context=GenerationContext(
                input_lengths=input_lengths,
                sequences=out.sequences,
                tokenizer=config.tokenizer,
                stop_strings=list(stop_strs),
                cap=cap,
                eos_token_ids=config.eos_ids,
                scores=out.scores,
                model=config.model,
                entropy_mode=config.entropy_mode,
            ),
        )
        decoded_texts.append(decoded_text)
        entropy_series.append(row_entropies)
        stop_reasons.append(stop_reason)

    return decoded_texts, entropy_series, input_lengths, out.sequences, stop_reasons


def _select_first_pass_choice(
    *,
    prob: str,
    ex_idx: int,
    existing_samples: DefaultDict[str, set],
    existing_pass1: Dict[tuple, str],
    new_pass1_by_ex_and_k: Dict[tuple, str],
    pass1_full_rows: List[str],
    row_to_ex_idx: List[int],
    num_samples: int,
    second_pass_use_sample_idx: int,
) -> str:
    """Choose which pass-1 output to show in the pass-2 cue."""
    k_choice = max(0, min(second_pass_use_sample_idx, num_samples - 1))
    prev = existing_pass1.get((prob, k_choice))
    if prev is not None:
        return prev

    prev = new_pass1_by_ex_and_k.get((ex_idx, k_choice))
    if prev is not None:
        return prev

    have_sorted = sorted(existing_samples.get(prob, set()))
    if have_sorted:
        prev = existing_pass1.get((prob, have_sorted[0]))
        if prev is not None:
            return prev

    new_ks = sorted(
        k
        for (eidx, k) in new_pass1_by_ex_and_k
        if eidx == ex_idx
    )
    if new_ks:
        prev = new_pass1_by_ex_and_k.get((ex_idx, new_ks[0]))
        if prev is not None:
            return prev

    for row_index, full_row in enumerate(pass1_full_rows):
        if row_to_ex_idx[row_index] == ex_idx:
            return full_row

    return ""


def _build_work_items_for_batch(
    examples: Iterable[Dict[str, Any]],
    existing_samples: DefaultDict[str, set],
    num_samples: int,
) -> List[Dict[str, Any]]:
    """Normalize raw examples and determine which samples still need inference."""
    work_items: List[Dict[str, Any]] = []
    for raw_example in examples:
        prob = (
            raw_example.get("problem")
            or raw_example.get("question")
            or raw_example.get("query")
            or raw_example.get("prompt")
            or raw_example.get("instruction")
        )
        gold = (
            raw_example.get("answer")
            or raw_example.get("final_answer")
            or raw_example.get("target")
            or raw_example.get("boxed_answer")
            or raw_example.get("solution")
        )
        if gold and not any(
            key in raw_example
            for key in ("answer", "final_answer", "target", "boxed_answer", "solution")
        ):
            boxed_match = (
                re.search(r"\\boxed\{([^}]*)\}", str(gold))
                or re.search(r"\\boxed\(([^)]*)\)", str(gold))
            )
            if boxed_match:
                gold = boxed_match.group(1)
        if not prob:
            continue
        have = existing_samples.get(prob, set())
        if len(have) >= num_samples:
            continue
        todo = [sample_idx for sample_idx in range(num_samples) if sample_idx not in have]
        if not todo:
            continue
        example_copy = dict(raw_example)
        example_copy["_normalized_problem"] = prob
        example_copy["_normalized_gold"] = gold
        example_copy["_todo_samples"] = todo
        work_items.append(example_copy)
    return work_items


def _build_first_pass_prefixes(
    tokenizer: AutoTokenizer,
    work_items: Sequence[Dict[str, Any]],
) -> Tuple[List[str], List[int], List[int]]:
    """Construct think-pass prefixes and associated bookkeeping arrays."""
    base1 = [
        chat_base_for_pass1(tokenizer, example["_normalized_problem"])
        for example in work_items
    ]
    pre1_think: List[str] = []
    row_to_ex_idx: List[int] = []
    row_target_sample_idx: List[int] = []
    for ex_idx, example in enumerate(work_items):
        for sample_idx in example["_todo_samples"]:
            pre1_think.append(base1[ex_idx] + "<think>\n")
            row_to_ex_idx.append(ex_idx)
            row_target_sample_idx.append(sample_idx)
    return pre1_think, row_to_ex_idx, row_target_sample_idx


def _run_first_pass(
    *,
    config: BatchGenerationConfig,
    think_cap: int,
    answer_cap: int,
    pre1_think: List[str],
) -> FirstPassOutputs:
    """Run pass-1 think and answer generations for a batch of prefixes."""
    think1_texts, think1_ents, _, _, think1_stop = _gen_batch(
        pre1_think,
        think_cap,
        ["</think>"],
        config,
    )

    pre1_answer: List[str] = []
    for row_index, prefix in enumerate(pre1_think):
        pre1_answer.append(prefix + think1_texts[row_index] + "</think>\n<answer>\n")
    answer1_texts, answer1_ents, _, _, answer1_stop = _gen_batch(
        pre1_answer,
        answer_cap,
        ["</answer>"],
        config,
    )
    pass1_full_rows = _build_first_pass_rows(think1_texts, answer1_texts)
    return FirstPassOutputs(
        full_rows=pass1_full_rows,
        think_entropies=think1_ents,
        think_stop_reasons=think1_stop,
        answer_entropies=answer1_ents,
        answer_stop_reasons=answer1_stop,
    )


def _build_first_pass_rows(
    think1_texts: Sequence[str],
    answer1_texts: Sequence[str],
) -> List[str]:
    """Pack first-pass think/answer texts into full rows."""
    return [
        f"<think>{think1_texts[row_index]}</think>\n<answer>{answer1_texts[row_index]}</answer>"
        for row_index in range(len(think1_texts))
    ]


def _prepare_second_pass_inputs(
    *,
    tokenizer: AutoTokenizer,
    work_items: Sequence[Dict[str, Any]],
    existing_samples: DefaultDict[str, set],
    existing_pass1: Dict[tuple, str],
    new_pass1_by_ex_and_k: Dict[tuple, str],
    pass1_full_rows: Sequence[str],
    row_to_ex_idx: Sequence[int],
    num_samples: int,
    second_pass_use_sample_idx: int,
    second_pass_phrase: str,
) -> Tuple[List[str], List[str], str]:
    """Prepare inputs and metadata required for the second pass."""
    cue_str = second_pass_phrase.strip() + " "
    firstpass_choice_text_per_ex: List[str] = []
    for ex_idx, example in enumerate(work_items):
        prob = example["_normalized_problem"]
        prev = _select_first_pass_choice(
            prob=prob,
            ex_idx=ex_idx,
            existing_samples=existing_samples,
            existing_pass1=existing_pass1,
            new_pass1_by_ex_and_k=new_pass1_by_ex_and_k,
            pass1_full_rows=list(pass1_full_rows),
            row_to_ex_idx=list(row_to_ex_idx),
            num_samples=num_samples,
            second_pass_use_sample_idx=second_pass_use_sample_idx,
        )
        firstpass_choice_text_per_ex.append(prev)

    base2_per_ex = [
        chat_base_for_pass2(
            tokenizer,
            example["_normalized_problem"],
            firstpass_choice_text_per_ex[ex_idx],
            second_pass_phrase.strip(),
        )
        for ex_idx, example in enumerate(work_items)
    ]
    return firstpass_choice_text_per_ex, base2_per_ex, cue_str


def _run_second_pass(
    *,
    config: BatchGenerationConfig,
    think_cap: int,
    answer_cap: int,
    pre1_think: Sequence[str],
    row_to_ex_idx: Sequence[int],
    base2_per_ex: Sequence[str],
    cue_str: str,
) -> SecondPassOutputs:
    """Run pass-2 think and answer generations for a batch of prefixes."""
    pre2_think: List[str] = []
    for row_index, _ in enumerate(pre1_think):
        ex_idx = row_to_ex_idx[row_index]
        base2 = base2_per_ex[ex_idx]
        pre2_think.append(base2 + "<think>\n" + cue_str)

    think2_texts_only_new, think2_ents, _, _, think2_stop = _gen_batch(
        pre2_think,
        think_cap,
        ["</think>"],
        config,
    )
    think2_texts = [cue_str + text for text in think2_texts_only_new]

    pre2_answer: List[str] = []
    for row_index, prefix in enumerate(pre2_think):
        pre2_answer.append(prefix + think2_texts_only_new[row_index] + "</think>\n<answer>\n")
    answer2_texts, answer2_ents, _, _, answer2_stop = _gen_batch(
        pre2_answer,
        answer_cap,
        ["</answer>"],
        config,
    )

    pass2_full_rows = [
        f"<think>{think2_texts[row_index]}</think>\n<answer>{answer2_texts[row_index]}</answer>"
        for row_index in range(len(pre2_think))
    ]
    return SecondPassOutputs(
        full_rows=pass2_full_rows,
        think_entropies=think2_ents,
        think_stop_reasons=think2_stop,
        answer_entropies=answer2_ents,
        answer_stop_reasons=answer2_stop,
    )


def _write_results_for_batch(
    *,
    outpath: str,
    work_items: Sequence[Dict[str, Any]],
    row_to_ex_idx: Sequence[int],
    row_target_sample_idx: Sequence[int],
    first_pass: FirstPassOutputs,
    two_pass_enabled: bool,
    cue_str: str,
    firstpass_choice_text_per_ex: Sequence[str],
    second_pass: Optional[SecondPassOutputs],
    step: int,
    split_name: str,
) -> None:
    """Write combined pass-1 and optional pass-2 rows for a batch to disk."""
    with open(outpath, "a", encoding="utf-8") as outfile:
        for row_index, full_row in enumerate(first_pass.full_rows):
            ex_idx = row_to_ex_idx[row_index]
            example = work_items[ex_idx]
            prob = example["_normalized_problem"]
            gold = example["_normalized_gold"]
            canon_gold = _canon_math(gold)
            sample_idx = row_target_sample_idx[row_index]

            meta_p1 = PassMeta(
                problem=prob,
                canon_gold=canon_gold,
                injected_cue=False,
                prev_output=None,
                cue_prefix_str="",
                stop_reason_think=first_pass.think_stop_reasons[row_index],
                stop_reason_answer=first_pass.answer_stop_reasons[row_index],
            )
            pass1_result = _pack_pass_result(
                full_text=full_row,
                ent_think=first_pass.think_entropies[row_index],
                ent_answer=first_pass.answer_entropies[row_index],
                meta=meta_p1,
            )

            second_pass_result: Optional[Dict[str, Any]] = None
            if two_pass_enabled and second_pass is not None:
                meta_p2 = PassMeta(
                    problem=prob,
                    canon_gold=canon_gold,
                    injected_cue=True,
                    prev_output=firstpass_choice_text_per_ex[ex_idx],
                    cue_prefix_str=cue_str,
                    stop_reason_think=second_pass.think_stop_reasons[row_index],
                    stop_reason_answer=second_pass.answer_stop_reasons[row_index],
                )
                second_pass_result = _pack_pass_result(
                    full_text=second_pass.full_rows[row_index],
                    ent_think=second_pass.think_entropies[row_index],
                    ent_answer=second_pass.answer_entropies[row_index],
                    meta=meta_p2,
                )
                second_pass_result["improved_over_pass1"] = bool(
                    second_pass_result.get("is_correct_pred"),
                ) and not bool(pass1_result.get("is_correct_pred"))

            row = {
                "problem": prob,
                "gold_answer": gold,
                "gold_answer_canon": canon_gold,
                **build_two_pass_row_base(
                    step=step,
                    split_name=split_name,
                    sample_idx=sample_idx,
                    pass1=pass1_result,
                    pass2=second_pass_result,
                ),
            }
            json.dump(row, outfile, ensure_ascii=False)
            outfile.write("\n")


# ───────────────────── Inference Loop (two-phase per pass) ─────────────────────
def run_inference_on_split(
    split_name: str,
    examples,  # datasets.Dataset
    tokenizer,
    model,     # DSModelWrapper or plain HF model
    step: int,
    outdir: str,
    batch_size: int = 8,
    num_samples: int = 1,
    temperature: float = 0.0,
    top_p: float = 0.95,
    entropy_mode: str = "reconsider",
    eos_ids: Optional[List[int]] = None,
    two_pass: bool = False,
    second_pass_phrase: str = "Wait, we need to reconsider. Let's think this through step by step.",
    second_pass_use_sample_idx: int = 0,
    think_cap: int = 750,
    answer_cap: int = 50,
) -> None:
    """Run two-pass think+answer inference over a dataset split."""
    if (temperature is None or float(temperature) == 0.0) and num_samples > 1:
        logging.warning(
            "temperature=0 with num_samples=%d → all samples will be identical (greedy).",
            num_samples,
        )

    outpath = os.path.join(outdir, f"step{step:04d}_{split_name}.jsonl")
    existing_samples, existing_pass1 = _scan_existing_results(outpath)
    logger.info("Resume scan: %d problems already present", len(existing_samples))

    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    two_pass_enabled = bool(two_pass)
    config = BatchGenerationConfig(
        tokenizer=tokenizer,
        model=model,
        eos_ids=eos_ids,
        entropy_mode=entropy_mode,
        temperature=temperature,
        top_p=top_p,
    )

    for start_index in range(0, len(examples), batch_size):
        idx_hi = min(start_index + batch_size, len(examples))
        slice_ds = examples.select(range(start_index, idx_hi))

        work_items = _build_work_items_for_batch(
            slice_ds,
            existing_samples,
            num_samples,
        )
        if not work_items:
            continue

        pre1_think, row_to_ex_idx, row_target_sample_idx = _build_first_pass_prefixes(
            tokenizer,
            work_items,
        )
        first_pass = _run_first_pass(
            config=config,
            think_cap=think_cap,
            answer_cap=answer_cap,
            pre1_think=pre1_think,
        )

        new_pass1_by_ex_and_k: Dict[tuple, str] = {}
        for row_index, full_row in enumerate(first_pass.full_rows):
            ex_idx = row_to_ex_idx[row_index]
            sample_idx = row_target_sample_idx[row_index]
            new_pass1_by_ex_and_k[(ex_idx, sample_idx)] = full_row

        if two_pass_enabled:
            (
                firstpass_choice_text_per_ex,
                base2_per_ex,
                cue_str,
            ) = _prepare_second_pass_inputs(
                tokenizer=tokenizer,
                work_items=work_items,
                existing_samples=existing_samples,
                    existing_pass1=existing_pass1,
                    new_pass1_by_ex_and_k=new_pass1_by_ex_and_k,
                    pass1_full_rows=first_pass.full_rows,
                    row_to_ex_idx=row_to_ex_idx,
                    num_samples=num_samples,
                    second_pass_use_sample_idx=second_pass_use_sample_idx,
                    second_pass_phrase=second_pass_phrase,
                )
            second_pass = _run_second_pass(
                config=config,
                think_cap=think_cap,
                answer_cap=answer_cap,
                pre1_think=pre1_think,
                row_to_ex_idx=row_to_ex_idx,
                base2_per_ex=base2_per_ex,
                cue_str=cue_str,
            )
        else:
            firstpass_choice_text_per_ex = [""] * len(work_items)
            cue_str = ""
            second_pass = None

        _write_results_for_batch(
            outpath=outpath,
            work_items=work_items,
            row_to_ex_idx=row_to_ex_idx,
            row_target_sample_idx=row_target_sample_idx,
            first_pass=first_pass,
            two_pass_enabled=two_pass_enabled,
            cue_str=cue_str,
            firstpass_choice_text_per_ex=firstpass_choice_text_per_ex,
            second_pass=second_pass,
            step=step,
            split_name=split_name,
        )

    logger.info("Finished split=%s; wrote %s", split_name, outpath)

def load_math500(cache_dir: str, split: str, seed: int, dataset_path: Optional[str] = None):
    """
    Thin wrapper delegating to math_core.load_math500 to keep a single
    implementation of the MATH-500 loading logic.
    """
    return _load_math500_core(cache_dir, split, seed, dataset_path)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create an argument parser for the DeepSpeed-backed Llama MATH runner."""
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
    """Initialise the tokenizer and derived EOS id set."""
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
    """Initialise the HF model, wrap it in DeepSpeed, and load checkpoints."""
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
    """Load the requested dataset (MATH-500 or arbitrary HF dataset)."""
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
    """CLI entrypoint for the DeepSpeed-backed Llama MATH runner."""
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
