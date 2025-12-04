#!/usr/bin/env python3
"""
Evaluation loop and generation orchestration for the crossword checker
(``src.utils.check``).
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union


try:
    # Package execution: src.utils.check_eval
    from . import check_dataset
    from .check_helpers import (
        GenerationConfig,
        GenerationRuntime,
        SamplingConfig,
        StopOnGeneratedSubstring,
        _concat_anchor,
        _decode_generated_only,
        _generate,
        _sanitize_generation_config,
        _token_entropy_stream,
        _token_logprobs_stream,
    )
except ImportError:  # pragma: no cover - running as a stand-alone script
    import check_dataset  # type: ignore[import-error]
    from check_helpers import (  # type: ignore[import-error]
        GenerationConfig,
        GenerationRuntime,
        SamplingConfig,
        StopOnGeneratedSubstring,
        _concat_anchor,
        _decode_generated_only,
        _generate,
        _sanitize_generation_config,
        _token_entropy_stream,
        _token_logprobs_stream,
    )

logger = logging.getLogger(__name__)


@dataclass
class EvalStats:
    """Simple container for evaluation counters."""

    correct: int = 0
    seen: int = 0
    n_no_answer: int = 0
    n_nonmatch: int = 0


@dataclass
class EvalContext:
    """Context needed for running evaluation on a split."""

    args: argparse.Namespace
    runtime: GenerationRuntime
    stopping_criteria_list_cls: Any
    dataset: Any
    out_path: Path
    eos_ids: List[int]


@dataclass
class BatchLoopState:
    """Mutable state carried across evaluation batches."""

    n_total: int
    num_samples: int
    printed_examples: int
    stats: EvalStats


@dataclass
class BatchArtifacts:
    """Per-batch data needed for scoring and logging."""

    batch: Any
    start_index: int
    grouped_data: Dict[str, List[List[Any]]]
    fout: Any


@dataclass
class RowScore:
    """Aggregated per-row statistics and sample payloads."""

    n_answered: int
    n_correct: int
    any_correct: bool
    samples: List[Dict[str, Union[int, str, float, bool]]]


def _load_libraries() -> Tuple[Any, Any, Any, Any]:
    """Dynamically import torch and transformers classes."""
    torch_mod = import_module("torch")
    transformers_mod = import_module("transformers")

    auto_tokenizer_cls = getattr(transformers_mod, "AutoTokenizer")
    auto_model_cls = getattr(transformers_mod, "AutoModelForCausalLM")
    stopping_criteria_list_cls = getattr(
        transformers_mod,
        "StoppingCriteriaList",
    )

    return torch_mod, auto_tokenizer_cls, auto_model_cls, stopping_criteria_list_cls


def _configure_torch_for_eval(args: argparse.Namespace, torch_mod: Any) -> None:
    """Configure torch for deterministic and fast matmul."""
    torch_mod.manual_seed(args.seed)
    torch_mod.backends.cuda.matmul.allow_tf32 = True
    try:
        torch_mod.set_float32_matmul_precision("high")
    except (AttributeError, TypeError, RuntimeError):
        # Older torch versions or CPU-only builds may not support this.
        pass


def _load_tokenizer_and_model_for_eval(
    args: argparse.Namespace,
    torch_mod: Any,
    auto_tokenizer_cls: Any,
    auto_model_cls: Any,
) -> Tuple[Any, Any]:
    """Load tokenizer and model for local evaluation."""
    dtype_map = {
        "bfloat16": torch_mod.bfloat16,
        "float16": torch_mod.float16,
        "float32": torch_mod.float32,
    }
    dtype = dtype_map[args.dtype]
    device_map = "auto" if torch_mod.cuda.is_available() else None

    logger.info("Loading tokenizer from %s …", args.model_path)
    tok = auto_tokenizer_cls.from_pretrained(
        str(args.model_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"

    logger.info("Loading model from %s …", args.model_path)
    model = auto_model_cls.from_pretrained(
        str(args.model_path),
        torch_dtype=dtype,
        attn_implementation="sdpa",
        trust_remote_code=True,
        device_map=device_map,
        local_files_only=True,
    )
    if hasattr(model.config, "sliding_window"):
        model.config.sliding_window = None
    _sanitize_generation_config(model)
    model.config.forced_eos_token_id = None
    model.config.forced_bos_token_id = None
    model.eval()
    return tok, model


def _compute_eos_token_ids(tok: Any) -> List[int]:
    """Return a robust list of EOS token ids for Qwen-style chat models."""
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    eos_ids = [tok.eos_token_id]
    if im_end_id is not None and im_end_id != -1:
        eos_ids.append(im_end_id)
    return eos_ids


def _prepare_dataset_split(args: argparse.Namespace) -> Any:
    """Load dataset split and apply optional limiting."""
    logger.info("Loading dataset …")
    dataset = check_dataset.load_cryptonite_split(
        args.dataset_name,
        args.split,
        args.cryptonite_zip,
    )
    n_total = len(dataset)
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, n_total)))
        n_total = len(dataset)
    logger.info("Loaded split '%s' with %d examples.", args.split, n_total)
    return dataset


def _prepare_out_path(args: argparse.Namespace) -> Path:
    """Create parent directory for output file and return its path."""
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def _build_eval_context(args: argparse.Namespace) -> EvalContext:
    """Construct an EvalContext with all heavy resources loaded."""
    (
        torch_mod,
        auto_tokenizer_cls,
        auto_model_cls,
        stopping_criteria_list_cls,
    ) = _load_libraries()
    _configure_torch_for_eval(args, torch_mod)
    tok, model = _load_tokenizer_and_model_for_eval(
        args,
        torch_mod,
        auto_tokenizer_cls,
        auto_model_cls,
    )
    eos_ids = _compute_eos_token_ids(tok)
    dataset = _prepare_dataset_split(args)
    out_path = _prepare_out_path(args)
    runtime = GenerationRuntime(
        torch_mod=torch_mod,
        model=model,
        tokenizer=tok,
    )
    return EvalContext(
        args=args,
        runtime=runtime,
        stopping_criteria_list_cls=stopping_criteria_list_cls,
        dataset=dataset,
        out_path=out_path,
        eos_ids=eos_ids,
    )


def _group_by_row(
    batch_size: int,
    num_samples: int,
    texts: List[str],
    metrics: Dict[str, List[float]],
) -> Dict[str, List[List[Any]]]:
    """Reshape flat per-sample lists back to per-row lists."""
    grouped_texts = [texts[index * num_samples : (index + 1) * num_samples] for index in range(batch_size)]
    grouped_logsum = [
        metrics["logsum"][index * num_samples : (index + 1) * num_samples] for index in range(batch_size)
    ]
    grouped_logavg = [
        metrics["logavg"][index * num_samples : (index + 1) * num_samples] for index in range(batch_size)
    ]
    grouped_genlen = [
        metrics["genlen"][index * num_samples : (index + 1) * num_samples] for index in range(batch_size)
    ]
    grouped_entavg = [
        metrics["entropy"][index * num_samples : (index + 1) * num_samples] for index in range(batch_size)
    ]
    return {
        "texts": grouped_texts,
        "logsum": grouped_logsum,
        "logavg": grouped_logavg,
        "genlen": grouped_genlen,
        "entropy": grouped_entavg,
    }


def _build_metrics_dict(
    runtime: GenerationRuntime,
    sequences: Any,
    scores: Any,
    input_lens: Any,
    eos_ids: List[int],
) -> Tuple[List[str], Dict[str, List[float]]]:
    """Decode generated sequences and compute per-sequence metrics."""
    torch_mod = runtime.torch_mod
    texts = _decode_generated_only(runtime.tokenizer, sequences, input_lens)
    logsum, logavg, genlen = _token_logprobs_stream(
        torch_mod,
        scores,
        sequences,
        input_lens,
        eos_ids=eos_ids,
    )
    entropy, _ = _token_entropy_stream(
        torch_mod,
        scores,
        sequences,
        input_lens,
        eos_ids=eos_ids,
    )
    metrics: Dict[str, List[float]] = {
        "logsum": logsum,
        "logavg": logavg,
        "genlen": genlen,
        "entropy": entropy,
    }
    return texts, metrics


def _sampling_fallback(
    context: EvalContext,
    tensors: Dict[str, Any],
    grouped_data: Dict[str, List[List[Any]]],
) -> Dict[str, List[List[Any]]]:
    """Retry rows without an answer using sampling (single-sample mode)."""
    args = context.args
    need_retry = [
        index for index, rows in enumerate(grouped_data["texts"]) if not check_dataset.extract_answer_last(rows[0])
    ]
    if not need_retry or not args.fallback_sampling:
        return grouped_data

    ids_sub = tensors["input_ids"][need_retry]
    att_sub = tensors["attention"][need_retry]
    stops_sub = context.stopping_criteria_list_cls(
        [
            StopOnGeneratedSubstring(
                context.runtime.tokenizer,
                att_sub.sum(dim=-1).tolist(),
                "</answer>",
            )
        ]
    )

    generation = _generate(
        context.runtime,
        ids_sub,
        att_sub,
        GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            num_beams=1,
            min_new_tokens=max(args.min_new_tokens, 12),
            stop_criteria=stops_sub,
            eos_token_ids=context.eos_ids,
            num_return_sequences=1,
            sampling=SamplingConfig(
                ban_eos_steps=max(0, args.ban_eos_steps // 2),
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            ),
        ),
    )
    texts2, metrics = _build_metrics_dict(
        context.runtime,
        generation[0],
        generation[1],
        generation[2],
        context.eos_ids,
    )

    for local_index, batch_index in enumerate(need_retry):
        grouped_data["texts"][batch_index] = [texts2[local_index]]
        grouped_data["logsum"][batch_index] = [metrics["logsum"][local_index]]
        grouped_data["logavg"][batch_index] = [metrics["logavg"][local_index]]
        grouped_data["genlen"][batch_index] = [metrics["genlen"][local_index]]
        grouped_data["entropy"][batch_index] = [metrics["entropy"][local_index]]

    return grouped_data


def _scaffold_fallback(
    context: EvalContext,
    dialogs: List[List[Dict[str, str]]],
    grouped_data: Dict[str, List[List[Any]]],
) -> Dict[str, List[List[Any]]]:
    """Second-stage scaffold to force an <answer> span (single-sample mode)."""
    args = context.args
    need_retry = [
        batch_index
        for batch_index in range(len(grouped_data["texts"]))
        if not check_dataset.extract_answer_last(grouped_data["texts"][batch_index][0])
    ]
    if not need_retry or not args.force_answer:
        return grouped_data

    enc2 = context.runtime.tokenizer.apply_chat_template(
        [dialogs[index] for index in need_retry],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
    )
    if hasattr(enc2, "input_ids"):
        ids2 = enc2["input_ids"].to(context.runtime.model.device)
        att2 = enc2.get(
            "attention_mask",
            context.runtime.torch_mod.ones_like(enc2["input_ids"]),
        ).to(context.runtime.model.device)
    else:
        ids2 = enc2.to(context.runtime.model.device)
        if context.runtime.tokenizer.pad_token_id is not None:
            att2 = (ids2 != context.runtime.tokenizer.pad_token_id).long()
        else:
            att2 = context.runtime.torch_mod.ones_like(ids2)

    ids2, att2 = _concat_anchor(
        context.runtime.torch_mod,
        context.runtime.tokenizer,
        ids2,
        att2,
        "<think>\n" if args.anchor_think else None,
    )
    ids2, att2 = _concat_anchor(
        context.runtime.torch_mod,
        context.runtime.tokenizer,
        ids2,
        att2,
        "\n</think>\n<answer>\n",
    )

    stops2 = context.stopping_criteria_list_cls(
        [
            StopOnGeneratedSubstring(
                context.runtime.tokenizer,
                att2.sum(dim=-1).tolist(),
                "</answer>",
            )
        ]
    )

    generation = _generate(
        context.runtime,
        ids2,
        att2,
        GenerationConfig(
            max_new_tokens=max(
                32,
                min(128, args.max_new_tokens // 3),
            ),
            num_beams=1,
            min_new_tokens=8,
            stop_criteria=stops2,
            eos_token_ids=context.eos_ids,
            num_return_sequences=1,
            sampling=SamplingConfig(
                ban_eos_steps=4,
                do_sample=False,
            ),
        ),
    )
    texts3, metrics = _build_metrics_dict(
        context.runtime,
        generation[0],
        generation[1],
        generation[2],
        context.eos_ids,
    )

    for local_index, batch_index in enumerate(need_retry):
        grouped_data["texts"][batch_index] = [texts3[local_index]]
        grouped_data["logsum"][batch_index] = [metrics["logsum"][local_index]]
        grouped_data["logavg"][batch_index] = [metrics["logavg"][local_index]]
        grouped_data["genlen"][batch_index] = [metrics["genlen"][local_index]]
        grouped_data["entropy"][batch_index] = [metrics["entropy"][local_index]]

    return grouped_data


def _apply_single_sample_fallbacks(
    context: EvalContext,
    dialogs: List[List[Dict[str, str]]],
    tensors: Dict[str, Any],
    grouped_data: Dict[str, List[List[Any]]],
) -> Dict[str, List[List[Any]]]:
    """Apply both sampling and scaffold fallbacks in single-sample mode."""
    updated = _sampling_fallback(context, tensors, grouped_data)
    return _scaffold_fallback(context, dialogs, updated)


def _encode_dialogs_for_batch(
    context: EvalContext,
    dialogs: List[List[Dict[str, str]]],
) -> Tuple[Any, Any]:
    """Encode chat dialogs into input ids and attention masks."""
    tok = context.runtime.tokenizer
    torch_mod = context.runtime.torch_mod
    enc = tok.apply_chat_template(
        dialogs,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
    )

    if hasattr(enc, "input_ids"):
        input_ids = enc["input_ids"].to(context.runtime.model.device)
        attention = enc.get(
            "attention_mask",
            torch_mod.ones_like(enc["input_ids"]),
        ).to(context.runtime.model.device)
    elif isinstance(enc, torch_mod.Tensor):
        input_ids = enc.to(context.runtime.model.device)
        if tok.pad_token_id is not None:
            attention = (input_ids != tok.pad_token_id).long()
        else:
            attention = torch_mod.ones_like(input_ids)
    else:
        raise TypeError(f"Unexpected encoding type: {type(enc)}")

    return input_ids, attention


def _generate_for_batch(
    context: EvalContext,
    batch: Any,
    num_samples: int,
) -> Dict[str, List[List[Any]]]:
    """Run generation (with fallbacks) for a single batch."""
    args = context.args
    dialogs = [check_dataset.build_messages(row["problem"], row["answer"]) for row in batch]
    input_ids, attention = _encode_dialogs_for_batch(context, dialogs)
    torch_mod = context.runtime.torch_mod

    input_ids, attention = _concat_anchor(
        torch_mod,
        context.runtime.tokenizer,
        input_ids,
        attention,
        "<think>\n" if args.anchor_think else None,
    )

    stops = context.stopping_criteria_list_cls(
        [
            StopOnGeneratedSubstring(
                context.runtime.tokenizer,
                [prompt_length for prompt_length in attention.sum(dim=-1).tolist() for _ in range(num_samples)],
                "</answer>",
            )
        ]
    )

    generation = _generate(
        context.runtime,
        input_ids,
        attention,
        GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            num_beams=1 if num_samples > 1 else args.num_beams,
            min_new_tokens=args.min_new_tokens,
            stop_criteria=stops,
            eos_token_ids=context.eos_ids,
            num_return_sequences=num_samples,
            sampling=SamplingConfig(
                ban_eos_steps=args.ban_eos_steps,
                do_sample=num_samples > 1,
                temperature=args.temperature if num_samples > 1 else None,
                top_p=(args.top_p if (num_samples > 1 and args.top_p not in (None, 0)) else None),
                top_k=(args.top_k if (num_samples > 1 and args.top_k not in (None, 0)) else None),
            ),
        ),
    )

    texts, metrics = _build_metrics_dict(
        context.runtime,
        generation[0],
        generation[1],
        generation[2],
        context.eos_ids,
    )

    grouped_data = _group_by_row(
        batch_size=len(batch),
        num_samples=num_samples,
        texts=texts,
        metrics=metrics,
    )

    if num_samples == 1:
        tensors = {"input_ids": input_ids, "attention": attention}
        grouped_data = _apply_single_sample_fallbacks(
            context,
            dialogs,
            tensors,
            grouped_data,
        )

    return grouped_data


def _compute_row_score(
    grouped: Dict[str, List[List[Any]]],
    row_index: int,
    gold_norm: str,
) -> RowScore:
    """Aggregate per-sample predictions and metrics for a single row."""
    texts = grouped["texts"][row_index]
    logsum = grouped["logsum"][row_index]
    logavg = grouped["logavg"][row_index]
    genlen = grouped["genlen"][row_index]
    entropy = grouped["entropy"][row_index]

    samples: List[Dict[str, Union[int, str, float, bool]]] = []
    n_answered = 0
    n_correct = 0

    for sample_index, text in enumerate(texts):
        raw_pred = check_dataset.extract_answer_with_fallback(text) or ""
        pred_norm = check_dataset.normalise(raw_pred) if raw_pred else ""
        if raw_pred:
            n_answered += 1
        if pred_norm == gold_norm:
            n_correct += 1

        samples.append(
            {
                "k": sample_index,
                "pred_text": text,
                "pred_answer_raw": raw_pred,
                "pred_answer_norm": pred_norm,
                "is_correct": pred_norm == gold_norm,
                "gen_len": genlen[sample_index],
                "logprob_sum": logsum[sample_index],
                "logprob_avg": logavg[sample_index],
                "entropy_avg": entropy[sample_index],
            }
        )

    return RowScore(
        n_answered=n_answered,
        n_correct=n_correct,
        any_correct=n_correct > 0,
        samples=samples,
    )


def _score_single_row(
    artifacts: BatchArtifacts,
    state: BatchLoopState,
    row_index: int,
) -> None:
    """Score a single row within a batch and write its payload."""
    grouped = artifacts.grouped_data
    row = artifacts.batch[row_index]
    gold_norm = row["answer"]
    score = _compute_row_score(grouped, row_index, gold_norm)

    state.stats.seen += 1
    if score.n_answered == 0:
        state.stats.n_no_answer += 1
    elif not score.any_correct:
        state.stats.n_nonmatch += 1
    else:
        state.stats.correct += 1

    payload = {
        "idx": artifacts.start_index + row_index,
        "clue": row["problem"].split("\n<think>")[0].strip(),
        "enum": len(gold_norm),
        "gold_answer": gold_norm,
        "num_samples": len(score.samples),
        "n_answered": score.n_answered,
        "n_correct": score.n_correct,
        "any_correct": bool(score.any_correct),
        "samples": score.samples,
    }
    artifacts.fout.write(json.dumps(payload, ensure_ascii=False) + "\n")

    if (score.n_answered == 0 or not score.any_correct) and state.printed_examples < 100:
        tail = (score.samples[0]["pred_text"] or "")[-300:]
        print("\n--- DEBUG SAMPLE ---")
        print("CLUE:", payload["clue"])
        print("ENUM:", payload["enum"], "GOLD:", gold_norm)
        print("GEN TAIL (s0):", tail if tail else "<empty>")
        print(
            "EXTRACTED (s0):",
            score.samples[0]["pred_answer_raw"] if score.samples else "None",
        )
        print("--------------------\n")
        state.printed_examples += 1


def _log_batch_summary(
    torch_mod: Any,
    grouped_logavg: List[List[float]],
    state: BatchLoopState,
) -> None:
    """Compute and print aggregate batch statistics."""
    flat_logavg = list(itertools.chain.from_iterable(grouped_logavg))
    finite_logavg = [
        value for value in flat_logavg if not torch_mod.isnan(torch_mod.tensor(value)) and value != float("-inf")
    ]
    if finite_logavg:
        batch_avg = sum(finite_logavg) / len(finite_logavg)
    else:
        batch_avg = 0.0

    accuracy = 100.0 * state.stats.correct / max(1, state.stats.seen)
    print(
        f"[{state.stats.seen:5d}/{state.n_total}] acc={accuracy:5.2f}%  "
        f"(last batch avg logp={batch_avg:.3f})  "
        f"[no-ans: {state.stats.n_no_answer}  nonmatch: {state.stats.n_nonmatch}]   "
        f"(R={state.num_samples})",
        flush=True,
    )


def _score_and_log_batch(
    context: EvalContext,
    artifacts: BatchArtifacts,
    state: BatchLoopState,
) -> None:
    """Score a single batch, write outputs, and log progress."""
    for row_index in range(len(artifacts.batch)):
        _score_single_row(artifacts, state, row_index)

    artifacts.fout.flush()
    try:
        os.fsync(artifacts.fout.fileno())
    except OSError:
        # Best-effort; not critical for correctness.
        pass

    _log_batch_summary(
        context.runtime.torch_mod,
        artifacts.grouped_data["logavg"],
        state,
    )


def _run_eval_loop(context: EvalContext) -> EvalStats:
    """Main evaluation loop over the dataset split."""
    args = context.args
    dataset = context.dataset
    out_path = context.out_path
    n_total = len(dataset)
    stats = EvalStats()
    state = BatchLoopState(
        n_total=n_total,
        num_samples=max(1, int(args.num_samples)),
        printed_examples=0,
        stats=stats,
    )

    with out_path.open("a", encoding="utf-8", buffering=1) as fout:
        for start in range(0, n_total, args.batch_size):
            batch = dataset.select(range(start, min(n_total, start + args.batch_size)))
            grouped_data = _generate_for_batch(
                context,
                batch,
                state.num_samples,
            )
            artifacts = BatchArtifacts(
                batch=batch,
                start_index=start,
                grouped_data=grouped_data,
                fout=fout,
            )
            _score_and_log_batch(context, artifacts, state)

    return state.stats


def run(args: argparse.Namespace) -> Tuple[EvalStats, Path]:
    """Public entry point used by ``src.utils.check.main``."""
    context = _build_eval_context(args)
    stats = _run_eval_loop(context)
    return stats, context.out_path
