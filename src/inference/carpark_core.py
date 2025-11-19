#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-pass batch inference for Rush Hour (car-parking) using Qwen2.5-style chat
LMs that produce: <think> ... </think><answer> ... </answer>.

This version supports resume/fill behaviour:
- If the output JSONL already contains some rows for an example_id, we detect
  which sample_idx values are present and write ONLY the missing ones.
- After a run, every example will have exactly --num_samples rows.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from src.inference.backends import HFBackend
from src.inference.carpark_rush_utils import (
    _canon_rush_generic,
    _canon_rush_gold,
    _is_valid_rush,
    rush_soft_match_reward,
)
from src.inference.common import (
    PassOutputs,
    StopOnSubstrings,
    add_token_and_tag_fields,
    build_entropy_pass_base,
    build_generate_kwargs,
    build_math_inference_config_kwargs,
    build_two_pass_row_base,
    decode_and_score_batch,
    extract_blocks as _extract_blocks,
    finite_mean as _finite_mean,
    iter_jsonl_objects,
    require_datasets,
    setup_script_logger,
    tokenize_prefixes_for_generate,
)
from src.inference.task_registry import CARPARK_SYSTEM_PROMPT
from src.inference.unified_runner_base import run_carpark_main

try:
    torch = importlib.import_module("torch")
except ImportError as torch_import_exc:  # pragma: no cover - hard dependency
    raise RuntimeError(
        "carpark_core requires 'torch'; install it to use this script.",
    ) from torch_import_exc

F = torch.nn.functional

try:
    transformers_mod = importlib.import_module("transformers")
except ImportError as transformers_import_exc:  # pragma: no cover - hard dependency
    raise RuntimeError(
        "carpark_core requires 'transformers'; install it to use this script.",
    ) from transformers_import_exc

AutoModelForCausalLM = transformers_mod.AutoModelForCausalLM
AutoTokenizer = transformers_mod.AutoTokenizer
StoppingCriteriaList = transformers_mod.StoppingCriteriaList


# ───────────────────────── System prompt (from training) ─────────────────────────

SYSTEM_PROMPT = CARPARK_SYSTEM_PROMPT


# Optional reconsideration markers (analytics)
_RECONSIDER_PATTERNS = [
    ("wait_line", re.compile(r"(?im)^\s*wait[,\.\-–—… ]", re.I)),
    ("wait_reconsider", re.compile(r"\bwait\b.*\breconsider\b", re.I | re.S)),
    ("step_by_step", re.compile(r"\bstep[-\s]?by[-\s]?step\b", re.I)),
    ("recheck", re.compile(r"\bre[-\s]?check(ing)?\b", re.I)),
]


logger = setup_script_logger(__name__)


# ───────────────────────── Chat builders ─────────────────────────

def _ensure_messages(obj: Union[str, List[Dict[str, str]]]) -> List[Dict[str, str]]:
    """Dataset may store messages as JSON string or as a Python list."""
    if isinstance(obj, list):
        return obj
    if isinstance(obj, str):
        try:
            parsed = json.loads(obj)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            return []
    raise ValueError("messages field is neither list nor JSON-encoded list")


def chat_base_for_pass1_from_messages(tokenizer, messages: List[Dict[str, str]]) -> str:
    """Build a chat prompt for pass-1 from serialized messages."""
    msgs = list(messages)
    has_sys = any(message.get("role") == "system" for message in msgs)
    if not has_sys:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + msgs
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def chat_base_for_pass2_from_messages(
    tokenizer,
    messages: List[Dict[str, str]],
    prev_output: str,
    cue: str,
) -> str:
    """Build a chat prompt for pass-2 that includes the pass-1 output and cue."""
    msgs = list(messages)
    has_sys = any(message.get("role") == "system" for message in msgs)
    if not has_sys:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + msgs
    msgs = msgs + [
        {"role": "assistant", "content": prev_output},
        {"role": "user", "content": cue},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ───────────────────────── Inference config ─────────────────────────


@dataclass
class IOConfig:
    """Output/split configuration."""

    split_name: str
    output_dir: str
    step: int


@dataclass
class ColumnConfig:
    """Dataset column names."""

    prompt_col: str
    solution_col: str


@dataclass
class GenerationSamplingConfig:
    """Sampling-related generation hyperparameters."""

    temperature: float
    top_p: float
    entropy_mode: str
    eos_ids: Optional[List[int]]


@dataclass
class GenerationLimitsConfig:
    """Generation limits and batch sizing."""

    batch_size: int
    num_samples: int
    think_cap: int
    answer_cap: int


@dataclass
class SecondPassConfig:
    """Settings controlling optional second-pass reconsideration."""

    two_pass: bool
    phrase: str
    use_sample_idx: int


@dataclass
class CarparkInferenceConfig:
    """Configuration for the two-pass Rush Hour inference loop."""

    io_config: IOConfig
    columns: ColumnConfig
    limits: GenerationLimitsConfig
    sampling: GenerationSamplingConfig
    second_pass: SecondPassConfig

    @classmethod
    def from_flat(
        cls,
        **flat_kwargs: Any,
    ) -> "CarparkInferenceConfig":
        """Construct a CarparkInferenceConfig from flat arguments."""
        return cls(
            io_config=IOConfig(
                split_name=str(flat_kwargs["split_name"]),
                output_dir=str(flat_kwargs["output_dir"]),
                step=int(flat_kwargs["step"]),
            ),
            columns=ColumnConfig(
                prompt_col=str(flat_kwargs["prompt_col"]),
                solution_col=str(flat_kwargs["solution_col"]),
            ),
            limits=GenerationLimitsConfig(
                batch_size=int(flat_kwargs["batch_size"]),
                num_samples=int(flat_kwargs["num_samples"]),
                think_cap=int(flat_kwargs["think_cap"]),
                answer_cap=int(flat_kwargs["answer_cap"]),
            ),
            sampling=GenerationSamplingConfig(
                temperature=float(flat_kwargs["temperature"]),
                top_p=float(flat_kwargs["top_p"]),
                entropy_mode=str(flat_kwargs["entropy_mode"]),
                eos_ids=flat_kwargs.get("eos_ids"),
            ),
            second_pass=SecondPassConfig(
                two_pass=bool(flat_kwargs["two_pass"]),
                phrase=str(flat_kwargs["second_pass_phrase"]),
                use_sample_idx=int(flat_kwargs["second_pass_use_sample_idx"]),
            ),
        )

    # Convenience properties mirroring the previous dataclass fields.
    @property
    def split_name(self) -> str:
        """Return the dataset split name."""
        return self.io_config.split_name

    @property
    def output_dir(self) -> str:
        """Return the output directory for results."""
        return self.io_config.output_dir

    @property
    def step(self) -> int:
        """Return the step index used in output filenames."""
        return self.io_config.step

    @property
    def prompt_col(self) -> str:
        """Return the name of the prompt column."""
        return self.columns.prompt_col

    @property
    def solution_col(self) -> str:
        """Return the name of the solution/answer column."""
        return self.columns.solution_col

    @property
    def batch_size(self) -> int:
        """Return the generation batch size."""
        return self.limits.batch_size

    @property
    def num_samples(self) -> int:
        """Return the number of samples per example."""
        return self.limits.num_samples

    @property
    def temperature(self) -> float:
        """Return the sampling temperature."""
        return self.sampling.temperature

    @property
    def top_p(self) -> float:
        """Return the nucleus sampling top_p value."""
        return self.sampling.top_p

    @property
    def entropy_mode(self) -> str:
        """Return the entropy scoring mode."""
        return self.sampling.entropy_mode

    @property
    def eos_ids(self) -> Optional[List[int]]:
        """Return the list of EOS token IDs used during generation."""
        return self.sampling.eos_ids

    @property
    def two_pass(self) -> bool:
        """Return whether second-pass generation is enabled."""
        return self.second_pass.two_pass

    @property
    def second_pass_phrase(self) -> str:
        """Return the cue phrase used to start the second pass."""
        return self.second_pass.phrase

    @property
    def second_pass_use_sample_idx(self) -> int:
        """Return which pass-1 sample index feeds pass-2."""
        return self.second_pass.use_sample_idx

    @property
    def think_cap(self) -> int:
        """Return the max new tokens for the think phase."""
        return self.limits.think_cap

    @property
    def answer_cap(self) -> int:
        """Return the max new tokens for the answer phase."""
        return self.limits.answer_cap


def _make_gen_kwargs(
    cap: int,
    tokenizer,
    config: CarparkInferenceConfig,
) -> Dict[str, Any]:
    """Build generation kwargs with a temperature=0 → greedy safeguard."""
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    return build_generate_kwargs(
        cap=cap,
        pad_token_id=pad_token_id,
        eos_ids=config.eos_ids,
        entropy_mode=config.entropy_mode,
        temperature=config.temperature,
        top_p=config.top_p,
    )


@dataclass
class InferenceContext:
    """Bundle model, tokenizer, and config for generation helpers."""

    tokenizer: Any
    model: Any
    config: CarparkInferenceConfig


@dataclass
class ResultsContext:
    """Bundle shared state needed when writing per-example results."""

    outpath: str
    inference: InferenceContext
    num_samples: int
    cue_str: str
    firstpass_choice: List[str]
    existing_by_example: Dict[str, set[int]]


@dataclass
class SecondPassContext:
    """Bundle inputs needed to run the optional second pass."""

    batch_items: List[Dict[str, Any]]
    inference: InferenceContext
    firstpass_choice: List[str]
    num_samples: int
    cue_str: str


def _gen_batch(
    prefixes: List[str],
    cap: int,
    stop_strs: List[str],
    context: InferenceContext,
) -> Tuple[List[str], List[List[float]], torch.Tensor, torch.Tensor, List[str]]:
    """Generate a batch of continuations and token-entropy series."""
    tokenizer = context.tokenizer
    model = context.model
    config = context.config

    inputs, input_lengths = tokenize_prefixes_for_generate(
        tokenizer,
        prefixes,
        max_length=4096,
    )
    stop = (
        StoppingCriteriaList([StopOnSubstrings(tokenizer, stop_strs)])
        if stop_strs
        else None
    )
    gen_kwargs = _make_gen_kwargs(cap, tokenizer, config)
    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs, stopping_criteria=stop)

    decoded_texts, entropy_series, stop_reasons = decode_and_score_batch(
        tokenizer=tokenizer,
        sequences=out.sequences,
        scores=out.scores,
        input_lengths=input_lengths,
        stop_strings=stop_strs,
        cap=cap,
        eos_ids=config.eos_ids,
        entropy_mode=config.entropy_mode,
        model=model,
    )

    return decoded_texts, entropy_series, input_lengths, out.sequences, stop_reasons


def _repeat_for_samples(prefixes: List[str], num_samples: int) -> List[str]:
    return [prefix for prefix in prefixes for _ in range(num_samples)]


def _norm_fields(
    example: dict,
    prompt_col: str,
    solution_col: str,
) -> Tuple[List[Dict[str, str]], Any]:
    """Normalize raw record fields into (messages, solution)."""
    messages = example.get(prompt_col)
    solution = example.get(solution_col)
    try:
        messages = _ensure_messages(messages)
    except ValueError:
        problem = example.get("problem") or example.get("board") or example.get("prompt") or ""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(problem)},
        ]
    return messages, solution


def _load_existing_example_index(outpath: str) -> Dict[str, set[int]]:
    """Scan an existing JSONL file and return {example_id -> {sample_idx,...}}."""
    existing_by_example: Dict[str, set[int]] = defaultdict(set)
    if not os.path.exists(outpath):
        return existing_by_example
    for record in iter_jsonl_objects(outpath):
        example_id = record.get("example_id")
        sample_idx = record.get("sample_idx")
        if example_id is None or not isinstance(sample_idx, int):
            continue
        existing_by_example[str(example_id)].add(sample_idx)
    return existing_by_example


@dataclass
class SampleRowContext:
    """Context required to build a single output row."""

    example: Dict[str, Any]
    pass1: PassOutputs
    pass2: Optional[PassOutputs]
    results_ctx: ResultsContext
    batch_index: int
    sample_idx: int
    gold_set: set[str]


def _pack_pass_result(
    full_text: str,
    ent_think: List[float],
    ent_answer: List[float],
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble per-pass result dict with entropy and reconsideration markers."""
    token_entropies_all = (ent_think or []) + (ent_answer or [])

    think_block, answer_block = _extract_blocks(full_text)
    think_text = think_block or ""
    pred_answer_text = (answer_block or "").strip()

    markers = _find_reconsider_markers(think_text, meta)
    pred_canon = _canon_rush_generic(pred_answer_text)

    base = build_entropy_pass_base(
        prev_output=meta.get("prev_output"),
        full_text=full_text,
        pred_answer_text=pred_answer_text,
        pred_canon=pred_canon,
        entropy_overall=_finite_mean(token_entropies_all) if token_entropies_all else None,
        entropy_think=_finite_mean(ent_think) if ent_think else None,
        entropy_answer=_finite_mean(ent_answer) if ent_answer else None,
    )
    tokens_think = len(ent_think or [])
    tokens_answer = len(ent_answer or [])
    base.update(
        {
            "stop_reason_think": meta.get("stop_reason_think"),
            "stop_reason_answer": meta.get("stop_reason_answer"),
            "has_reconsider_cue": bool(markers),
            "reconsider_markers": markers,
            "is_valid_pred": _is_valid_rush(pred_canon),
            "is_correct_pred": False,
        },
    )
    return add_token_and_tag_fields(
        base,
        tokens_total=len(token_entropies_all),
        tokens_think=tokens_think,
        tokens_answer=tokens_answer,
        full_text=full_text,
    )


def _find_reconsider_markers(think_text: str, meta: Dict[str, Any]) -> List[str]:
    """Return a list of reconsideration markers found in the think block."""
    markers: List[str] = []
    if think_text:
        skip_chars = len(meta.get("cue_prefix_str", "")) if meta.get("injected_cue") else 0
        search_text = think_text[skip_chars:] if skip_chars > 0 else think_text
        for name, pattern in _RECONSIDER_PATTERNS:
            if pattern.search(search_text):
                markers.append(name)
                break
    if meta.get("injected_cue"):
        markers = ["injected_cue"] + markers
    return markers


def _run_pass1_for_batch(
    batch_items: List[Dict[str, Any]],
    context: InferenceContext,
) -> Tuple[PassOutputs, int]:
    """Run pass-1 (think + answer) for a batch of examples."""
    num_samples = int(context.config.num_samples)
    base_prompts = [
        chat_base_for_pass1_from_messages(context.tokenizer, example["messages"])
        for example in batch_items
    ]
    pre1_think = _repeat_for_samples(
        [prompt + "<think>\n" for prompt in base_prompts],
        num_samples,
    )
    think1_texts, think1_ents, _, _, think1_stop = _gen_batch(
        pre1_think,
        context.config.think_cap,
        ["</think>"],
        context,
    )

    pre1_answer: List[str] = [
        pre_think + think_text + "</think>\n<answer>\n"
        for pre_think, think_text in zip(pre1_think, think1_texts)
    ]
    answer1_texts, answer1_ents, _, _, answer1_stop = _gen_batch(
        pre1_answer,
        context.config.answer_cap,
        ["</answer>"],
        context,
    )

    pass1_full_texts = [
        f"<think>{think_text}</think>\n<answer>{answer_text}</answer>"
        for think_text, answer_text in zip(think1_texts, answer1_texts)
    ]
    return (
        PassOutputs(
            full_texts=pass1_full_texts,
            ent_think=think1_ents,
            ent_answer=answer1_ents,
            stop_reason_think=think1_stop,
            stop_reason_answer=answer1_stop,
        ),
        num_samples,
    )


def _build_first_pass_choice(
    batch_items: List[Dict[str, Any]],
    pass1_full: List[str],
    num_samples: int,
    config: CarparkInferenceConfig,
) -> List[str]:
    """Select which pass-1 sample feeds pass-2 per example."""
    choice_texts: List[str] = []
    for batch_index, _ in enumerate(batch_items):
        chosen_sample = max(
            0,
            min(config.second_pass_use_sample_idx, num_samples - 1),
        )
        row_index = batch_index * num_samples + chosen_sample
        choice_texts.append(pass1_full[row_index])
    return choice_texts


def _build_second_pass_prompts(ctx: SecondPassContext) -> List[str]:
    """Build chat prompts for pass-2 for each batch item."""
    return [
        chat_base_for_pass2_from_messages(
            ctx.inference.tokenizer,
            example["messages"],
            ctx.firstpass_choice[index],
            ctx.inference.config.second_pass_phrase.strip(),
        )
        for index, example in enumerate(ctx.batch_items)
    ]


def _build_second_pass_think_prefixes(
    prompts: List[str],
    ctx: SecondPassContext,
) -> List[str]:
    """Expand prompts and cue string into think-prefixes for all samples."""
    return _repeat_for_samples(
        [prompt + "<think>\n" + ctx.cue_str for prompt in prompts],
        ctx.num_samples,
    )


def _run_pass2_for_batch(ctx: SecondPassContext) -> PassOutputs:
    """Run optional pass-2 reconsideration for a batch."""
    prompts = _build_second_pass_prompts(ctx)
    pre_think_prefixes = _build_second_pass_think_prefixes(prompts, ctx)
    think_texts, ent_think, _, _, stop_think = _gen_batch(
        pre_think_prefixes,
        ctx.inference.config.think_cap,
        ["</think>"],
        ctx.inference,
    )

    pre_answer_prefixes = [
        prefix + text + "</think>\n<answer>\n"
        for prefix, text in zip(pre_think_prefixes, think_texts)
    ]
    answer_texts, ent_answer, _, _, stop_answer = _gen_batch(
        pre_answer_prefixes,
        ctx.inference.config.answer_cap,
        ["</answer>"],
        ctx.inference,
    )

    full_texts = [
        f"<think>{ctx.cue_str}{think_text}</think>\n<answer>{answer_text}</answer>"
        for think_text, answer_text in zip(think_texts, answer_texts)
    ]

    return PassOutputs(
        full_texts=full_texts,
        ent_think=ent_think,
        ent_answer=ent_answer,
        stop_reason_think=stop_think,
        stop_reason_answer=stop_answer,
    )


def _build_row_for_sample(ctx: SampleRowContext) -> Dict[str, Any]:
    """Assemble the output row for a single (example, sample_idx)."""
    config = ctx.results_ctx.inference.config
    row_index = ctx.batch_index * ctx.results_ctx.num_samples + ctx.sample_idx
    gold_set = ctx.gold_set

    pass1_result = _pack_pass_result(
        full_text=ctx.pass1.full_texts[row_index],
        ent_think=ctx.pass1.ent_think[row_index],
        ent_answer=ctx.pass1.ent_answer[row_index],
        meta={
            "injected_cue": False,
            "prev_output": None,
            "cue_prefix_str": "",
            "stop_reason_think": ctx.pass1.stop_reason_think[row_index],
            "stop_reason_answer": ctx.pass1.stop_reason_answer[row_index],
        },
    )
    pass1_result["is_correct_pred"] = bool(
        pass1_result.get("pred_answer_canon")
        and pass1_result["pred_answer_canon"] in gold_set
    )
    (
        pass1_result["soft_reward"],
        pass1_result["soft_reward_detail"],
    ) = rush_soft_match_reward(
        pass1_result["pred_answer"],
        ctx.example["solution"],
    )

    pass2_result: Optional[Dict[str, Any]] = None
    if config.two_pass and ctx.pass2 is not None:
        pass2_result = _pack_pass_result(
            full_text=ctx.pass2.full_texts[row_index],
            ent_think=ctx.pass2.ent_think[row_index],
            ent_answer=ctx.pass2.ent_answer[row_index],
            meta={
                "injected_cue": True,
                "prev_output": ctx.results_ctx.firstpass_choice[ctx.batch_index],
                "cue_prefix_str": ctx.results_ctx.cue_str,
                "stop_reason_think": ctx.pass2.stop_reason_think[row_index],
                "stop_reason_answer": ctx.pass2.stop_reason_answer[row_index],
            },
        )
        pass2_result["is_correct_pred"] = bool(
            pass2_result.get("pred_answer_canon")
            and pass2_result["pred_answer_canon"] in gold_set
        )
        (
            pass2_result["soft_reward"],
            pass2_result["soft_reward_detail"],
        ) = rush_soft_match_reward(
            pass2_result["pred_answer"],
            ctx.example["solution"],
        )
        pass2_result["improved_over_pass1"] = (
            bool(pass2_result["is_correct_pred"])
            and not bool(pass1_result["is_correct_pred"])
        )

    return {
        "example_id": ctx.example["id"],
        "gold_answer": ctx.example["solution"],
        "gold_answer_canon_set": sorted(list(gold_set)),
        **build_two_pass_row_base(
            step=config.step,
            split_name=config.split_name,
            sample_idx=ctx.sample_idx,
            pass1=pass1_result,
            pass2=pass2_result,
        ),
    }


def _write_results_for_batch(
    batch_items: List[Dict[str, Any]],
    pass1: PassOutputs,
    pass2: Optional[PassOutputs],
    results_ctx: ResultsContext,
) -> None:
    """Write JSONL rows for a batch, only for missing sample indices."""
    with open(results_ctx.outpath, "a", encoding="utf-8") as handle:
        for batch_index, example in enumerate(batch_items):
            gold_set = _canon_rush_gold(example["solution"])
            for sample_idx in example["missing_indices"]:
                row_ctx = SampleRowContext(
                    example=example,
                    pass1=pass1,
                    pass2=pass2,
                    results_ctx=results_ctx,
                    batch_index=batch_index,
                    sample_idx=sample_idx,
                    gold_set=gold_set,
                )
                row = _build_row_for_sample(row_ctx)
                json.dump(row, handle, ensure_ascii=False)
                handle.write("\n")
                results_ctx.existing_by_example[example["id"]].add(sample_idx)


def _build_batch_items_for_range(
    examples,
    start_idx: int,
    batch_size: int,
    context: InferenceContext,
    existing_by_example: Dict[str, set[int]],
) -> List[Dict[str, Any]]:
    """Construct batch_items for a contiguous range of examples."""
    end_idx = min(start_idx + batch_size, len(examples))
    batch_ds = examples.select(range(start_idx, end_idx))

    batch_items: List[Dict[str, Any]] = []
    for offset, raw_example in enumerate(batch_ds):
        messages, solution = _norm_fields(
            raw_example,
            context.config.prompt_col,
            context.config.solution_col,
        )
        example_id = str(raw_example.get("id", f"idx_{start_idx + offset}"))
        have = existing_by_example.get(example_id, set())
        missing = [
            sample_idx
            for sample_idx in range(context.config.num_samples)
            if sample_idx not in have
        ]
        if not missing:
            continue
        batch_items.append(
            {
                "id": example_id,
                "messages": messages,
                "solution": solution,
                "missing_indices": missing,
            },
        )
    return batch_items


def _run_inference_on_split_core(
    examples,
    context: InferenceContext,
) -> None:
    """
    Run Rush Hour inference over a dataset, respecting existing results and
    filling missing sample indices per example up to `config.num_samples`.
    """
    outpath = os.path.join(
        context.config.output_dir,
        f"step{context.config.step:04d}_{context.config.split_name}.jsonl",
    )
    existing_by_example = _load_existing_example_index(outpath)
    logger.info(
        "→ %s | %d examples | resume: loaded %d existing example IDs",
        context.config.split_name,
        len(examples),
        len(existing_by_example),
    )

    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    for start_idx in range(0, len(examples), context.config.batch_size):
        batch_items = _build_batch_items_for_range(
            examples,
            start_idx,
            context.config.batch_size,
            context,
            existing_by_example,
        )
        if not batch_items:
            continue

        pass1_outputs, num_samples = _run_pass1_for_batch(batch_items, context)
        firstpass_choice = _build_first_pass_choice(
            batch_items,
            pass1_outputs.full_texts,
            num_samples,
            context.config,
        )

        cue_str = context.config.second_pass_phrase.strip() + " "
        if context.config.two_pass:
            second_pass_ctx = SecondPassContext(
                batch_items=batch_items,
                inference=context,
                firstpass_choice=firstpass_choice,
                num_samples=num_samples,
                cue_str=cue_str,
            )
            pass2_outputs = _run_pass2_for_batch(second_pass_ctx)
        else:
            total_rows = len(batch_items) * num_samples
            pass2_outputs = PassOutputs(
                full_texts=[""] * total_rows,
                ent_think=[[] for _ in range(total_rows)],
                ent_answer=[[] for _ in range(total_rows)],
                stop_reason_think=[""] * total_rows,
                stop_reason_answer=[""] * total_rows,
            )

        results_ctx = ResultsContext(
            outpath=outpath,
            inference=context,
            num_samples=num_samples,
            cue_str=cue_str,
            firstpass_choice=firstpass_choice,
            existing_by_example=existing_by_example,
        )
        _write_results_for_batch(
            batch_items=batch_items,
            pass1=pass1_outputs,
            pass2=pass2_outputs if context.config.two_pass else None,
            results_ctx=results_ctx,
        )


# ───────────────────── Inference Loop (public entrypoint) ─────────────────────


def run_inference_on_split(**kwargs) -> None:
    """Public wrapper matching the signature used by unified runners."""
    config_kwargs = build_math_inference_config_kwargs(
        batch_size=kwargs.get("batch_size", 8),
        num_samples=kwargs.get("num_samples", 1),
        temperature=kwargs.get("temperature", 0.0),
        top_p=kwargs.get("top_p", 0.95),
        entropy_mode=kwargs.get("entropy_mode", "reconsider"),
        eos_ids=kwargs.get("eos_ids"),
        two_pass=kwargs.get("two_pass", False),
        second_pass_phrase=kwargs.get(
            "second_pass_phrase",
            "Wait, we need to reconsider. Let's think this through step by step.",
        ),
        second_pass_use_sample_idx=kwargs.get("second_pass_use_sample_idx", 0),
        think_cap=kwargs.get("think_cap", 750),
        answer_cap=kwargs.get("answer_cap", 50),
    )
    config = CarparkInferenceConfig.from_flat(
        split_name=kwargs["split_name"],
        output_dir=kwargs["outdir"],
        step=kwargs["step"],
        prompt_col=kwargs.get("prompt_col", "messages"),
        solution_col=kwargs.get("solution_col", "solution"),
        **config_kwargs,
    )
    context = InferenceContext(
        tokenizer=kwargs["tokenizer"],
        model=kwargs["model"],
        config=config,
    )
    _run_inference_on_split_core(kwargs["examples"], context)


# ─────────────────────────── Dataset loader ───────────────────────────


def load_rush_dataset(
    dataset_id: str,
    split: str,
    cache_dir: str,
    prompt_col: str = "messages",
    solution_col: str = "solution",
):
    """Load Rush Hour dataset and ensure required columns are present."""
    _, load_dataset = require_datasets()
    dataset = load_dataset(
        dataset_id,
        split=split,
        cache_dir=cache_dir,
    )
    columns = set(dataset.column_names)
    if prompt_col not in columns or solution_col not in columns:
        raise ValueError(
            f"Dataset missing required columns: {prompt_col}, {solution_col}. "
            f"Found: {sorted(columns)}",
        )
    return dataset


# ─────────────────────────── Main CLI ───────────────────────────


def main(argv: Optional[List[str]] = None) -> None:
    """
    CLI entrypoint for carpark inference.

    Delegates to the shared unified carpark runner so CLI wiring stays centralized.
    """
    run_carpark_main(lambda: sys.modules[__name__], HFBackend, argv)


if __name__ == "__main__":
    main()
