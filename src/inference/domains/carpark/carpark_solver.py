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

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.inference.utils.carpark_rush_utils import rush_soft_match_reward
from src.inference.utils.common import (
    GenerationLimits,
    PassOutputs,
    add_token_and_tag_fields,
    build_entropy_pass_base,
    build_extra_pass_results_for_cues,
    build_math_inference_config_kwargs,
    build_second_pass_cue_strings,
    build_two_pass_row_base,
    empty_pass_outputs,
    extract_blocks as _extract_blocks,
    finite_mean as _finite_mean,
    repeat_for_samples as _repeat_for_samples,
    require_torch,
    require_transformers,
    run_generate_batch,
    setup_script_logger,
)
from src.inference.utils.math_pass_utils import DEFAULT_SECOND_PASS_PHRASE
from src.inference.utils.task_registry import CARPARK_SYSTEM_PROMPT

from .carpark_board import (
    _canon_rush_generic,
    _canon_rush_gold,
    _is_valid_rush,
)
from .carpark_data import (
    build_batch_items_for_range,
    load_existing_example_index,
)

torch = require_torch("carpark_core")
transformers_mod = require_transformers("carpark_core")
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


def chat_base_for_pass1_from_messages(tokenizer, messages: List[Dict[str, str]]) -> str:
    """
    Build a chat prompt for pass 1 from serialized messages.

    :param tokenizer: Chat tokenizer providing ``apply_chat_template``.
    :param messages: List of message dictionaries (orchestrated as role/content pairs).
    :returns: A formatted prompt string suitable for first-pass generation.
    """
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
    """
    Build a chat prompt for pass 2 that includes the pass-1 output and cue.

    :param tokenizer: Chat tokenizer providing ``apply_chat_template``.
    :param messages: Original conversation messages for the example.
    :param prev_output: First-pass reasoning and answer to show as assistant output.
    :param cue: Cue text encouraging reconsideration in the second pass.
    :returns: A formatted prompt string suitable for second-pass generation.
    """
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
    limits: GenerationLimits
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
            limits=GenerationLimits(
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
        """
        Dataset split name.

        :returns: Name of the dataset split (for example, ``\"train\"`` or ``\"test\"``).
        """
        return self.io_config.split_name

    @property
    def output_dir(self) -> str:
        """
        Output directory for results.

        :returns: Directory path where JSONL results are written.
        """
        return self.io_config.output_dir

    @property
    def step(self) -> int:
        """
        Step index used in output filenames.

        :returns: Integer training or checkpoint step identifier.
        """
        return self.io_config.step

    @property
    def prompt_col(self) -> str:
        """
        Name of the prompt column in the dataset.

        :returns: Column name containing chat messages or prompts.
        """
        return self.columns.prompt_col

    @property
    def solution_col(self) -> str:
        """
        Name of the solution/answer column in the dataset.

        :returns: Column name containing gold solutions.
        """
        return self.columns.solution_col

    @property
    def batch_size(self) -> int:
        """
        Generation batch size.

        :returns: Number of examples processed per generation batch.
        """
        return self.limits.batch_size

    @property
    def num_samples(self) -> int:
        """
        Number of samples per example.

        :returns: Number of generations drawn for each example.
        """
        return self.limits.num_samples

    @property
    def temperature(self) -> float:
        """
        Sampling temperature.

        :returns: Temperature used when sampling tokens.
        """
        return self.sampling.temperature

    @property
    def top_p(self) -> float:
        """
        Nucleus-sampling ``top_p`` value.

        :returns: Cumulative probability threshold for nucleus sampling.
        """
        return self.sampling.top_p

    @property
    def entropy_mode(self) -> str:
        """
        Entropy scoring mode.

        :returns: Name of the entropy computation mode.
        """
        return self.sampling.entropy_mode

    @property
    def eos_ids(self) -> Optional[List[int]]:
        """
        EOS token IDs used during generation.

        :returns: List of EOS token IDs, or ``None`` if not configured.
        """
        return self.sampling.eos_ids

    @property
    def two_pass(self) -> bool:
        """
        Whether second-pass generation is enabled.

        :returns: ``True`` if a reconsideration second pass is enabled.
        """
        return self.second_pass.two_pass

    @property
    def second_pass_phrase(self) -> str:
        """
        Cue phrase used to start the second pass.

        :returns: Phrase injected into second-pass prompts.
        """
        return self.second_pass.phrase

    @property
    def second_pass_use_sample_idx(self) -> int:
        """
        Pass-1 sample index that feeds pass 2.

        :returns: Zero-based index of the pass-1 sample highlighted in pass 2.
        """
        return self.second_pass.use_sample_idx

    @property
    def think_cap(self) -> int:
        """
        Maximum number of new tokens for the think phase.

        :returns: Token cap used when generating ``<think>`` content.
        """
        return self.limits.think_cap

    @property
    def answer_cap(self) -> int:
        """
        Maximum number of new tokens for the answer phase.

        :returns: Token cap used when generating ``<answer>`` content.
        """
        return self.limits.answer_cap

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
    return run_generate_batch(
        prefixes=prefixes,
        cap=cap,
        stop_strings=stop_strs,
        tokenizer=context.tokenizer,
        model=context.model,
        config_like=context.config,
        max_length=4096,
        torch_module=torch,
        stopping_criteria_list_cls=StoppingCriteriaList,
    )


@dataclass
class SampleRowPasses:
    """Per-row pass outputs used when building result rows."""

    pass1: PassOutputs
    pass2: Optional[PassOutputs]
    extra_passes: Optional[List[Tuple[str, PassOutputs]]]


@dataclass
class SampleRowContext:
    """Context required to build a single output row."""

    example: Dict[str, Any]
    results_ctx: ResultsContext
    batch_index: int
    sample_idx: int
    gold_set: set[str]
    passes: SampleRowPasses


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
        full_text=full_text,
        pred_answer_text=pred_answer_text,
        pred_canon=pred_canon,
        prev_output=meta.get("prev_output"),
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
    cue_phrase = ctx.cue_str.strip()
    return [
        chat_base_for_pass2_from_messages(
            ctx.inference.tokenizer,
            example["messages"],
            ctx.firstpass_choice[index],
            cue_phrase,
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

    pass1_outputs = ctx.passes.pass1
    pass2_outputs = ctx.passes.pass2
    extra_passes = ctx.passes.extra_passes

    pass1_result = _pack_pass_result(
        full_text=pass1_outputs.full_texts[row_index],
        ent_think=pass1_outputs.ent_think[row_index],
        ent_answer=pass1_outputs.ent_answer[row_index],
        meta={
            "injected_cue": False,
            "prev_output": None,
            "cue_prefix_str": "",
            "stop_reason_think": pass1_outputs.stop_reason_think[row_index],
            "stop_reason_answer": pass1_outputs.stop_reason_answer[row_index],
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
    if config.two_pass and pass2_outputs is not None:
        pass2_result = _pack_pass_result(
            full_text=pass2_outputs.full_texts[row_index],
            ent_think=pass2_outputs.ent_think[row_index],
            ent_answer=pass2_outputs.ent_answer[row_index],
            meta={
                "injected_cue": True,
                "prev_output": ctx.results_ctx.firstpass_choice[ctx.batch_index],
                "cue_prefix_str": ctx.results_ctx.cue_str,
                "stop_reason_think": pass2_outputs.stop_reason_think[row_index],
                "stop_reason_answer": pass2_outputs.stop_reason_answer[row_index],
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

    # Optional multi-cue reconsideration passes (pass2a / pass2b / pass2c).
    def _pack_extra_result(
        cue_str_extra: str,
        outputs_extra: PassOutputs,
    ) -> Dict[str, Any]:
        extra_res = _pack_pass_result(
            full_text=outputs_extra.full_texts[row_index],
            ent_think=outputs_extra.ent_think[row_index],
            ent_answer=outputs_extra.ent_answer[row_index],
            meta={
                "injected_cue": True,
                "prev_output": ctx.results_ctx.firstpass_choice[ctx.batch_index],
                "cue_prefix_str": cue_str_extra,
                "stop_reason_think": outputs_extra.stop_reason_think[row_index],
                "stop_reason_answer": outputs_extra.stop_reason_answer[row_index],
            },
        )
        extra_res["is_correct_pred"] = bool(
            extra_res.get("pred_answer_canon")
            and extra_res["pred_answer_canon"] in gold_set
        )
        (
            extra_res["soft_reward"],
            extra_res["soft_reward_detail"],
        ) = rush_soft_match_reward(
            extra_res["pred_answer"],
            ctx.example["solution"],
        )
        extra_res["improved_over_pass1"] = (
            bool(extra_res["is_correct_pred"])
            and not bool(pass1_result["is_correct_pred"])
        )
        return extra_res

    extra_pass_results: Dict[str, Dict[str, Any]] = build_extra_pass_results_for_cues(
        two_pass=config.two_pass,
        extra_passes=extra_passes,
        pack_result_for_extra=_pack_extra_result,
    )

    # For convenience, expose the main pass2 as pass2c when we have ≥3 cues.
    if config.two_pass and pass2_result is not None and len(extra_pass_results) >= 2:
        extra_pass_results.setdefault("pass2c", pass2_result)

    messages = ctx.example.get("messages")
    if isinstance(messages, list):
        problem_text = " ".join(str(m.get("content", "")) for m in messages)
    else:
        problem_text = str(messages)

    row_dict: Dict[str, Any] = {
        "example_id": ctx.example["id"],
        "problem": problem_text,
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
    for key in ("pass2a", "pass2b", "pass2c"):
        if key in extra_pass_results:
            row_dict[key] = extra_pass_results[key]

    return row_dict


def _write_results_for_batch(
    batch_items: List[Dict[str, Any]],
    pass1: PassOutputs,
    pass2: Optional[PassOutputs],
    results_ctx: ResultsContext,
    extra_passes: Optional[List[Tuple[str, PassOutputs]]] = None,
) -> None:
    """Write JSONL rows for a batch, only for missing sample indices."""
    with open(results_ctx.outpath, "a", encoding="utf-8") as handle:
        for batch_index, example in enumerate(batch_items):
            gold_set = _canon_rush_gold(example["solution"])
            for sample_idx in example["missing_indices"]:
                row_ctx = SampleRowContext(
                    example=example,
                    results_ctx=results_ctx,
                    batch_index=batch_index,
                    sample_idx=sample_idx,
                    gold_set=gold_set,
                    passes=SampleRowPasses(
                        pass1=pass1,
                        pass2=pass2,
                        extra_passes=extra_passes,
                    ),
                )
                row = _build_row_for_sample(row_ctx)
                json.dump(row, handle, ensure_ascii=False)
                handle.write("\n")
                results_ctx.existing_by_example[example["id"]].add(sample_idx)


def _compute_second_pass_outputs_for_carpark(
    *,
    context: InferenceContext,
    batch_items: List[Dict[str, Any]],
    firstpass_choice: List[str],
    num_samples: int,
) -> Tuple[List[str], Optional[PassOutputs], Optional[List[Tuple[str, PassOutputs]]]]:
    """
    Run optional second-pass reconsideration for a carpark batch.

    Returns:
        cue_strs: list of cue strings (with trailing spaces)
        main_pass2: the PassOutputs corresponding to the last cue (for pass2)
        extra_passes: list of (cue_str, PassOutputs) for earlier cues
    """
    cue_strs = build_second_pass_cue_strings(context.config.second_pass_phrase)
    extra_passes: List[Tuple[str, PassOutputs]] = []
    main_pass2: Optional[PassOutputs] = None

    if context.config.two_pass and cue_strs:
        for idx, cue_str in enumerate(cue_strs):
            second_pass_ctx = SecondPassContext(
                batch_items=batch_items,
                inference=context,
                firstpass_choice=firstpass_choice,
                num_samples=num_samples,
                cue_str=cue_str,
            )
            outputs_i = _run_pass2_for_batch(second_pass_ctx)
            if idx == len(cue_strs) - 1:
                main_pass2 = outputs_i
            else:
                extra_passes.append((cue_str, outputs_i))
    else:
        total_rows = len(batch_items) * num_samples
        main_pass2 = empty_pass_outputs(total_rows)

    return cue_strs, main_pass2, extra_passes


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
    existing_by_example = load_existing_example_index(outpath)
    logger.info(
        "→ %s | %d examples | resume: loaded %d existing example IDs",
        context.config.split_name,
        len(examples),
        len(existing_by_example),
    )

    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    for start_idx in range(0, len(examples), context.config.batch_size):
        batch_items = build_batch_items_for_range(
            examples=examples,
            start_idx=start_idx,
            batch_size=context.config.batch_size,
            prompt_col=context.config.prompt_col,
            solution_col=context.config.solution_col,
            num_samples=context.config.num_samples,
            existing_by_example=existing_by_example,
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

        cue_strs, main_pass2, extra_passes = _compute_second_pass_outputs_for_carpark(
            context=context,
            batch_items=batch_items,
            firstpass_choice=firstpass_choice,
            num_samples=num_samples,
        )

        results_ctx = ResultsContext(
            outpath=outpath,
            inference=context,
            num_samples=num_samples,
            cue_str=cue_strs[-1] if cue_strs else "",
            firstpass_choice=firstpass_choice,
            existing_by_example=existing_by_example,
        )
        _write_results_for_batch(
            batch_items=batch_items,
            pass1=pass1_outputs,
            pass2=main_pass2 if context.config.two_pass else None,
            results_ctx=results_ctx,
            extra_passes=extra_passes if extra_passes else None,
        )


# ───────────────────── Inference Loop (public entrypoint) ─────────────────────


def run_inference_on_split(**kwargs) -> None:
    """
    Public wrapper matching the signature used by unified runners.

    This function adapts flat keyword arguments from the unified CLI into a
    :class:`CarparkInferenceConfig` and runs the full two-pass inference loop.

    :param kwargs: Keyword-only arguments including ``split_name``, ``examples``,
        ``tokenizer``, ``model``, ``step``, ``outdir``, and common sampling
        options (``batch_size``, ``num_samples``, ``temperature``, ``top_p``,
        ``entropy_mode``, ``eos_ids``, ``two_pass``, ``second_pass_phrase``,
        ``second_pass_use_sample_idx``, ``think_cap``, ``answer_cap``).
    :returns: ``None``. Results are written to a JSONL file under ``outdir``.
    """
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
            DEFAULT_SECOND_PASS_PHRASE,
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


# Intentionally no CLI or dataset loader here; those live in
# :mod:`src.inference.domains.carpark.carpark_core` and
# :mod:`src.inference.domains.carpark.carpark_data`.
