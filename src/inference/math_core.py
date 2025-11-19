#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-pass batch inference for Qwen2.5-style chat LMs that produce:
  <think> ... </think><answer> ... </answer>

Now with resume/fill:
- If a results file exists (step####_<split>.jsonl), the script checks how many
  samples exist per problem and ONLY generates the missing ones up to
  --num_samples (e.g., 8). It preserves prior rows and appends the new ones.

Other features (unchanged)
--------------------------
- Uses your system prompt for BOTH passes (first message in the chat).
- Two-phase generation per pass (think → answer) with explicit stops.
- Per-pass hard caps: think <= 750, answer <= 50 new tokens.
- Pass 2: shows Pass-1 output as an assistant turn, user supplies a cue,
  and the new <think> starts with the cue (we prefill the cue inside <think>).
- Correctness: gold counted correct if it appears ANYWHERE inside <answer>
  after canonicalization (substring check).
- Cue-robust analytics: injected cue at the start of pass-2 <think> does NOT
  trigger “reconsider” detection.
- EOS includes <|im_end|> and <|endoftext|>.
- NaN-safe token entropy; optional "reconsider" slicing.
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, TYPE_CHECKING
from src.inference.backends import _load_torch_and_transformers
from src.inference.common import (
    PassOutputs,
    SamplingConfigBase,
    build_math_pass_meta,
    build_second_pass_think_prefixes,
    canon_math as _canon_math,
    empty_pass_outputs,
    extract_problem_and_answer,
    finite_mean as _finite_mean,
    load_local_json_dataset,
    pack_math_pass_result as _pack_pass_result,
    require_datasets,
    run_generate_batch,
    scan_existing_pass1_results,
    setup_script_logger,
)
from src.inference.task_registry import MATH_SYSTEM_PROMPT

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch
    from torch.nn import functional as F  # type: ignore[unused-import]
    from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList

# ───────────────────────── System prompt ─────────────────────────
SYSTEM_PROMPT = MATH_SYSTEM_PROMPT

# ───────────────────────── Utilities ─────────────────────────
_fmean = _finite_mean


@dataclass
class MathSamplingConfig(SamplingConfigBase):
    """Sampling / generation controls shared by math inference loops."""

    batch_size: int = 8
    num_samples: int = 1


@dataclass
class MathTwoPassConfig:
    """Configuration of the optional second pass."""

    enabled: bool = False
    phrase: str = (
        "Wait, we need to reconsider. Let's think this through step by step."
    )
    use_sample_idx: int = 0
    think_cap: int = 750
    answer_cap: int = 50


class MathInferenceConfig:
    """Configuration for the two-pass math inference loop."""

    def __init__(
        self,
        *,
        split_name: str,
        output_dir: str,
        step: int,
        **kwargs: Any,
    ) -> None:
        self.split_name = split_name
        self.output_dir = output_dir
        self.step = step
        self.sampling = MathSamplingConfig(
            batch_size=kwargs.pop("batch_size", 8),
            num_samples=kwargs.pop("num_samples", 1),
            temperature=kwargs.pop("temperature", 0.0),
            top_p=kwargs.pop("top_p", 0.95),
            entropy_mode=kwargs.pop("entropy_mode", "reconsider"),
            eos_ids=kwargs.pop("eos_ids", None),
        )
        self.two_pass_cfg = MathTwoPassConfig(
            enabled=kwargs.pop("two_pass", False),
            phrase=kwargs.pop(
                "second_pass_phrase",
                "Wait, we need to reconsider. Let's think this through step by step.",
            ),
            use_sample_idx=kwargs.pop("second_pass_use_sample_idx", 0),
            think_cap=kwargs.pop("think_cap", 750),
            answer_cap=kwargs.pop("answer_cap", 50),
        )
        if kwargs:
            raise TypeError(f"Unexpected MathInferenceConfig kwargs: {sorted(kwargs.keys())}")

    # Convenience properties mirroring the legacy flat attributes -----------------
    @property
    def batch_size(self) -> int:
        """Batch size used for inference."""
        return self.sampling.batch_size

    @property
    def num_samples(self) -> int:
        """Number of samples to generate per problem."""
        return self.sampling.num_samples

    @property
    def temperature(self) -> float:
        """Sampling temperature for generation."""
        return self.sampling.temperature

    @property
    def top_p(self) -> float:
        """Top-p nucleus sampling parameter."""
        return self.sampling.top_p

    @property
    def entropy_mode(self) -> str:
        """Controls how token-level entropy is computed or disabled."""
        return self.sampling.entropy_mode

    @property
    def eos_ids(self) -> Optional[List[int]]:
        """List of token ids that should trigger EOS during generation."""
        return self.sampling.eos_ids

    @property
    def two_pass(self) -> bool:
        """Whether the reconsideration second pass is enabled."""
        return self.two_pass_cfg.enabled

    @property
    def second_pass_phrase(self) -> str:
        """Cue phrase injected at the start of second-pass think blocks."""
        return self.two_pass_cfg.phrase

    @property
    def second_pass_use_sample_idx(self) -> int:
        """Preferred sample index from pass 1 to surface in pass 2."""
        return self.two_pass_cfg.use_sample_idx

    @property
    def think_cap(self) -> int:
        """Maximum number of newly generated tokens for <think>."""
        return self.two_pass_cfg.think_cap

    @property
    def answer_cap(self) -> int:
        """Maximum number of newly generated tokens for <answer>."""
        return self.two_pass_cfg.answer_cap

# ───────────────────────── Logging ─────────────────────────
logger = setup_script_logger(__name__)


# ───────────────────────── Prompt builders (WITH system msg) ─────────────────────────
def chat_base_for_pass1(tokenizer, problem: str) -> str:
    """Build the base chat prompt for pass 1 (problem only)."""
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Problem: {problem}"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

def chat_base_for_pass2(tokenizer, problem: str, prev_output: str, cue: str) -> str:
    """Build the base chat prompt for pass 2 (problem + prior reasoning + cue)."""
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Problem: {problem}"},
            {"role": "assistant", "content": prev_output},
            {"role": "user", "content": cue},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

# ─────────────────────────── Results scanning ───────────────────────────
# >>> RESUME/FILL LOGIC <<<
def _scan_existing_results(results_path: str) -> tuple[DefaultDict[str, set], Dict[tuple, str]]:
    """Wrapper around shared scan_existing_pass1_results helper."""
    return scan_existing_pass1_results(results_path)


@dataclass
class BatchSpec:
    """Specification for a batch generation call."""

    prefixes: List[str]
    cap: int
    stop_strs: List[str]


@dataclass
class MathInferenceContext:
    """Bundle tokenizer, model, and config for generation helpers."""

    tokenizer: Any
    model: Any
    config: MathInferenceConfig


@dataclass
class BatchLayout:
    """Mapping from generated rows back to original work items and samples."""

    work_items: List[Dict[str, Any]]
    row_to_ex_idx: List[int]
    row_target_sample_idx: List[int]


@dataclass
class ExistingPassState:
    """Track existing pass-1 samples found on disk for resume."""

    existing_samples: DefaultDict[str, set]
    existing_pass1: Dict[tuple, str]


@dataclass
class TwoPassBatchOutputs:
    """Container combining first- and optional second-pass outputs."""

    pass1: PassOutputs
    pass2: Optional[PassOutputs]


@dataclass
class BatchWriteContext:
    """Context object holding shared state for result writing."""

    outpath: str
    config: MathInferenceConfig
    cue_str: str
    existing_state: ExistingPassState
    firstpass_choice_text_per_ex: List[str]


@dataclass
class SecondPassInputs:
    """Inputs required to run the optional second pass."""

    layout: BatchLayout
    pre1_think: List[str]
    firstpass_choice_text_per_ex: List[str]
    cue_str: str


def _gen_batch(
    batch_spec: BatchSpec,
    context: MathInferenceContext,
) -> Tuple[List[str], List[List[float]], Any, Any, List[str]]:
    """Generate a batch of continuations and token-entropy series."""
    torch_mod, _, _, stopping_criteria_list_cls = _load_torch_and_transformers()
    return run_generate_batch(
        prefixes=batch_spec.prefixes,
        cap=batch_spec.cap,
        stop_strings=batch_spec.stop_strs,
        tokenizer=context.tokenizer,
        model=context.model,
        config_like=context.config,
        max_length=4096,
        torch_module=torch_mod,
        stopping_criteria_list_cls=stopping_criteria_list_cls,
    )


def _build_work_items_for_slice(
    examples_slice,
    existing_samples: Dict[str, set],
    config: MathInferenceConfig,
) -> List[Dict[str, Any]]:
    """Construct per-problem work items for a dataset slice."""
    work_items: List[Dict[str, Any]] = []
    for raw_example in examples_slice:
        problem, gold = _norm_fields(raw_example)
        if not problem:
            continue
        have = existing_samples.get(problem, set())
        if len(have) >= config.num_samples:
            continue
        missing_indices = [
            sample_idx
            for sample_idx in range(config.num_samples)
            if sample_idx not in have
        ]
        if not missing_indices:
            continue
        example_copy = dict(raw_example)
        example_copy["_normalized_problem"] = problem
        example_copy["_normalized_gold"] = gold
        example_copy["_todo_samples"] = missing_indices
        work_items.append(example_copy)
    return work_items


def _build_pass1_prefixes(
    work_items: List[Dict[str, Any]],
    tokenizer,
) -> Tuple[List[str], List[int], List[int]]:
    """Build pass-1 <think> prefixes and row→example/sample index mappings."""
    base_prompts = [
        chat_base_for_pass1(tokenizer, item["_normalized_problem"])
        for item in work_items
    ]

    pre1_think: List[str] = []
    row_to_ex_idx: List[int] = []
    row_target_sample_idx: List[int] = []
    for ex_idx, work_item in enumerate(work_items):
        for sample_idx in work_item["_todo_samples"]:
            pre1_think.append(base_prompts[ex_idx] + "<think>\n")
            row_to_ex_idx.append(ex_idx)
            row_target_sample_idx.append(sample_idx)
    return pre1_think, row_to_ex_idx, row_target_sample_idx


def _run_pass1_generations(
    pre1_think: List[str],
    context: MathInferenceContext,
) -> PassOutputs:
    """Run pass-1 think and answer generations for a prepared prefix batch."""
    config = context.config
    think1_texts, think1_ents, _, _, think1_stop = _gen_batch(
        BatchSpec(prefixes=pre1_think, cap=config.think_cap, stop_strs=["</think>"]),
        context,
    )

    pre1_answer: List[str] = []
    for pre_think, think_text in zip(pre1_think, think1_texts):
        prefix = pre_think + think_text + "</think>\n<answer>\n"
        pre1_answer.append(prefix)
    answer1_texts, answer1_ents, _, _, answer1_stop = _gen_batch(
        BatchSpec(prefixes=pre1_answer, cap=config.answer_cap, stop_strs=["</answer>"]),
        context,
    )

    full_texts = [
        f"<think>{think_text}</think>\n<answer>{answer_text}</answer>"
        for think_text, answer_text in zip(think1_texts, answer1_texts)
    ]

    return PassOutputs(
        full_texts=full_texts,
        ent_think=think1_ents,
        ent_answer=answer1_ents,
        stop_reason_think=think1_stop,
        stop_reason_answer=answer1_stop,
    )


def _run_pass1_for_batch(
    work_items: List[Dict[str, Any]],
    context: MathInferenceContext,
) -> Tuple[PassOutputs, List[int], List[int], List[str]]:
    """Run pass-1 (think + answer) for a batch of math problems."""
    pre1_think, row_to_ex_idx, row_target_sample_idx = _build_pass1_prefixes(
        work_items,
        context.tokenizer,
    )
    outputs = _run_pass1_generations(pre1_think, context)
    return outputs, row_to_ex_idx, row_target_sample_idx, pre1_think


@dataclass
class FirstPassChoiceInputs:
    """Inputs required to choose a representative pass-1 sample per example."""

    layout: BatchLayout
    existing_state: ExistingPassState
    new_pass1_by_ex_and_sample: Dict[Tuple[int, int], str]
    pass1_full_texts: List[str]
    config: MathInferenceConfig


def _index_new_pass1_by_example_and_sample(
    pass1_full_texts: List[str],
    row_to_ex_idx: List[int],
    row_target_sample_idx: List[int],
) -> Dict[Tuple[int, int], str]:
    """Build an index mapping (example_idx, sample_idx) to pass-1 full text."""
    mapping: Dict[Tuple[int, int], str] = {}
    for row_index, full_text in enumerate(pass1_full_texts):
        example_idx = row_to_ex_idx[row_index]
        sample_idx = row_target_sample_idx[row_index]
        mapping[(example_idx, sample_idx)] = full_text
    return mapping


def _select_first_pass_choice(
    problem: str,
    ex_idx: int,
    inputs: FirstPassChoiceInputs,
) -> str:
    """Choose which pass-1 output to show in the pass-2 cue for one example."""
    config = inputs.config
    existing_state = inputs.existing_state
    mapping = inputs.new_pass1_by_ex_and_sample

    num_samples = max(1, int(config.num_samples))
    k_choice = max(0, min(config.second_pass_use_sample_idx, num_samples - 1))

    prev_text = existing_state.existing_pass1.get((problem, k_choice))
    if prev_text is not None:
        return prev_text

    prev_text = mapping.get((ex_idx, k_choice))
    if prev_text is not None:
        return prev_text

    have_sorted = sorted(existing_state.existing_samples.get(problem, set()))
    if have_sorted:
        prev_text = existing_state.existing_pass1.get((problem, have_sorted[0]))
        if prev_text is not None:
            return prev_text

    new_indices = sorted(
        sample_idx
        for (example_idx, sample_idx) in mapping
        if example_idx == ex_idx
    )
    if new_indices:
        prev_text = mapping.get((ex_idx, new_indices[0]))
        if prev_text is not None:
            return prev_text

    for row_index, full_text in enumerate(inputs.pass1_full_texts):
        if inputs.layout.row_to_ex_idx[row_index] == ex_idx:
            return full_text

    return ""


def _build_first_pass_choice(
    *,
    layout: BatchLayout,
    pass1_full_texts: List[str],
    existing_state: ExistingPassState,
    config: MathInferenceConfig,
) -> List[str]:
    """Select which pass-1 sample feeds pass-2 per example."""
    if not config.two_pass:
        return [""] * len(layout.work_items)

    new_pass1_by_ex_and_idx = _index_new_pass1_by_example_and_sample(
        pass1_full_texts,
        layout.row_to_ex_idx,
        layout.row_target_sample_idx,
    )
    choice_inputs = FirstPassChoiceInputs(
        layout=layout,
        existing_state=existing_state,
        new_pass1_by_ex_and_sample=new_pass1_by_ex_and_idx,
        pass1_full_texts=pass1_full_texts,
        config=config,
    )

    choice_texts: List[str] = []
    for ex_idx, work_item in enumerate(layout.work_items):
        problem = work_item["_normalized_problem"]
        prev_text = _select_first_pass_choice(problem, ex_idx, choice_inputs)
        choice_texts.append(prev_text or "")
    return choice_texts


def _build_second_pass_base_prompts(
    *,
    tokenizer,
    work_items: List[Dict[str, Any]],
    firstpass_choice_text_per_ex: List[str],
    phrase: str,
) -> List[str]:
    """Build base chat prompts for pass 2 for each example."""
    return [
        chat_base_for_pass2(
            tokenizer,
            work_item["_normalized_problem"],
            firstpass_choice_text_per_ex[ex_idx],
            phrase,
        )
        for ex_idx, work_item in enumerate(work_items)
    ]


def _run_second_pass_generations(
    *,
    context: MathInferenceContext,
    pre2_think: List[str],
) -> Tuple[List[str], List[List[float]], List[str], List[str], List[List[float]], List[str]]:
    """Run second-pass think and answer generations for prepared prefixes."""
    config = context.config
    think2_texts_only_new, think2_ents, _, _, think2_stop = _gen_batch(
        BatchSpec(prefixes=pre2_think, cap=config.think_cap, stop_strs=["</think>"]),
        context,
    )

    pre2_answer = [
        pre_think + think_new + "</think>\n<answer>\n"
        for pre_think, think_new in zip(pre2_think, think2_texts_only_new)
    ]
    answer2_texts, answer2_ents, _, _, answer2_stop = _gen_batch(
        BatchSpec(prefixes=pre2_answer, cap=config.answer_cap, stop_strs=["</answer>"]),
        context,
    )
    return (
        think2_texts_only_new,
        think2_ents,
        think2_stop,
        answer2_texts,
        answer2_ents,
        answer2_stop,
    )


def _run_pass2_for_batch(
    *,
    context: MathInferenceContext,
    second_pass_inputs: SecondPassInputs,
) -> PassOutputs:
    """Run pass-2 (think + answer) for a batch of math problems."""
    config = context.config
    if not config.two_pass:
        total_rows = len(second_pass_inputs.pre1_think)
        return empty_pass_outputs(total_rows)

    phrase = config.second_pass_phrase.strip()
    base2_per_ex = _build_second_pass_base_prompts(
        tokenizer=context.tokenizer,
        work_items=second_pass_inputs.layout.work_items,
        firstpass_choice_text_per_ex=second_pass_inputs.firstpass_choice_text_per_ex,
        phrase=phrase,
    )
    pre2_think = build_second_pass_think_prefixes(
        base2_per_ex=base2_per_ex,
        pre1_think=second_pass_inputs.pre1_think,
        row_to_ex_idx=second_pass_inputs.layout.row_to_ex_idx,
        cue_str=second_pass_inputs.cue_str,
    )
    (
        think2_texts_only_new,
        think2_ents,
        think2_stop,
        answer2_texts,
        answer2_ents,
        answer2_stop,
    ) = _run_second_pass_generations(
        context=context,
        pre2_think=pre2_think,
    )
    think2_texts = [second_pass_inputs.cue_str + text for text in think2_texts_only_new]

    full_texts = [
        f"<think>{think_text}</think>\n<answer>{answer_text}</answer>"
        for think_text, answer_text in zip(think2_texts, answer2_texts)
    ]

    return PassOutputs(
        full_texts=full_texts,
        ent_think=think2_ents,
        ent_answer=answer2_ents,
        stop_reason_think=think2_stop,
        stop_reason_answer=answer2_stop,
    )


def _norm_fields(example: dict) -> Tuple[Optional[str], Optional[str]]:
    """Normalize raw record fields into (problem, gold_answer)."""
    problem, gold = extract_problem_and_answer(example)
    if gold and not any(
        key in example for key in ("answer", "final_answer", "target", "boxed_answer", "solution")
    ):
        match = re.search(r"\\boxed\{([^}]*)\}", str(gold))
        if not match:
            match = re.search(r"\\boxed\(([^)]*)\)", str(gold))
        if match:
            gold = match.group(1)
    return problem, gold


def _write_results_for_batch(
    *,
    layout: BatchLayout,
    outputs: TwoPassBatchOutputs,
    context: BatchWriteContext,
) -> None:
    """Write pass-1/2 results for a batch to JSONL."""
    with open(context.outpath, "a", encoding="utf-8") as outfile:
        for row_index, full_row in enumerate(outputs.pass1.full_texts):
            sample_idx = layout.row_target_sample_idx[row_index]
            item = layout.work_items[layout.row_to_ex_idx[row_index]]
            prob = item["_normalized_problem"]
            gold = item["_normalized_gold"]
            canon_gold = _canon_math(gold)

            pass1_result = _pack_pass_result(
                full_text=full_row,
                ent_think=outputs.pass1.ent_think[row_index],
                ent_answer=outputs.pass1.ent_answer[row_index],
                meta=build_math_pass_meta(
                    problem=prob,
                    canon_gold=canon_gold,
                    injected_cue=False,
                    prev_output=None,
                    cue_prefix_str="",
                    stop_reason_think=outputs.pass1.stop_reason_think[row_index],
                    stop_reason_answer=outputs.pass1.stop_reason_answer[row_index],
                ),
            )

            pass2_result = None
            if context.config.two_pass and outputs.pass2 is not None:
                pass2_result = _pack_pass_result(
                    full_text=outputs.pass2.full_texts[row_index],
                    ent_think=outputs.pass2.ent_think[row_index],
                    ent_answer=outputs.pass2.ent_answer[row_index],
                    meta=build_math_pass_meta(
                        problem=prob,
                        canon_gold=canon_gold,
                        injected_cue=True,
                        prev_output=context.firstpass_choice_text_per_ex[
                            layout.row_to_ex_idx[row_index]
                        ],
                        cue_prefix_str=context.cue_str,
                        stop_reason_think=outputs.pass2.stop_reason_think[row_index],
                        stop_reason_answer=outputs.pass2.stop_reason_answer[row_index],
                    ),
                )
                pass2_result["improved_over_pass1"] = bool(
                    pass2_result.get("is_correct_pred"),
                ) and not bool(
                    pass1_result.get("is_correct_pred"),
                )

            json.dump(
                {
                    "problem": prob,
                    "gold_answer": gold,
                    "gold_answer_canon": canon_gold,
                    "step": context.config.step,
                    "split": context.config.split_name,
                    "sample_idx": sample_idx,
                    "pass1": pass1_result,
                    "pass2": pass2_result,
                },
                outfile,
                ensure_ascii=False,
            )
            outfile.write("\n")

            context.existing_state.existing_samples.setdefault(prob, set()).add(sample_idx)
            context.existing_state.existing_pass1[(prob, sample_idx)] = pass1_result["output"]

    filled = sum(len(item["_todo_samples"]) for item in layout.work_items)
    logger.info(
        "Filled %d missing samples across %d problems in this batch.",
        filled,
        len(layout.work_items),
    )


def _run_inference_batch(
    *,
    slice_ds,
    context: MathInferenceContext,
    outpath: str,
    existing_state: ExistingPassState,
) -> None:
    """Run inference for a single dataset slice and append JSONL rows."""
    work_items = _build_work_items_for_slice(
        slice_ds,
        existing_state.existing_samples,
        context.config,
    )
    if not work_items:
        return

    pass1_outputs, row_to_ex_idx, row_target_sample_idx, pre1_think = _run_pass1_for_batch(
        work_items,
        context,
    )
    layout = BatchLayout(
        work_items=work_items,
        row_to_ex_idx=row_to_ex_idx,
        row_target_sample_idx=row_target_sample_idx,
    )

    cue_str = context.config.second_pass_phrase.strip() + " "
    firstpass_choice_text_per_ex = _build_first_pass_choice(
        layout=layout,
        pass1_full_texts=pass1_outputs.full_texts,
        existing_state=existing_state,
        config=context.config,
    )

    pass2_outputs = _run_pass2_for_batch(
        context=context,
        second_pass_inputs=SecondPassInputs(
            layout=layout,
            pre1_think=pre1_think,
            firstpass_choice_text_per_ex=firstpass_choice_text_per_ex,
            cue_str=cue_str,
        ),
    )

    write_context = BatchWriteContext(
        outpath=outpath,
        config=context.config,
        cue_str=cue_str,
        existing_state=existing_state,
        firstpass_choice_text_per_ex=firstpass_choice_text_per_ex,
    )
    _write_results_for_batch(
        layout=layout,
        outputs=TwoPassBatchOutputs(
            pass1=pass1_outputs,
            pass2=pass2_outputs if context.config.two_pass else None,
        ),
        context=write_context,
    )


# ───────────────────── Inference Loop (two-phase per pass) ─────────────────────
def run_inference_on_split(
    examples,  # datasets.Dataset
    tokenizer,
    model,
    config: MathInferenceConfig,
) -> None:
    """
    Run math inference over a dataset, respecting existing results and filling
    missing sample indices per problem up to `config.num_samples`.
    """

    if (config.temperature is None or float(config.temperature) == 0.0) and config.num_samples > 1:
        logger.warning(
            "temperature=0 with num_samples=%d → all samples will be identical (greedy).",
            config.num_samples,
        )

    outpath = os.path.join(
        config.output_dir,
        f"step{config.step:04d}_{config.split_name}.jsonl",
    )
    existing_samples, existing_pass1 = _scan_existing_results(outpath)
    logger.info("Resume scan: %d problems already present", len(existing_samples))

    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    existing_state = ExistingPassState(
        existing_samples=existing_samples,
        existing_pass1=existing_pass1,
    )
    context = MathInferenceContext(
        tokenizer=tokenizer,
        model=model,
        config=config,
    )

    for start_idx in range(0, len(examples), config.batch_size):
        end_idx = min(start_idx + config.batch_size, len(examples))
        slice_ds = examples.select(range(start_idx, end_idx))
        _run_inference_batch(
            slice_ds=slice_ds,
            context=context,
            outpath=outpath,
            existing_state=existing_state,
        )

def load_math500(cache_dir: str, split: str, seed: int, dataset_path: Optional[str] = None):
    """Load MATH-500 (or a competition-math fallback) and normalize fields."""
    _, load_dataset = require_datasets()

    if dataset_path:
        logger.info("Loading MATH-500 from local file: %s", dataset_path)
        return load_local_json_dataset(dataset_path)

    candidates = [
        "HuggingFaceH4/MATH-500",
        "AI-MO/MATH-500",
        "lighteval/MATH-500",
        "openai/math-500",
        "TIGER-Lab/MATH-500",
    ]
    for repo in candidates:
        try:
            logger.info("Trying remote MATH-500 candidate: %s", repo)
            ds_full = load_dataset(repo, split=split, cache_dir=cache_dir)
            colnames = set(ds_full.column_names)

            def _norm(example):
                problem, answer = extract_problem_and_answer(example)
                return {"problem": problem, "answer": answer}

            normalized_ds = ds_full.map(_norm, remove_columns=list(colnames))
            normalized_ds = normalized_ds.filter(
                lambda row: row["problem"] is not None and row["answer"] is not None,
            )
            if len(normalized_ds) == 0:
                raise ValueError(f"{repo} contained no usable (problem,answer) pairs")
            logger.info("Loaded MATH-500 from %s | N=%d", repo, len(normalized_ds))
            return normalized_ds
        except (OSError, ValueError, RuntimeError) as load_exc:
            logger.warning("Skipping %s (%r)", repo, load_exc)

    try:
        ds_full = load_dataset("hendrycks/competition_math", split=split, cache_dir=cache_dir)
        max_examples = min(500, len(ds_full))
        return ds_full.shuffle(seed=seed).select(range(max_examples))
    except (OSError, RuntimeError) as load_exc:
        raise RuntimeError(f"Could not load MATH-500 or fallback dataset: {load_exc}") from load_exc
