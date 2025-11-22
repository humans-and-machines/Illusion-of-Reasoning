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

import re
from importlib import import_module
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, TYPE_CHECKING

from src.inference.backends import _load_torch_and_transformers
from src.inference.utils.common import (
    DEFAULT_SECOND_PASS_PHRASE,
    PassOutputs,
    SamplingConfigBase,
    build_math_pass_meta,
    build_second_pass_think_prefixes,
    empty_pass_outputs,
    extract_problem_and_answer,
    finite_mean as _finite_mean,
    pack_math_pass_result as _pack_math_pass_result,
    run_generate_batch,
    scan_existing_pass1_results,
    setup_script_logger,
)
from src.inference.utils.task_registry import MATH_SYSTEM_PROMPT

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
    """
    Sampling and generation controls shared by math inference loops.

    :param batch_size: Number of examples to process per generation batch.
    :param num_samples: Number of samples to draw per problem.
    """

    batch_size: int = 8
    num_samples: int = 1


@dataclass
class MathTwoPassConfig:
    """
    Configuration of the optional second pass.

    :param enabled: Whether to run a second reconsideration pass.
    :param phrase: Cue phrase injected into second-pass prompts.
    :param use_sample_idx: Preferred pass-1 sample index to surface in pass 2.
    :param think_cap: Maximum number of new tokens to generate for ``<think>``.
    :param answer_cap: Maximum number of new tokens to generate for ``<answer>``.
    """

    enabled: bool = False
    phrase: str = DEFAULT_SECOND_PASS_PHRASE
    use_sample_idx: int = 0
    think_cap: int = 750
    answer_cap: int = 50


class MathInferenceConfig:
    """
    Configuration for the two-pass math inference loop.

    This wraps sampling-related options and second-pass settings while exposing a
    legacy, attribute-based interface for callers.

    :param split_name: Dataset split name (for example, ``\"test\"`` or ``\"validation\"``).
    :param output_dir: Directory where JSONL results will be written.
    :param step: Training or checkpoint step identifier used in the output filename.
    :param kwargs: Additional configuration options such as ``batch_size``,
        ``num_samples``, ``temperature``, ``top_p``, ``entropy_mode``,
        ``eos_ids``, ``two_pass``, ``second_pass_phrase``,
        ``second_pass_use_sample_idx``, ``think_cap``, and ``answer_cap``.
    """

    def __init__(
        self,
        *,
        split_name: str,
        output_dir: str,
        step: int,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a :class:`MathInferenceConfig` from high-level arguments.

        :param split_name: Dataset split name (for example, ``\"test\"`` or ``\"validation\"``).
        :param output_dir: Directory where JSONL results will be written.
        :param step: Training or checkpoint step identifier used in filenames.
        :param kwargs: Additional configuration options such as ``batch_size``,
            ``num_samples``, ``temperature``, ``top_p``, ``entropy_mode``,
            ``eos_ids``, ``two_pass``, ``second_pass_phrase``,
            ``second_pass_use_sample_idx``, ``think_cap``, and ``answer_cap``.
        :raises TypeError: If unexpected keyword arguments are provided.
        """
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
                DEFAULT_SECOND_PASS_PHRASE,
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
        """
        Batch size used for inference.

        :returns: Number of examples processed per generation batch.
        """
        return self.sampling.batch_size

    @property
    def num_samples(self) -> int:
        """
        Number of samples to generate per problem.

        :returns: Number of samples drawn for each problem.
        """
        return self.sampling.num_samples

    @property
    def temperature(self) -> float:
        """
        Sampling temperature for generation.

        :returns: Temperature used when sampling tokens.
        """
        return self.sampling.temperature

    @property
    def top_p(self) -> float:
        """
        Top-p nucleus sampling parameter.

        :returns: Cumulative probability threshold for nucleus sampling.
        """
        return self.sampling.top_p

    @property
    def entropy_mode(self) -> str:
        """
        Controls how token-level entropy is computed or disabled.

        :returns: Name of the entropy computation mode.
        """
        return self.sampling.entropy_mode

    @property
    def eos_ids(self) -> Optional[List[int]]:
        """
        Token IDs that should trigger EOS during generation.

        :returns: List of token IDs treated as end-of-sequence markers, if any.
        """
        return self.sampling.eos_ids

    @property
    def two_pass(self) -> bool:
        """
        Whether the reconsideration second pass is enabled.

        :returns: ``True`` if the second pass should be run, otherwise ``False``.
        """
        return self.two_pass_cfg.enabled

    @property
    def second_pass_phrase(self) -> str:
        """
        Cue phrase injected at the start of second-pass think blocks.

        :returns: Phrase inserted into second-pass prompts before model reasoning.
        """
        return self.two_pass_cfg.phrase

    @property
    def second_pass_use_sample_idx(self) -> int:
        """
        Preferred sample index from pass 1 to surface in pass 2.

        :returns: Zero-based index of the pass-1 sample to highlight.
        """
        return self.two_pass_cfg.use_sample_idx

    @property
    def think_cap(self) -> int:
        """
        Maximum number of newly generated tokens for ``<think>``.

        :returns: Token cap used when generating ``<think>`` content.
        """
        return self.two_pass_cfg.think_cap

    @property
    def answer_cap(self) -> int:
        """
        Maximum number of newly generated tokens for ``<answer>``.

        :returns: Token cap used when generating ``<answer>`` content.
        """
        return self.two_pass_cfg.answer_cap

# ───────────────────────── Logging ─────────────────────────
logger = setup_script_logger(__name__)


# ───────────────────────── Prompt builders (WITH system msg) ─────────────────────────
def chat_base_for_pass1(tokenizer, problem: str) -> str:
    """
    Build the base chat prompt for pass 1.

    The resulting prompt includes the system message and a single user turn
    containing the math problem.

    :param tokenizer: Chat tokenizer providing ``apply_chat_template``.
    :param problem: Raw problem text to inject into the user turn.
    :returns: A formatted prompt string suitable for generation.
    """
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Problem: {problem}"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def chat_base_for_pass2(tokenizer, problem: str, prev_output: str, cue: str) -> str:
    """
    Build the base chat prompt for pass 2.

    The prompt includes the problem, a prior assistant reasoning trace, and a
    user-provided cue that will seed the new ``<think>`` block.

    :param tokenizer: Chat tokenizer providing ``apply_chat_template``.
    :param problem: Raw problem text to inject into the user turn.
    :param prev_output: Prior assistant reasoning and answer to show as context.
    :param cue: Cue text that encourages reconsideration in the second pass.
    :returns: A formatted prompt string suitable for second-pass generation.
    """
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
    """
    Scan an existing JSONL results file to support resume/fill behavior.

    :param results_path: Path to the JSONL file containing prior results.
    :returns: A pair ``(existing_samples, existing_pass1)`` where
        ``existing_samples`` maps normalized problems to the set of seen sample
        indices and ``existing_pass1`` maps ``(problem, sample_idx)`` pairs to
        full pass-1 outputs.
    """
    return scan_existing_pass1_results(results_path)


@dataclass
class BatchSpec:
    """
    Specification for a batch generation call.

    :param prefixes: Prefix strings to feed into the model.
    :param cap: Maximum number of new tokens to generate per prefix.
    :param stop_strs: Stop strings that terminate generation when produced.
    """

    prefixes: List[str]
    cap: int
    stop_strs: List[str]


@dataclass
class MathInferenceContext:
    """
    Bundle tokenizer, model, and configuration for math generation helpers.

    :param tokenizer: Tokenizer used to encode prompts and decode outputs.
    :param model: Underlying language model used for generation.
    :param config: High-level math inference configuration.
    """

    tokenizer: Any
    model: Any
    config: MathInferenceConfig


@dataclass
class BatchLayout:
    """
    Mapping from generated rows back to original work items and samples.

    :param work_items: Per-example dictionaries containing normalized fields.
    :param row_to_ex_idx: For each generated row, index of the source example.
    :param row_target_sample_idx: For each row, target sample index for that example.
    """

    work_items: List[Dict[str, Any]]
    row_to_ex_idx: List[int]
    row_target_sample_idx: List[int]


@dataclass
class ExistingPassState:
    """
    Track existing pass-1 samples found on disk for resume.

    :param existing_samples: Mapping from problem to the set of completed sample indices.
    :param existing_pass1: Mapping from ``(problem, sample_idx)`` to full pass-1 text.
    """

    existing_samples: DefaultDict[str, set]
    existing_pass1: Dict[tuple, str]


@dataclass
class TwoPassBatchOutputs:
    """
    Container combining first- and optional second-pass outputs.

    :param pass1: First-pass outputs, always present.
    :param pass2: Second-pass outputs if ``two_pass`` is enabled, otherwise ``None``.
    """

    pass1: PassOutputs
    pass2: Optional[PassOutputs]


@dataclass
class BatchWriteContext:
    """
    Context object holding shared state for result writing.

    :param outpath: Path to the JSONL results file being written.
    :param config: Inference configuration used for this run.
    :param cue_strs: Sequence of cue strings used for extra reconsideration passes.
    :param existing_state: Snapshot of existing pass-1 samples on disk.
    :param firstpass_choice_text_per_ex: Chosen pass-1 text per example for pass 2.
    """

    outpath: str
    config: MathInferenceConfig
    cue_strs: List[str]
    existing_state: ExistingPassState
    firstpass_choice_text_per_ex: List[str]


@dataclass
class SecondPassInputs:
    """
    Inputs required to run the optional second pass.

    :param layout: Layout mapping rows back to examples and sample indices.
    :param pre1_think: Prefixes used for first-pass ``<think>`` generations.
    :param firstpass_choice_text_per_ex: Chosen pass-1 full text per example.
    :param cue_str: Cue string to inject into second-pass reasoning.
    """

    layout: BatchLayout
    pre1_think: List[str]
    firstpass_choice_text_per_ex: List[str]
    cue_str: str


def _gen_batch(
    batch_spec: BatchSpec,
    context: MathInferenceContext,
) -> Tuple[List[str], List[List[float]], Any, Any, List[str]]:
    """
    Generate a batch of continuations and token-entropy series.

    :param batch_spec: Specification of prefixes, caps, and stop strings.
    :param context: Generation context containing tokenizer, model, and config.
    :returns: A tuple ``(texts, entropies, model_outputs, scores, stop_reasons)`` where
        ``texts`` are decoded continuations, ``entropies`` are per-token entropy
        values, and the remaining elements hold backend-specific outputs and
        per-row stop reasons.
    """
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
    """
    Construct per-problem work items for a dataset slice.

    Existing results are respected: only missing sample indices per problem
    (relative to ``config.num_samples``) are scheduled for generation.

    :param examples_slice: Slice of the dataset to process.
    :param existing_samples: Mapping from problem to the set of completed sample indices.
    :param config: Inference configuration used to determine number of samples.
    :returns: A list of work-item dictionaries, each annotated with normalized
        problem/answer fields and outstanding sample indices.
    """
    work_items: List[Dict[str, Any]] = []
    for raw_example in examples_slice:
        problem, gold = _norm_fields(raw_example)
        # Skip records without a usable problem/answer pair. Tests expect
        # examples lacking an explicit answer field to be ignored here.
        if not problem or gold is None:
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


def _pack_pass_result(
    *,
    problem: str,
    full_text: str,
    ent_think: List[float],
    ent_answer: List[float],
    injected_cue: bool,
    canon_gold: Optional[str],
    prev_output: Optional[str],
    cue_prefix_str: str,
    stop_reason_think: Optional[str],
    stop_reason_answer: Optional[str],
) -> Dict[str, Any]:
    """
    Convenience wrapper around :func:`pack_math_pass_result`.

    This preserves the older, more explicit keyword-based interface used in
    tests while delegating to the shared packing helpers in
    :mod:`src.inference.utils.math_pass_utils`.
    """
    meta = build_math_pass_meta(
        problem=problem,
        canon_gold=canon_gold,
        injected_cue=injected_cue,
        prev_output=prev_output,
        cue_prefix_str=cue_prefix_str,
        stop_reason_think=stop_reason_think,
        stop_reason_answer=stop_reason_answer,
    )
    return _pack_math_pass_result(
        full_text=full_text,
        ent_think=ent_think,
        ent_answer=ent_answer,
        meta=meta,
    )


def _build_pass1_prefixes(
    work_items: List[Dict[str, Any]],
    tokenizer,
) -> Tuple[List[str], List[int], List[int]]:
    """
    Build pass-1 ``<think>`` prefixes and row-to-example/sample mappings.

    :param work_items: Normalized work items with outstanding sample indices.
    :param tokenizer: Chat tokenizer providing ``apply_chat_template``.
    :returns: A tuple ``(prefixes, row_to_ex_idx, row_target_sample_idx)`` where
        ``prefixes`` are the think-prefix prompts and the index arrays map rows
        back to their originating examples and sample indices.
    """
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
    """Run pass-1 think and answer generations for a batch."""
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
) -> Tuple[PassOutputs, BatchLayout, List[str]]:
    """Run pass-1 (think + answer) for a batch of math problems."""
    pre1_think, row_to_ex_idx, row_target_sample_idx = _build_pass1_prefixes(
        work_items,
        context.tokenizer,
    )
    outputs = _run_pass1_generations(pre1_think, context)
    layout = BatchLayout(
        work_items=work_items,
        row_to_ex_idx=row_to_ex_idx,
        row_target_sample_idx=row_target_sample_idx,
    )
    return outputs, layout, pre1_think


@dataclass
class FirstPassChoiceInputs:
    """
    Inputs required to choose a representative pass-1 sample per example.

    :param layout: Batch layout describing the relationship between rows and examples.
    :param existing_state: Snapshot of existing pass-1 results on disk.
    :param new_pass1_by_ex_and_sample: Mapping from ``(example_idx, sample_idx)``
        to newly generated full pass-1 text.
    :param pass1_full_texts: All pass-1 full texts for the current batch.
    :param config: Inference configuration controlling sample selection.
    """

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
    """Choose which pass-1 output to surface in the cue for one example."""
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
    """Select which pass-1 sample feeds pass 2 for each example."""
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
    """Run pass 2 (think + answer) for a batch of math problems."""
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
    """Normalize raw record fields into a (problem, gold_answer) pair."""
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


def _build_extra_pass_results_for_row(
    *,
    row_index: int,
    prob: str,
    canon_gold: Optional[str],
    layout: BatchLayout,
    context: BatchWriteContext,
    extra_passes: Optional[List[Tuple[str, PassOutputs]]],
    pass1_result: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
    """
    Thin wrapper delegating to the implementation in ``math_core_runner``.
    """
    math_core_runner = import_module("src.inference.domains.math.math_core_runner")

    return math_core_runner.build_extra_pass_results_for_row(
        row_index=row_index,
        prob=prob,
        canon_gold=canon_gold,
        layout=layout,
        context=context,
        extra_passes=extra_passes,
        pass1_result=pass1_result,
    )


def _write_results_for_batch(
    *,
    layout: BatchLayout,
    outputs: TwoPassBatchOutputs,
    context: BatchWriteContext,
    extra_passes: Optional[List[Tuple[str, PassOutputs]]] = None,
) -> None:
    """
    Thin wrapper delegating to the implementation in ``math_core_runner``.
    """
    math_core_runner = import_module("src.inference.domains.math.math_core_runner")

    math_core_runner.write_results_for_batch(
        layout=layout,
        outputs=outputs,
        context=context,
        extra_passes=extra_passes,
    )


def _compute_second_pass_outputs(
    *,
    context: MathInferenceContext,
    layout: BatchLayout,
    pre1_think: List[str],
    firstpass_choice_text_per_ex: List[str],
) -> Tuple[Optional[PassOutputs], Optional[List[Tuple[str, PassOutputs]]], List[str]]:
    """
    Thin wrapper delegating to the implementation in ``math_core_runner``.
    """
    math_core_runner = import_module("src.inference.domains.math.math_core_runner")

    return math_core_runner.compute_second_pass_outputs(
        context=context,
        layout=layout,
        pre1_think=pre1_think,
        firstpass_choice_text_per_ex=firstpass_choice_text_per_ex,
    )


def _run_inference_batch(
    *,
    slice_ds,
    context: MathInferenceContext,
    outpath: str,
    existing_state: ExistingPassState,
) -> None:
    """
    Thin wrapper delegating to the implementation in ``math_core_runner``.
    """
    math_core_runner = import_module("src.inference.domains.math.math_core_runner")

    math_core_runner.run_inference_batch(
        slice_ds=slice_ds,
        context=context,
        outpath=outpath,
        existing_state=existing_state,
    )


# ───────────────────── Inference Loop (two-phase per pass) ─────────────────────
def run_inference_on_split(
    examples,  # datasets.Dataset
    tokenizer,
    model,
    config: MathInferenceConfig,
) -> None:
    """
    Run math inference over a dataset split.

    This wrapper delegates to :mod:`src.inference.domains.math.math_core_runner` while
    preserving the original public entry point.
    """
    math_core_runner = import_module("src.inference.domains.math.math_core_runner")

    math_core_runner.run_inference_on_split(
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
    Thin wrapper delegating to the MATH-500 loader in :mod:`math_core_runner`.
    """
    math_core_runner = import_module("src.inference.domains.math.math_core_runner")

    return math_core_runner.load_math500(
        cache_dir=cache_dir,
        split=split,
        seed=seed,
        dataset_path=dataset_path,
    )
