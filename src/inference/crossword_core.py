#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Crossword inference core utilities and unified-runner entrypoint."""

import importlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from packaging import version
from src.inference.backends import HFBackend
from src.inference.common import (
    RECONSIDER_PATTERNS as _RECONSIDER_PATTERNS,
    StopOnSubstrings,
    add_token_and_tag_fields,
    build_entropy_pass_base,
    contains_canon as _contains_canon,
    decode_and_score_batch,
    extract_blocks as _extract_blocks,
    find_markers_and_context as _find_markers_and_context,
    finite_mean as _finite_mean,
    load_local_json_dataset,
    setup_script_logger,
    tokenize_prefixes_for_generate,
)
from src.inference.unified_runner_base import run_crossword_main

try:
    torch = importlib.import_module("torch")
except ImportError as exc:  # pragma: no cover - hard dependency
    raise RuntimeError(
        "crossword_core requires PyTorch; install the 'torch' package."
    ) from exc

try:
    transformers_mod = importlib.import_module("transformers")
except ImportError as exc:  # pragma: no cover - hard dependency
    raise RuntimeError(
        "crossword_core requires 'transformers'; install it to use this script."
    ) from exc

StoppingCriteriaList = transformers_mod.StoppingCriteriaList

# ───────────────────────── System prompt (CROSSWORD) ─────────────────────────
SYSTEM_PROMPT = """  You are an expert *cryptic-crossword solver*.

  Do this (repeat until fully consistent):

  A) DEVICE TRIAGE
    • List plausible devices from {anagram, container, reversal, hidden, charade,
      deletion, homophone, double def, &lit, letter selection, substitution, abbr}.
    • For each, quote the indicator word(s). Reject with a reason.

  B) PARSE
    • Mark the **definition** (start or end).
    • Mark the **wordplay** with exact fodder + operations.

  C) CHECKS
    • Enumeration must match exactly.
    • Letter accounting must be exact (anagram multiset or stepwise build).

  D) DECIDE
    • Pick the candidate best matching definition, indicator(s), and enumeration.
    • Do NOT assume anagram without a clear anagrind and fully used fodder.

  E) RECONSIDER (if any check fails)
    • Begin the next <think> with:
      "Wait, we need to reconsider. Let's think this through step by step."
    • Say why it failed, then re-run A–D with an alternative device/parse.

  FORMAT (no deviations):
    • Reasoning only in <think>…</think>
    • Final entry ONLY (UPPER-CASE) in <answer>…</answer>
  ------------------------------------------------------------
  HIDDEN
  Clue: Close, as seen in plaNET EARly (4)
  <think>Device: HIDDEN; indicator “as seen in”.
  Def: “Close”. Wordplay: hidden in “plaNET EARly” → NEAR.
  Enumeration: (4) OK.</think>
  <answer> NEAR </answer>

  Clue: Mix TEA for a hot drink (3)
  <think>Device: ANAGRAM; indicator “Mix”. Fodder TEA → TEA.
  Def: “a hot drink”. Accounting exact; (3) OK.</think>
  <answer> TEA </answer>

  Clue: Pet, when back, is a god (3)
  <think>Device: REVERSAL; indicator “when back”.
  Wordplay: GOD ← → DOG. Def: “Pet”. (3) OK.</think>
  <answer> DOG </answer>

  Clue: Animal, by the sound of “dear” (4)
  <think>Device triage: {homophone ✓ (“by the sound of”), hidden ✗, anagram ✗, …}
  Def: “Animal”. Wordplay: “dear” (sounds like) → DEER. Enumeration (4) OK.</think>
  <answer>DEER</answer>

  Clue: Shoe liner at home on fish (6)
  <think>Device triage: {hidden ? (“on” is not a hidden indicator), anagram ✗ (no anagrind),
  charade ✓ (“at home”=IN, “on”=next to), homophone ✗, …}
  Attempt (HIDDEN) rejected: no indicator; also hidden spans don’t give (6).
  Candidate attempt (wrong path): — fails enumeration/indicator, so we must rethink.
  Re-evaluate as CHARADES: IN (“at home”) + SOLE (“fish”) → INSOLE.
  Accounting: INSOLE letters: I N S O L E (6).
  Definition “Shoe liner” fits. Enumeration (6) OK.</think>
  <answer>INSOLE</answer>
"""

@dataclass
class CrosswordCapsConfig:
    """Caps and sampling counts for crossword inference."""

    batch_size: int = 8
    num_samples: int = 1
    think_cap: int = 750
    answer_cap: int = 50


@dataclass
class CrosswordSamplingConfig:
    """Sampling configuration for crossword generation."""

    temperature: float = 0.0
    top_p: float = 0.95
    entropy_mode: str = "reconsider"


@dataclass
class CrosswordTwoPassConfig:
    """Configuration for optional second-pass reconsideration."""

    enabled: bool = False
    phrase: str = (
        "Wait, we need to reconsider. Let's think this through step by step."
    )
    sample_index: int = 0


@dataclass
class CrosswordInferenceConfig:
    """Configuration for the crossword inference loop."""

    split_name: str
    output_dir: str
    step: int
    eos_ids: Optional[List[int]] = None
    caps: CrosswordCapsConfig = field(default_factory=CrosswordCapsConfig)
    sampling: CrosswordSamplingConfig = field(default_factory=CrosswordSamplingConfig)
    two_pass: CrosswordTwoPassConfig = field(default_factory=CrosswordTwoPassConfig)


# ───────────────────────── Utilities ─────────────────────────
# Crossword-friendly canon: casefold; strip spaces, hyphens, punctuation.
RE_PUNCT = re.compile(r"[^a-z0-9]", re.I)


def _canon_cross(text: Optional[str]) -> Optional[str]:
    """Canonicalize crossword answers for robust comparison."""
    if text is None:
        return None
    lowered = text.strip().lower()
    lowered = lowered.replace("–", "-").replace("—", "-")
    return RE_PUNCT.sub("", lowered)


def _compute_entropy_info(
    ent_think: List[float],
    ent_answer: List[float],
) -> Dict[str, Any]:
    """Aggregate entropy-related statistics for a single pass."""
    token_entropies = (ent_think or []) + (ent_answer or [])
    tokens_think = len(ent_think or [])
    tokens_answer = len(ent_answer or [])
    entropy_overall = _finite_mean(token_entropies) if token_entropies else None
    entropy_think = _finite_mean(ent_think) if ent_think else None
    entropy_answer = _finite_mean(ent_answer) if ent_answer else None
    return {
        "tok_ents_all": token_entropies,
        "tokens_think": tokens_think,
        "tokens_answer": tokens_answer,
        "entropy_overall": entropy_overall,
        "entropy_think": entropy_think,
        "entropy_answer": entropy_answer,
    }


def _find_reconsider_info(
    *,
    think_text: str,
    clue: str,
    injected_cue: bool,
    cue_prefix_str: str,
) -> Dict[str, Any]:
    """Locate reconsideration markers and their surrounding context."""
    skip_chars = len(cue_prefix_str) if injected_cue else 0
    markers, pos_in_think, reconsider_context, reconsider_excerpt = _find_markers_and_context(
        think_text,
        f"Clue: {clue}",
        _RECONSIDER_PATTERNS,
        skip_prefix_chars=skip_chars,
    )
    if injected_cue:
        markers = ["injected_cue"] + (markers or [])
    return {
        "markers": markers,
        "pos_in_think": pos_in_think,
        "reconsider_context": reconsider_context,
        "reconsider_excerpt": reconsider_excerpt,
    }


@dataclass
class BatchGenerationContext:
    """Lightweight wrapper for tokenizer/model/config used in _gen_batch."""

    tokenizer: Any
    model: Any
    config: CrosswordInferenceConfig


# ───────────────────────── Logging & DS patch ─────────────────────────
logger = setup_script_logger(__name__)

try:
    if version.parse(torch.__version__) >= version.parse("2.6.0"):
        torch_serialization = importlib.import_module("torch.serialization")
        deepspeed_zero_config = importlib.import_module("deepspeed.runtime.zero.config")
        deepspeed_loss_scaler = importlib.import_module("deepspeed.runtime.fp16.loss_scaler")
        add_safe_globals = getattr(torch_serialization, "add_safe_globals")
        zero_stage_enum_cls = getattr(deepspeed_zero_config, "ZeroStageEnum")
        loss_scaler_cls = getattr(deepspeed_loss_scaler, "LossScaler")
        add_safe_globals([zero_stage_enum_cls, loss_scaler_cls])
        logger.info("DeepSpeed ZeRO patch enabled")
except (ImportError, AttributeError) as exc:
    logger.warning("DeepSpeed patch disabled (missing deps/attrs): %r", exc)

# ───────────────────────── Prompt builders ─────────────────────────
def chat_base_for_pass1(tokenizer, clue: str, enumeration: Optional[str]) -> str:
    """Build the chat-formatted prompt for crossword pass 1."""
    enum_text = f" ({enumeration})" if enumeration else ""
    user_text = f"Clue: {clue}{enum_text}"
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def chat_base_for_pass2(
    tokenizer,
    clue: str,
    enumeration: Optional[str],
    prev_output: str,
    cue: str,
) -> str:
    """Build the chat-formatted prompt for crossword pass 2."""
    enum_text = f" ({enumeration})" if enumeration else ""
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Clue: {clue}{enum_text}"},
            {"role": "assistant", "content": prev_output},
            {"role": "user", "content": cue},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

def _make_gen_kwargs(cap: int, tokenizer, config: CrosswordInferenceConfig) -> Dict[str, Any]:
    """Build generate() kwargs for a given token cap and configuration.

    If temperature <= 0 → greedy (do_sample=False) and omit temperature/top_p.
    Else → sampling with provided temperature/top_p.
    """
    do_sample = (
        config.sampling.temperature is not None
        and float(config.sampling.temperature) > 0.0
    )
    kwargs: Dict[str, Any] = {
        "max_new_tokens": cap,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": config.eos_ids,
        "do_sample": do_sample,
        "return_dict_in_generate": True,
        "output_scores": config.sampling.entropy_mode != "none",
        "num_return_sequences": 1,
    }
    if do_sample:
        kwargs["temperature"] = float(config.sampling.temperature)
        if config.sampling.top_p is not None:
            kwargs["top_p"] = float(config.sampling.top_p)
    return kwargs


def _gen_batch(
    prefixes: List[str],
    cap: int,
    stop_strs: List[str],
    generation: BatchGenerationContext,
) -> Tuple[List[str], List[List[float]], Any, Any, List[str]]:
    """Generate a batch of continuations and token-entropy series."""
    inputs, input_lengths = tokenize_prefixes_for_generate(
        generation.tokenizer,
        prefixes,
        max_length=5500,
    )

    stop = (
        StoppingCriteriaList(
            [StopOnSubstrings(generation.tokenizer, stop_strs)],
        )
        if stop_strs
        else None
    )
    gen_kwargs = _make_gen_kwargs(cap, generation.tokenizer, generation.config)

    with torch.inference_mode():
        out = generation.model.generate(
            **inputs,
            **gen_kwargs,
            stopping_criteria=stop,
        )

    decoded_texts, entropy_series, stop_reasons = decode_and_score_batch(
        tokenizer=generation.tokenizer,
        sequences=out.sequences,
        scores=out.scores,
        input_lengths=input_lengths,
        stop_strings=stop_strs,
        cap=cap,
        eos_ids=generation.config.eos_ids,
        entropy_mode=generation.config.sampling.entropy_mode,
        model=generation.model,
    )

    return decoded_texts, entropy_series, input_lengths, out.sequences, stop_reasons


def _repeat_for_samples(values: List[str], num_samples: int) -> List[str]:
    """Repeat each prefix num_samples times (row-major expansion)."""
    return [value for value in values for _ in range(num_samples)]


def _norm_fields(example: dict):
    """Normalize raw record fields into (clue, answer, enumeration)."""
    clue = (
        example.get("clue")
        or example.get("problem")
        or example.get("question")
        or example.get("prompt")
    )
    gold = (
        example.get("answer")
        or example.get("target")
        or example.get("solution")
    )
    enumeration = (
        example.get("enumeration")
        or example.get("enum")
        or example.get("lengths")
    )
    if isinstance(enumeration, (list, tuple)):
        enumeration = " ".join(str(part) for part in enumeration)
    return clue, gold, enumeration


def _pack_pass_result(
    *,
    full_text: str,
    ent_think: List[float],
    ent_answer: List[float],
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble per-pass result dict with entropy and reconsideration markers."""
    think_text, answer = _extract_blocks(full_text)
    think_text = think_text or ""
    pred_answer_text = answer or ""

    reconsider_info = _find_reconsider_info(
        think_text=think_text,
        clue=meta["clue"],
        injected_cue=bool(meta.get("injected_cue")),
        cue_prefix_str=meta.get("cue_prefix_str", ""),
    )
    entropy_info = _compute_entropy_info(ent_think, ent_answer)

    pred_canon = _canon_cross(pred_answer_text)
    is_correct_pred = _contains_canon(pred_canon, meta.get("canon_gold"))

    base = build_entropy_pass_base(
        prev_output=meta.get("prev_output"),
        full_text=full_text,
        pred_answer_text=pred_answer_text,
        pred_canon=pred_canon,
        entropy_overall=entropy_info["entropy_overall"],
        entropy_think=entropy_info["entropy_think"],
        entropy_answer=entropy_info["entropy_answer"],
    )
    base.update(
        {
            "enumeration": meta.get("enumeration"),
            "stop_reason_think": meta.get("stop_reason_think"),
            "stop_reason_answer": meta.get("stop_reason_answer"),
            "has_reconsider_cue": bool(reconsider_info["markers"]),
            "reconsider_markers": reconsider_info["markers"] or [],
            "reconsider_pos": reconsider_info["pos_in_think"],
            "reconsider_context": reconsider_info["reconsider_context"],
            "reconsider_excerpt": reconsider_info["reconsider_excerpt"],
            "is_correct_pred": is_correct_pred,
            "is_correct_after_reconsideration": bool(reconsider_info["markers"])
            and bool(is_correct_pred),
        },
    )
    return add_token_and_tag_fields(
        base,
        tokens_total=len(entropy_info["tok_ents_all"]),
        tokens_think=entropy_info["tokens_think"],
        tokens_answer=entropy_info["tokens_answer"],
        full_text=full_text,
    )


def _scan_existing_problems(outpath: str) -> set[str]:
    """Scan an existing JSONL results file and recover seen problems."""
    seen: set[str] = set()
    if not os.path.exists(outpath):
        return seen
    with open(outpath, encoding="utf-8") as results_file:
        for line in results_file:
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            problem_text = parsed.get("problem")
            if isinstance(problem_text, str):
                seen.add(problem_text)
    return seen


def _build_batch_from_slice(batch_ds, seen: set[str]) -> List[Dict[str, Any]]:
    """Normalize a dataset slice into a batch of crossword examples."""
    batch: List[Dict[str, Any]] = []
    for raw_example in batch_ds:
        clue, gold, enumeration = _norm_fields(raw_example)
        if not clue or clue in seen:
            continue
        example = dict(raw_example)
        example["_clue"] = clue
        example["_gold"] = gold
        example["_enum"] = enumeration
        batch.append(example)
    return batch


def _run_first_pass_for_batch(
    batch: List[Dict[str, Any]],
    generation: BatchGenerationContext,
    caps: CrosswordCapsConfig,
) -> Tuple[
    List[str],
    List[List[float]],
    List[List[float]],
    List[str],
    List[str],
]:
    """Run the first think+answer pass for a batch."""
    total_rows = len(batch) * caps.num_samples

    base1 = [
        chat_base_for_pass1(generation.tokenizer, example["_clue"], example["_enum"])
        for example in batch
    ]
    pre1_think = _repeat_for_samples(
        [prompt + "<think>\n" for prompt in base1],
        caps.num_samples,
    )
    think1_texts, think1_ents, _, _, think1_stop = _gen_batch(
        pre1_think,
        caps.think_cap,
        ["</think>"],
        generation,
    )

    pre1_answer = [
        pre1_think[row_index] + think1_texts[row_index] + "</think>\n<answer>\n"
        for row_index in range(total_rows)
    ]
    answer1_texts, answer1_ents, _, _, answer1_stop = _gen_batch(
        pre1_answer,
        caps.answer_cap,
        ["</answer>"],
        generation,
    )
    pass1_full = [
        f"<think>{think1_texts[row_index]}</think>\n<answer>{answer1_texts[row_index]}</answer>"
        for row_index in range(total_rows)
    ]
    return pass1_full, think1_ents, answer1_ents, think1_stop, answer1_stop


def _compute_firstpass_choice(
    pass1_full: List[str],
    batch_size: int,
    num_samples: int,
    two_pass_cfg: CrosswordTwoPassConfig,
) -> List[str]:
    """Select which pass-1 sample to expose in the pass-2 cue."""
    choices: List[str] = []
    sample_index = max(0, min(two_pass_cfg.sample_index, num_samples - 1))
    for batch_index in range(batch_size):
        row_index = batch_index * num_samples + sample_index
        choices.append(pass1_full[row_index])
    return choices


@dataclass
class SecondPassInputs:
    """Inputs required to run the second pass for a batch."""

    batch: List[Dict[str, Any]]
    firstpass_choice: List[str]
    num_samples: int


def _run_second_pass_for_batch(
    inputs: SecondPassInputs,
    generation: BatchGenerationContext,
    caps: CrosswordCapsConfig,
    two_pass_cfg: CrosswordTwoPassConfig,
) -> Tuple[
    List[str],
    List[List[float]],
    List[List[float]],
    List[str],
    List[str],
]:
    """Run the second think+answer pass for a batch."""
    state: Dict[str, Any] = {
        "batch": inputs.batch,
        "firstpass_choice": inputs.firstpass_choice,
        "num_samples": inputs.num_samples,
    }
    state["total_rows"] = len(state["batch"]) * state["num_samples"]
    state["cue_str"] = two_pass_cfg.phrase.strip() + " "

    state["base2"] = [
        chat_base_for_pass2(
            generation.tokenizer,
            example["_clue"],
            example["_enum"],
            state["firstpass_choice"][batch_index],
            two_pass_cfg.phrase.strip(),
        )
        for batch_index, example in enumerate(state["batch"])
    ]
    state["pre2_think"] = _repeat_for_samples(
        [prompt + "<think>\n" + state["cue_str"] for prompt in state["base2"]],
        state["num_samples"],
    )
    (
        state["think2_texts_only"],
        state["think2_ents"],
        _,
        _,
        state["think2_stop"],
    ) = _gen_batch(
        state["pre2_think"],
        caps.think_cap,
        ["</think>"],
        generation,
    )
    think2_texts = [
        state["cue_str"] + text for text in state["think2_texts_only"]
    ]

    state["pre2_answer"] = [
        state["pre2_think"][row_index]
        + state["think2_texts_only"][row_index]
        + "</think>\n<answer>\n"
        for row_index in range(state["total_rows"])
    ]
    (
        state["answer2_texts"],
        state["answer2_ents"],
        _,
        _,
        state["answer2_stop"],
    ) = _gen_batch(
        state["pre2_answer"],
        caps.answer_cap,
        ["</answer>"],
        generation,
    )
    pass2_full = [
        (
            f"<think>{think2_texts[row_index]}</think>\n"
            f"<answer>{state['answer2_texts'][row_index]}</answer>"
        )
        for row_index in range(state["total_rows"])
    ]
    return (
        pass2_full,
        state["think2_ents"],
        state["answer2_ents"],
        state["think2_stop"],
        state["answer2_stop"],
    )


def _write_results_for_batch(
    *,
    batch: List[Dict[str, Any]],
    firstpass_choice: List[str],
    results: Dict[str, Any],
    context: Dict[str, Any],
) -> None:
    """Write per-example results for a batch to disk."""
    for batch_index, example in enumerate(batch):
        canon_gold = _canon_cross(example["_gold"])
        for sample_idx in range(context["num_samples"]):
            row_index = batch_index * context["num_samples"] + sample_idx
            pass1_meta = {
                "clue": example["_clue"],
                "enumeration": example["_enum"],
                "injected_cue": False,
                "canon_gold": canon_gold,
                "prev_output": None,
                "cue_prefix_str": "",
                "stop_reason_think": results["think1_stop"][row_index],
                "stop_reason_answer": results["answer1_stop"][row_index],
            }
            pass1 = _pack_pass_result(
                full_text=results["pass1_full"][row_index],
                ent_think=results["think1_ents"][row_index],
                ent_answer=results["answer1_ents"][row_index],
                meta=pass1_meta,
            )
            pass2: Optional[Dict[str, Any]] = None
            if context["config"].two_pass.enabled:
                pass2_meta = {
                    "clue": example["_clue"],
                    "enumeration": example["_enum"],
                    "injected_cue": True,
                    "canon_gold": canon_gold,
                    "prev_output": firstpass_choice[batch_index],
                    "cue_prefix_str": context["config"].two_pass.phrase.strip(),
                    "stop_reason_think": results["think2_stop"][row_index],
                    "stop_reason_answer": results["answer2_stop"][row_index],
                }
                pass2 = _pack_pass_result(
                    full_text=results["pass2_full"][row_index],
                    ent_think=results["think2_ents"][row_index],
                    ent_answer=results["answer2_ents"][row_index],
                    meta=pass2_meta,
                )
                pass2["improved_over_pass1"] = bool(
                    pass2.get("is_correct_pred"),
                ) and not bool(pass1.get("is_correct_pred"))

            row = {
                "problem": example["_clue"],
                "gold_answer": example["_gold"],
                "gold_answer_canon": canon_gold,
                "enumeration": example["_enum"],
                "step": context["config"].step,
                "split": context["config"].split_name,
                "sample_idx": sample_idx,
                "pass1": pass1,
                "pass2": pass2,
            }
            with open(context["outpath"], "a", encoding="utf-8") as out_file:
                json.dump(row, out_file, ensure_ascii=False)
                out_file.write("\n")
        context["seen"].add(example["_clue"])


def _process_batch(
    *,
    batch_ds,
    seen: set[str],
    generation: BatchGenerationContext,
    config: CrosswordInferenceConfig,
    outpath: str,
) -> None:
    """Process a single dataset slice: run passes and write outputs."""
    batch = _build_batch_from_slice(batch_ds, seen)
    if not batch:
        return
    first_pass_results = dict(
        zip(
            (
                "pass1_full",
                "think1_ents",
                "answer1_ents",
                "think1_stop",
                "answer1_stop",
            ),
            _run_first_pass_for_batch(
                batch=batch,
                generation=generation,
                caps=config.caps,
            ),
        ),
    )

    firstpass_choice = _compute_firstpass_choice(
        pass1_full=first_pass_results["pass1_full"],
        batch_size=len(batch),
        num_samples=config.caps.num_samples,
        two_pass_cfg=config.two_pass,
    )

    if config.two_pass.enabled:
        second_inputs = SecondPassInputs(
            batch=batch,
            firstpass_choice=firstpass_choice,
            num_samples=config.caps.num_samples,
        )
        second_pass_results = dict(
            zip(
                (
                    "pass2_full",
                    "think2_ents",
                    "answer2_ents",
                    "think2_stop",
                    "answer2_stop",
                ),
                _run_second_pass_for_batch(
                    inputs=second_inputs,
                    generation=generation,
                    caps=config.caps,
                    two_pass_cfg=config.two_pass,
                ),
            ),
        )
    else:
        total_rows = len(batch) * config.caps.num_samples
        second_pass_results = {
            "pass2_full": [""] * total_rows,
            "think2_ents": [[] for _ in range(total_rows)],
            "answer2_ents": [[] for _ in range(total_rows)],
            "think2_stop": [""] * total_rows,
            "answer2_stop": [""] * total_rows,
        }

    results_state: Dict[str, Any] = {
        **first_pass_results,
        **second_pass_results,
    }
    output_context = {
        "config": config,
        "outpath": outpath,
        "num_samples": config.caps.num_samples,
        "seen": seen,
    }
    _write_results_for_batch(
        batch=batch,
        firstpass_choice=firstpass_choice,
        results=results_state,
        context=output_context,
    )


# ───────────────────── Inference Loop ─────────────────────
def run_inference_on_split(
    examples,
    tokenizer,
    model,
    config: CrosswordInferenceConfig,
) -> None:
    """Run crossword inference over a dataset and save JSONL results."""
    outpath = os.path.join(
        config.output_dir,
        f"step{config.step:04d}_{config.split_name}.jsonl",
    )
    seen = _scan_existing_problems(outpath)

    logger.info(
        "→ %s | %d examples (skipping %d already done)",
        config.split_name,
        len(examples),
        len(seen),
    )

    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    generation = BatchGenerationContext(
        tokenizer=tokenizer,
        model=model,
        config=config,
    )

    for idx_start in range(0, len(examples), config.caps.batch_size):
        idx_end = min(idx_start + config.caps.batch_size, len(examples))
        batch_ds = examples.select(range(idx_start, idx_end))
        _process_batch(
            batch_ds=batch_ds,
            seen=seen,
            generation=generation,
            config=config,
            outpath=outpath,
        )

def load_crossword_local(dataset_path: str):
    """Load crossword examples from a local JSONL file."""
    logger.info("Loading CROSSWORD JSONL from: %s", dataset_path)
    return load_local_json_dataset(dataset_path)

# ───────────────────────── Main ─────────────────────────
def main(argv: Optional[List[str]] = None) -> None:
    """Legacy CLI entrypoint for crossword_core.

    Delegates to the shared unified crossword runner so CLI wiring stays centralized.
    """
    run_crossword_main(lambda: sys.modules[__name__], HFBackend, argv)

if __name__ == "__main__":
    main()
