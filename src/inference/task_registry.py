#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task registry for inference scripts.

This keeps task-specific knobs (prompts, stop tokens, caps, canonicalizers,
and dataset loaders) in one place so individual entrypoints can become thin
wrappers over a unified driver.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from src.inference.common import OPENR1_PROMPT_TEMPLATE


def _resolve_callable(path: Optional[str]) -> Optional[Callable]:
    """Import dotted-path 'module:attr' lazily and return the referenced callable."""
    if not path:
        return None
    if ":" not in path:
        raise ValueError(f"Callable path must look like 'module:attr', got {path}")
    module_name, attr_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


@dataclass
class DatasetSpec:
    """Dataset loader specification for a task."""

    loader: str  # dotted path to a loader function
    default_id: Optional[str] = None
    prompt_column: Optional[str] = None
    answer_column: Optional[str] = None
    split: str = "test"

    def loader_fn(self) -> Callable:
        """Resolve and return the loader function described by this spec."""
        loader_func = _resolve_callable(self.loader)
        if loader_func is None:
            raise ValueError(f"Could not resolve dataset loader: {self.loader}")
        return loader_func


@dataclass
class TaskSpec:
    """Configuration for a logical inference task (prompt, caps, dataset, etc.)."""

    name: str
    config: Dict[str, Any]
    dataset: Optional[DatasetSpec] = None
    notes: str = ""

    @property
    def system_prompt(self) -> Optional[str]:
        """System prompt string for the task, if configured."""
        return self.config.get("system_prompt")

    @property
    def stop_think(self) -> List[str]:
        """Stop strings for the think phase."""
        return self.config.get("stop_think", [])

    @property
    def stop_answer(self) -> List[str]:
        """Stop strings for the answer phase."""
        return self.config.get("stop_answer", [])

    @property
    def two_pass(self) -> bool:
        """Whether the task uses a two-pass protocol."""
        return bool(self.config.get("two_pass", False))

    @property
    def think_cap(self) -> Optional[int]:
        """Maximum think tokens, if applicable."""
        return self.config.get("think_cap")

    @property
    def answer_cap(self) -> Optional[int]:
        """Maximum answer tokens, if applicable."""
        return self.config.get("answer_cap")

    @property
    def max_output_tokens(self) -> Optional[int]:
        """Maximum total output tokens for single-pass APIs."""
        return self.config.get("max_output_tokens")

    def canon_pred_fn(self) -> Optional[Callable]:
        """Return the canonicalization function for predictions, if configured."""
        return _resolve_callable(self.config.get("canonicalize_pred"))

    def canon_gold_fn(self) -> Optional[Callable]:
        """Return the canonicalization function for gold answers, if configured."""
        return _resolve_callable(self.config.get("canonicalize_gold"))


# ----------------------- Prompts -----------------------
MATH_SYSTEM_PROMPT = """You are an expert *mathematics problem-solver*.

  Every time you receive a problem you must:
  • Analyse it thoroughly.
    – Pinpoint the **goal** (what quantity/set/form is requested).
    – Pinpoint the **givens/constraints** (domains, integrality, non-negativity, geometric conditions).
    – Choose the **methods** to apply (algebraic manipulation, factorization, inequalities, counting, modular arithmetic, geometry, calculus, etc.).
    – Write out the full derivation that leads to the final result.

  • Check that the result satisfies all original constraints (no extraneous roots, correct domain, simplified form, exact arithmetic).

  • Respond in **exactly** the tag-based format shown below – no greeting, no commentary outside the tags.
    – The final answer goes inside `<answer>` **only**.
    – Use **exact** math (fractions, radicals, π, e). Avoid unnecessary decimals.
    – Canonical forms: integers as plain numbers; reduced fractions a/b with b>0; simplified radicals; rationalized denominators; sets/tuples with standard notation; intervals in standard notation.
    – If there is **no solution**, write `NO SOLUTION`. If the problem is **underdetermined**, write `I DON'T KNOW`.

  • You have a hard cap of **750 output tokens**. Be concise but complete.

  ------------------------------------------------------------
  TAG TEMPLATE (copy this shape for every problem)
  <think>
  YOUR reasoning process goes here:
  1. quote the relevant bits of the problem
  2. name the mathematical tool(s) you apply
  3. show each intermediate step until the result is reached
  </think>
  <answer>
  THEANSWER
  </answer>
"""

CROSSWORD_SYSTEM_PROMPT = """  You are an expert *cryptic-crossword solver*.

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
    • Begin the next <think> with: "Wait, we need to reconsider. Let's think this through step by step."
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
  Accounting: INSOLE letters: I N S O L E (6). Definition “Shoe liner” fits. Enumeration (6) OK.</think>
  <answer>INSOLE</answer>
"""

CARPARK_SYSTEM_PROMPT = (
    "You are an expert Rush Hour ({N}×{N}) solver.\n"
    "INPUTS\n"
    "• Board (row-major string with 'o','A','B'..'Z', optional 'x')\n"
    "• Board size (e.g., 4×4/5×5/6×6)\n"
    "• Optimum moves {moves}\n"
    "OUTPUT\n"
    "• Exactly ONE optimal sequence in <answer> only.\n"
    "• Token = <PIECE><DIR><STEPS> (e.g., A>2,B<1,Cv3)\n"
    "• DIR: '<' left, '>' right, '^' up, 'v' down\n"
    "• No spaces/prose/extra lines in <answer>.\n"
    "GOAL\n"
    "• Right end of 'A' reaches the right edge.\n"
    "OPTIMALITY & TIE-BREAK\n"
    "• Use exactly {moves} tokens.\n"
    "• If multiple optimal, choose lexicographically smallest ASCII comma-list.\n"
    "VALIDATION\n"
    "• REGEX: ^[A-Z][<>^v]\\d+(,[A-Z][<>^v]\\d+)*$\n"
    "• AXES: A is 2-long horizontal; others length 2/3, fixed H or V.\n"
    "• COLLISION: no overlaps; 'x' is immovable.\n"
    "• GOAL: applying all tokens reaches the goal.\n"
    "• LENGTH: #tokens = {moves} (when provided).\n"
    "RETHINK\n"
    "1) In <think>, propose S1 and run all VALIDATION checks.\n"
    "2) If any fail, name the failure, propose S2, re-check.\n"
    "3) If any fail, propose S3, re-check. Repeat as needed.\n"
    "4) Put ONLY the final passing sequence in <answer>.\n"
    "EXAMPLE (guidance only)\n"
    "• Board 4×4: oAABCooBCoooDDoo, {moves}=2\n"
    "<think>\n"
    "S1: A>1 → GOAL✗ (blocked by B). Wait, hang on\n"
    "S2: Bv2,A>1 → all ✓.\n"
    "</think>\n"
    "<answer>\n"
    "Bv2,A>1\n"
    "</answer>\n"
)

# ----------------------- Registry -----------------------
TASK_REGISTRY: Dict[str, TaskSpec] = {
    # Math (open-source HF models; two-pass optional)
    "math-qwen": TaskSpec(
        name="math-qwen",
        config={
            "system_prompt": MATH_SYSTEM_PROMPT,
            "stop_think": ["</think>"],
            "stop_answer": ["</answer>"],
            "two_pass": True,
            "think_cap": 750,
            "answer_cap": 50,
            "canonicalize_pred": "src.inference.common:canon_math",
            "canonicalize_gold": "src.inference.common:canon_math",
        },
        dataset=DatasetSpec(
            loader="src.inference.math_core:load_math500",
            default_id="MATH-500",
            prompt_column="problem",
            answer_column="answer",
            split="test",
        ),
        notes="Qwen-style chat formatting with resume/fill; supports two-pass cue.",
    ),
    # Math (Llama checkpoints via ZeRO)
    "math-llama": TaskSpec(
        name="math-llama",
        config={
            "system_prompt": MATH_SYSTEM_PROMPT,
            "stop_think": ["</think>"],
            "stop_answer": ["</answer>"],
            "two_pass": True,
            "think_cap": 750,
            "answer_cap": 50,
            "canonicalize_pred": "src.inference.common:canon_math",
            "canonicalize_gold": "src.inference.common:canon_math",
        },
        dataset=DatasetSpec(
            loader="src.inference.math_llama_core:load_math500",
            default_id="MATH-500",
            prompt_column="problem",
            answer_column="answer",
            split="test",
        ),
        notes="ZeRO-3 loader with two-pass support and resume/fill.",
    ),
    # Math via Azure-hosted DeepSeek-R1 (Responses / Chat Completions)
    "math-azure": TaskSpec(
        name="math-azure",
        config={
            "system_prompt": MATH_SYSTEM_PROMPT,
            "stop_think": ["</think>"],
            "stop_answer": ["</answer>"],
            "two_pass": False,
            "max_output_tokens": 900,
            "canonicalize_pred": "src.inference.common:canon_math",
            "canonicalize_gold": "src.inference.common:canon_math",
        },
        dataset=DatasetSpec(
            loader="src.inference.math_deepseek_azure:load_math500",
            default_id="MATH-500",
            prompt_column="problem",
            answer_column="answer",
            split="test",
        ),
        notes="Single-pass generation against Azure Responses/Chat with resume/fill.",
    ),
    # Rush Hour (car-park) two-pass inference
    "carpark": TaskSpec(
        name="carpark",
        config={
            "system_prompt": CARPARK_SYSTEM_PROMPT,
            "stop_think": ["</think>"],
            "stop_answer": ["</answer>"],
            "two_pass": True,
            "think_cap": 750,
            "answer_cap": 50,
            "canonicalize_pred": "src.inference.carpark_core:_canon_rush_generic",
            "canonicalize_gold": "src.inference.carpark_core:_canon_rush_gold",
        },
        dataset=DatasetSpec(
            loader="src.inference.carpark_core:load_rush_dataset",
            default_id="od2961/rush4-5-6-balanced",
            prompt_column="messages",
            answer_column="solution",
            split="test",
        ),
        notes="Rush Hour resume/fill + two-pass with optional cue.",
    ),
    # Cryptic crossword solver (two-pass capable)
    "crossword": TaskSpec(
        name="crossword",
        config={
            "system_prompt": CROSSWORD_SYSTEM_PROMPT,
            "stop_think": ["</think>"],
            "stop_answer": ["</answer>"],
            "two_pass": True,
            "think_cap": 750,
            "answer_cap": 50,
            "canonicalize_pred": "src.inference.crossword_core:_canon_cross",
            "canonicalize_gold": "src.inference.crossword_core:_canon_cross",
        },
        dataset=DatasetSpec(
            loader="src.inference.crossword_core:load_crossword_local",
            default_id="CROSSWORD-LOCAL",
            prompt_column="clue",
            answer_column="answer",
            split="test",
        ),
        notes="Cryptic Crossword with reconsider cue analytics.",
    ),
    # Baseline one-pass Open-R1 style inference (single prompt template)
    "openr1": TaskSpec(
        name="openr1",
        config={
            "system_prompt": OPENR1_PROMPT_TEMPLATE,
            "stop_think": [],
            "stop_answer": [],
            "two_pass": False,
            "think_cap": None,
            "answer_cap": None,
            "canonicalize_pred": None,
            "canonicalize_gold": None,
        },
        dataset=DatasetSpec(
            loader="datasets:load_dataset",
            default_id="open-r1/OpenR1-Math-220k",
            prompt_column="problem",
            answer_column="answer",
            split="train",
        ),
        notes="Simple batch inference with entropy; uses raw prompt template.",
    ),
}
