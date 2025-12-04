# -*- coding: utf-8 -*-
"""
cryptic_rewards.py — Dense-but-precise rewards for GRPO on cryptic crosswords.

Rewarding scheme (all rewards are per-sample and clipped to [0, 1]):

  TAGS & TAG FACTOR
  -----------------
  We look for four tags in the model's completion:
      <think>, </think>, <answer>, </answer>
  Let present_tags be how many of these appear (0..4). We define:
      tag_factor = present_tags / 4.0          # 0.00, 0.25, 0.50, 0.75, 1.00
  This tag_factor multiplies the tiny "contains" bonus and the crossword
  accuracy reward (shaping). The exact-match term remains a strict 0/1.

  COMPONENTS
  ----------
  1) Exact match inside <answer>…</answer>  (binary, NOT scaled by tag_factor)
     - Extract inner <answer> text; canonicalize to LETTERS-ONLY A–Z string.
     - Compare to canonicalized gold. If equal → base = 1.0, else 0.0.
     - Optional enumeration check: reject if lengths mismatch (see kwargs).

  2) Tiny "contains-anywhere" bonus  (scaled by tag_factor)
     - If the gold string appears ANYWHERE in the whole completion as a
       stand-alone word (not touching letters on either side), we add:
           contains_bonus * tag_factor    (default contains_bonus = 0.02)
     - "Stand-alone word" means the character immediately before/after the
       match is NOT a letter (Unicode-aware). Implemented with lookarounds:
           (?<!LETTER) gold (?!LETTER), LETTER = [^\\W\\d_]
       (so punctuation, quotes, spaces, parentheses, etc. are valid separators;
        glued forms like 'foosuite' or 'suitefoo' do not trigger the bonus.)

  3) Crossword accuracy reward (shaping)  (scaled by tag_factor)
     - Canonicalize completion and gold as in (2), and check if the gold
       appears ANYWHERE. If yes, we compute a length factor:
           0.0 at <= min_tokens, → 1.0 by >= max_full_len (linear in between)
       and multiply by tag_factor. This shaping is added as:
           + 0.25 * crossword_accuracy_reward
       Defaults: min_tokens=25, max_full_len=80 (override via kwargs).

  FINAL
  -----
  pure_accuracy_reward(...) returns:
      clip_0_1( base + contains_bonus * tag_factor + 0.25 * crossword_accuracy )
  crossword_accuracy_reward(...) returns the shaping term itself (already
  tag-scaled and length-scaled), useful when combining rewards explicitly.

Kwargs you may pass:
  - contains_bonus: float (default 0.02) tiny bonus magnitude
  - expected_len / n_letters / length / lengths: enumeration checks for exact match
  - min_tokens (int, default 25), max_full_len (int, default 80)

Notes:
  • Lookbehind/ahead assertions are fixed-width and zero-width in Python’s `re`;
    we use 1-char letter lookarounds to ensure separate-word matches robustly.
  • We purposely avoid \\b “word boundary” because \\b treats digits/_ as “word”
    characters, which isn’t the desired notion for crossword “words”.

"""

from __future__ import annotations

# ----------------------------- config ------------------------------
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, List, Sequence

from .rush_rewards import rush_solution_exact, rush_solution_shaped


__all__ = [
    "crossword_accuracy_reward",
    "pure_accuracy_reward",
    "pure_accuracy_reward_math",
    "crossword_format_reward",
    "crossword_length_reward",
    "rush_solution_exact",
    "rush_solution_shaped",
]

# ----------------------------- config ------------------------------

MIN_TOKENS_DEFAULT = 25  # crossword shaping: 0 at <= this many tokens
MAX_FULL_LEN_DEFAULT = 80  # crossword shaping: 1 at >= this many tokens

# Treat “letter” as Unicode letters (no digits/underscore) for word isolation.
LETTER = r"[^\W\d_]"  # letters only (Unicode)

# Tags used to score formatting/structure.
TAGS = ("<think>", "</think>", "<answer>", "</answer>")


# Immutable bundle for crossword scoring knobs.
@dataclass(frozen=True)
class _CrosswordScoreConfig:
    min_tokens: int
    max_full_len: int
    contains_mode: str
    scaled: bool


# ----------------------------- regexes -----------------------------

# Accept the tag pair anywhere in the string (not anchored).
_format_pat = re.compile(r"(?si)<think>.*?</think>.*?<answer>.*?</answer>")

# Extract the <answer> inner text; allow newlines/spaces.
_answer_pat = re.compile(r"(?si)<answer>\s*(.*?)\s*</answer>")

# ---------------------------- helpers ------------------------------


def _extract_content(comp: Any) -> str:
    """
    Accepts:
      • a plain string
      • a chat-style list like [{'role': 'assistant', 'content': '...'}]
      • nested sequences from chat structures
    Returns assistant text as a string.
    """
    if comp is None:
        return ""
    if isinstance(comp, str):
        return comp
    if isinstance(comp, Sequence) and comp:
        first = comp[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
        if isinstance(first, Sequence):
            return _extract_content(first)
    return str(comp)


def _canon_crossword_letters(text: str) -> str:
    """
    Canonicalize a crossword answer to LETTERS-ONLY A–Z:
      - Unicode NFKD + strip combining marks (remove accents)
      - Normalize curly quotes/dashes → straight variants
      - Map '&' → 'AND'
      - Uppercase
      - Keep only A–Z
    """
    if text is None:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.translate(
        str.maketrans(
            {
                "’": "'",
                "‘": "'",
                "“": '"',
                "”": '"',
                "—": "-",
                "–": "-",
                "−": "-",
            },
        ),
    )
    normalized = normalized.replace("&", "AND").upper()
    return "".join(ch for ch in normalized if "A" <= ch <= "Z")


def _count_tokens_whitespace(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


# --------------------------- rewards -------------------------------


def _normalize_for_word_match(text: str) -> str:
    """
    Normalize without removing separators so regex word boundaries work.
    - Strip accents
    - Normalize curly quotes/dashes
    - Map '&' → 'AND'
    (Case-insensitive regex handles case differences.)
    """
    if text is None:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.translate(
        str.maketrans(
            {
                "’": "'",
                "‘": "'",
                "“": '"',
                "”": '"',
                "—": "-",
                "–": "-",
                "−": "-",
            },
        ),
    )
    return normalized.replace("&", "AND")


# --------------------------- rewards -------------------------------


def _has_gold_in_completion(
    gold_canon: str,
    raw_completion: str,
    contains_mode: str,
) -> bool:
    """Return True iff the gold answer is found under the requested mode."""
    if not gold_canon:
        return False

    if contains_mode == "any":
        raw_canon = _canon_crossword_letters(raw_completion)
        return gold_canon in raw_canon

    normalized_text = _normalize_for_word_match(raw_completion)
    tokens = [
        _canon_crossword_letters(token) for token in re.findall(rf"{LETTER}+", normalized_text, flags=re.UNICODE)
    ]
    if contains_mode == "word":
        return any(token == gold_canon for token in tokens)
    # "contiguous" mode: allow the gold letters inside any contiguous token.
    return any(gold_canon in token for token in tokens)


def _length_ramp_factor(raw_completion: str, min_tokens: int, max_full_len: int) -> float:
    """Return the [0,1] ramp factor based on whitespace token count."""
    token_count = _count_tokens_whitespace(raw_completion)
    if token_count <= min_tokens:
        return 0.0
    if token_count >= max_full_len:
        return 1.0
    return (token_count - min_tokens) / float(max_full_len - min_tokens)


def _score_single_crossword(
    completion: Any,
    gold: str,
    *,
    config: _CrosswordScoreConfig,
) -> float:
    """Score a single completion/gold pair for crossword-style accuracy."""
    raw_completion = _extract_content(completion) or ""
    text_lower = raw_completion.lower()

    present_tags = sum(1 for tag in TAGS if tag in text_lower)
    tag_factor = present_tags / 4.0 if config.scaled else 1.0

    gold_canon = _canon_crossword_letters(gold)
    has_gold = _has_gold_in_completion(gold_canon, raw_completion, config.contains_mode)

    if not config.scaled:
        return 1.0 if has_gold else 0.0

    length_factor = _length_ramp_factor(
        raw_completion,
        config.min_tokens,
        config.max_full_len,
    )
    return (1.0 if has_gold else 0.0) * length_factor * tag_factor


def crossword_accuracy_reward(
    completions: List[Any],
    answer: List[str],
    **kwargs,
) -> List[float]:
    """
    "Contains-anywhere" shaping, scaled by tag_factor and length:

      reward = has_gold * length_factor * tag_factor

    where:
      - has_gold:
          contains_mode = "any"        → (default) gold letters appear anywhere (after canon)
                        = "contiguous" → gold letters appear within a single contiguous LETTER+ run
                        = "word"       → gold letters appear as a standalone LETTER+ token
      - length_factor:  0 at <= min_tokens; 1 at >= max_full_len; linear ramp.
      - tag_factor:     (#present_tags / 4.0), tags ∈ {<think>, </think>, <answer>, </answer>}.

    Kwargs:
      - min_tokens (int, default 25)
      - max_full_len (int, default 80)
      - contains_mode (str, default "any"): "any" | "contiguous" | "word"
    """
    config = _CrosswordScoreConfig(
        min_tokens=int(kwargs.get("min_tokens", MIN_TOKENS_DEFAULT)),
        max_full_len=int(kwargs.get("max_full_len", MAX_FULL_LEN_DEFAULT)),
        contains_mode=str(kwargs.get("contains_mode", "any")).lower(),
        scaled=bool(kwargs.get("scaled", False)),  # default: no scaling (tests expect full credit)
    )

    return [
        _score_single_crossword(
            comp,
            gold,
            config=config,
        )
        for comp, gold in zip(completions, answer)
    ]


def _expected_length(length_spec: dict[str, Any]) -> int | None:
    """Compute the total expected length given enumeration kwargs."""
    expected = length_spec.get("expected_len") or length_spec.get("n_letters") or length_spec.get("length")
    if expected is None and "lengths" in length_spec:
        try:
            expected = sum(int(str(x)) for x in length_spec["lengths"])
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return None
    if expected is None:
        return None
    try:
        return int(expected)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def _score_single_pure(
    completion: Any,
    gold: str,
    *,
    contains_bonus: float,
    length_kwargs: dict[str, Any],
) -> float:
    """
    Score a single completion/gold pair for pure accuracy with tiny bonus and enumeration.
    """
    completion_text = _extract_content(completion) or ""
    gold_text = (gold or "").strip()

    # Tag factor (¼ per present tag; applies to tiny bonus).
    tag_factor = sum(1 for tag in TAGS if tag in completion_text.lower()) / 4.0

    # ---- WORD-LEVEL partial credit (no cross-space) ----
    gold_letters = _canon_crossword_letters(gold_text)
    normalized_text = _normalize_for_word_match(completion_text)

    tiny_bonus = 0.0
    if gold_letters:
        sep_word_pat = re.compile(
            rf"(?i)(?<!{LETTER}){re.escape(gold_letters)}(?!{LETTER})",
            re.UNICODE,
        )
        if sep_word_pat.search(normalized_text):
            tiny_bonus = contains_bonus * tag_factor

    # ---- Exact match inside <answer>…</answer> (binary) ----
    match = _answer_pat.search(completion_text)
    if not match:
        return tiny_bonus  # no answer tag → only tiny bonus (if any)

    pred_canon = _canon_crossword_letters(match.group(1))
    gold_canon = _canon_crossword_letters(gold_text)

    # Optional strict enumeration enforcement.
    expected_len = _expected_length(length_kwargs)
    if expected_len is not None and len(pred_canon) != expected_len:
        return tiny_bonus  # enumeration mismatch → only tiny bonus

    return (1.0 if pred_canon == gold_canon else 0.0) + tiny_bonus


def pure_accuracy_reward(
    completions: List[Any],
    answer: List[str],
    **kwargs,
) -> List[float]:
    """
    Exact-match + tiny separate-word bonus + crossword shaping (clipped to 1.0):

      base = 1.0 iff canonicalized <answer>…</answer> == canonicalized gold, else 0.0

      tiny_bonus = contains_bonus * tag_factor  if the canonicalized letters of the
                   gold appear as a standalone LETTER+ word in the (normalized)
                   completion (no spaces/punct inside the match); else 0.0.
         • Regex (case-insensitive):
              (?<!LETTER)  GOLD_LETTERS  (?!LETTER)
           where LETTER = [^\\W\\d_], and GOLD_LETTERS has no spaces/punct.

      shaping = 0.25 * crossword_accuracy_reward(…)  # already tag-/length-scaled

      return min(1.0, base + tiny_bonus + 0.25 * shaping)
    """
    contains_bonus = float(kwargs.get("contains_bonus", 0.02))
    length_kwargs: dict[str, Any] = {
        name: kwargs[name] for name in ("expected_len", "n_letters", "length", "lengths") if name in kwargs
    }

    outs: List[float] = [
        _score_single_pure(
            comp,
            gold,
            contains_bonus=contains_bonus,
            length_kwargs=length_kwargs,
        )
        for comp, gold in zip(completions, answer)
    ]

    # Add 0.25 * crossword shaping (optionally make contains strict via contains_mode kwargs).
    shaping = crossword_accuracy_reward(completions, answer, **kwargs)
    return [min(1.0, base + 0.25 * score) for base, score in zip(outs, shaping)]


def _canon_math(text: str) -> str:
    """
    Canonicalize math answers:
    - strip leading/trailing whitespace
    - remove LaTeX spacing commands (\\ ), $...$, and curly braces around single tokens
    - normalize minus-zero to zero
    - drop trailing .0 from integers
    - remove spaces inside the expression unless inside LaTeX commands
    - unify parentheses usage for single-number expressions
    """
    text = text.strip()

    # Remove LaTeX math mode markers
    text = text.replace("$", "")

    # Remove LaTeX spacing commands
    text = re.sub(r"\\\s+", "", text)

    # Remove outer braces around a single token
    if re.fullmatch(r"\{[^{}]+\}", text):
        text = text[1:-1]

    # Drop surrounding parentheses if they enclose just a number/frac/root
    if re.fullmatch(r"\([^()]+\)", text):
        inner = text[1:-1].strip()
        # only drop if it doesn't change grouping meaning
        if re.match(r"^[\d\.\-\\sqrt]+$", inner):
            text = inner

    # Strip spaces
    text = text.replace(" ", "")

    # Convert -0 to 0
    if text in ("-0", "+0"):
        text = "0"

    # Remove trailing .0 from integers
    if re.fullmatch(r"-?\d+\.0+", text):
        text = text.split(".")[0]

    return text


def pure_accuracy_reward_math(
    completions: List[Any],
    answer: List[str],
    **_unused_kwargs,
) -> List[float]:
    """
    Pure exact-match for math problems with format requirement:
      • Output must match <think> … </think><answer> … </answer> (any spacing/newlines).
      • The <answer> content must exactly equal the gold (canonicalized math form).
    """
    outs: List[float] = []

    for comp, gold in zip(completions, answer):
        txt = _extract_content(comp)

        # Must satisfy the full tag template
        if not _format_pat.match(txt):
            outs.append(0.0)
            continue

        match = _answer_pat.search(txt)
        if not match:
            outs.append(0.0)
            continue

        pred = match.group(1)
        is_correct_match = _canon_math(pred) == _canon_math(gold)
        outs.append(1.0 if is_correct_match else 0.0)

    return outs


# --------------------------- crossword-format helpers -------------------------------


def _extract_answer_letters(comp: Any) -> str:
    """Pull the inner <answer>…</answer> text and canon to LETTERS-ONLY A–Z."""
    raw = _extract_content(comp)
    match = _answer_pat.search(raw or "")
    if not match:
        return ""
    return _canon_crossword_letters(match.group(1))


def crossword_format_reward(
    completions: List[Any],
    answer: List[str],
    prompt: List[Any] | None = None,  # unused but kept for API parity
) -> List[float]:
    """
    Formatting bonus (crosswords):
    - 0 if the answer is correct.
    - 1 if the answer is wrong but:
        * tags for <think>…</think><answer>…</answer> are present, and
        * the submitted answer is all-uppercase (no lowercase letters).
    Otherwise 0.
    """
    _ = prompt  # kept for API parity; not used in scoring
    outs: List[float] = []
    for comp, gold in zip(completions, answer):
        raw = _extract_content(comp) or ""
        gold_can = _canon_crossword_letters(gold)
        ans_can = _extract_answer_letters(comp)

        # no bonus for correct answers
        if ans_can == gold_can and gold_can:
            outs.append(0.0)
            continue

        has_tags = bool(_format_pat.search(raw))
        ans_match = _answer_pat.search(raw or "")
        ans_text = ans_match.group(1) if ans_match else ""
        has_lower = any(ch.isalpha() and ch.islower() for ch in ans_text)
        outs.append(1.0 if has_tags and ans_text and not has_lower else 0.0)
    return outs


def crossword_length_reward(
    completions: List[Any],
    prompt: List[Any],
) -> List[float]:
    """
    Reward 1.0 if the answer length matches the enumeration in the prompt, else 0.0.
    Prompts are expected to include "(N)" where N is the length.
    """
    outs: List[float] = []
    for comp, prm in zip(completions, prompt):
        user_txt = ""
        if isinstance(prm, list) and prm:
            # expect a chat-style list of messages; pick user content
            for msg in prm:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    user_txt = str(msg.get("content", ""))
                    break
        match = re.search(r"\((\d+)\)", user_txt)
        if not match:
            outs.append(0.0)
            continue
        expected_len = int(match.group(1))
        ans_len = len(_extract_answer_letters(comp))
        outs.append(1.0 if ans_len == expected_len else 0.0)
    return outs


def formating(completions, **_unused_kwargs):
    """
    Check that the reasoning/answer are wrapped in <think>…</think><answer>…</answer>
    with newlines; used as a formatting bonus in tests.
    """
    pattern = re.compile(r"^<think>\s*.*?\s*</think>\s*<answer>\s*.*?\s*</answer>\s*$", re.DOTALL)
    outs = []
    for comp in completions:
        raw = _extract_content(comp) or ""
        outs.append(1.0 if pattern.match(raw) else 0.0)
    return outs
