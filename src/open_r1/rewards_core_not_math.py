from __future__ import annotations
# -*- coding: utf-8 -*-
"""
cryptic_rewards.py  —  Dense-but-precise rewards for GRPO on cryptic crosswords.

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


# ----------------------------- config ------------------------------
import re
import unicodedata
from typing import Any, List, Sequence

# ----------------------------- config ------------------------------

MIN_TOKENS_DEFAULT = 25     # crossword shaping: 0 at <= this many tokens
MAX_FULL_LEN_DEFAULT = 80   # crossword shaping: 1 at >= this many tokens

# Treat “letter” as Unicode letters (no digits/underscore) for word isolation.
LETTER = r"[^\W\d_]"  # letters only (Unicode)
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


def _canon_crossword_letters(s: str) -> str:
    """
    Canonicalize a crossword answer to LETTERS-ONLY A–Z:
      - Unicode NFKD + strip combining marks (remove accents)
      - Normalize curly quotes/dashes → straight variants
      - Map '&' → 'AND'
      - Uppercase
      - Keep only A–Z
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"',
                                   "—": "-", "–": "-", "−": "-"}))
    s = s.replace("&", "AND").upper()
    return "".join(ch for ch in s if "A" <= ch <= "Z")


def _count_tokens_whitespace(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))

# --------------------------- rewards -------------------------------


def _normalize_for_word_match(s: str) -> str:
    """
    Normalize without removing separators so regex word boundaries work.
    - Strip accents
    - Normalize curly quotes/dashes
    - Map '&' → 'AND'
    (Case-insensitive regex handles case differences.)
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"',
                                   "—": "-", "–": "-", "−": "-"}))
    return s.replace("&", "AND")

# --------------------------- rewards -------------------------------

def crossword_accuracy_reward(
    completions: List[Any],
    answer:      List[str],
    **kw,
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
    outs: List[float] = []

    min_tokens = int(kw.get("min_tokens", MIN_TOKENS_DEFAULT))
    max_full   = int(kw.get("max_full_len", MAX_FULL_LEN_DEFAULT))
    contains_mode = str(kw.get("contains_mode", "any")).lower()

    TAGS = ("<think>", "</think>", "<answer>", "</answer>")

    for gold, comp in zip(answer, completions):
        raw = _extract_content(comp) or ""
        txt_lower = raw.lower()

        present_tags = sum(1 for t in TAGS if t in txt_lower)
        tag_factor   = present_tags / 4.0

        gold_can = _canon_crossword_letters(gold)

        # --- has_gold according to contains_mode ---
        if contains_mode == "any":
            raw_can  = _canon_crossword_letters(raw)
            has_gold = bool(gold_can) and (gold_can in raw_can)
        else:
            raw_norm = _normalize_for_word_match(raw)
            # contiguous LETTER+ tokens from normalized text
            tokens = [ _canon_crossword_letters(t)
                       for t in re.findall(fr"{LETTER}+", raw_norm, flags=re.UNICODE) ]
            if contains_mode == "word":
                has_gold = bool(gold_can) and any(tok == gold_can for tok in tokens)
            else:  # "contiguous": allow substring inside a single LETTER+ token
                has_gold = bool(gold_can) and any(gold_can in tok for tok in tokens)

        # Length ramp (whitespace-token proxy)
        n_tokens = _count_tokens_whitespace(raw)
        if n_tokens <= min_tokens:
            length_factor = 0.0
        elif n_tokens >= max_full:
            length_factor = 1.0
        else:
            length_factor = (n_tokens - min_tokens) / float(max_full - min_tokens)

        outs.append( (1.0 if has_gold else 0.0) * length_factor * tag_factor )

    return outs

def pure_accuracy_reward(
    completions: List[Any],
    answer:      List[str],
    **kw,
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
    outs: List[float] = []
    contains_bonus = float(kw.get("contains_bonus", 0.02))
    TAGS = ("<think>", "</think>", "<answer>", "</answer>")

    for comp, gold in zip(completions, answer):
        txt = _extract_content(comp) or ""
        txt_lower = txt.lower()
        g   = (gold or "").strip()

        # tag factor (¼ per present tag; applies to tiny bonus)
        present_tags = sum(1 for t in TAGS if t in txt_lower)
        tag_factor   = present_tags / 4.0

        # ---- WORD-LEVEL partial credit (no cross-space) ----
        # Use canonicalized letters for the gold and a normalized text that preserves separators.
        gold_letters = _canon_crossword_letters(g)        # A–Z only, no spaces/hyphens
        raw_norm     = _normalize_for_word_match(txt)     # keeps separators for boundaries

        # Standalone word of letters-only (prevents hits like 'fooa nsfoo' or 'XSUITEX')
        sep_word_pat = re.compile(
            rf"(?i)(?<!{LETTER}){re.escape(gold_letters)}(?!{LETTER})",
            re.UNICODE
        )
        has_sep_word = bool(gold_letters) and bool(sep_word_pat.search(raw_norm))
        tiny_bonus   = contains_bonus * tag_factor if has_sep_word else 0.0

        # ---- Exact match inside <answer>…</answer> (binary) ----
        m = _answer_pat.search(txt)
        if not m:
            outs.append(tiny_bonus)      # no answer tag → only tiny bonus (if any)
            continue

        pred_c = _canon_crossword_letters(m.group(1))
        gold_c = _canon_crossword_letters(g)

        # Optional strict enumeration enforcement
        expected_len = kw.get("expected_len") or kw.get("n_letters") or kw.get("length")
        if expected_len is None and "lengths" in kw:
            try:
                expected_len = sum(int(x) for x in kw["lengths"])
            except Exception:
                expected_len = None
        if expected_len is not None:
            try:
                if len(pred_c) != int(expected_len):
                    outs.append(tiny_bonus)  # enumeration mismatch → only tiny bonus
                    continue
            except Exception:
                pass

        base = 1.0 if (pred_c == gold_c) else 0.0
        outs.append(base + tiny_bonus)

    # Add 0.25 * crossword shaping (optionally make contains strict via contains_mode kw)
    shaping = crossword_accuracy_reward(completions, answer, **kw)
    return [min(1.0, base + 0.25 * s) for base, s in zip(outs, shaping)]

import unicodedata
import re
from typing import Any, List
    
def _canon_crossword_letters(s: str) -> str:
    """
    Canonicalize a crossword answer to LETTERS-ONLY A–Z string:
      - Unicode NFKD + strip combining marks (remove accents)
      - Normalize curly quotes/dashes; map '&' -> 'AND'
      - Uppercase
      - Drop everything that's not A–Z
    """
    if s is None:
        return ""

    # Normalize unicode + strip accents
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    # Normalize common punctuation variants
    s = s.translate(str.maketrans({
        "’": "'", "‘": "'", "“": '"', "”": '"',
        "—": "-", "–": "-", "−": "-",
    }))

    # Map symbols that semantically carry letters
    s = s.replace("&", "AND")

    # Uppercase, then keep letters only
    s = s.upper()
    return "".join(ch for ch in s if "A" <= ch <= "Z")



import re
from typing import Any, Dict, Iterable, List, Optional, Union

# ── helpers you referenced ──────────────────────────────────────────────
MOVE_RE  = re.compile(r"^[A-Z][<>^v]\d+$")
TOK_LIST = re.compile(r"\s*,\s*")
ANS_TAG  = re.compile(r"(?is)<answer>(.*?)</answer>")

def _extract_answer_text(text: str) -> str:
    """Return inner text of <answer>…</answer> if present; else the whole string."""
    m = ANS_TAG.search(text or "")
    return (m.group(1) if m else (text or "")).strip()

def _canon_token(tok: str) -> Optional[str]:
    """Uppercase, strip spaces, ensure it looks like PIECE+DIR+STEPS."""
    t = (tok or "").strip().upper().replace(" ", "")
    return t if MOVE_RE.match(t) else None

from typing import List, Optional, Any
import re

def _canon_seq(x: Any) -> Optional[List[str]]:
    """
    Parse a Rush Hour move sequence out of raw model text or a Python list.
    Normalizes:
      - extracts <answer>...</answer> if present (ignores <think>...)
      - accepts commas or whitespace as separators
      - maps U/D/L/R and Unicode arrows to ^/v/</>
      - uppercases piece letters; strips spaces; removes leading zeros in steps
    Returns list like ["E^1","G<1","Bv1","A>3"] or None if nothing parseable.
    """
    # Coerce to a single string
    if isinstance(x, (list, tuple)):
        s = ",".join(str(t) for t in x)
    else:
        s = str(x)

    # Strip <think> and isolate <answer> if present
    s = re.sub(r"(?is)<think>.*?</think>", "", s)
    m = re.search(r"(?is)<answer>\s*(.*?)\s*</answer>", s)
    if m:
        s = m.group(1)

    # Normalize arrows
    s = (s
         .replace("↑", "^").replace("↓", "v")
         .replace("←", "<").replace("→", ">"))

    # Find tokens in order, allowing spaces like "A > 3"
    tokens = []
    for m in re.finditer(r"([A-Za-z])\s*([><\^vUDLR])\s*([0-9]+)", s):
        piece = m.group(1).upper()
        d_raw = m.group(2)  # one of <,>,^,v or U/D/L/R (sometimes V)
        # Map symbols as-is; map letters case-insensitively to canonical symbols
        if d_raw in ("<", ">", "^", "v"):
            d = d_raw
        else:
            d = {"U":"^", "D":"v", "L":"<", "R":">", "V":"v"}.get(d_raw.upper(), d_raw)
            
        steps = str(int(m.group(3)))  # drop leading zeros
        tokens.append(f"{piece}{d}{steps}")

    return tokens or None

import re
import math
from typing import List, Tuple, Optional, Dict

TOKEN_RE = re.compile(r"^[A-Z][<>^v]\d+$")

# ---------- Parsing helpers ----------

def _extract_answer_block(text: str) -> Optional[str]:
    """Prefer <answer>...</answer>; otherwise find the first token list in text."""
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.S | re.I)
    if m:
        cand = m.group(1).strip()
        if _looks_like_token_list(cand):
            return cand
    # fallback: any token list looking like A>2,B<1,...
    m2 = re.search(r"([A-Za-z][<>^v]\d+(?:\s*,\s*[A-Za-z][<>^v]\d+)*)", text)
    return m2.group(1) if m2 else None

def _looks_like_token_list(s: str) -> bool:
    s = s.strip().upper().replace(" ", "")
    if not s:
        return False
    parts = s.split(",")
    return all(TOKEN_RE.match(p) for p in parts)

def _canon_seq(seq: str) -> Optional[List[str]]:
    """
    Canonicalize a move sequence string to a list of tokens:
      - Uppercase letters
      - Remove spaces
      - Validate each token
      - Merge consecutive identical (car,dir) tokens by summing steps
    Returns None if invalid.
    """
    if not isinstance(seq, str):
        return None
    s = seq.strip().upper().replace(" ", "")
    if not s:
        return None
    parts = s.split(",")
    toks = []
    for p in parts:
        if not TOKEN_RE.match(p):
            return None
        car = p[0]
        d = p[1]
        n = int(p[2:])
        if n <= 0:
            return None
        toks.append((car, d, n))

    # merge consecutive same (car,dir)
    merged = []
    for car, d, n in toks:
        if merged and merged[-1][0] == car and merged[-1][1] == d:
            merged[-1] = (car, d, merged[-1][2] + n)
        else:
            merged.append((car, d, n))
    return [f"{c}{d}{k}" for (c, d, k) in merged]


def _len_tokens(tokens: Optional[List[str]]) -> int:
    return 0 if not tokens else len(tokens)

# rush_reward.py
# Complete Rush Hour reward utilities:
# - Robust token parsing & canonicalization
# - Prompt/board extraction (handles '4x4' or '4×4', 'Board:', 'Minimal/Optimal moves')
# - Board simulator with walls 'x' and empty 'o'
# - Potential-based shaping: Φ = blockers + distance_to_exit (decrease is rewarded)
# - Rewards:
#     * rush_solution_shaped: dense [0,1], preserves optimal solution ordering
#     * rush_solution_exact:  1 iff exact canonical match; else 0
#
# Usage (TRL-style):
#   reward_fn: rush_reward:rush_solution_shaped
#
# Notes:
# - Tokens look like:  A>2,B<1,Cv3
#   piece ∈ 'A'..'Z'; dir ∈ {<, >, ^, v}; steps ∈ {1,2,...}
# - We accept 'V' and normalize to 'v'.
# - We prefer answers inside <answer>...</answer>, but will fall back to the first token list.

import re
import math
from typing import List, Tuple, Optional, Dict, Any, Iterable

# ---------- Token utilities ----------

# Directions allowed; accept uppercase 'V' too
_TOKEN_RE = re.compile(r"^[A-Z][<>^vV]\d+$")

def _parse_and_normalize_token(p: str) -> Optional[Tuple[str, str, int]]:
    """
    Parse a single token (e.g., 'Bv3', 'A>2', 'C^1') and normalize to:
      (piece_upper, dir_norm, steps_int)
    dir_norm ∈ {'<','>','^','v'}
    """
    if not isinstance(p, str):
        return None
    p = p.strip().replace(" ", "")
    if not _TOKEN_RE.match(p):
        return None
    piece = p[0].upper()
    d = p[1]
    d = 'v' if d in ('v', 'V') else d
    try:
        n = int(p[2:])
    except Exception:
        return None
    if n <= 0:
        return None
    return (piece, d, n)

def _looks_like_token_list(s: str) -> bool:
    s = (s or "").strip().replace(" ", "")
    if not s:
        return False
    parts = s.split(",")
    for p in parts:
        tup = _parse_and_normalize_token(p)
        if tup is None:
            return False
    return True

def _canon_seq(seq: Any) -> Optional[List[str]]:
    """
    Canonicalize a move sequence (string like "Bv1,A>1" or list of tokens)
    into a list of normalized tokens ["Bv1","A>1"], merging consecutive
    duplicates of the same (piece, dir) by summing steps.
    Returns None if invalid.
    """
    if isinstance(seq, (list, tuple)):
        parts = [str(x) for x in seq]
    elif isinstance(seq, str):
        s = seq.strip()
        s = s.replace(" ", "")
        if not s:
            return None
        parts = s.split(",")
    else:
        return None

    toks: List[Tuple[str, str, int]] = []
    for p in parts:
        tup = _parse_and_normalize_token(p)
        if tup is None:
            return None
        toks.append(tup)

    # Merge consecutive same (piece, dir)
    merged: List[Tuple[str, str, int]] = []
    for (piece, d, n) in toks:
        if merged and merged[-1][0] == piece and merged[-1][1] == d:
            merged[-1] = (piece, d, merged[-1][2] + n)
        else:
            merged.append((piece, d, n))
    return [f"{c}{d}{k}" for (c, d, k) in merged]

def _extract_answer_block(text: str) -> Optional[str]:
    """Prefer <answer>...</answer>; otherwise find the first token list."""
    if not isinstance(text, str):
        return None
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.S | re.I)
    if m:
        cand = m.group(1).strip()
        if _looks_like_token_list(cand):
            return cand
    # fallback: tolerant regex for tokens
    m2 = re.search(r"([A-Za-z][<>^vV]\d+(?:\s*,\s*[A-Za-z][<>^vV]\d+)*)", text)
    return m2.group(1) if m2 else None

# Accept commas, whitespace, newlines, or semicolons between tokens.
_TOKEN_SCAN = re.compile(r"([A-Za-z])\s*([><\^vVUDLR])\s*([0-9]+)")

def _canon_seq_from_text(text: Any) -> Optional[List[str]]:
    """Extract tokens anywhere in the prediction (favoring <answer>...</answer> if present)."""
    if isinstance(text, (list, tuple)):
        text = ",".join(str(t) for t in text)
    s = str(text or "")
    # Prefer the <answer> block if present
    m = re.search(r"(?is)<answer>\s*(.*?)\s*</answer>", s)
    s = m.group(1) if m else s

    toks: List[Tuple[str,str,int]] = []
    for m in _TOKEN_SCAN.finditer(s):
        piece = m.group(1).upper()
        d_raw = m.group(2)  # one of <,>,^,v or U/D/L/R (sometimes V)
        # Map symbols as-is; map letters case-insensitively to canonical symbols
        if d_raw in ("<", ">", "^", "v"):
            d = d_raw
        else:
            d = {"U":"^", "D":"v", "L":"<", "R":">", "V":"v"}.get(d_raw.upper(), d_raw)
            
        n     = int(m.group(3))
        if n <= 0: 
            continue
        toks.append((piece, d, n))

    if not toks:
        return None

    # merge consecutive same (piece,dir)
    merged: List[Tuple[str,str,int]] = []
    for (c, d, n) in toks:
        if merged and merged[-1][0]==c and merged[-1][1]==d:
            merged[-1] = (c, d, merged[-1][2] + n)
        else:
            merged.append((c, d, n))
    return [f"{c}{d}{k}" for (c, d, k) in merged]


def _len_tokens(tokens: Optional[List[str]]) -> int:
    return 0 if not tokens else len(tokens)


# ---------- Prompt extraction ----------

def _stringify_prompt(prompts: Any) -> str:
    """
    Robustly produce a single string from common prompt representations:
    - str
    - list[str]
    - list[dict] with 'content' fields
    - dict with 'prompt' / 'content' fields
    """
    if isinstance(prompts, str):
        return prompts
    if isinstance(prompts, dict):
        for key in ("content", "prompt", "text"):
            if key in prompts and isinstance(prompts[key], str):
                return prompts[key]
    if isinstance(prompts, (list, tuple)):
        chunks: List[str] = []
        for item in prompts:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                # OpenAI-style chat message: {'role': 'user', 'content': '...'}
                val = item.get("content")
                if isinstance(val, str):
                    chunks.append(val)
                elif isinstance(val, (list, tuple)):
                    # some SDKs split content into parts
                    chunks.extend([str(x) for x in val])
            else:
                chunks.append(str(item))
        return "\n".join(chunks)
    return str(prompts)

def _extract_puzzle_from_prompt(prompt: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    N = None
    mN = re.search(r"Board\s*size\s*:\s*(\d+)\s*[xX×]\s*(\d+)", prompt)
    if mN:
        rN, cN = int(mN.group(1)), int(mN.group(2))
        if rN == cN:
            N = rN

    # NEW: accept board on next line; allow whitespace inside, then strip it
    mB = re.search(r"(?is)Board\s*:\s*([A-Za-zx\s]+)", prompt)
    board = None
    if mB:
        board_raw = mB.group(1)
        board = re.sub(r"\s+", "", board_raw)  # remove newlines/spaces

    # If we know N and captured more than needed, crop to N*N
    if N is not None and board:
        need = N * N
        if len(board) >= need:
            board = board[:need]
        else:
            # try to find a contiguous run of length N*N anywhere (fallback)
            patt = re.compile(rf"\b([A-Za-zx]{{{need}}})\b")
            m = patt.search(prompt)
            if m:
                board = m.group(1)

    # If N unknown but board exists, infer from perfect square length
    if N is None and board:
        root = int(round(math.sqrt(len(board))))
        if root * root == len(board):
            N = root

    mg = re.search(r"(?:Minimal|Optimal)\s+(?:length\s*\(moves\)|moves?)\s*:\s*(\d+)", prompt, flags=re.I)
    gold_moves = int(mg.group(1)) if mg else None
    return board, N, gold_moves

# ---------- Board & simulation ----------

class Board:
    """
    Rush Hour board:
      - grid: NxN row-major string
          'o' = empty, 'x' = wall, 'A'..'Z' = cars (length 2 or 3 typically)
      - 'A' is the target car (assumed horizontal); goal is A's rightmost cell at col N-1.
    """

    def __init__(self, grid_str: str, N: int):
        self.N = N
        s = (grid_str or "").strip()
        if len(s) != N * N:
            raise ValueError(f"Board string length {len(s)} != N*N ({N*N})")
        self.grid: List[List[str]] = [list(s[r * N:(r + 1) * N]) for r in range(N)]
        self.cars: Dict[str, List[Tuple[int, int]]] = self._index_cars()
        self.orient: Dict[str, str] = self._orientations()  # 'H' or 'V'

    def clone(self) -> "Board":
        b = Board("o" * (self.N * self.N), self.N)
        b.grid = [row[:] for row in self.grid]
        b.cars = {k: v[:] for k, v in self.cars.items()}
        b.orient = dict(self.orient)
        return b

    def _index_cars(self) -> Dict[str, List[Tuple[int, int]]]:
        cars: Dict[str, List[Tuple[int, int]]] = {}
        for r in range(self.N):
            for c in range(self.N):
                ch = self.grid[r][c]
                if ch != 'o' and ch != 'x':
                    cars.setdefault(ch, []).append((r, c))
        for k in cars:
            cars[k].sort()
        return cars

    def _orientations(self) -> Dict[str, str]:
        orient: Dict[str, str] = {}
        for car, cells in self.cars.items():
            if len(cells) == 1:
                orient[car] = 'H'  # default for single-cells
                continue
            rows = {r for (r, _) in cells}
            cols = {c for (_, c) in cells}
            if len(rows) == 1:
                orient[car] = 'H'
            elif len(cols) == 1:
                orient[car] = 'V'
            else:
                # malformed; pick H to avoid crashing
                orient[car] = 'H'
        return orient

    def is_solved(self) -> bool:
        if 'A' not in self.cars:
            return False
        max_c = max(c for (_, c) in self.cars['A'])
        return max_c == self.N - 1

    def a_row_and_rightmost(self) -> Tuple[int, int]:
        if 'A' not in self.cars or not self.cars['A']:
            return (-1, -1)
        row = self.cars['A'][0][0]
        rightmost = max(c for (_, c) in self.cars['A'])
        return row, rightmost

    def blockers_and_distance(self) -> Tuple[int, int]:
        """
        Heuristic pieces:
          - blockers: count of cars/walls in A's row strictly to A's right
          - distance: (N - 1 - rightmost_A)
        """
        row, rightmost = self.a_row_and_rightmost()
        if row < 0:
            return (0, 0)
        dist = (self.N - 1) - rightmost
        blockers = 0
        for c in range(rightmost + 1, self.N):
            ch = self.grid[row][c]
            if ch == 'x':
                blockers += 1
            elif ch != 'o':
                blockers += 1
        return blockers, dist

    def _cells_for(self, car: str) -> List[Tuple[int, int]]:
        return self.cars.get(car, [])

    def _clear_cells(self, cells: Iterable[Tuple[int, int]]) -> None:
        for r, c in cells:
            self.grid[r][c] = 'o'

    def _occupy_cells(self, car: str, cells: Iterable[Tuple[int, int]]) -> None:
        cells = list(cells)
        for r, c in cells:
            self.grid[r][c] = car
        self.cars[car] = sorted(cells)

    def _step_move(self, car: str, d: str) -> bool:
        """
        Move 'car' one cell in direction d if legal.
        Returns False if illegal (wrong axis, off board, or blocked).
        """
        cells = self._cells_for(car)
        if not cells:
            return False

        ori = self.orient.get(car, 'H')
        if d in '<>' and ori != 'H':
            return False
        if d in '^v' and ori != 'V':
            return False

        if d == '<':
            leftmost = min(c for (_, c) in cells)
            r = cells[0][0]
            nc = leftmost - 1
            if nc < 0 or self.grid[r][nc] != 'o':
                return False
            new_cells = [(r, c - 1) for (r, c) in cells]
        elif d == '>':
            rightmost = max(c for (_, c) in cells)
            r = cells[0][0]
            nc = rightmost + 1
            if nc >= self.N or self.grid[r][nc] != 'o':
                return False
            new_cells = [(r, c + 1) for (r, c) in cells]
        elif d == '^':
            top = min(r for (r, _) in cells)
            c = cells[0][1]
            nr = top - 1
            if nr < 0 or self.grid[nr][c] != 'o':
                return False
            new_cells = [(r - 1, c) for (r, c) in cells]
        else:  # 'v'
            bot = max(r for (r, _) in cells)
            c = cells[0][1]
            nr = bot + 1
            if nr >= self.N or self.grid[nr][c] != 'o':
                return False
            new_cells = [(r + 1, c) for (r, c) in cells]

        self._clear_cells(cells)
        self._occupy_cells(car, new_cells)
        return True

    def apply_token(self, token: str) -> bool:
        car, d, steps = token[0], token[1], int(token[2:])
        for _ in range(steps):
            if not self._step_move(car, d):
                return False
        return True


def _simulate_prefix(board: Board, tokens: List[str]) -> Tuple[int, bool, Board]:
    """
    Apply tokens in order until one fails or goal is reached.
    Returns (valid_prefix_len, solved, final_board_state).
    """
    b = board.clone()
    valid = 0
    for t in tokens:
        ok = b.apply_token(t)
        if not ok:
            return valid, False, b
        valid += 1
        if b.is_solved():
            return valid, True, b
    return valid, b.is_solved(), b


# ---------- Gold handling ----------
def _canon_gold_candidates(gold: Any) -> List[List[str]]:
    if gold is None:
        return []
    # unwrap ["A>2,Bv5"] → "A>2,Bv5"
    if isinstance(gold, (list, tuple)) and len(gold) == 1 and isinstance(gold[0], str):
        gold = gold[0]
    if isinstance(gold, str) or (isinstance(gold, (list, tuple)) and (not gold or isinstance(gold[0], str))):
        seq = _canon_seq(gold)
        return [seq] if seq is not None else []

    cands: List[List[str]] = []
    if isinstance(gold, (list, tuple)):
        for g in gold:
            seq = _canon_seq(g)
            if seq is not None:
                cands.append(seq)
    return cands

# ---------- Rewards ----------

# --- Think-length helpers (bonus for longer reasoning) ---
_THINK_BLOCK = re.compile(r"(?is)<think>\s*(.*?)\s*</think>")

def _count_think_tokens(text: str) -> int:
    m = _THINK_BLOCK.search(text or "")
    if not m:
        return 0
    # whitespace-delimited proxy for token count (keeps it fast & robust)
    return len(re.findall(r"\S+", m.group(1)))

def rush_solution_shaped(
    *, 
    prompts,
    completions,
    answer=None,
    gold=None,
    # NEW: allow bypassing prompt parsing entirely:
    board_str: str | None = None,
    N: int | None = None,
    gold_moves: int | None = None,
    **kw,
) -> List[float]:
    """
    Dense, shaped reward in [0,1] for Rush Hour that works WITH or WITHOUT a board.

    When a board is provided (board_str+N), legality and Φ-progress are used.
    When no board is provided, the reward still gives:
      - exact match (vs gold candidates)
      - prefix/LCP credit vs gold
      - gold-only "solve optimality": shorter sequences are better based on gold_moves
    """
    # weights (sum ≤ 1 to keep final score in [0,1]; tune if desired)
    w_exact  = float(kw.get("w_exact", 0.65))
    w_solve  = float(kw.get("w_solve", 0.20))
    w_prefix = float(kw.get("w_prefix", 0.10))
    w_phi    = float(kw.get("w_phi", 0.05))

    # Normalize inputs
    if isinstance(completions, str):
        completions = [completions]
    else:
        completions = list(completions)

    # Canonicalize gold candidates
    gold_cands = _canon_gold_candidates(gold or answer or kw.get("answers") or kw.get("gold_answers"))
    gold_min_len = min((_len_tokens(gc) for gc in gold_cands if gc is not None), default=None)

    # Prefer explicit board/moves if provided; else try parsing prompt for convenience
    board = None
    gm_from_prompt = None
    if board_str is not None and N is not None:
        try:
            board = Board(board_str, int(N))
        except Exception:
            board = None
    else:
        prompt_text = _stringify_prompt(prompts)
        b2, N2, gm_from_prompt = _extract_puzzle_from_prompt(prompt_text)
        if b2 and N2:
            try:
                board = Board(b2, int(N2))
            except Exception:
                board = None

    # Choose a gold_moves target for boardless solve shaping
    target_moves = gold_moves if (gold_moves is not None) else (gm_from_prompt if gm_from_prompt is not None else gold_min_len)

    out_scores: List[float] = []
    for pred in completions:
        pred_text = str(pred or "")
        pred_can  = _canon_seq_from_text(pred_text)

        # 0) Formatting bonus when we see at least one valid token but parser fails
        if pred_can is None:
            # --- always initialize locals to avoid UnboundLocalError ---
            fmt_bonus   = 0.0
            think_bonus = 0.0

            # tiny formatting bonus if we can spot at least one token pattern
            try:
                n_tokens = len(list(_TOKEN_SCAN.finditer(pred_text)))
            except Exception:
                n_tokens = 0
            fmt_bonus = 0.01 if n_tokens > 0 else 0.0  # == min(0.01, 0.01 * n_tokens)

            # bonus for longer <think>
            tmin = int(kw.get("think_min_tokens", 25))
            tmax = int(kw.get("think_full_tokens", 100))
            tcap = float(kw.get("think_bonus_cap", 0.02))

            n_think = _count_think_tokens(pred_text)
            if n_think > tmin:
                ramp = (n_think - tmin) / max(1, (tmax - tmin))
                think_bonus = tcap * min(1.0, max(0.0, ramp))

            # keep this branch "tiny" so it never dominates signal
            out_scores.append(min(0.01, fmt_bonus + think_bonus))
            continue

        # 1) Exact match
        exact = 1.0 if any(pred_can == gc for gc in gold_cands if gc is not None) else 0.0

        # 2) Prefix credit (LCP vs best gold candidate)
        prefix = 0.0
        if gold_cands:
            best = 0
            denom = 1
            for gc in gold_cands:
                if not gc:
                    continue
                lcp = 0
                for a, b in zip(pred_can, gc):
                    if a == b:
                        lcp += 1
                    else:
                        break
                best  = max(best, lcp)
                denom = max(denom, _len_tokens(gc))
            prefix = best / max(1, denom)

        # 3) Solve & Φ terms
        solve_term = 0.0
        phi_term   = 0.0
        if board is not None:
            # Use legality-aware simulation when board is available
            valid_k, solved, b_final = _simulate_prefix(board, pred_can)

            # Prefix is at least the legal prefix fraction of our own sequence
            prefix = max(prefix, valid_k / max(1, _len_tokens(pred_can)))

            if solved:
                L = _len_tokens(pred_can)
                m = target_moves if target_moves is not None else L
                solve_term = 1.0 if L <= m else 1.0 / (1.0 + (L - m))

            blk0, dist0 = board.blockers_and_distance()
            blk1, dist1 = b_final.blockers_and_distance()
            h0, h1 = (blk0 + dist0), (blk1 + dist1)
            if h0 > 0:
                phi_term = max(0.0, (h0 - h1) / h0)
            else:
                phi_term = 1.0 if solved else 0.0

        else:
            # --- GOLD-ONLY SHAPING (no board) ---
            # Reward being at or under minimal/target moves softly
            if target_moves is not None:
                L = _len_tokens(pred_can)
                m = int(target_moves)
                solve_term = 1.0 if L <= m else 1.0 / (1.0 + (L - m))
            # phi_term stays 0 without a board

        score = (
            w_exact  * exact +
            w_solve  * solve_term +
            w_prefix * prefix +
            w_phi    * phi_term
        )
        out_scores.append(float(max(0.0, min(1.0, score))))

    return out_scores

def rush_solution_exact(*, prompts, completions, answer=None, gold=None, **kwargs):
    if isinstance(completions, str):
        completions = [completions]
    else:
        completions = list(completions)

    gold_cands = _canon_gold_candidates(gold or answer or kwargs.get("answers") or kwargs.get("gold_answers"))

    scores: List[float] = []
    for pred in completions:
        pred_text = str(pred or "")
        pred_can  = _canon_seq_from_text(pred_text)

        # 0) Formatting bonus when we see at least one valid token but parser fails
        if pred_can is None:
            # --- always initialize locals to avoid UnboundLocalError ---
            fmt_bonus   = 0.0
            think_bonus = 0.0

            # tiny formatting bonus if we can spot at least one token pattern
            try:
                n_tokens = len(list(_TOKEN_SCAN.finditer(pred_text)))
            except Exception:
                n_tokens = 0
            fmt_bonus = 0.01 if n_tokens > 0 else 0.0  # equivalent to min(0.01, 0.01 * n_tokens)

            # bonus for longer <think>
            tmin = int(kwargs.get("think_min_tokens", 25))
            tmax = int(kwargs.get("think_full_tokens", 100))
            tcap = float(kwargs.get("think_bonus_cap", 0.02))

            n_think = _count_think_tokens(pred_text)
            if n_think > tmin:
                ramp = (n_think - tmin) / max(1, (tmax - tmin))
                think_bonus = tcap * min(1.0, max(0.0, ramp))

            # keep this branch "tiny" so it never dominates signal
            scores.append(min(0.01, fmt_bonus + think_bonus))
            continue

        ok = any(pred_can == gc for gc in gold_cands if gc is not None)
        scores.append(1.0 if ok else 0.0)

    return scores


def _canon_math(s: str) -> str:
    """
    Canonicalize math answers:
    - strip leading/trailing whitespace
    - remove LaTeX spacing commands (\ ), $...$, and curly braces around single tokens
    - normalize minus-zero to zero
    - drop trailing .0 from integers
    - remove spaces inside the expression unless inside LaTeX commands
    - unify parentheses usage for single-number expressions
    """
    s = s.strip()

    # Remove LaTeX math mode markers
    s = s.replace('$', '')

    # Remove LaTeX spacing commands
    s = re.sub(r"\\\s+", "", s)

    # Remove outer braces around a single token
    if re.fullmatch(r"\{[^{}]+\}", s):
        s = s[1:-1]

    # Drop surrounding parentheses if they enclose just a number/frac/root
    if re.fullmatch(r"\([^()]+\)", s):
        s_inner = s[1:-1].strip()
        # only drop if it doesn't change grouping meaning
        if re.match(r"^[\d\.\-\\sqrt]+$", s_inner):
            s = s_inner

    # Strip spaces
    s = s.replace(" ", "")

    # Convert -0 to 0
    if s in ("-0", "+0"):
        s = "0"

    # Remove trailing .0 from integers
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split('.')[0]

    return s

def pure_accuracy_reward_math(
    completions: List[Any],
    answer:      List[str],
    **kw,
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

        m = _answer_pat.search(txt)
        if not m:
            outs.append(0.0)
            continue

        pred = m.group(1)
        ok = (_canon_math(pred) == _canon_math(gold))
        outs.append(1.0 if ok else 0.0)

    return outs