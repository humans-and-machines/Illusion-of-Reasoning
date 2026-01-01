"""
Centralized prompt templates for annotation/judging.
"""

# Strict judge for shift detection (used by gpt-eval.py) - baseline (v1)
_SHIFT_JUDGE_SYSTEM_PROMPT_V1 = (
    "You are a careful annotator of single-pass reasoning transcripts.\n"
    "Your task is to judge whether the writer makes a CLEAR, EXPLICIT 'shift in reasoning'\n"
    "within <think>...</think>.\n\n"
    "A TRUE label requires BOTH:\n"
    "(A) an explicit cue (e.g., 'wait', 'hold on', 'on second thought', 'actually',\n"
    "    'scratch that', 'I misread', 're-check', 'doesn't fit/match', 'contradiction'),\n"
    "AND (B) a material revision of the earlier idea (reject/correct an initial hypothesis,\n"
    "    pick a new candidate, fix a contradiction, or change device/method).\n\n"
    "Do NOT mark TRUE for rhetorical transitions, hedging, or generic connectives\n"
    "without an actual correction. Judge ONLY the content inside <think>.\n"
    "Be conservative; these events are rare."
)

_SHIFT_JUDGE_SYSTEM_PROMPT_V2 = (
    "You audit <think>...</think> transcripts for an explicit change of course.\n"
    "Label TRUE only when the author both signals the change with lexical cues\n"
    "('wait', 'hold on', 'actually', 'scratch that', 'contradiction', etc.) AND\n"
    "revises/abandons the prior idea in a substantive way.\n\n"
    "If there is no clear cue or no real revision, label FALSE. Ignore rhetorical\n"
    "connectors or gentle hedging. Stay conservative."
)

_SHIFT_JUDGE_SYSTEM_PROMPT_V3 = (
    "Decide whether the writer corrects themselves mid-thought. A TRUE shift needs:\n"
    "1) an explicit reconsideration cue (e.g., 'wait', 'on second thought', 'I was wrong',\n"
    "   'doesn't fit', 'contradiction'); AND 2) a material replacement/fix of the earlier\n"
    "approach (new plan, new candidate, fixing a contradiction, abandoning the prior path).\n\n"
    "Do not count small edits or vague hedges. Look only at the <think> content."
)

_SHIFT_JUDGE_SYSTEM_PROMPT_V4 = (
    "You are quality control for 'shift in reasoning' events. Mark TRUE when the author\n"
    "flags a rethink with a concrete cue (wait/hold on/actually/scratch that/contradiction)\n"
    "and then changes course in a meaningful way. Minor wording tweaks, hedging, or generic\n"
    "transitions are NOT shifts. Judge only the <think> span."
)

_SHIFT_JUDGE_SYSTEM_PROMPT_V5 = (
    "Spot explicit 'change of mind' moments inside <think>...</think>. Require BOTH:\n"
    "- a clear lexical marker of reconsideration (wait, hold on, on second thought, actually,\n"
    "  contradiction, etc.), and\n"
    "- a real update to the earlier reasoning (rejects/corrects a candidate, swaps method,\n"
    "  repairs a contradiction).\n\n"
    "If either part is missing, output FALSE. Be strict; true shifts are rare."
)

_SHIFT_JUDGE_USER_TEMPLATE_V1 = """Problem/Clue (if available):
{problem}

PASS-1 <think> (truncated if long):
{think}

Heuristic cue candidates (may be empty): {cues}
first_marker_pos: {pos}

Return ONLY a compact JSON object with keys:
- shift_in_reasoning: true|false
- confidence: "low"|"medium"|"high"
- markers_found: string[]       (verbatim lexical cues you relied on)
- first_marker_index: integer   (character offset into <think>, -1 if absent)
- before_excerpt: string        (<=120 chars ending right before the first marker)
- after_excerpt: string         (<=140 chars starting at the first marker)
- explanation_short: string     (<=140 chars justification)
"""

_SHIFT_JUDGE_USER_TEMPLATE_V2 = """Problem/Clue (if available):
{problem}

PASS-1 <think> (truncated if long):
{think}

Possible cues (empty is fine): {cues}
first_marker_pos: {pos}

Respond with JSON ONLY, using exactly these keys:
- shift_in_reasoning: true|false
- confidence: "low"|"medium"|"high"
- markers_found: string[]
- first_marker_index: integer (-1 if none)
- before_excerpt: string (<=120 chars before marker)
- after_excerpt: string  (<=140 chars from marker)
- explanation_short: string (<=140 chars, concise justification)
"""

_SHIFT_JUDGE_USER_TEMPLATE_V3 = """Problem/Clue (if available):
{problem}

PASS-1 <think> (truncated if long):
{think}

Candidate cues: {cues}
first_marker_pos: {pos}

Return a JSON object only:
- shift_in_reasoning (true|false)
- confidence ("low"|"medium"|"high")
- markers_found (string[])
- first_marker_index (int, -1 if absent)
- before_excerpt (<=120 chars right before marker)
- after_excerpt (<=140 chars starting at marker)
- explanation_short (<=140 chars, terse rationale)
"""

_SHIFT_JUDGE_USER_TEMPLATE_V4 = """Problem/Clue (if available):
{problem}

PASS-1 <think> (truncated if long):
{think}

Lexical cue hints: {cues}
first_marker_pos: {pos}

Output ONLY JSON with fields:
- shift_in_reasoning: true|false
- confidence: "low"|"medium"|"high"
- markers_found: string[]
- first_marker_index: integer (-1 if no cue)
- before_excerpt: string (<=120 chars ending before marker)
- after_excerpt: string  (<=140 chars starting at marker)
- explanation_short: string (<=140 chars, why you chose the label)
"""

_SHIFT_JUDGE_USER_TEMPLATE_V5 = """Problem/Clue (if available):
{problem}

PASS-1 <think> (truncated if long):
{think}

Heuristic cues (optional): {cues}
first_marker_pos: {pos}

Return JSON only with keys:
- shift_in_reasoning: true|false
- confidence: "low"|"medium"|"high"
- markers_found: string[]
- first_marker_index: integer (-1 if none)
- before_excerpt: string (<=120 chars before marker)
- after_excerpt: string  (<=140 chars from marker onward)
- explanation_short: string (<=140 chars, brief justification)
"""

SHIFT_JUDGE_PROMPT_VARIANTS = {
    "v1": {"system": _SHIFT_JUDGE_SYSTEM_PROMPT_V1, "user": _SHIFT_JUDGE_USER_TEMPLATE_V1},
    "v2": {"system": _SHIFT_JUDGE_SYSTEM_PROMPT_V2, "user": _SHIFT_JUDGE_USER_TEMPLATE_V2},
    "v3": {"system": _SHIFT_JUDGE_SYSTEM_PROMPT_V3, "user": _SHIFT_JUDGE_USER_TEMPLATE_V3},
    "v4": {"system": _SHIFT_JUDGE_SYSTEM_PROMPT_V4, "user": _SHIFT_JUDGE_USER_TEMPLATE_V4},
    "v5": {"system": _SHIFT_JUDGE_SYSTEM_PROMPT_V5, "user": _SHIFT_JUDGE_USER_TEMPLATE_V5},
}

SHIFT_JUDGE_PROMPT_VARIANT_KEYS = tuple(SHIFT_JUDGE_PROMPT_VARIANTS.keys())


def canonicalize_shift_judge_variant(variant: str | None) -> str:
    """Normalize a variant key; falls back to ``v1`` when unknown."""
    key = (str(variant).strip().lower() if variant else "") or "v1"
    if key.isdigit():
        key = f"v{key}"
    return key if key in SHIFT_JUDGE_PROMPT_VARIANTS else "v1"


def get_shift_judge_prompts(variant: str | None):
    """
    Return (system_prompt, user_template) for the requested variant.

    Accepts ``v1``..``v5`` or bare digits (``1``..``5``). Falls back to v1.
    """
    key = canonicalize_shift_judge_variant(variant)
    system_user = SHIFT_JUDGE_PROMPT_VARIANTS.get(key) or SHIFT_JUDGE_PROMPT_VARIANTS["v1"]
    return system_user["system"], system_user["user"]


# Backwards-compatible aliases for the baseline prompt (v1)
SHIFT_JUDGE_SYSTEM_PROMPT = SHIFT_JUDGE_PROMPT_VARIANTS["v1"]["system"]
SHIFT_JUDGE_USER_TEMPLATE = SHIFT_JUDGE_PROMPT_VARIANTS["v1"]["user"]

# Conservative shift prompt used in evaluation.py (Princeton sandbox)
SHIFT_PROMPT = """You are a strict arbiter for detecting *explicit* shifts in reasoning.

RULES (be conservative):
1) Count a shift ONLY if the writer clearly aborts or negates an earlier approach
   using explicit cues like: "wait", "hold on", "on second thought", "instead",
   "scratch that", "I was wrong", "doesn't work", "contradiction", etc.
2) Minor self-corrections (typos, sign tweaks) do NOT count.
3) Vague hedging without an explicit cue does NOT count.

INPUT:
---------
Problem:
{problem}

Pass-1 think text (may include other text around it):
{think}

TASK:
Return **exactly** this JSON object (no extra text):

{{
  "shift": "YES" or "NO",
  "cue": "<≤24 chars phrase copied from the text, or '—'>",
  "justification": "<≤50 chars reason, conservative>"
}}
"""
