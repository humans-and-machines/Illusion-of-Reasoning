"""
Centralized prompt templates for annotation/judging.
"""

# Strict judge for shift detection (used by gpt-eval.py)
SHIFT_JUDGE_SYSTEM_PROMPT = (
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

SHIFT_JUDGE_USER_TEMPLATE = """Problem/Clue (if available):
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
