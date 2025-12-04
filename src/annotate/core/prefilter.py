"""Shared regex utilities for shift-cue detection and tag extraction."""

import re
from typing import List, Optional, Tuple


RE_THINK = re.compile(r"(?si)<think>(.*?)</think>")
RE_ANSWER = re.compile(r"(?si)<answer>(.*?)</answer>")

# Strict cue list (be conservative; no generic 'but' alone)
SHIFT_CAND_PATTERNS: List[Tuple[str, re.Pattern]] = [
    # ───────── Pauses / immediate self-interruptions ─────────
    ("wait", re.compile(r"(?i)(?:^|\W)wait(?:\W|$)")),
    ("hold on", re.compile(r"(?i)\bhold (?:on|up)\b")),
    ("hang on", re.compile(r"(?i)\bhang on\b")),
    ("one sec", re.compile(r"(?i)\b(?:one|a)\s+(?:sec|second)\b")),
    ("just a sec", re.compile(r"(?i)\bjust (?:a )?(?:sec|second)\b")),
    ("give me a moment", re.compile(r"(?i)\bgive me (?:a )?moment\b")),
    ("pause", re.compile(r"(?i)\bpause\b")),
    ("on second thought", re.compile(r"(?i)\bon (?:second|further) thought\b")),
    ("second guess", re.compile(r"(?i)\bsecond-?guess(?:ing)?\b")),
    ("reconsider", re.compile(r"(?i)\breconsider\b")),
    ("rethink", re.compile(r"(?i)\bre-?think(?:ing)?\b")),
    # ───────── Explicit self-corrections / pivots ─────────
    ("actually", re.compile(r"(?i)(?:^|\W)actually(?:\W|$)")),
    ("in fact", re.compile(r"(?i)\bin fact\b")),
    ("rather", re.compile(r"(?i)(?:^|\W)rather(?:\W|$)")),
    ("instead", re.compile(r"(?i)(?:^|\W)instead(?:\W|$)")),
    ("instead_of", re.compile(r"(?i)\binstead of\b")),
    ("better", re.compile(r"(?i)(?:^|\W)better(?:\W|$)")),
    ("prefer", re.compile(r"(?i)\bI (?:would )?prefer\b")),
    ("let me correct", re.compile(r"(?i)\blet'?s? (?:correct|fix) (?:that|this)\b")),
    ("correction_keyword", re.compile(r"(?i)\bcorrection\b")),
    ("correction_colon", re.compile(r"(?i)\bcorrection:\b")),
    ("to correct", re.compile(r"(?i)\bto correct\b")),
    ("fix that", re.compile(r"(?i)\bfix (?:that|this)\b")),
    ("change to", re.compile(r"(?i)\bchange (?:that|this)?\s*to\b")),
    ("switch to", re.compile(r"(?i)\bswitch (?:to|over)\b")),
    ("replace with", re.compile(r"(?i)\breplace (?:it|that|this)?\s*with\b")),
    ("try instead", re.compile(r"(?i)\btry (?:this|that )?instead\b")),
    ("consider instead", re.compile(r"(?i)\bconsider (?:instead|alternatively)\b")),
    ("alternate", re.compile(r"(?i)\balternat(?:e|ive)\b")),
    ("new candidate", re.compile(r"(?i)\bnew (?:candidate|answer|approach|plan)\b")),
    ("update_colon", re.compile(r"(?i)\bupdate:\b")),
    # ───────── Negations of previous statement / immediate reversal ─────────
    ("no_comma", re.compile(r"(?i)(?:^|\W)no[,!\.]?\s+(?:that|this|it)\b")),
    ("nope", re.compile(r"(?i)\bnope\b")),
    ("nah", re.compile(r"(?i)\bnah\b")),
    ("never mind", re.compile(r"(?i)\bnever mind\b|\bnvm\b")),
    ("disregard", re.compile(r"(?i)\b(?:disregard|ignore) (?:that|this|the previous|above)\b")),
    ("scratch that", re.compile(r"(?i)\bscratch that\b")),
    ("strike that", re.compile(r"(?i)\bstrike that\b")),
    ("forget that", re.compile(r"(?i)\bforget (?:that|this)\b")),
    ("I retract", re.compile(r"(?i)\bI retract\b")),
    ("I take that back", re.compile(r"(?i)\bI take (?:that|it) back\b")),
    ("I stand corrected", re.compile(r"(?i)\bI stand corrected\b")),
    ("not X but Y", re.compile(r"(?i)\bnot\s+\w+(?:\s+\w+)?\s*,?\s+(?:but|rather)\b")),
    # ───────── Admission of error / fault ─────────
    ("wrong_self", re.compile(r"(?i)\bi (?:was|am) wrong\b")),
    ("wrong_generic", re.compile(r"(?i)\bthat(?:'s| is)? wrong\b")),
    ("incorrect", re.compile(r"(?i)\bincorrect\b|\bnot correct\b")),
    ("mistake_generic", re.compile(r"(?i)\b(?:my )?mistake\b|\bI made a mistake\b")),
    ("my bad", re.compile(r"(?i)\bmy bad\b")),
    ("oops", re.compile(r"(?i)\b(?:oops|whoops|uh[-\s]*oh)\b")),
    ("apologies", re.compile(r"(?i)\bapolog(?:y|ies|ise|ize)\b")),
    ("erroneous", re.compile(r"(?i)\berroneous\b")),
    (
        "error_on_my_part",
        re.compile(r"(?i)\b(?:an|the) error (?:on|in) (?:my|the) (?:part|work)\b"),
    ),
    # ───────── Mis-* patterns (specific failure types) ─────────
    ("misread", re.compile(r"(?i)\bmis-?read\b|\bI misread\b")),
    ("miscount", re.compile(r"(?i)\bmis-?count(?:ed|ing)?\b")),
    ("miscalc", re.compile(r"(?i)\bmis-?calculat(?:e|ed|ion)\b|\bcalc(?:ulation)? error\b")),
    ("misapply", re.compile(r"(?i)\bmis-?appl(?:y|ied|ication)\b")),
    ("misparse", re.compile(r"(?i)\bmis-?pars(?:e|ed|ing)\b")),
    ("misspell", re.compile(r"(?i)\bmis-?spell(?:ed|ing)?\b|\bmisspelt\b|\bmisspelled\b")),
    ("misindex", re.compile(r"(?i)\bmis-?index(?:ed|ing)?\b")),
    ("misuse_rule", re.compile(r"(?i)\bmis-?us(?:e|ed|ing)\b")),
    ("confused_with", re.compile(r"(?i)\bI (?:confused|mixed up) .* with\b", re.S)),
    ("conflated", re.compile(r"(?i)\bI conflated\b")),
    ("typo", re.compile(r"(?i)\btypo\b")),
    ("off by one", re.compile(r"(?i)\boff[-\s]?by[-\s]?one\b")),
    # ───────── Constraint/length/pattern mismatch (xword-friendly) ─────────
    ("doesnt fit", re.compile(r"(?i)\bdoes(?:n'?t| not) (?:fit|match)(?: length| pattern)?\b")),
    ("letters dont fit", re.compile(r"(?i)\bletters? do(?:es)?n'?t (?:fit|match)\b")),
    ("pattern mismatch", re.compile(r"(?i)\bpattern (?:mis)?match\b")),
    ("length mismatch", re.compile(r"(?i)\blength (?:mis)?match\b")),
    ("too many letters", re.compile(r"(?i)\btoo many letters\b")),
    ("too few letters", re.compile(r"(?i)\b(?:not enough|too few) letters\b")),
    ("wrong length", re.compile(r"(?i)\bwrong length\b")),
    ("violates enumeration", re.compile(r"(?i)\bviolates? (?:the )?enumeration\b")),
    ("doesnt parse", re.compile(r"(?i)\bdoes(?:n'?t| not) parse\b")),
    ("definition mismatch", re.compile(r"(?i)\bdefinition (?:doesn'?t|does not) match\b")),
    ("not an anagram of", re.compile(r"(?i)\bnot an anagram of\b")),
    ("anagram doesnt work", re.compile(r"(?i)\banagram (?:doesn'?t|does not) (?:work|fit)\b")),
    ("fodder mismatch", re.compile(r"(?i)\bfodder (?:doesn'?t|does not) (?:match|fit)\b")),
    # ───────── Logical contradiction / impossibility ─────────
    ("contradiction", re.compile(r"(?i)\bcontradict(?:s|ion|ory)\b")),
    ("inconsistent", re.compile(r"(?i)\binconsistent\b")),
    ("cant be", re.compile(r"(?i)\bcan'?t be\b|\bcannot be\b")),
    ("impossible", re.compile(r"(?i)\bimpossible\b")),
    ("doesnt make sense", re.compile(r"(?i)\bdoes(?:n'?t| not) make sense\b")),
    ("doesnt add up", re.compile(r"(?i)\bdoes(?:n'?t| not) add up\b")),
    ("cannot both", re.compile(r"(?i)\bcan(?:not|'?t) both\b")),
    ("leads to", re.compile(r"(?i)\bleads to (?:a )?contradiction\b")),
    ("this implies not", re.compile(r"(?i)\bthis implies .* (?:is|are) not\b", re.S)),
    # ───────── Re-check / review / backtrack ─────────
    ("re-check", re.compile(r"(?i)\bre-?check(?:ing|ed)?\b")),
    ("double-check", re.compile(r"(?i)\bdouble-?check(?:ing|ed)?\b")),
    ("check again", re.compile(r"(?i)\bcheck(?:ing)? again\b")),
    ("re-evaluate", re.compile(r"(?i)\bre-?evaluat(?:e|ed|ing|ion)\b")),
    ("re-examine", re.compile(r"(?i)\bre-?examin(?:e|ed|ing|ation)\b")),
    ("on review", re.compile(r"(?i)\b(?:on|upon) (?:review|reflection|reconsideration)\b")),
    ("backtrack", re.compile(r"(?i)\bbacktrack(?:ing|ed)?\b")),
    ("start over", re.compile(r"(?i)\bstart over\b|\brestart\b|\breset\b|\bfrom scratch\b")),
    # ───────── “Previous idea was X, but …” templates ─────────
    (
        "i_thought_but",
        re.compile(
            r"(?i)\bI (?:first|initially|originally) thought\b.*\b(?:but|however)\b",
            re.S,
        ),
    ),
    (
        "previously_but",
        re.compile(r"(?i)\bpreviously\b.*\b(?:but|however)\b", re.S),
    ),
    (
        "earlier_but",
        re.compile(r"(?i)\bearlier\b.*\b(?:but|however)\b", re.S),
    ),
    (
        "however+fix",
        re.compile(
            r"(?i)\bhowever\b.*\b(?:correct|fix|instead|rather|change)\b",
            re.S,
        ),
    ),
    ("but_instead", re.compile(r"(?i)\bbut\b.*\binstead\b", re.S)),
    ("but_rather", re.compile(r"(?i)\bbut\b.*\brather\b", re.S)),
    # ───────── Oversight / omission admissions ─────────
    ("i forgot", re.compile(r"(?i)\bI forgot\b")),
    ("i missed", re.compile(r"(?i)\bI (?:missed|overlooked)\b")),
    ("i didnt notice", re.compile(r"(?i)\bI did(?:n'?t| not) notice\b")),
    ("i ignored", re.compile(r"(?i)\bI (?:ignored|skipped)\b")),
    ("i misremembered", re.compile(r"(?i)\bI mis-?remembered\b")),
    ("i misheard", re.compile(r"(?i)\bI mis-?heard\b")),
    # ───────── Directional errors / swaps ─────────
    ("reversed", re.compile(r"(?i)\brevers(?:e|ed|ing)\b")),
    ("got backwards", re.compile(r"(?i)\bgot .* backwards\b", re.S)),
    ("swapped", re.compile(r"(?i)\bswapp?ed\b")),
    ("mixed up", re.compile(r"(?i)\bmix(?:ed)? up\b")),
    # ───────── “turns out / realization” ─────────
    ("turns out", re.compile(r"(?i)\bturns out\b")),
    ("i realize", re.compile(r"(?i)\bI (?:now )?real(?:i[sz]e|ising|izing)\b")),
    ("on reflection", re.compile(r"(?i)\bon reflection\b")),
    ("after all", re.compile(r"(?i)\bafter all\b")),
    # ───────── Generic “No, that doesn’t …” templates ─────────
    ("no_that_doesnt", re.compile(r"(?i)(?:^|\W)no[,!\.]?\s+that (?:doesn'?t|does not)\b")),
    ("no_this_doesnt", re.compile(r"(?i)(?:^|\W)no[,!\.]?\s+this (?:doesn'?t|does not)\b")),
    ("no_it_doesnt", re.compile(r"(?i)(?:^|\W)no[,!\.]?\s+it (?:doesn'?t|does not)\b")),
    # ───────── “This fails because …” ─────────
    ("fails_because", re.compile(r"(?i)\bfails? because\b")),
    ("won't work", re.compile(r"(?i)\bwon'?t work\b")),
    ("not working", re.compile(r"(?i)\bnot working\b")),
    ("dead end", re.compile(r"(?i)\bdead end\b")),
    # Keep the original broad "no,..." pattern (for audit/back-compat)
    ("original_no_comma", re.compile(r"(?i)(?:^|\W)no[,!\.]?\s")),
]


def extract_think(txt: str) -> Optional[str]:
    """Return the <think> block contents, if present."""
    match = RE_THINK.search(txt or "")
    return match.group(1).strip() if match else None


def find_shift_cues(think: str) -> Tuple[List[str], Optional[int]]:
    """Find cue names and the earliest character index in the think text."""
    if not think:
        return [], None
    hits: List[str] = []
    first_pos = None
    for name, pat in SHIFT_CAND_PATTERNS:
        match = pat.search(think)
        if match:
            hits.append(name)
            pos = match.start()
            if first_pos is None or pos < first_pos:
                first_pos = pos
    return hits, first_pos


def find_cue_hits(think: str) -> List[str]:
    """Return only the names of cues found in think text."""
    return find_shift_cues(think)[0]


__all__ = [
    "RE_THINK",
    "RE_ANSWER",
    "SHIFT_CAND_PATTERNS",
    "extract_think",
    "find_shift_cues",
    "find_cue_hits",
]
