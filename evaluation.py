#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
annotate_reasoning_shifts.py
────────────────────────────
Detects PASS-1 "shift in reasoning" (explicit rethinking) in math results and appends
LLM-adjudicated flags to each row.

• Input: result JSONL files under a results_root (e.g., GRPO-1.5B/step50/step0050_test.jsonl)
  Each line should have either:
    Schema A (flat):    { problem, gold_answer, step, split, sample_idx, output, ... }
    Schema B (two-pass):{ ..., pass1:{ output: "<think>...</think><answer>...</answer>", ... }, ... }

• Output: for each input JSONL, writes sibling file with suffix *_shifted.jsonl
  Each row gains the following PASS-1 fields:
    - shift_cand_hits:  [pattern names matched by regex prefilter]
    - shift_llm:        bool (LLM-confirmed shift)
    - shift_phrase:     short cue phrase selected by LLM (or first regex hit)
    - shift_justification: brief reason from LLM (≤50 chars)
    - shift_decided_by: "llm" | "regex-none"     (regex-none when no cue; LLM not called)
    - shift_model:      deployment name used (e.g., "gpt-4o")

• Calls DeepSeek via Princeton AI-Sandbox (Azure-style). We use a sticky fallback that
  never re-tries /responses after a 404 and defaults to Chat Completions.

Usage:
  python annotate_reasoning_shifts.py \
      /n/fs/similarity/open-r1/results/GRPO-1.5B \
      --split test \
      --seed 123 \
      --always_llm         # (optional) ask LLM even without regex cues
      --overwrite          # (optional) re-annotate rows that already have fields

Notes:
  - We ONLY inspect PASS-1 (the think text if present).
  - We’re conservative: a shift requires an explicit cue (e.g., "wait", "on second thought").
"""

import argparse
import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────── AzureOpenAI (Princeton Sandbox) ───────────────────
from openai import AzureOpenAI

SANDBOX_API_KEY  = "1e30d0e4d7564ba984e8adff48053009"
SANDBOX_ENDPOINT = "https://api-ai-sandbox.princeton.edu/"
SANDBOX_API_VER  = "2025-03-01-preview"
DEPLOYMENT_NAME  = "gpt-4o"   # <<< use the deployed name; "gpt-5o" will 404 if not deployed

_client = AzureOpenAI(
    api_key        = SANDBOX_API_KEY,
    azure_endpoint = SANDBOX_ENDPOINT,
    api_version    = SANDBOX_API_VER,
)

_RESPONSES_AVAILABLE = None  # sticky probe (we'll effectively disable for Princeton)

def ds_call(messages, max_tokens: int, temperature: float = 0.0) -> str:
    """
    Prefer Azure Chat Completions on the Princeton sandbox.
    Try /responses at most once; after 404, stick to chat.completions.
    """
    global _RESPONSES_AVAILABLE

    # Hard-disable Responses for known Princeton host (avoids the 404 spam entirely)
    if _RESPONSES_AVAILABLE is None:
        _RESPONSES_AVAILABLE = (
            ("api-ai-sandbox.princeton.edu" not in SANDBOX_ENDPOINT)
        )

    if _RESPONSES_AVAILABLE:
        try:
            # Not generally available on Princeton; kept for portability.
            resp = _client.responses.create(
                model=DEPLOYMENT_NAME,
                input=[{"role": "user", "content": messages[-1]["content"]}],
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            try:
                return resp.output_text
            except Exception:
                chunks = []
                for out in getattr(resp, "output", []):
                    for blk in getattr(out, "content", []):
                        if getattr(blk, "type", "") == "output_text":
                            chunks.append(getattr(blk, "text", ""))
                return "".join(chunks).strip()
        except Exception as e:
            logging.info("Responses API not available (%s). Falling back to chat.completions.", e)
            _RESPONSES_AVAILABLE = False

    # Stable path on Princeton sandbox
    resp = _client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=messages,
    )
    return resp.choices[0].message.content.strip()

# ───────────────────── Regex prefilter (candidate cues) ─────────────────────
# Requires: import re
from typing import List, Tuple

SHIFT_CAND_PATTERNS: List[Tuple[str, re.Pattern]] = [
    # ───────── Pauses / immediate self-interruptions ─────────
    ("wait",                  re.compile(r"(?i)(?:^|\W)wait(?:\W|$)")),
    ("hold on",               re.compile(r"(?i)\bhold (?:on|up)\b")),
    ("hang on",               re.compile(r"(?i)\bhang on\b")),
    ("one sec",               re.compile(r"(?i)\b(?:one|a)\s+(?:sec|second)\b")),
    ("just a sec",            re.compile(r"(?i)\bjust (?:a )?(?:sec|second)\b")),
    ("give me a moment",      re.compile(r"(?i)\bgive me (?:a )?moment\b")),
    ("pause",                 re.compile(r"(?i)\bpause\b")),
    ("on second thought",     re.compile(r"(?i)\bon (?:second|further) thought\b")),
    ("second guess",          re.compile(r"(?i)\bsecond-?guess(?:ing)?\b")),
    ("reconsider",            re.compile(r"(?i)\breconsider\b")),
    ("rethink",               re.compile(r"(?i)\bre-?think(?:ing)?\b")),

    # ───────── Explicit self-corrections / pivots ─────────
    ("actually",              re.compile(r"(?i)(?:^|\W)actually(?:\W|$)")),
    ("in fact",               re.compile(r"(?i)\bin fact\b")),
    ("rather",                re.compile(r"(?i)(?:^|\W)rather(?:\W|$)")),
    ("instead",               re.compile(r"(?i)(?:^|\W)instead(?:\W|$)")),
    ("instead_of",            re.compile(r"(?i)\binstead of\b")),
    ("better",                re.compile(r"(?i)(?:^|\W)better(?:\W|$)")),
    ("prefer",                re.compile(r"(?i)\bI (?:would )?prefer\b")),
    ("let me correct",        re.compile(r"(?i)\blet'?s? (?:correct|fix) (?:that|this)\b")),
    ("correction_keyword",    re.compile(r"(?i)\bcorrection\b")),
    ("correction_colon",      re.compile(r"(?i)\bcorrection:\b")),
    ("to correct",            re.compile(r"(?i)\bto correct\b")),
    ("fix that",              re.compile(r"(?i)\bfix (?:that|this)\b")),
    ("change to",             re.compile(r"(?i)\bchange (?:that|this)?\s*to\b")),
    ("switch to",             re.compile(r"(?i)\bswitch (?:to|over)\b")),
    ("replace with",          re.compile(r"(?i)\breplace (?:it|that|this)?\s*with\b")),
    ("try instead",           re.compile(r"(?i)\btry (?:this|that )?instead\b")),
    ("consider instead",      re.compile(r"(?i)\bconsider (?:instead|alternatively)\b")),
    ("alternate",             re.compile(r"(?i)\balternat(?:e|ive)\b")),
    ("new candidate",         re.compile(r"(?i)\bnew (?:candidate|answer|approach|plan)\b")),
    ("update_colon",          re.compile(r"(?i)\bupdate:\b")),

    # ───────── Negations of previous statement / immediate reversal ─────────
    ("no_comma",              re.compile(r"(?i)(?:^|\W)no[,!\.]?\s+(?:that|this|it)\b")),
    ("nope",                  re.compile(r"(?i)\bnope\b")),
    ("nah",                   re.compile(r"(?i)\bnah\b")),
    ("never mind",            re.compile(r"(?i)\bnever mind\b|\bnvm\b")),
    ("disregard",             re.compile(r"(?i)\b(?:disregard|ignore) (?:that|this|the previous|above)\b")),
    ("scratch that",          re.compile(r"(?i)\bscratch that\b")),
    ("strike that",           re.compile(r"(?i)\bstrike that\b")),
    ("forget that",           re.compile(r"(?i)\bforget (?:that|this)\b")),
    ("I retract",             re.compile(r"(?i)\bI retract\b")),
    ("I take that back",      re.compile(r"(?i)\bI take (?:that|it) back\b")),
    ("I stand corrected",     re.compile(r"(?i)\bI stand corrected\b")),
    ("not X but Y",           re.compile(r"(?i)\bnot\s+\w+(?:\s+\w+)?\s*,?\s+(?:but|rather)\b")),

    # ───────── Admission of error / fault ─────────
    ("wrong_self",            re.compile(r"(?i)\bi (?:was|am) wrong\b")),
    ("wrong_generic",         re.compile(r"(?i)\bthat(?:'s| is)? wrong\b")),
    ("incorrect",             re.compile(r"(?i)\bincorrect\b|\bnot correct\b")),
    ("mistake_generic",       re.compile(r"(?i)\b(?:my )?mistake\b|\bI made a mistake\b")),
    ("my bad",                re.compile(r"(?i)\bmy bad\b")),
    ("oops",                  re.compile(r"(?i)\b(?:oops|whoops|uh[-\s]*oh)\b")),
    ("apologies",             re.compile(r"(?i)\bapolog(?:y|ies|ise|ize)\b")),
    ("erroneous",             re.compile(r"(?i)\berroneous\b")),
    ("error_on_my_part",      re.compile(r"(?i)\b(?:an|the) error (?:on|in) (?:my|the) (?:part|work)\b")),

    # ───────── Mis-* patterns (specific failure types) ─────────
    ("misread",               re.compile(r"(?i)\bmis-?read\b|\bI misread\b")),
    ("miscount",              re.compile(r"(?i)\bmis-?count(?:ed|ing)?\b")),
    ("miscalc",               re.compile(r"(?i)\bmis-?calculat(?:e|ed|ion)\b|\bcalc(?:ulation)? error\b")),
    ("misapply",              re.compile(r"(?i)\bmis-?appl(?:y|ied|ication)\b")),
    ("misparse",              re.compile(r"(?i)\bmis-?pars(?:e|ed|ing)\b")),
    ("misspell",              re.compile(r"(?i)\bmis-?spell(?:ed|ing)?\b|\bmisspelt\b|\bmisspelled\b")),
    ("misindex",              re.compile(r"(?i)\bmis-?index(?:ed|ing)?\b")),
    ("misuse_rule",           re.compile(r"(?i)\bmis-?us(?:e|ed|ing)\b")),
    ("confused_with",         re.compile(r"(?i)\bI (?:confused|mixed up) .* with\b", re.S)),
    ("conflated",             re.compile(r"(?i)\bI conflated\b")),
    ("typo",                  re.compile(r"(?i)\btypo\b")),
    ("off by one",            re.compile(r"(?i)\boff[-\s]?by[-\s]?one\b")),

    # ───────── Constraint/length/pattern mismatch (xword-friendly) ─────────
    ("doesnt fit",            re.compile(r"(?i)\bdoes(?:n'?t| not) (?:fit|match)(?: length| pattern)?\b")),
    ("letters dont fit",      re.compile(r"(?i)\bletters? do(?:es)?n'?t (?:fit|match)\b")),
    ("pattern mismatch",      re.compile(r"(?i)\bpattern (?:mis)?match\b")),
    ("length mismatch",       re.compile(r"(?i)\blength (?:mis)?match\b")),
    ("too many letters",      re.compile(r"(?i)\btoo many letters\b")),
    ("too few letters",       re.compile(r"(?i)\b(?:not enough|too few) letters\b")),
    ("wrong length",          re.compile(r"(?i)\bwrong length\b")),
    ("violates enumeration",  re.compile(r"(?i)\bviolates? (?:the )?enumeration\b")),
    ("doesnt parse",          re.compile(r"(?i)\bdoes(?:n'?t| not) parse\b")),
    ("definition mismatch",   re.compile(r"(?i)\bdefinition (?:doesn'?t|does not) match\b")),
    ("not an anagram of",     re.compile(r"(?i)\bnot an anagram of\b")),
    ("anagram doesnt work",   re.compile(r"(?i)\banagram (?:doesn'?t|does not) (?:work|fit)\b")),
    ("fodder mismatch",       re.compile(r"(?i)\bfodder (?:doesn'?t|does not) (?:match|fit)\b")),

    # ───────── Logical contradiction / impossibility ─────────
    ("contradiction",         re.compile(r"(?i)\bcontradict(?:s|ion|ory)\b")),
    ("inconsistent",          re.compile(r"(?i)\binconsistent\b")),
    ("cant be",               re.compile(r"(?i)\bcan'?t be\b|\bcannot be\b")),
    ("impossible",            re.compile(r"(?i)\bimpossible\b")),
    ("doesnt make sense",     re.compile(r"(?i)\bdoes(?:n'?t| not) make sense\b")),
    ("doesnt add up",         re.compile(r"(?i)\bdoes(?:n'?t| not) add up\b")),
    ("cannot both",           re.compile(r"(?i)\bcan(?:not|'?t) both\b")),
    ("leads to",              re.compile(r"(?i)\bleads to (?:a )?contradiction\b")),
    ("this implies not",      re.compile(r"(?i)\bthis implies .* (?:is|are) not\b", re.S)),

    # ───────── Re-check / review / backtrack ─────────
    ("re-check",              re.compile(r"(?i)\bre-?check(?:ing|ed)?\b")),
    ("double-check",          re.compile(r"(?i)\bdouble-?check(?:ing|ed)?\b")),
    ("check again",           re.compile(r"(?i)\bcheck(?:ing)? again\b")),
    ("re-evaluate",           re.compile(r"(?i)\bre-?evaluat(?:e|ed|ing|ion)\b")),
    ("re-examine",            re.compile(r"(?i)\bre-?examin(?:e|ed|ing|ation)\b")),
    ("on review",             re.compile(r"(?i)\b(?:on|upon) (?:review|reflection|reconsideration)\b")),
    ("backtrack",             re.compile(r"(?i)\bbacktrack(?:ing|ed)?\b")),
    ("start over",            re.compile(r"(?i)\bstart over\b|\brestart\b|\breset\b|\bfrom scratch\b")),

    # ───────── “Previous idea was X, but …” templates ─────────
    ("i_thought_but",         re.compile(r"(?i)\bI (?:first|initially|originally) thought\b.*\b(?:but|however)\b", re.S)),
    ("previously_but",        re.compile(r"(?i)\bpreviously\b.*\b(?:but|however)\b", re.S)),
    ("earlier_but",           re.compile(r"(?i)\bearlier\b.*\b(?:but|however)\b", re.S)),
    ("however+fix",           re.compile(r"(?i)\bhowever\b.*\b(?:correct|fix|instead|rather|change)\b", re.S)),
    ("but_instead",           re.compile(r"(?i)\bbut\b.*\binstead\b", re.S)),
    ("but_rather",            re.compile(r"(?i)\bbut\b.*\brather\b", re.S)),

    # ───────── Oversight / omission admissions ─────────
    ("i forgot",              re.compile(r"(?i)\bI forgot\b")),
    ("i missed",              re.compile(r"(?i)\bI (?:missed|overlooked)\b")),
    ("i didnt notice",        re.compile(r"(?i)\bI did(?:n'?t| not) notice\b")),
    ("i ignored",             re.compile(r"(?i)\bI (?:ignored|skipped)\b")),
    ("i misremembered",       re.compile(r"(?i)\bI mis-?remembered\b")),
    ("i misheard",            re.compile(r"(?i)\bI mis-?heard\b")),

    # ───────── Directional errors / swaps ─────────
    ("reversed",              re.compile(r"(?i)\brevers(?:e|ed|ing)\b")),
    ("got backwards",         re.compile(r"(?i)\bgot .* backwards\b", re.S)),
    ("swapped",               re.compile(r"(?i)\bswapp?ed\b")),
    ("mixed up",              re.compile(r"(?i)\bmix(?:ed)? up\b")),

    # ───────── “turns out / realization” ─────────
    ("turns out",             re.compile(r"(?i)\bturns out\b")),
    ("i realize",             re.compile(r"(?i)\bI (?:now )?real(?:i[sz]e|ising|izing)\b")),
    ("on reflection",         re.compile(r"(?i)\bon reflection\b")),
    ("after all",             re.compile(r"(?i)\bafter all\b")),

    # ───────── Generic “No, that doesn’t …” templates ─────────
    ("no_that_doesnt",        re.compile(r"(?i)(?:^|\W)no[,!\.]?\s+that (?:doesn'?t|does not)\b")),
    ("no_this_doesnt",        re.compile(r"(?i)(?:^|\W)no[,!\.]?\s+this (?:doesn'?t|does not)\b")),
    ("no_it_doesnt",          re.compile(r"(?i)(?:^|\W)no[,!\.]?\s+it (?:doesn'?t|does not)\b")),

    # ───────── “This fails because …” ─────────
    ("fails_because",         re.compile(r"(?i)\bfails? because\b")),
    ("won't work",            re.compile(r"(?i)\bwon'?t work\b")),
    ("not working",           re.compile(r"(?i)\bnot working\b")),
    ("dead end",              re.compile(r"(?i)\bdead end\b")),

    # Keep originals (for completeness & back-compat)
    ("original_no_comma",     re.compile(r"(?i)(?:^|\W)no[,!\.]?\s")),
]


RE_THINK = re.compile(r"(?si)<think>(.*?)</think>")

# ───────────────────── LLM prompt (strict, conservative) ─────────────────────
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

# ────────────────────────── Helpers ──────────────────────────
def extract_pass1_text(rec: Dict[str, Any]) -> Optional[str]:
    """Get the pass-1 full text; prefer the <think> block if present."""
    txt = None
    if isinstance(rec.get("pass1"), dict):
        txt = rec["pass1"].get("output") or rec["pass1"].get("full_text") or rec["pass1"].get("raw")
    txt = txt or rec.get("output") or rec.get("full_text") or rec.get("raw")
    if not txt:
        return None
    m = RE_THINK.search(txt)
    return (m.group(1).strip() if m else txt.strip())

def find_cue_hits(text: str) -> List[str]:
    hits = []
    for name, pat in SHIFT_CAND_PATTERNS:
        if pat.search(text):
            hits.append(name)
    return hits

def llm_decide_shift(problem: str, think_text: str, model_name: str) -> Tuple[bool, str, str]:
    """Call LLM once; parse a strict JSON object."""
    msg = SHIFT_PROMPT.format(problem=problem[:2000], think=think_text[:4000])
    raw = ds_call(
        messages=[
            {"role": "system", "content": "You are concise and conservative."},
            {"role": "user",   "content": msg},
        ],
        max_tokens=140,
        temperature=0.0,
    )
    # Strip code fences if any
    raw = raw.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.splitlines()[1:-1]).strip()
    try:
        obj = json.loads(raw)
    except Exception:
        logging.warning("Non-JSON LLM output; treating as NO. Raw: %s", raw[:200])
        return False, "—", "parse-fail"
    shift = str(obj.get("shift", "NO")).upper() == "YES"
    cue = str(obj.get("cue", "—")).strip() or "—"
    just = str(obj.get("justification", "")).strip()[:50]
    return shift, cue, just or ("explicit cue" if shift else "no explicit cue")

def should_skip(rec: Dict[str, Any], overwrite: bool) -> bool:
    if overwrite:
        return False
    # If we already annotated, skip
    return all(k in rec for k in ("shift_llm", "shift_phrase", "shift_justification"))

def scan_files(root: Path, split: Optional[str]) -> List[Path]:
    files = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if not fn.endswith(".jsonl"):
                continue
            # Avoid re-processing already annotated outputs
            if fn.endswith("_shifted.jsonl"):
                continue
            if split and split not in fn:
                continue
            files.append(Path(dp) / fn)
    files.sort()
    return files

# ────────────────────────── Main ──────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_root", help="Directory containing step*/.../*.jsonl")
    ap.add_argument("--split", default="test", help="Substring filter for filenames (e.g. 'test')")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--always_llm", action="store_true",
                    help="Call LLM even when no regex cue hits (defaults to NO for speed).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-annotate rows even if fields already present.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    random.seed(args.seed)

    root = Path(args.results_root)
    files = scan_files(root, args.split)
    if not files:
        logging.error("No JSONL files found under %s (split filter: %r).", root, args.split)
        sys.exit(1)

    logging.info("Found %d files to annotate.", len(files))

    for path in files:
        logging.info("Annotating: %s", path)
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue

        # Create randomized processing order (but keep original order in output)
        order = list(range(len(rows)))
        random.shuffle(order)

        # Work on a copy of rows; write annotations back into 'rows'
        for idx in order:
            rec = rows[idx]
            if should_skip(rec, args.overwrite):
                continue

            think_text = extract_pass1_text(rec) or ""
            problem = rec.get("problem", "") or rec.get("clue", "")

            cand_hits = find_cue_hits(think_text)
            shift_llm = False
            shift_phrase = "—"
            shift_just = "no explicit cue"
            decided_by = "regex-none"

            # If we see candidate cues OR user insists, ask the LLM to confirm.
            if cand_hits or args.always_llm:
                shift_llm, shift_phrase, shift_just = llm_decide_shift(problem, think_text, DEPLOYMENT_NAME)
                decided_by = "llm"

            rec["shift_cand_hits"]     = cand_hits
            rec["shift_llm"]           = bool(shift_llm)
            rec["shift_phrase"]        = shift_phrase
            rec["shift_justification"] = shift_just
            rec["shift_decided_by"]    = decided_by
            rec["shift_model"]         = DEPLOYMENT_NAME

        out = path.with_name(path.stem + "_shifted.jsonl")
        with out.open("w", encoding="utf-8") as w:
            for r in rows:
                w.write(json.dumps(r, ensure_ascii=False) + "\n")
        logging.info("Wrote %s", out)

if __name__ == "__main__":
    main()
