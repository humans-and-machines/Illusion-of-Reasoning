#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Annotate pass-1 'shift in reasoning' for inference JSONL outputs — with K prompt variants,
tight rubric (soft validation), parameterized prefilter, retries with backoff, sidecar dumps,
and pairwise κ over the successful subset.

This version reduces variance:
  • Prompt variants are minimally different.
  • Evidence/variant shuffling is OFF by default (flags can re-enable).
  • Ensemble = simple majority by default (flag can use split-merge).

Usage (reduced-variance defaults):
  python gpt-cohens-kappa.py /path/to/file_or_dir \
    --k 5 --prefilter_mode medium --max_sep 350 \
    --dump_variants ./variant_dumps --report_json interprompt_report.json
"""

import os
import re
import json
import time
import sys
import argparse
import random
import logging
import hashlib
import os.path as _osp
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Tuple

# ─────────────────── API config (env-driven) ───────────────────
sandbox_api_key  = os.getenv("SANDBOX_API_KEY", "")
sandbox_endpoint = os.getenv("SANDBOX_ENDPOINT", "https://api-ai-sandbox.princeton.edu/")
sandbox_api_ver  = os.getenv("SANDBOX_API_VER", "2025-03-01-preview")
DEEPSEEK_MODEL   = os.getenv("SANDBOX_DEPLOYMENT", "gpt-4o")

# ─────────────────── Azure OpenAI SDK ───────────────────
try:
    from openai import AzureOpenAI
except Exception:
    AzureOpenAI = None
    print("ERROR: openai>=1.x with AzureOpenAI is required. pip install openai", file=sys.stderr)

@contextmanager
def timed(label):
    start = time.time()
    yield
    logging.debug(f"[TIMING] {label}: {time.time() - start:,.2f}s")

MAX_FIELD_LEN = 4096
def _clamp(txt: str, lim: int = MAX_FIELD_LEN) -> str:
    txt = txt or ""
    return txt[-lim:] if len(txt) > lim else txt

_client = None
def _client_lazy():
    global _client
    if _client is None:
        if not sandbox_api_key:
            raise RuntimeError("Set SANDBOX_API_KEY (and optionally SANDBOX_ENDPOINT/DEPLOYMENT) in your environment or .env")
        if AzureOpenAI is None:
            raise RuntimeError("AzureOpenAI client not available. Install openai>=1.x.")
        _client = AzureOpenAI(
            api_key        = sandbox_api_key,
            azure_endpoint = sandbox_endpoint,
            api_version    = sandbox_api_ver,
        )
    return _client

def _dump_filtered(prompt: str, reason: str = "UNPARSEABLE"):
    ts = time.strftime("%Y%m%d_%H%M%S")
    dg = hashlib.md5(prompt.encode()).hexdigest()[:8]
    fn = f"filtered_prompt_{ts}_{dg}.txt"
    with open(fn, "w", encoding="utf-8") as f:
        f.write(prompt)
    logging.warning("Judge filtered/failed (%s); saved prompt to %s", reason, fn)

# ─────────────────── Extractors ───────────────────
RE_THINK  = re.compile(r"(?si)<think>(.*?)</think>")
RE_ANSWER = re.compile(r"(?si)<answer>(.*?)</answer>")

def _extract_think(txt: str) -> Optional[str]:
    m = RE_THINK.search(txt or "")
    return m.group(1).strip() if m else None

def _json_from_text(s: str) -> Optional[Dict[str, Any]]:
    s = (s or "").strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            return json.loads(s)
        except Exception:
            pass
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j != -1 and j > i:
        try:
            return json.loads(s[i:j+1])
        except Exception:
            return None
    return None

# ─────────────────── Tight rubric (soft validation) ───────────────────
PROMPT_SYSTEM = (
    "You are a strict annotator of single-pass reasoning transcripts. Decide whether there is a CLEAR, "
    "EXPLICIT 'shift in reasoning' inside <think>…</think>.\n\n"
    "POLICY (rare TRUE):\n"
    "  TRUE requires BOTH:\n"
    "   (A) an explicit cue (e.g., 'wait', 'hold on', 'on second thought', 'actually', 'scratch that',\n"
    "       'I misread', 're-check', 'doesn't fit/match', 'contradiction'); AND\n"
    "   (B) a material revision of an earlier idea (reject/correct a prior hypothesis, switch candidate, fix contradiction).\n"
    "  FALSE for rhetorical transitions or hedges ('however', 'anyway', 'maybe', 'perhaps') without a correction.\n\n"
    "OUTPUT: Reply ONLY with a single JSON object:\n"
    "{\n"
    '  "label": true|false,\n'
    '  "reason_category": "self-correction"|"constraint-mismatch"|"contradiction-resolution"|"recompute"|"other",\n'
    '  "cue_text": string,            # ≤40 chars; REQUIRED IFF label=true\n'
    '  "cue_char_index": integer,     # char offset into <think>; -1 if absent\n'
    '  "prior_hypothesis_quote": string,   # ≤80 chars; REQUIRED IFF label=true\n'
    '  "revision_quote": string,           # ≤80 chars; REQUIRED IFF label=true\n'
    '  "markers_found": string[],\n'
    '  "explanation_short": string,   # ≤140 chars; state cue + what changed\n'
    '  "confidence": "low"|"medium"|"high",\n'
    '  "rationale_quality": 1|2|3|4|5,\n'
    '  "rule_violations": string[]\n'
    "}\n\n"
    "VALIDATION RULES:\n"
    " • If label=true but fields are missing, STILL return label=true but list violations like "
    '   "missing-required-fields:cue_text,revision_quote".\n'
    " • If unsure or evidence incomplete, return label=false with a violation note.\n"
    " • Temperature must be 0. JSON only; no Markdown/code fences."
)

# Minimal-variation variants (near-identical wording)
PROMPT_VARIANTS = [
    ("v1", "Apply the policy exactly. JSON ONLY."),
    ("v2", "Apply the same policy. JSON ONLY."),
    ("v3", "Policy as stated. JSON ONLY."),
    ("v4", "Use the policy verbatim. JSON ONLY."),
    ("v5", "Return only the JSON per policy."),
]

# ─────────────────── Patterns ───────────────────
SHIFT_CAND_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("wait",                  re.compile(r"(?i)(?:^|\W)wait(?:\W|$)")),
    ("hold on",               re.compile(r"(?i)\bhold (?:on|up)\b")),
    ("hang on",               re.compile(r"(?i)\bhang on\b")),
    ("on second thought",     re.compile(r"(?i)\bon (?:second|further) thought\b")),
    ("reconsider",            re.compile(r"(?i)\breconsider\b")),
    ("rethink",               re.compile(r"(?i)\bre-?think(?:ing)?\b")),
    ("actually",              re.compile(r"(?i)(?:^|\W)actually(?:\W|$)")),
    ("instead",               re.compile(r"(?i)(?:^|\W)instead(?:\W|$)")),
    ("let me correct",        re.compile(r"(?i)\blet'?s? (?:correct|fix) (?:that|this)\b")),
    ("correction",            re.compile(r"(?i)\bcorrection\b")),
    ("scratch that",          re.compile(r"(?i)\bscratch that\b")),
    ("replace with",          re.compile(r"(?i)\breplace (?:it|that|this)?\s*with\b")),
    ("try instead",           re.compile(r"(?i)\btry (?:this|that )?instead\b")),
    ("new candidate",         re.compile(r"(?i)\bnew (?:candidate|answer|approach)\b")),
    ("update:",               re.compile(r"(?i)\bupdate:\b")),
    ("never mind",            re.compile(r"(?i)\bnever mind\b|\bnvm\b")),
    ("disregard",             re.compile(r"(?i)\b(?:disregard|ignore) (?:that|this|the previous|above)\b")),
    ("I stand corrected",     re.compile(r"(?i)\bI stand corrected\b")),
    ("not X but Y",           re.compile(r"(?i)\bnot\s+\w+(?:\s+\w+)?\s*,?\s+(?:but|rather)\b")),
    ("I was wrong",           re.compile(r"(?i)\bi (?:was|am) wrong\b")),
    ("incorrect",             re.compile(r"(?i)\bincorrect\b|\bnot correct\b")),
    ("mistake",               re.compile(r"(?i)\b(?:my )?mistake\b|\bI made a mistake\b")),
    ("oops",                  re.compile(r"(?i)\b(?:oops|whoops|uh[-\s]*oh)\b")),
    ("misread",               re.compile(r"(?i)\bmis-?read\b|\bI misread\b")),
    ("miscount",              re.compile(r"(?i)\bmis-?count(?:ed|ing)?\b")),
    ("miscalc",               re.compile(r"(?i)\bmis-?calculat(?:e|ed|ion)\b")),
    ("doesn't fit",           re.compile(r"(?i)\bdoes(?:n'?t| not) (?:fit|match)(?: length| pattern)?\b")),
    ("letters don't fit",     re.compile(r"(?i)\bletters? do(?:es)?n'?t (?:fit|match)\b")),
    ("pattern mismatch",      re.compile(r"(?i)\bpattern (?:mis)?match\b")),
    ("length mismatch",       re.compile(r"(?i)\blength (?:mis)?match\b")),
    ("doesn't parse",         re.compile(r"(?i)\bdoes(?:n'?t| not) parse\b")),
    ("definition mismatch",   re.compile(r"(?i)\bdefinition (?:doesn'?t|does not) match\b")),
    ("contradiction",         re.compile(r"(?i)\bcontradict(?:s|ion|ory)\b")),
    ("inconsistent",          re.compile(r"(?i)\binconsistent\b")),
    ("can't be",              re.compile(r"(?i)\bcan'?t be\b|\bcannot be\b")),
    ("doesn't add up",        re.compile(r"(?i)\bdoes(?:n'?t| not) add up\b")),
    ("re-check",              re.compile(r"(?i)\bre-?check(?:ing|ed)?\b")),
    ("double-check",          re.compile(r"(?i)\bdouble-?check(?:ing|ed)?\b")),
    ("re-evaluate",           re.compile(r"(?i)\bre-?evaluat(?:e|ed|ing|ion)\b")),
    ("backtrack",             re.compile(r"(?i)\bbacktrack(?:ing|ed)?\b")),
    ("start over",            re.compile(r"(?i)\bstart over\b|\brestart\b|\bfrom scratch\b")),
]

NON_SHIFT_PATTERNS_STRICT = [
    re.compile(r"(?i)\bmaybe\b"), re.compile(r"(?i)\bperhaps\b"),
    re.compile(r"(?i)\bi think\b"), re.compile(r"(?i)\bi guess\b"),
    re.compile(r"(?i)\bit could be\b"), re.compile(r"(?i)\bprobably\b"),
    re.compile(r"(?i)\bpossibly\b"), re.compile(r"(?i)\bmight be\b"),
    re.compile(r"(?i)\bhowever\b"), re.compile(r"(?i)\banyway\b"),
    re.compile(r"(?i)\bregardless\b"), re.compile(r"(?i)\bthat said\b"),
    re.compile(r"(?i)\bin conclusion\b"), re.compile(r"(?i)\bto sum up\b"),
    re.compile(r"(?i)\bso\b(?=\s)"), re.compile(r"(?i)\bbecause\b"),
    re.compile(r"(?i)\btherefore\b"), re.compile(r"(?i)\bnext\b"),
    re.compile(r"(?i)\bthen\b"), re.compile(r"(?i)\bfirst\b|\bsecond\b|\bthird\b"),
    re.compile(r"(?i)\blet'?s (?:move on|continue|consider)\b"),
    re.compile(r"(?i)\bok(?:ay)?[,!]?\b|\balright[,!]?\b"),
    re.compile(r"(?i)\bnow\b(?=\s)"),
    re.compile(r"(?i)\b(?:let'?s )?check\b(?!.*\b(?:wrong|incorrect|doesn'?t|cannot|reject|scratch|instead|new candidate)\b)"),
]
NON_SHIFT_PATTERNS_SOFT = [
    re.compile(r"(?i)\bmaybe\b"), re.compile(r"(?i)\bperhaps\b"),
    re.compile(r"(?i)\bi think\b"), re.compile(r"(?i)\bi guess\b"),
    re.compile(r"(?i)\bit could be\b"), re.compile(r"(?i)\bprobably\b"),
    re.compile(r"(?i)\bpossibly\b"), re.compile(r"(?i)\bmight be\b"),
    re.compile(r"(?i)\banyway\b"), re.compile(r"(?i)\bregardless\b"),
    re.compile(r"(?i)\bthat said\b"),
    re.compile(r"(?i)\blet'?s (?:move on|continue|consider)\b"),
    re.compile(r"(?i)\bok(?:ay)?[,!]?\b|\balright[,!]?\b"),
    re.compile(r"(?i)\b(?:let'?s )?check\b(?!.*\b(?:wrong|incorrect|doesn'?t|cannot|reject|scratch|instead|new candidate)\b)"),
]

REVISION_PATTERNS = [
    re.compile(r"(?i)\bnot\s+\w+(?:\s+\w+)?\s*,?\s+(?:but|rather)\b"),
    re.compile(r"(?i)\bscratch that\b|(?:strike|forget) that"),
    re.compile(r"(?i)\bnew (?:candidate|answer|approach)\b"),
    re.compile(r"(?i)\b(replace|change|switch)\s+(?:it|that|this)?\s*(?:to|with)\b"),
    re.compile(r"(?i)\btry (?:this|that )?instead\b"),
    re.compile(r"(?i)\b(recompute|recalculate|correct)\b"),
    re.compile(r"(?i)\bdoes(?:n'?t| not) (?:fit|match)\b"),
    re.compile(r"(?i)\bcontradiction\b|\binconsistent\b|\bcan(?:not|'?t) be\b"),
]

def _find_shift_cues(think: str) -> Tuple[List[str], Optional[int]]:
    if not think:
        return [], None
    hits, first_pos = [], None
    for name, pat in SHIFT_CAND_PATTERNS:
        m = pat.search(think)
        if m:
            hits.append(name)
            pos = m.start()
            if first_pos is None or pos < first_pos:
                first_pos = pos
    return hits, first_pos

def _has_revision_signal(think: str) -> bool:
    if not think:
        return False
    return any(p.search(think) for p in REVISION_PATTERNS)

def _cue_and_revision_close(think: str, cue_pos: Optional[int], max_sep: int = 220) -> bool:
    if cue_pos is None or cue_pos < 0:
        return False
    window = think[cue_pos: cue_pos + max_sep]
    return _has_revision_signal(window)

def _has_strong_cue(think: str) -> bool:
    s = (think or "").lower()
    return any(k in s for k in ["scratch that", "stand corrected", "not ", "disregard", "forget that", "replace", "try instead"])

def _nonshift_hit_mode(think: str, mode: str) -> bool:
    pats = NON_SHIFT_PATTERNS_STRICT if mode == "strict" else (NON_SHIFT_PATTERNS_SOFT if mode == "medium" else [])
    return any(p.search(think or "") for p in pats)

# ─────────────────── Model calls (temp=0) with retries ───────────────────
def _call_openai_responses(client, user_content: str) -> Optional[str]:
    resp = client.responses.create(
        model=DEEPSEEK_MODEL,
        input=[{"role": "system", "content": PROMPT_SYSTEM},
               {"role": "user", "content": user_content}],
        temperature=0.0,
        max_output_tokens=500,
    )
    content = getattr(resp, "output_text", None)
    if content is None and hasattr(resp, "output") and hasattr(resp.output, "text"):
        content = resp.output.text
    return content

def _call_openai_chat(client, user_content: str) -> Optional[str]:
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        temperature=0.0,
        max_tokens=500,
        messages=[
            {"role": "system", "content": PROMPT_SYSTEM},
            {"role": "user", "content": user_content},
        ],
    )
    return resp.choices[0].message.content

def _render_evidence(problem: str, think: str, cues: List[str], pos: Optional[int],
                     shuffle: bool, rng: random.Random) -> str:
    blocks = [
        ("Problem/Clue", _clamp(problem or "(unknown)", 800)),
        ("PASS-1 <think> (truncated)", _clamp(think or "", 1400)),
        ("Heuristic cue candidates", ", ".join(cues) if cues else "(none)"),
        ("first_marker_pos", str(-1 if pos is None else int(pos))),
    ]
    if shuffle:
        rng.shuffle(blocks)
    return "\n\n".join(f"{k}:\n{v}" for k, v in blocks)

def build_user_prompt(problem: str, think: str, cues: List[str], pos: Optional[int],
                      variant_text: str, rng: random.Random, retry_ix: int,
                      shuffle_evidence: bool) -> str:
    evidence = _render_evidence(problem, think, cues, pos, shuffle_evidence, rng)
    retry_hint = ""
    if retry_ix > 0:
        retry_hint = (
            f"\n\nRETRY #{retry_ix}: Your previous reply was not valid JSON. "
            "Respond ONLY with a single JSON object matching the schema."
        )
    return f"{variant_text}{retry_hint}\n\n{evidence}\n"

def judge_once_with_retries(problem: str, think: str, cues: List[str], pos: Optional[int],
                            variant_text: str, rng: random.Random,
                            max_retries: int, base_backoff: float,
                            shuffle_evidence: bool, deterministic: bool) -> Dict[str, Any]:
    client = _client_lazy()
    last_user_content = ""
    for attempt in range(max_retries + 1):
        user_content = build_user_prompt(problem, think, cues, pos, variant_text, rng, attempt, shuffle_evidence)
        last_user_content = user_content
        content = None
        try:
            content = _call_openai_responses(client, user_content)
        except Exception:
            try:
                content = _call_openai_chat(client, user_content)
            except Exception as e_fb:
                if attempt < max_retries:
                    sleep_s = base_backoff * (2 ** attempt)
                    if not deterministic:
                        sleep_s *= (1.0 + rng.uniform(0, 0.25))
                    time.sleep(sleep_s)
                    continue
                _dump_filtered(user_content + "\n\n[ERROR] " + repr(e_fb), reason="API_ERROR")
                return {
                    "label": False, "correctness": "uncertain", "shift_type": "other",
                    "confidence": "low", "rationale_quality": 1, "markers_found": [],
                    "first_marker_index": -1 if pos is None else int(pos),
                    "before_excerpt": "", "after_excerpt": "",
                    "explanation_short": "Model call failed after retries; default FALSE.",
                    "_ok": False, "_failure": "API_ERROR",
                }

        obj = _json_from_text(content or "")
        if isinstance(obj, dict):
            # Normalize defaults
            obj.setdefault("label", False)
            obj.setdefault("reason_category", "other")
            obj.setdefault("correctness", "uncertain")
            obj.setdefault("shift_type", obj.get("reason_category", "other"))
            obj.setdefault("confidence", "low")
            obj.setdefault("rationale_quality", 3)
            obj.setdefault("markers_found", [])
            obj.setdefault("cue_text", "")
            obj.setdefault("cue_char_index", -1 if pos is None else int(pos))
            obj.setdefault("prior_hypothesis_quote", "")
            obj.setdefault("revision_quote", "")
            obj.setdefault("explanation_short", "")
            obj.setdefault("rule_violations", [])

            # Guardrail: require explicit cue somewhere
            has_explicit = bool(cues or (obj.get("markers_found") or []) or obj.get("cue_text"))
            if not has_explicit:
                obj["label"] = False
                rv = obj.get("rule_violations") or []
                if "no-explicit-cue" not in rv:
                    rv.append("no-explicit-cue")
                obj["rule_violations"] = rv
                obj["confidence"] = "low"

            # Clamp fields
            for k, lim in [("cue_text", 40), ("prior_hypothesis_quote", 80), ("revision_quote", 80)]:
                if isinstance(obj.get(k), str):
                    obj[k] = _clamp(obj[k], lim)
            obj["before_excerpt"] = _clamp(obj.get("before_excerpt", ""), 240)
            obj["after_excerpt"]  = _clamp(obj.get("after_excerpt", ""), 280)
            obj["explanation_short"] = _clamp(obj.get("explanation_short", ""), 300)

            # SOFT VALIDATION: do NOT demote label=true when fields are missing. Just flag.
            if obj.get("label", False):
                missing = []
                if not obj.get("cue_text"): missing.append("cue_text")
                if not obj.get("prior_hypothesis_quote"): missing.append("prior_hypothesis_quote")
                if not obj.get("revision_quote"): missing.append("revision_quote")
                if missing:
                    rv = obj.get("rule_violations") or []
                    rv.append("missing-required-fields:" + ",".join(missing))
                    obj["rule_violations"] = rv
                    if obj.get("confidence") == "high":
                        obj["confidence"] = "medium"

            obj["_ok"] = True
            return obj

        # Unparseable → retry/backoff
        if attempt < max_retries:
            sleep_s = base_backoff * (2 ** attempt)
            if not deterministic:
                sleep_s *= (1.0 + rng.uniform(0, 0.25))
            time.sleep(sleep_s)
            continue

        _dump_filtered(last_user_content + "\n\n[RAW]\n" + (content or ""), reason="UNPARSEABLE")
        return {
            "label": False, "correctness": "uncertain", "shift_type": "other",
            "confidence": "low", "rationale_quality": 1, "markers_found": [],
            "first_marker_index": -1 if pos is None else int(pos),
            "before_excerpt": "", "after_excerpt": "",
            "explanation_short": "Unparseable after retries; default FALSE.",
            "_ok": False, "_failure": "UNPARSEABLE",
        }

# ─────────────────── File scanning ───────────────────
def nat_step_from_path(path: str) -> Optional[int]:
    m = re.search(r"step(\d+)", path)
    return int(m.group(1)) if m else None

def scan_inputs(root_or_file: str, split: Optional[str]) -> List[str]:
    if os.path.isfile(root_or_file):
        return [root_or_file] if (not split or split in os.path.basename(root_or_file)) else []
    files = []
    for dp, _, fns in os.walk(root_or_file):
        for fn in fns:
            if fn.endswith(".jsonl") and (not split or split in fn):
                files.append(os.path.join(dp, fn))
    files.sort(key=lambda p: (nat_step_from_path(p) or 0, p), reverse=True)
    return files

# ─────────────────── Sidecar helpers ───────────────────
def _sidecar_path(input_file: str, dump_dir: str) -> str:
    base = _osp.basename(input_file)
    stem = base[:-6] if base.endswith(".jsonl") else base
    return _osp.join(dump_dir, f"{stem}.variants.jsonl")

def _atomic_write_lines(path: str, lines: List[str]):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln)
            if not ln.endswith("\n"):
                f.write("\n")
    os.replace(tmp, path)

# ─────────────────── Agreement metrics ───────────────────
def majority_label(votes: List[bool]) -> Tuple[bool, bool]:
    t = sum(1 for v in votes if v)
    f = len(votes) - t
    if t == f:
        return False, True
    return (t > f), False

def split_merge_aggregate(votes: List[bool], rng: random.Random) -> Tuple[bool, Dict[str, Any]]:
    # kept for compatibility; not used unless --ensemble_method split-merge
    idx = list(range(len(votes)))
    rng.shuffle(idx)
    shuffled = [votes[i] for i in idx]
    mid = len(shuffled)//2
    a, b = shuffled[:mid], shuffled[mid:]
    maj_a, tie_a = majority_label(a)
    maj_b, tie_b = majority_label(b if b else a)
    if (not tie_a) and (not tie_b) and (maj_a == maj_b):
        final = maj_a; strategy = "agree_halves"
    else:
        final, tie_all = majority_label(votes)
        strategy = "global_majority" if not tie_all else "tie_default_false"
    stability = sum(1 for v in votes if v == final) / max(1, len(votes))
    return final, {"order": idx, "halves_majority": {"A": maj_a, "B": maj_b},
                   "strategy": strategy, "stability": stability}

def cohen_kappa(labels_a: List[Any], labels_b: List[Any]) -> float:
    if len(labels_a) != len(labels_b) or len(labels_a) == 0:
        return 0.0
    A = [str(x) for x in labels_a]; B = [str(x) for x in labels_b]
    classes = sorted(set(A) | set(B)); n = len(A)
    po = sum(1 for i in range(n) if A[i] == B[i]) / n
    from collections import Counter
    ca, cb = Counter(A), Counter(B)
    pe = sum((ca[c]/n) * (cb[c]/n) for c in classes)
    denom = (1.0 - pe)
    if denom <= 1e-12: return 0.0
    return (po - pe) / denom

def pairwise_agreement_and_kappa(items_labels: List[List[Optional[bool]]],
                                 items_ok: List[List[bool]]) -> Tuple[float, float, Dict[str, int]]:
    if not items_labels: return 0.0, 0.0, {}
    K = len(items_labels[0])
    pair_pos, pair_k, used = [], [], {}
    for a in range(K):
        for b in range(a+1, K):
            la, lb = [], []
            for labels, oks in zip(items_labels, items_ok):
                if oks[a] and oks[b]:
                    va, vb = labels[a], labels[b]
                    if va is not None and vb is not None:
                        la.append(va); lb.append(vb)
            if not la: continue
            n = len(la)
            po = sum(1 for i in range(n) if la[i] == lb[i]) / n
            kappa = cohen_kappa(la, lb)
            pair_pos.append(po); pair_k.append(kappa); used[f"{a}-{b}"] = n
    po_mean = sum(pair_pos)/len(pair_pos) if pair_pos else 0.0
    k_mean  = sum(pair_k)/len(pair_k) if pair_k else 0.0
    return po_mean, k_mean, used

def bootstrap_pairwise_kappa(items_labels: List[List[Optional[bool]]],
                             items_ok: List[List[bool]],
                             B: int, seed: int) -> Tuple[float, float, float]:
    if not items_labels: return 0.0, 0.0, 0.0
    rng = random.Random(seed); N = len(items_labels)
    samples = []
    for _ in range(B):
        idxs = [rng.randrange(N) for _ in range(N)]
        subL = [items_labels[i] for i in idxs]
        subO = [items_ok[i] for i in idxs]
        _, k_mean, _ = pairwise_agreement_and_kappa(subL, subO)
        samples.append(k_mean)
    samples.sort()
    mean_val, _, _ = pairwise_agreement_and_kappa(items_labels, items_ok)
    lo = samples[int(0.025*B)] if B >= 40 else min(samples)
    hi = samples[int(0.975*B)-1] if B >= 40 else max(samples)
    return mean_val, lo, hi

# ─────────────────── Core annotation ───────────────────
def record_id_for_logs(obj: Dict[str, Any]) -> str:
    return obj.get("row_key") or obj.get("problem") or obj.get("clue") or f"idx={obj.get('dataset_index','?')}"

def annotate_file(path: str, seed: int, k: int, max_calls: Optional[int], dry_run: bool, jitter: float,
                  overwrite: bool, rng_global: random.Random, metrics_sink: Dict[str, Any],
                  dump_variants_dir: Optional[str],
                  max_retries: int, backoff_base: float,
                  prefilter_mode: str, max_sep: int, dump_prefiltered: Optional[str],
                  shuffle_evidence: bool, shuffle_variants: bool, ensemble_method: str,
                  deterministic: bool):
    logging.info("Annotating: %s", path)
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    records: List[Dict[str, Any]] = []
    for ln in lines:
        try:
            records.append(json.loads(ln))
        except Exception:
            records.append({"__raw__": ln})

    todo_idxs: List[int] = []
    for i, rec in enumerate(records):
        metrics_sink["total_examples_seen"] += 1
        if "__raw__" in rec:
            continue
        p1 = rec.get("pass1") or {}
        already = ("shift_in_reasoning_v1" in p1) or ("shift_variants_v1" in p1)
        if already and not overwrite:
            metrics_sink["skipped_already_annotated"] += 1
            continue
        out = p1.get("output")
        if not out:
            metrics_sink["skipped_no_think"] += 1
            continue
        think = _extract_think(out)
        if not think:
            p1["shift_in_reasoning_v1"] = False
            p1["shift_ensemble_method"] = "no_think_block_default_false"
            p1["shift_k"] = 0
            rec["pass1"] = p1
            metrics_sink["skipped_no_think"] += 1
            continue

        cues, pos = _find_shift_cues(think)

        # Non-shift block per mode
        if _nonshift_hit_mode(think, prefilter_mode):
            metrics_sink["prefilter_block_nonshift"] += 1
            reason = f"prefilter_nonshift_block[{prefilter_mode}]"
            p1.update({
                "shift_in_reasoning_v1": False,
                "shift_markers_v1": [],
                "shift_first_marker_char": -1 if pos is None else int(pos),
                "shift_before_excerpt": "",
                "shift_after_excerpt": "",
                "shift_rationale_gpt": "Non-shift rhetorical/hedge detected; conservative FALSE.",
                "shift_rationale_gpt_model": DEEPSEEK_MODEL,
                "shift_rationale_gpt_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "shift_ensemble_method": reason,
                "shift_k": 0,
                "shift_variants_v1": [],
                "shift_vote_counts": {"true": 0, "false": 0},
                "shift_stability": 1.0,
            })
            rec["pass1"] = p1
            if dump_prefiltered:
                fp = _osp.join(dump_prefiltered, "prefiltered.jsonl")
                with open(fp, "a", encoding="utf-8") as _f:
                    json.dump({
                        "source_path": _osp.abspath(path),
                        "record_id": record_id_for_logs(rec),
                        "reason": reason, "pos": pos, "cues": cues, "think": _clamp(think, 600)
                    }, _f, ensure_ascii=False); _f.write("\n")
            continue

        # Evidence sufficiency by mode
        ok = False
        if prefilter_mode == "strict":
            ok = bool(cues) and _has_revision_signal(think) and _cue_and_revision_close(think, pos, max_sep)
        elif prefilter_mode == "medium":
            ok = bool(cues) and (_has_revision_signal(think) or _has_strong_cue(think))
            if ok and bool(cues) and _has_revision_signal(think):
                ok = _cue_and_revision_close(think, pos, max_sep) or _has_strong_cue(think)
        else:  # loose
            ok = bool(cues) or _has_revision_signal(think)

        if not ok:
            metrics_sink["prefilter_block_no_revision"] += 1
            reason = f"prefilter_block[{prefilter_mode}]"
            p1.update({
                "shift_in_reasoning_v1": False,
                "shift_markers_v1": cues or [],
                "shift_first_marker_char": -1 if pos is None else int(pos),
                "shift_before_excerpt": "",
                "shift_after_excerpt": "",
                "shift_rationale_gpt": "Insufficient prefilter evidence under mode; FALSE.",
                "shift_rationale_gpt_model": DEEPSEEK_MODEL,
                "shift_rationale_gpt_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "shift_ensemble_method": reason,
                "shift_k": 0,
                "shift_variants_v1": [],
                "shift_vote_counts": {"true": 0, "false": 0},
                "shift_stability": 1.0,
            })
            rec["pass1"] = p1
            if dump_prefiltered:
                fp = _osp.join(dump_prefiltered, "prefiltered.jsonl")
                with open(fp, "a", encoding="utf-8") as _f:
                    json.dump({
                        "source_path": _osp.abspath(path),
                        "record_id": record_id_for_logs(rec),
                        "reason": reason, "pos": pos, "cues": cues,
                        "has_revision_anywhere": _has_revision_signal(think),
                        "strong_cue": _has_strong_cue(think),
                        "think": _clamp(think, 600)
                    }, _f, ensure_ascii=False); _f.write("\n")
            continue

        # Passed prefilter
        p1["_shift_prefilter_markers"] = cues
        p1["_shift_prefilter_pos"] = pos
        rec["pass1"] = p1
        todo_idxs.append(i)

    # Processing order: fixed by default
    idxs = todo_idxs[:]
    rng_local = random.Random(seed)
    if shuffle_variants or shuffle_evidence:  # small nod to determinism unless user opts in
        rng_local.shuffle(idxs)
    if max_calls is not None:
        idxs = idxs[:max_calls]

    calls = 0
    dump_rows: List[Dict[str, Any]] = []

    for idx in idxs:
        rec = records[idx]
        p1 = rec.get("pass1") or {}
        out = p1.get("output") or ""
        think = _extract_think(out) or ""
        problem = rec.get("problem") or rec.get("clue") or ""
        cues = p1.get("_shift_prefilter_markers") or []
        pos  = p1.get("_shift_prefilter_pos")

        if dry_run:
            logging.info("DRY-RUN would annotate idx=%s id=%s", idx, record_id_for_logs(rec))
            continue

        # Build variants (fixed order by default)
        variant_texts = [txt for _, txt in PROMPT_VARIANTS]
        while len(variant_texts) < k:
            variant_texts += [txt for _, txt in PROMPT_VARIANTS]
        variant_texts = variant_texts[:k]
        if shuffle_variants:
            rng_local.shuffle(variant_texts)

        variant_results: List[Dict[str, Any]] = []
        with timed(f"judge_k={k} idx={idx}"):
            for vtext in variant_texts:
                r = judge_once_with_retries(
                    problem, think, cues, pos, vtext, rng_local,
                    max_retries=max_retries, base_backoff=backoff_base,
                    shuffle_evidence=shuffle_evidence, deterministic=not (shuffle_evidence or shuffle_variants)
                )
                r["label"] = bool(r.get("label", False))
                try:
                    rq = int(r.get("rationale_quality", 3))
                    r["rationale_quality"] = max(1, min(5, rq))
                except Exception:
                    r["rationale_quality"] = 3
                variant_results.append(r)
                calls += 1
                if not r.get("_ok", True):
                    metrics_sink["failed_calls"] += 1
                if (shuffle_evidence or shuffle_variants) and jitter > 0:
                    time.sleep(rng_local.uniform(0.0, jitter))

        votes = [vr["label"] for vr in variant_results]
        ok_flags = [bool(vr.get("_ok", True)) for vr in variant_results]

        # Ensemble: simple majority by default
        if ensemble_method == "split-merge":
            final_label, agg_meta = split_merge_aggregate(votes, rng_local)
        else:
            final_label, tie = majority_label(votes)
            agg_meta = {"strategy": "global_majority" if not tie else "tie_default_false",
                        "stability": sum(1 for v in votes if v == final_label)/max(1, len(votes)),
                        "order": list(range(len(votes)))}

        # Representative variant: best rationale_quality, then earliest cue index
        best_idx = max(range(len(variant_results)),
                       key=lambda i: (variant_results[i].get("rationale_quality", 0),
                                      -1 if variant_results[i].get("cue_char_index", 10**9) is None
                                      else -variant_results[i].get("cue_char_index", 10**9)))
        best = variant_results[best_idx]

        # Stamp
        p1["shift_in_reasoning_v1"]   = bool(final_label)
        p1["shift_markers_v1"]        = list(best.get("markers_found", []))
        p1["shift_first_marker_char"] = int(best.get("cue_char_index", -1 if pos is None else pos))
        p1["shift_before_excerpt"]    = _clamp(best.get("prior_hypothesis_quote", ""), 240)
        p1["shift_after_excerpt"]     = _clamp(best.get("revision_quote", ""), 280)
        p1["shift_rationale_gpt"]     = _clamp(best.get("explanation_short", ""), 300)
        p1["shift_rationale_gpt_model"] = DEEPSEEK_MODEL
        p1["shift_rationale_gpt_time"]  = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        p1["shift_ensemble_method"] = "majority" if ensemble_method != "split-merge" else "split-merge"
        p1["shift_k"] = k
        p1["shift_variants_v1"] = [{
            "variant_id": f"v{i}",
            "label": vr["label"],
            "reason_category": vr.get("reason_category", "other"),
            "correctness": vr.get("correctness", "uncertain"),
            "shift_type": vr.get("shift_type", vr.get("reason_category", "other")),
            "confidence": vr.get("confidence", "low"),
            "rationale_quality": vr.get("rationale_quality", 3),
            "markers_found": vr.get("markers_found", []),
            "cue_text": vr.get("cue_text", ""),
            "cue_char_index": vr.get("cue_char_index", -1 if pos is None else pos),
            "prior_hypothesis_quote": vr.get("prior_hypothesis_quote", ""),
            "revision_quote": vr.get("revision_quote", ""),
            "explanation_short": vr.get("explanation_short", ""),
            "rule_violations": vr.get("rule_violations", []),
            "_ok": vr.get("_ok", True),
            "_failure": vr.get("_failure", None),
        } for i, vr in enumerate(variant_results)]
        p1["shift_vote_counts"] = {"true": int(sum(votes)), "false": int(len(votes) - sum(votes))}
        p1["shift_stability"] = agg_meta["stability"]
        p1["shift_vote_order"] = agg_meta.get("order", list(range(len(votes))))
        rec["pass1"] = p1

        # Store for agreement
        metrics_sink["items_labels"].append(votes)
        metrics_sink["items_ok"].append(ok_flags)
        metrics_sink["judged_examples"] += 1

        # Sidecar
        if dump_variants_dir:
            dump_rows.append({
                "source_path": _osp.abspath(path),
                "record_id": record_id_for_logs(rec),
                "dataset_index": rec.get("dataset_index", None),
                "k": k,
                "votes": votes,
                "ensemble_label": bool(final_label),
                "vote_counts": {"true": int(sum(votes)), "false": int(len(votes) - sum(votes))},
                "stability": p1["shift_stability"],
                "variants": p1["shift_variants_v1"],
                "variant_ok_flags": ok_flags,
            })

    # Atomic write back
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in records:
            if "__raw__" in rec:
                f.write(rec["__raw__"])
            else:
                json.dump(rec, f, ensure_ascii=False); f.write("\n")
    os.replace(tmp, path)
    logging.info("Updated %s (judge calls: %d)", path, calls)

    # Sidecar
    if dump_variants_dir and dump_rows:
        sidecar = _sidecar_path(path, dump_variants_dir)
        old = []
        if _osp.exists(sidecar):
            with open(sidecar, "r", encoding="utf-8") as f:
                old = f.readlines()
        new_lines = old + [json.dumps(row, ensure_ascii=False) for row in dump_rows]
        _atomic_write_lines(sidecar, new_lines)
        logging.info("Sidecar saved: %s (+%d rows)", sidecar, len(dump_rows))

# ─────────────────── CLI / main ───────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_path", help="JSONL file or directory (recursive).")
    ap.add_argument("--split", default=None, help="Substring filter for filenames.")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max_calls", type=int, default=None)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--jitter", type=float, default=0.0, help="Ignored unless shuffles are enabled.")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--report_json", default=None)
    ap.add_argument("--dump_variants", default=None)
    ap.add_argument("--max_retries", type=int, default=5)
    ap.add_argument("--backoff_base", type=float, default=0.35)

    # Prefilter controls
    ap.add_argument("--prefilter_mode", choices=["strict","medium","loose"], default="medium")
    ap.add_argument("--max_sep", type=int, default=350)
    ap.add_argument("--dump_prefiltered", default=None)

    # NEW: variance controls (defaults reduce variance)
    ap.add_argument("--shuffle_evidence", action="store_true", help="Shuffle evidence block order (default off).")
    ap.add_argument("--shuffle_variants", action="store_true", help="Shuffle variant order (default off).")
    ap.add_argument("--ensemble_method", choices=["majority","split-merge"], default="majority",
                    help="Ensemble final label from K votes. Default: majority.")
    ap.add_argument("--deterministic", action="store_true",
                    help="Force deterministic retries/backoffs; implied by no shuffle.")

    ap.add_argument("--loglevel", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.loglevel.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    files = scan_inputs(args.input_path, args.split)
    if not files:
        print("No JSONL inputs found; check path/split.", file=sys.stderr)
        sys.exit(1)

    if args.dump_variants:
        os.makedirs(args.dump_variants, exist_ok=True)
    if args.dump_prefiltered:
        os.makedirs(args.dump_prefiltered, exist_ok=True)

    rng_global = random.Random(args.seed)
    metrics_sink = {
        "items_labels": [],
        "items_ok": [],
        "total_examples_seen": 0,
        "judged_examples": 0,
        "skipped_no_think": 0,
        "skipped_already_annotated": 0,
        "prefilter_block_nonshift": 0,
        "prefilter_block_no_revision": 0,
        "failed_calls": 0,
    }

    # Process
    for path in files:
        annotate_file(
            path=path, seed=args.seed, k=args.k, max_calls=args.max_calls,
            dry_run=args.dry_run, jitter=args.jitter, overwrite=args.overwrite,
            rng_global=rng_global, metrics_sink=metrics_sink,
            dump_variants_dir=args.dump_variants,
            max_retries=args.max_retries, backoff_base=args.backoff_base,
            prefilter_mode=args.prefilter_mode, max_sep=args.max_sep,
            dump_prefiltered=args.dump_prefiltered,
            shuffle_evidence=args.shuffle_evidence,
            shuffle_variants=args.shuffle_variants,
            ensemble_method=args.ensemble_method,
            deterministic=args.deterministic or (not args.shuffle_evidence and not args.shuffle_variants),
        )

    # Agreement
    items_labels = metrics_sink["items_labels"]
    items_ok     = metrics_sink["items_ok"]
    if items_labels and not args.dry_run:
        po_mean, kappa_mean, used_counts = pairwise_agreement_and_kappa(items_labels, items_ok)
        kappa_bs, k_lo, k_hi = bootstrap_pairwise_kappa(items_labels, items_ok, B=args.bootstrap, seed=args.seed)
        summary = {
            "judged_examples": metrics_sink["judged_examples"],
            "k": args.k,
            "mean_pairwise_percent_agreement": po_mean,
            "mean_pairwise_cohens_kappa": kappa_mean,
            "bootstrap_CI_95pct": [k_lo, k_hi],
            "bootstrap_B": args.bootstrap,
            "seed": args.seed,
            "total_examples_seen": metrics_sink["total_examples_seen"],
            "skipped_no_think": metrics_sink["skipped_no_think"],
            "skipped_already_annotated": metrics_sink["skipped_already_annotated"],
            "prefilter_block_nonshift": metrics_sink["prefilter_block_nonshift"],
            "prefilter_block_no_revision": metrics_sink["prefilter_block_no_revision"],
            "failed_calls": metrics_sink["failed_calls"],
            "pairwise_effective_counts": used_counts,
            "prefilter_mode": args.prefilter_mode,
            "max_sep": args.max_sep,
            "shuffle_evidence": args.shuffle_evidence,
            "shuffle_variants": args.shuffle_variants,
            "ensemble_method": args.ensemble_method,
        }
        print("\n=== Inter-prompt Agreement Report (reduced variance) ===")
        print(json.dumps(summary, indent=2))
        if args.report_json:
            with open(args.report_json, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"[saved] {args.report_json}")
    else:
        logging.info("No judged items collected (dry_run=%s or no eligible items); skipping agreement report.", args.dry_run)

if __name__ == "__main__":
    main()
