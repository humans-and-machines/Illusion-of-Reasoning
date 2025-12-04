#!/usr/bin/env python3
"""
Dataset loading, normalisation, and prompt construction helpers for crossword
evaluation used by ``src.utils.check``.
"""

from __future__ import annotations

import json
import re
import string
import zipfile
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional


_PUNCT_RE = re.compile(rf"[{re.escape(string.punctuation)}]")

# training tag patterns
_ANS_PAT = re.compile(r"(?si)<answer>\s*([^<\n]+?)\s*</answer>")
_CAPS_FALLBACK = re.compile(r"\b[A-Z]{2,}\b")


def normalise(ans: str) -> str:
    """Normalise an answer string for comparison."""
    return _PUNCT_RE.sub("", ans).replace(" ", "").upper()


def extract_answer_last(txt: str) -> Optional[str]:
    """Extract the last <answer>...</answer> span from text."""
    last_match = None
    for match in _ANS_PAT.finditer(txt or ""):
        last_match = match
    return last_match.group(1).strip() if last_match else None


def extract_answer_with_fallback(txt: str) -> Optional[str]:
    """Extract answer span, falling back to final all-caps token."""
    match = extract_answer_last(txt)
    if match:
        return match
    caps = _CAPS_FALLBACK.findall(txt or "")
    return caps[-1] if caps else None


def _rows_from_jsonl_bytes(buf: bytes) -> List[dict]:
    """Parse a JSONL byte buffer into rows with normalised answers."""
    rows: List[dict] = []
    for line in buf.splitlines():
        text = line.decode("utf-8").strip()
        if not text:
            continue
        obj = json.loads(text)
        raw_ans = obj.get("answer") or obj.get("solution") or obj.get("soln")
        if raw_ans is None:
            raise ValueError(f"Missing answer key in jsonl row: {list(obj.keys())}")
        rows.append(
            {
                "problem": f"{obj['clue']}  \n<think>",
                "answer": normalise(raw_ans),
            }
        )
    return rows


def _import_datasets_module():
    """Import the optional 'datasets' package at runtime."""
    try:
        return import_module("datasets")
    except ImportError as exc:  # pragma: no cover - runtime-only dependency
        msg = (
            "The 'datasets' package is required to load the Cryptonite dataset. "
            "Install it with `pip install datasets`."
        )
        raise RuntimeError(msg) from exc


def _build_dataset_from_parts(
    datasets_mod: Any,
    parts: Dict[str, List[dict]],
    split: str,
) -> Any:
    """Construct a DatasetDict from pre-parsed JSONL parts and return the desired split."""
    dataset_dict = datasets_mod.DatasetDict(
        {key: datasets_mod.Dataset.from_list(value) for key, value in parts.items()}
    )
    return dataset_dict[split]


def load_cryptonite_split(
    dataset_name: Optional[str],
    split: str,
    cryptonite_zip: Optional[Path],
) -> Any:
    """Load the requested split either from HF datasets or a local ZIP."""
    datasets_mod = _import_datasets_module()

    if cryptonite_zip:
        zip_path = Path(cryptonite_zip)
        if not zip_path.exists():
            raise FileNotFoundError(zip_path)

        with zipfile.ZipFile(zip_path) as zip_file:
            members = [m for m in zip_file.namelist() if m.endswith(".jsonl")]
            if not members:
                raise ValueError("ZIP contains no .jsonl files")

            parts: Dict[str, List[dict]] = {}
            for member in members:
                blob = zip_file.read(member)
                if "train" in member:
                    parts["train"] = _rows_from_jsonl_bytes(blob)
                elif "valid" in member or "val" in member:
                    parts["validation"] = _rows_from_jsonl_bytes(blob)
                elif "test" in member:
                    parts["test"] = _rows_from_jsonl_bytes(blob)

            if "validation" not in parts and "train" in parts:
                train_rows = parts["train"]
                nval = max(1, int(0.10 * len(train_rows)))
                parts["validation"] = train_rows[-nval:]
                parts["train"] = train_rows[:-nval]

            return _build_dataset_from_parts(datasets_mod, parts, split)

    if not dataset_name:
        msg = "Either --cryptonite_zip or --dataset_name must be provided."
        raise ValueError(msg)

    return datasets_mod.load_dataset(dataset_name, split=split)


TRAIN_SYSTEM_PROMPT = """You are an expert *cryptic-crossword solver*.

Every time you receive a clue you must:
• Analyse it thoroughly.  
  – Pinpoint the **definition**.  
  – Pinpoint the **word-play** (anagram, container, reversal, homophone, charade, etc.).  
  – Write out the full derivation that turns the word-play into the answer.  

• Check that the answer’s length matches the enumeration in brackets.

• Respond in **exactly** the tag-based format shown below – no greeting, no commentary outside the tags.  
  – The final answer goes inside `<answer>` in UPPER-CASE.  
  – Never reveal the answer elsewhere.

------------------------------------------------------------
TAG TEMPLATE (copy this shape for every clue)
<think>
YOUR reasoning process goes here:  
1. quote the relevant bits of the clue  
2. name the cryptic device(s) you apply  
3. show each intermediate step until the answer is reached  

If you spot an error with the answer reached, iterate, repeating steps 1-3 as many
times as necessary until you are confident in your answer.
</think>
<answer>
THEANSWER        ← all-caps; length must match enumeration
</answer>"""


def build_messages(clue_with_think: str, gold_answer_norm: str) -> List[dict]:
    """Build training-format messages for a single crossword clue."""
    enumeration = len(gold_answer_norm)
    return [
        {"role": "system", "content": TRAIN_SYSTEM_PROMPT},
        {"role": "user", "content": f"{clue_with_think.strip()} ({enumeration})"},
    ]
