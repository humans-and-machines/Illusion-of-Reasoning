"""Shared helpers for extracting text/answers from pass dictionaries."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional


_TAGS_ANSWER = re.compile(r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>", re.IGNORECASE | re.DOTALL)


def extract_pass_text(pass_dict: Dict[str, Any]) -> Optional[str]:
    """
    Return the most relevant assistant text from a pass dictionary.
    """
    if not isinstance(pass_dict, dict):
        return None
    for key in (
        "output",
        "text",
        "content",
        "response",
        "model_output",
        "assistant_text",
        "prev_output",
    ):
        value = pass_dict.get(key)
        if isinstance(value, str) and value.strip():
            return value
    messages = pass_dict.get("messages")
    if isinstance(messages, list):
        for message_dict in reversed(messages):
            if (
                isinstance(message_dict, dict)
                and str(message_dict.get("role", "")).lower() == "assistant"
            ):
                content = message_dict.get("content") or message_dict.get("text")
                if isinstance(content, str) and content.strip():
                    return content
    choices = pass_dict.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict):
            content = message.get("content") or message.get("text")
            if isinstance(content, str) and content.strip():
                return content
    return None


def _extract_answer_from_fields(pass_dict: Dict[str, Any]) -> Optional[str]:
    for key in (
        "final_answer",
        "answer",
        "short_answer",
        "pred",
        "prediction",
        "pred_text",
        "parsed_answer",
        "extracted_answer",
    ):
        value = pass_dict.get(key)
        if isinstance(value, str) and value.strip():
            match = _TAGS_ANSWER.search(value)
            return match.group(1).strip() if match else value.strip()
    return None


def _extract_answer_from_output_field(pass_dict: Dict[str, Any]) -> Optional[str]:
    output = pass_dict.get("output")
    if not isinstance(output, str) or not output.strip():
        return None
    matches = _TAGS_ANSWER.findall(output)
    if matches:
        candidates = [segment.strip() for segment in matches if segment and segment.strip()]
        if candidates:
            return candidates[-1]
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if lines and len(lines[-1]) <= 200:
        return lines[-1]
    return None


def _extract_answer_from_text_fallback(pass_dict: Dict[str, Any]) -> Optional[str]:
    text = extract_pass_text(pass_dict)
    if not isinstance(text, str) or not text.strip():
        return None
    match = _TAGS_ANSWER.search(text)
    if match:
        return match.group(1).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and len(lines[-1]) <= 200:
        return lines[-1]
    return None


def extract_pass_answer(pass_dict: Dict[str, Any]) -> Optional[str]:
    """
    Attempt to extract a concise answer string from a pass dictionary.
    """
    if not isinstance(pass_dict, dict):
        return None
    for extractor in (
        _extract_answer_from_fields,
        _extract_answer_from_output_field,
        _extract_answer_from_text_fallback,
    ):
        answer = extractor(pass_dict)
        if answer is not None:
            return answer
    return None


__all__ = [
    "extract_pass_text",
    "extract_pass_answer",
]
