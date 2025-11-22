#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Helpers for normalizing model/domain tokens in directory names.
"""

from __future__ import annotations

import re
from typing import Optional


MODEL_LABELS = {
    "qwen7b": "Qwen2.5-7B",
    "llama8b": "Llama3.1-8B",
}

DOMAIN_NORMALIZE = {
    "xword": "Crossword",
    "crossword": "Crossword",
    "math": "Math",
    "carpark": "Carpark",
    "rush": "Carpark",
    "rushhour": "Carpark",
}

_TEMP_PATTERN = re.compile(r"temp-([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def detect_model_key(name_lower: str) -> Optional[str]:
    """Return the canonical model key (llama8b/qwen7b) if obvious from the name."""
    if "llama" in name_lower and "8b" in name_lower:
        return "llama8b"
    if "qwen" in name_lower and "7b" in name_lower:
        return "qwen7b"
    if "7b" in name_lower and "llama" not in name_lower:
        return "qwen7b"
    return None


def detect_domain(name_lower: str) -> Optional[str]:
    """Normalize a lowered directory name to Math/Crossword/Carpark."""
    for token, domain_name in DOMAIN_NORMALIZE.items():
        if token in name_lower:
            return domain_name
    return None


def detect_temperature(name_lower: str, low_alias: float) -> Optional[float]:
    """Extract temperature tokens, respecting ``low`` aliases."""
    match = _TEMP_PATTERN.search(name_lower)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    if "low-temp" in name_lower or "low_temp" in name_lower:
        return float(low_alias)
    return None


__all__ = [
    "MODEL_LABELS",
    "DOMAIN_NORMALIZE",
    "detect_model_key",
    "detect_domain",
    "detect_temperature",
]
