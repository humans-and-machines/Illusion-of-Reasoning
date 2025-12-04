#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text-only helpers split out from ``inference.common``.

These utilities do not depend on torch or transformers and are shared across
math/carpark/crossword inference entrypoints.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple


def find_markers_and_context(
    think_text: Optional[str],
    prompt_text: str,
    patterns: Sequence[Tuple[str, re.Pattern]],
    *,
    skip_prefix_chars: int = 0,
) -> Tuple[List[str], Optional[int], Optional[str], Optional[str]]:
    """
    Scan ``think_text`` for the earliest match among ``patterns``.

    :param think_text: Model reasoning text (may be ``None`` or empty).
    :param prompt_text: Prompt text used to build contextual prefixes.
    :param patterns: Sequence of ``(name, regex)`` marker patterns to search for.
    :param skip_prefix_chars: Number of characters at the start of ``think_text``
        to ignore when searching (for example, injected cues).
    :returns: Tuple ``(markers, earliest_pos, context_prefix, excerpt)`` where
        ``markers`` is the list of matched marker names, ``earliest_pos`` is
        the earliest character index of any match, ``context_prefix`` is a
        combined prompt+prefix string, and ``excerpt`` is a short window around
        the earliest match.
    """
    if not think_text:
        return [], None, None, None

    earliest_pos: Optional[int] = None
    markers: List[str] = []

    for name, pattern in patterns:
        match = pattern.search(
            think_text[skip_prefix_chars:] if skip_prefix_chars > 0 else think_text,
        )
        if not match:
            continue
        markers.append(name)
        pos_global = skip_prefix_chars + match.start() if skip_prefix_chars > 0 else match.start()
        if earliest_pos is None or pos_global < earliest_pos:
            earliest_pos = pos_global

    if not markers:
        return [], None, None, None

    if earliest_pos is None:
        return markers, None, think_text, think_text

    prefix = think_text[:earliest_pos]
    context = f"{prompt_text}\n\n{prefix}"
    window_start = max(0, earliest_pos - 60)
    window_end = min(len(think_text), earliest_pos + 60)
    excerpt = think_text[window_start:window_end]
    return markers, earliest_pos, context, excerpt
