#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backwards‑compatible wrapper for shift‑in‑reasoning annotation.

Historically, consumers imported the annotator via
``src.annotate.tasks.shifts``.  The modern implementation lives under
``src.annotate.core.shift_core`` (library) and ``src.annotate.cli.shift_cli``
(CLI).  This module re‑exports the public API from those modules so existing
imports keep working without triggering linter warnings about unused imports.
"""

from __future__ import annotations

from ... import _ANNOTATION_PUBLIC_API as _BASE_PUBLIC_API
from ...cli import shift_cli as _cli
from ...core import shift_core as _core


# Re‑export core library primitives.
AnnotateOpts = _core.AnnotateOpts
annotate_file = _core.annotate_file
llm_judge_shift = _core.llm_judge_shift
scan_jsonl = _core.scan_jsonl
nat_step_from_path = _core.nat_step_from_path
record_id_for_logs = _core.record_id_for_logs
_annotate_record_for_pass = _core._annotate_record_for_pass  # pylint: disable=protected-access
_json_from_text = _core._json_from_text  # pylint: disable=protected-access
_sanitize_jsonish = _core._sanitize_jsonish  # pylint: disable=protected-access


def build_argparser():
    """Backwards‑compatible alias for the CLI argparser builder."""

    return _cli.build_argparser()


def main() -> None:
    """Backwards‑compatible CLI entrypoint."""

    _cli.main()


__all__ = [
    # Core annotation API
    *_BASE_PUBLIC_API,
    "nat_step_from_path",
    "record_id_for_logs",
    "_annotate_record_for_pass",
    "_json_from_text",
    "_sanitize_jsonish",
    # CLI helpers
    "build_argparser",
    "main",
]


if __name__ == "__main__":
    main()
