"""
Core shift-annotation logic: prefilters, prompts, annotators, and cleaners.

Most callers should import via :mod:`src.annotate` instead of this subpackage.
"""

from .shift_core import AnnotateOpts, annotate_file, llm_judge_shift, scan_jsonl


__all__ = ["AnnotateOpts", "annotate_file", "llm_judge_shift", "scan_jsonl"]
