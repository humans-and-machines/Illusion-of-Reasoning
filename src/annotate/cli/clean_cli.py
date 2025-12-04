#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI wrapper to strip fallback shift annotations from JSONL results.

The core logic lives in :mod:`src.annotate.core.clean_core`. This module only
parses arguments and dispatches to ``clean_root``.
"""

from __future__ import annotations

import argparse

from ..core.clean_core import clean_root


def main() -> None:
    """CLI entrypoint for cleaning fallback shift labels."""
    parser = argparse.ArgumentParser(
        description=(
            "Strip fallback shift_in_reasoning_v1 annotations from JSONL results under a given root directory."
        ),
    )
    parser.add_argument(
        "results_root",
        help=("Root directory containing step*/.../*.jsonl (e.g., artifacts/results/gpt4o-math-portkey-temp005)."),
    )
    args = parser.parse_args()

    clean_root(args.results_root)


if __name__ == "__main__":
    main()
