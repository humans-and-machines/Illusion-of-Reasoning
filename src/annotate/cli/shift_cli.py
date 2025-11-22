#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI entrypoint for shift-in-reasoning annotation.

This is a thin wrapper over :mod:`src.annotate.core.shift_core` that handles
argument parsing, logging configuration and optional cleaning of fallback
shift labels.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from src.annotate.core.clean_core import clean_root as _clean_failed_root
from src.annotate.core.shift_core import (
    AnnotateOpts,
    DEFAULT_API_VERSION,
    DEFAULT_DEPLOYMENT,
    DEFAULT_ENDPOINT,
    DEFAULT_USE_V1,
    annotate_file,
    scan_jsonl,
)
from src.annotate.core.progress import count_progress


def build_argparser() -> argparse.ArgumentParser:
    """CLI argument builder for shift annotation."""
    arg_parser = argparse.ArgumentParser(
        description="Annotate shift-in-reasoning events in JSONL inference results.",
    )
    arg_parser.add_argument("results_root", help="Directory containing step*/.../*.jsonl")
    arg_parser.add_argument(
        "--split",
        default=None,
        help="Filter filenames that contain this substring (e.g., 'test').",
    )
    arg_parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Shuffle seed for random processing order.",
    )
    arg_parser.add_argument(
        "--max_calls",
        type=int,
        default=None,
        help="Optional cap on model calls.",
    )
    arg_parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Discover candidates but do not call the model or write changes.",
    )
    arg_parser.add_argument(
        "--jitter",
        type=float,
        default=0.25,
        help="Max random sleep (seconds) between calls; 0 to disable.",
    )
    arg_parser.add_argument("--loglevel", default="INFO")
    arg_parser.add_argument(
        "--force_relabel",
        action="store_true",
        help="Re-annotate records even if shift_in_reasoning_v1 already exists.",
    )
    arg_parser.add_argument(
        "--clean_failed_first",
        action="store_true",
        help=(
            "Run clean-shift-fallbacks on results_root before annotation "
            "(strips fallback FALSE shift labels from prior failed judge calls)."
        ),
    )
    arg_parser.add_argument(
        "--passes",
        default="pass1",
        help=(
            "Comma-separated pass keys to annotate "
            "(e.g., 'pass1', 'pass1,pass2,pass2a,pass2b,pass2c'). "
            "Defaults to 'pass1' for backwards compatibility."
        ),
    )
    arg_parser.add_argument(
        "--backend",
        choices=["azure", "portkey"],
        default="azure",
        help="LLM backend: 'azure' (default) or 'portkey' (AI Sandbox via portkey-ai).",
    )
    arg_parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="Azure OpenAI endpoint (e.g., https://<res>.openai.azure.com/).",
    )
    arg_parser.add_argument(
        "--deployment",
        default=DEFAULT_DEPLOYMENT,
        help="Azure OpenAI deployment name (e.g., 'gpt-4o').",
    )
    arg_parser.add_argument(
        "--api_version",
        default=DEFAULT_API_VERSION,
        help="Azure API version (legacy client only).",
    )
    arg_parser.add_argument(
        "--use_v1",
        type=int,
        default=DEFAULT_USE_V1,
        help="1=prefer v1 Responses API, 0=legacy client.",
    )
    return arg_parser


def main() -> None:
    """CLI entrypoint."""
    arg_parser = build_argparser()
    args = arg_parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.loglevel.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.clean_failed_first:
        logging.info(
            "Cleaning prior fallback shift labels under %s before annotation.",
            os.path.abspath(args.results_root),
        )
        _clean_failed_root(args.results_root)

    files = scan_jsonl(args.results_root, args.split)
    if not files:
        print("No JSONL files found; check path/split.")
        return

    opts = AnnotateOpts(
        seed=args.seed,
        max_calls=args.max_calls,
        dry_run=args.dry_run,
        jitter=args.jitter,
        force_relabel=args.force_relabel,
        client_cfg={
            "backend": args.backend,
            "endpoint": args.endpoint,
            "api_version": args.api_version,
            "use_v1": args.use_v1,
            "deployment": args.deployment,
        },
        passes=[
            p.strip()
            for p in (args.passes or "").split(",")
            if p.strip()
        ],
    )

    for path in files:
        total, seen = count_progress(Path(path))
        logging.info(
            "Existing GPT-4o annotations for %s: %d/%d (pending %d)",
            path,
            seen,
            total,
            total - seen,
        )
        annotate_file(path, opts)


__all__ = ["build_argparser", "main"]

if __name__ == "__main__":
    main()
