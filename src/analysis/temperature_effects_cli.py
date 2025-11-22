#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI argument construction for ``temperature_effects.py``.

This module keeps the main analysis script smaller and focused on the core
computation, while centralizing argument definitions used by the CLI entrypoint.
"""

from __future__ import annotations

import argparse

from src.analysis.utils import add_carpark_threshold_args, add_temp_scan_args


def build_temperature_effects_arg_parser() -> argparse.ArgumentParser:
    """
    Build and return the argument parser for the temperature-effects script.
    """
    parser = argparse.ArgumentParser()

    add_temp_scan_args(parser, include_math3=False)

    parser.add_argument("--label_crossword", type=str, default="Crossword")
    parser.add_argument("--label_math", type=str, default="Llama-8B-Math")
    parser.add_argument("--label_math2", type=str, default="Qwen-7B-Math")
    parser.add_argument("--label_carpark", type=str, default="Carpark")
    parser.add_argument("--include_math2", action="store_true", default=True)

    parser.add_argument("--split", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--dataset_name", default="MIXED")
    parser.add_argument("--model_name", default="7B_vs_8B")

    parser.add_argument("--gpt_mode", choices=["canonical", "broad"], default="canonical")
    parser.add_argument("--no_gpt_subset_native", action="store_true")
    parser.add_argument("--min_step", type=int, default=None)
    parser.add_argument("--max_step", type=int, default=None)

    add_carpark_threshold_args(parser)

    parser.add_argument("--low_alias", type=float, default=0.3)
    parser.add_argument(
        "--skip_substr",
        nargs="*",
        default=["compare-1shot", "1shot", "hf_cache"],
    )

    # plotting knobs
    parser.add_argument("--make_plot", action="store_true")
    parser.add_argument("--plot_title", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=300)

    # PARALLEL knobs
    parser.add_argument(
        "--workers",
        type=int,
        default=40,
        help="Number of worker processes/threads (default: 40).",
    )
    parser.add_argument(
        "--parallel",
        type=str,
        choices=["process", "thread"],
        default="process",
        help="Parallelism backend: 'process' (default) or 'thread'.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=4,
        help="Chunk size for executor.map.",
    )

    return parser


__all__ = ["build_temperature_effects_arg_parser"]
