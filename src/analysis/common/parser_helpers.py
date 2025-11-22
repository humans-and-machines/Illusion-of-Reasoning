#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Common argparse helpers shared across multiple analysis CLIs.
"""

from __future__ import annotations

import argparse

from src.analysis.utils import add_results_root_split_and_output_args


def standard_results_parser(
    dataset_default: str = "MATH-500",
    model_default: str = "Qwen2.5-1.5B",
) -> argparse.ArgumentParser:
    """
    Return a parser preconfigured with the typical results-root arguments.
    """
    parser = argparse.ArgumentParser()
    add_results_root_split_and_output_args(
        parser,
        dataset_default=dataset_default,
        model_default=model_default,
        results_root_optional=False,
    )
    return parser


def add_binning_argument(
    parser: argparse.ArgumentParser,
    *,
    default: str = "uniform",
) -> None:
    """
    Add the shared ``--binning`` option used by entropy/uncertainty plots.
    """
    parser.add_argument(
        "--binning",
        type=str,
        default=default,
        choices=["uniform", "quantile"],
        help="Binning strategy for entropy/perplexity axes.",
    )


def add_carpark_softscore_args(
    parser: argparse.ArgumentParser,
    *,
    op_default: str = "ge",
    threshold_default: float = 0.1,
) -> None:
    """
    Add shared Carpark soft-score threshold arguments used by multiple plots.
    """
    parser.add_argument(
        "--carpark_success_op",
        type=str,
        choices=["ge", "gt", "le", "lt"],
        default=op_default,
    )
    parser.add_argument(
        "--carpark_soft_threshold",
        type=float,
        default=threshold_default,
    )


def add_entropy_range_args(parser: argparse.ArgumentParser) -> None:
    """
    Add entropy/uncertainty binning arguments shared by multiple scripts.
    """
    parser.add_argument("--bins", type=int, default=10)
    add_binning_argument(parser)
    parser.add_argument(
        "--share_bins",
        type=str,
        default="global",
        choices=["global", "per_domain"],
    )
    parser.add_argument(
        "--entropy_min",
        type=float,
        default=None,
        help="Fixed minimum value for entropy bins (e.g., 0.0).",
    )
    parser.add_argument(
        "--entropy_max",
        type=float,
        default=None,
        help="Fixed maximum value for entropy bins (e.g., 4.0).",
    )


__all__ = [
    "standard_results_parser",
    "add_binning_argument",
    "add_carpark_softscore_args",
    "add_entropy_range_args",
]
