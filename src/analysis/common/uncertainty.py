#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared helpers for standardizing uncertainty columns across analysis scripts.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd


def _uncertainty_stats(
    data: pd.DataFrame,
    source_col: str,
) -> Tuple[float, float]:
    """Return mean/std used for uncertainty standardization."""
    mean_uncertainty = float(data[source_col].mean())
    std_uncertainty = float(data[source_col].std(ddof=0))
    return mean_uncertainty, std_uncertainty


def standardize_uncertainty(
    data: pd.DataFrame,
    source_col: str = "uncertainty",
    dest_col: str = "uncertainty_std",
) -> pd.DataFrame:
    """
    Return a copy of ``data`` with a standardized uncertainty column added.
    """
    mean_uncertainty, std_uncertainty = _uncertainty_stats(data, source_col)
    standardized = data.copy()
    standardized[dest_col] = (standardized[source_col] - mean_uncertainty) / (
        std_uncertainty + 1e-8
    )
    return standardized


def standardize_uncertainty_with_stats(
    data: pd.DataFrame,
    source_col: str = "uncertainty",
    dest_col: str = "uncertainty_std",
) -> Tuple[pd.DataFrame, float, float]:
    """
    Standardize the uncertainty column and also return the (mean, std) statistics.
    """
    mean_uncertainty, std_uncertainty = _uncertainty_stats(data, source_col)
    standardized = data.copy()
    standardized[dest_col] = (standardized[source_col] - mean_uncertainty) / (
        std_uncertainty + 1e-8
    )
    return standardized, mean_uncertainty, std_uncertainty


__all__ = ["standardize_uncertainty", "standardize_uncertainty_with_stats"]
