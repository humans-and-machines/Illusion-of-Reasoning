"""Shared helpers for Hugging Face crossword datasets.

Copyright (c) 2024 The Illusion-of-Reasoning contributors.
Licensed under the Apache License, Version 2.0. See LICENSE for details.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict


def dataset_classes() -> Tuple["Dataset", "DatasetDict"]:
    """
    Import Dataset classes lazily so pylint passes without the `datasets` package installed.

    :returns: Tuple of (Dataset, DatasetDict) classes loaded from the datasets package.
    :rtype: Tuple[Dataset, DatasetDict]
    :raises ImportError: If the `datasets` package is not available.
    """
    try:
        datasets_module = importlib.import_module("datasets")
    except ImportError as exc:
        raise ImportError(
            "Install the `datasets` package to convert crossword datasets."
        ) from exc
    return datasets_module.Dataset, datasets_module.DatasetDict
