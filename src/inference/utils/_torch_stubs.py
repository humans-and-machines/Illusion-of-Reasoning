#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fallback stubs for optional torch/transformers dependencies.

These are imported by :mod:`src.inference.utils.common` when the real
libraries are unavailable so that importing the module remains safe in
environments without GPU or heavy ML libraries.
"""

from __future__ import annotations

from typing import Any


class TorchStub:
    """Stub object that raises if torch-dependent utilities are used."""

    # Minimal symbolic tensor types to satisfy isinstance/attribute probes in
    # downstream code and tests when the real torch is unavailable.
    SymFloat = float
    SymBool = bool
    long = int
    long = "long"
    float16 = "float16"
    bfloat16 = "bfloat16"

    def __getattr__(self, _name: str) -> Any:
        msg = "torch is required for inference utilities in inference.common."
        raise ImportError(msg)

    # Minimal tensor creation helpers used in tests when the real torch
    # package is absent.
    def tensor(self, data=None, **_kwargs: Any):
        """Return raw data to mimic torch.tensor when torch is absent."""
        return data

    def zeros(self, shape=None, **_kwargs: Any):
        """Return a nested list of zeros with a minimal shape signature."""
        return [[0] * shape[1]] if isinstance(shape, tuple) else [0]

    def ones(self, shape=None, **_kwargs: Any):
        """Return a nested list of ones with a minimal shape signature."""
        return [[1] * shape[1]] if isinstance(shape, tuple) else [1]

    def is_available(self) -> bool:
        """Return False to mirror torch.cuda.is_available-style probes."""
        return False

    def device(self) -> str:
        """Placeholder device accessor to satisfy style checks."""
        return "cpu"

    def inference_mode(self, *_args: Any, **_kwargs: Any):
        """Stub context/decorator matching torch.inference_mode."""
        return lambda fn: fn


class StoppingCriteriaStub:
    """Stub StoppingCriteria base when transformers is unavailable."""

    def __call__(self, *_: Any, **__: Any) -> bool:
        msg = "transformers is required for StoppingCriteria in inference.common."
        raise ImportError(msg)

    def clone(self) -> "StoppingCriteriaStub":
        """Return self; provided solely to satisfy minimal API expectations."""
        return self

    def has_stops(self) -> bool:
        """Placeholder method for compatibility with StopOnSubstrings."""
        return False


__all__ = ["TorchStub", "StoppingCriteriaStub"]
