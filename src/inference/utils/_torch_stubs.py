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

    def __getattr__(self, _name: str) -> Any:
        msg = "torch is required for inference utilities in inference.common."
        raise ImportError(msg)

    def is_available(self) -> bool:
        """Return False to mirror torch.cuda.is_available-style probes."""
        return False

    def device(self) -> str:
        """Placeholder device accessor to satisfy style checks."""
        return "cpu"


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
