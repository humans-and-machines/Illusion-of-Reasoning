#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utility helpers for math_llama_core (kept separate to reduce module size)."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

try:
    torch = importlib.import_module("torch")
except ImportError as exc:  # pragma: no cover - hard dependency
    raise RuntimeError(
        "math_llama_utils requires 'torch'; install the 'torch' package.",
    ) from exc


@dataclass
class GeneratedBatchView:
    """Lightweight view of generation outputs used for entropy computation."""

    sequences: Any
    scores: Any
    model: Any


class DSModelWrapper:
    """Adapter so code can call `.generate` and `.parameters()` on a DeepSpeed engine."""

    def __init__(self, engine: Any):
        """Initialize the wrapper with a DeepSpeed engine."""
        self.engine = engine
        self.module = engine.module
        self.device = getattr(
            engine,
            "device",
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        self.config = getattr(self.module, "config", None)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying HF module."""
        if hasattr(self.module, name):
            return getattr(self.module, name)
        raise AttributeError(name)

    def parameters(self):
        """Expose the underlying module's parameters iterator."""
        return self.module.parameters()

    def generate(self, *args, **kwargs):
        """Forward generate calls to the underlying module."""
        return self.module.generate(*args, **kwargs)

    def eval(self):
        """Put the underlying module into eval mode and return self."""
        self.module.eval()
        return self
