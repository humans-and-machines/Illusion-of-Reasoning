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
    """
    Lightweight view of generation outputs used for entropy computation.

    :param sequences: Generated token ID sequences (for example, from ``generate``).
    :param scores: Per-step score tensors (logits or log-probs) aligned with ``sequences``.
    :param model: Underlying model used to compute entropies.
    """

    sequences: Any
    scores: Any
    model: Any


class DSModelWrapper:
    """
    Adapter so code can call ``.generate`` and ``.parameters()`` on a DeepSpeed engine.

    This wrapper exposes a Hugging Face-like interface around a DeepSpeed engine
    so downstream utilities can treat it like a regular ``AutoModelForCausalLM``.
    """

    def __init__(self, engine: Any):
        """
        Initialize the wrapper with a DeepSpeed engine.

        :param engine: DeepSpeed engine instance to wrap.
        """
        self.engine = engine
        self.module = engine.module
        self.device = getattr(
            engine,
            "device",
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        self.config = getattr(self.module, "config", None)

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to the underlying HF module.

        :param name: Attribute name being accessed.
        :returns: Attribute value from the underlying module if present.
        :raises AttributeError: If the attribute does not exist on the module.
        """
        if hasattr(self.module, name):
            return getattr(self.module, name)
        raise AttributeError(name)

    def parameters(self):
        """
        Expose the underlying module's parameters iterator.

        :returns: Iterator over model parameters.
        """
        return self.module.parameters()

    def generate(self, *args, **kwargs):
        """
        Forward ``generate`` calls to the underlying HF module.

        :param args: Positional arguments forwarded to ``module.generate``.
        :param kwargs: Keyword arguments forwarded to ``module.generate``.
        :returns: Result of the underlying ``generate`` call.
        """
        return self.module.generate(*args, **kwargs)

    def eval(self):
        """
        Put the underlying module into eval mode and return ``self``.

        :returns: The current :class:`DSModelWrapper` instance.
        """
        self.module.eval()
        return self
