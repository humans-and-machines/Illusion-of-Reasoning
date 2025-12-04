#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified math inference CLI.

Thin wrapper that wires :class:`HFBackend` into
``src.inference.runners.unified_runner_base.run_math_main``.
"""

from __future__ import annotations

from typing import Optional, Sequence

from src.inference.backends import HFBackend
from src.inference.runners.unified_runner_base import run_math_main


__all__ = ["main", "HFBackend", "run_math_main"]


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point for math inference CLI."""
    kwargs = {"backend_cls": HFBackend}
    if argv is not None:
        kwargs["argv"] = argv
    run_math_main(**kwargs)


if __name__ == "__main__":
    main()
