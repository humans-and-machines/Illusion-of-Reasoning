"""
Backwards-compatible shim for the shift annotation CLI.

Canonical implementation now lives in :mod:`src.annotate.cli.shift_cli`.
"""

from ..cli.shift_cli import build_argparser, main

__all__ = ["build_argparser", "main"]


if __name__ == "__main__":
    main()
