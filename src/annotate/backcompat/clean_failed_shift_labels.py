"""
Backwards-compatible shim for the cleaning CLI.

Canonical implementation now lives in :mod:`src.annotate.cli.clean_cli`.
"""

from ..cli.clean_cli import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
