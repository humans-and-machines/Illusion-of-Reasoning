"""Public GRPO CLI shim.

This module remains import-compatible (`python -m training.grpo`) while the
real implementation now lives under :mod:`src.training.cli.grpo`.
"""

from __future__ import annotations

from .cli import grpo as grpo_cli
from .configs import merge_dataclass_attributes


def main() -> None:
    """Delegate to the CLI entrypoint, allowing tests to patch merge helpers."""
    grpo_cli.main(merge_fn=merge_dataclass_attributes)


__all__ = ["main", "merge_dataclass_attributes"]


if __name__ == "__main__":
    main()
