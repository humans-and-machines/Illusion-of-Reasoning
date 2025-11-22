#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Legacy-compatible wrapper for the Figure 2 (uncertainty → correctness) script.

Canonical usage is now::

  python -m src.analysis.figure_2_uncertainty ...

Internally this loads and forwards to the new ``figure_2.py`` module so
existing scripts that call ``python src/analysis/figure-2.py`` continue to work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from src.analysis.utils import load_legacy_main_from_path


def _load_legacy_main() -> Callable[[], Any]:
    here = Path(__file__).resolve().parent
    script_path = here / "figure_2.py"
    return load_legacy_main_from_path(
        script_path,
        import_name="analysis_figure_2_legacy",
    )


def main() -> None:
    """Load and invoke the legacy Figure 2 script's ``main()`` function."""
    _load_legacy_main()()


if __name__ == "__main__":
    main()
