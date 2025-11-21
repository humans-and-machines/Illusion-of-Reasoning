#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Legacy entrypoint for the H3 uncertainty × intervention analysis.

This thin wrapper forwards to the canonical module
``src.analysis.h3_analysis``. For new runs, prefer:

  python -m src.analysis.rq3_analysis ...
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def main() -> None:
    here = Path(__file__).resolve().parent
    target = here / "h3_analysis.py"
    if not target.is_file():
        raise SystemExit(f"Expected h3_analysis.py next to this file (missing: {target})")

    spec = importlib.util.spec_from_file_location("h3_analysis_legacy", target)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load h3_analysis module from {target}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[call-arg]

    if not hasattr(mod, "main"):
        raise SystemExit("h3_analysis.py does not define a main() function.")

    # Reuse the current sys.argv so CLI behavior matches the original script.
    mod.main()  # type: ignore[call-arg]


if __name__ == "__main__":
    main()

