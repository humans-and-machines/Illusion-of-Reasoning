#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compatibility wrapper for the Forced Aha effect analysis.

The actual implementation lives in :mod:`src.analysis.forced_aha_effect_impl`;
this file simply delegates to keep existing CLI entry points working.
"""

from __future__ import annotations

from src.analysis.script_utils import ensure_script_context

from src.analysis.forced_aha_effect_impl import main

ensure_script_context()

if __name__ == "__main__":
    main()
