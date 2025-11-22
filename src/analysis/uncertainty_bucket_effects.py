#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compatibility wrapper for ``uncertainty_bucket_effects``.

The full implementation lives in :mod:`src.analysis.uncertainty_bucket_effects_impl`;
this module simply forwards ``main()`` so existing automation and docs do not
need to change their import paths.
"""

from __future__ import annotations

from src.analysis.script_utils import ensure_script_context

from src.analysis.uncertainty_bucket_effects_impl import main

ensure_script_context()

if __name__ == "__main__":
    main()
