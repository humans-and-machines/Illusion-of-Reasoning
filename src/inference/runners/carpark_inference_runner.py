#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Legacy Rush Hour (car-park) inference entrypoint.

This module forwards directly to :mod:`src.inference.domains.carpark.carpark_core` and is kept
only for CLI compatibility. For new runs, prefer the unified CLI
``src.inference.cli.unified_carpark``.
"""

from src.inference.domains.carpark.carpark_core import main


if __name__ == "__main__":
    main()
