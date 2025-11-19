#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Legacy entrypoint forwarding to carpark_core.main (kept for CLI compatibility).
"""
from src.inference.carpark_core import main


if __name__ == "__main__":
    main()
