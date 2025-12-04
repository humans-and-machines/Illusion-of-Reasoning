#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight logging and cache helpers shared by gateway utilities.
"""

from __future__ import annotations

import logging
import os


__all__ = ["setup_hf_cache_dir_env", "setup_script_logger"]


def setup_hf_cache_dir_env(base_dir: str = "./.hf_cache") -> str:
    """
    Initialize HuggingFace cache directory and related environment variables.

    :param base_dir: Base directory to use for the HF cache (directories such
        as ``HF_HOME`` and ``TRANSFORMERS_CACHE`` will be created under this).
    :returns: Absolute path to the HF cache directory so callers can pass it
        to transformers / datasets loaders.
    """
    hf_cache_dir = os.path.abspath(base_dir)
    os.environ.update(
        HF_HOME=hf_cache_dir,
        TRANSFORMERS_CACHE=os.path.join(hf_cache_dir, "transformers"),
        HF_HUB_CACHE=os.path.join(hf_cache_dir, "hub"),
    )
    return hf_cache_dir


def setup_script_logger(name: str) -> logging.Logger:
    """
    Configure a basic process-wide logger using LOGLEVEL env and return a module logger.

    This mirrors the common pattern used in the inference entrypoints.

    :param name: Logger name (typically ``__name__`` of the calling module).
    :returns: Configured :class:`logging.Logger` instance.
    """
    loglevel = os.getenv("LOGLEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, loglevel, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%-Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)
