"""
CLI entrypoints for annotation utilities.

These modules are primarily intended to be executed as scripts.
"""

from .shift_cli import build_argparser as shift_build_argparser
from .shift_cli import main as shift_main
from .clean_cli import main as clean_failed_main

__all__ = ["shift_build_argparser", "shift_main", "clean_failed_main"]
