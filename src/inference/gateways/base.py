"""
Shared helpers for gateway provider implementations.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.inference.utils.common import setup_script_logger
from src.inference.utils.task_registry import TASK_REGISTRY, TaskSpec


__all__ = ["get_task_spec", "setup_gateway_logger"]


def get_task_spec(task_name: str) -> TaskSpec:
    """
    Look up a task specification by name and raise if missing.
    """
    spec = TASK_REGISTRY.get(task_name)
    if spec is None:
        raise KeyError(f"Unknown task name: {task_name}")
    return spec


def setup_gateway_logger(name: Optional[str]) -> logging.Logger:
    """
    Create a standard script logger for gateway providers.
    """
    return setup_script_logger(name or __name__)
