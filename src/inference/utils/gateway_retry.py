#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retry helpers shared by gateway scripts.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from src.inference.utils.gateway_logging import setup_script_logger


__all__ = ["RetryContext", "RetrySettings", "call_with_retries", "call_with_gateway_retries"]


@dataclass(frozen=True)
class RetryContext:
    """
    Context used for logging retry attempts.
    """

    logger: logging.Logger
    sample_idx: int
    problem_snippet: str
    min_sleep: Optional[float] = None


@dataclass(frozen=True)
class RetrySettings:
    """
    Tunable retry settings shared across gateway callers.
    """

    max_retries: int
    retry_backoff: float
    exception_types: Sequence[type[BaseException]] = (Exception,)


def call_with_retries(func, *, settings: RetrySettings, context: RetryContext):
    """
    Call ``func()`` with simple retry-on-exception semantics shared by math gateways.

    :param func: Zero-argument callable to execute.
    :param settings: Retry settings bundle (counts and backoff).
    :param context: Logging context for this call.
    :returns: Result of ``func()`` if it eventually succeeds.
    :raises Exception: Re-raises the last exception if retries are exhausted.
    """
    exception_types: Tuple[type[BaseException], ...] = tuple(settings.exception_types) or (Exception,)
    attempt = 0
    while True:
        try:
            return func()
        except exception_types as exc:
            attempt += 1
            if attempt > settings.max_retries:
                context.logger.error(
                    "Failed after %d retries on sample_idx=%d | prob snippet=%.60s | err=%r",
                    attempt - 1,
                    context.sample_idx,
                    context.problem_snippet,
                    exc,
                )
                raise
            sleep_dur = settings.retry_backoff * attempt
            if context.min_sleep is not None:
                sleep_dur = max(context.min_sleep, sleep_dur)
            context.logger.warning(
                "Retry %d/%d for sample_idx=%d after error: %r (sleep %.1fs)",
                attempt,
                settings.max_retries,
                context.sample_idx,
                exc,
                sleep_dur,
            )
            time.sleep(sleep_dur)


def call_with_gateway_retries(
    func,
    args: argparse.Namespace,
    context: Optional[RetryContext] = None,
    **legacy_context: object,
):
    """
    Convenience wrapper around ``call_with_retries`` using standard CLI args.

    This deduplicates the common pattern used by math gateway runners.
    """
    if context is None:
        logger = legacy_context.get("logger")
        sample_idx = legacy_context.get("sample_idx")
        problem_snippet = legacy_context.get("problem_snippet")
        min_sleep = legacy_context.get("min_sleep")
        context = RetryContext(
            logger=logger or setup_script_logger(__name__),
            sample_idx=sample_idx if sample_idx is not None else -1,
            problem_snippet=problem_snippet or "",
            min_sleep=min_sleep,
        )
    return call_with_retries(
        func,
        settings=RetrySettings(
            max_retries=args.max_retries,
            retry_backoff=args.retry_backoff,
        ),
        context=context,
    )
