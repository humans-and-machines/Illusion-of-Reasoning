"""Shared matplotlib stub helpers to keep pylint duplication checks happy."""

from __future__ import annotations

from typing import Any


class AxisSettersMixin:
    """Mixin providing no-op axis setter methods for stubbed matplotlib."""

    def set_xlabel(self, *_args: Any, **_kwargs: Any):
        """No-op xlabel setter used in stub axes."""
        return None

    def set_ylabel(self, *_args: Any, **_kwargs: Any):
        """No-op ylabel setter used in stub axes."""
        return None

    def set_title(self, *_args: Any, **_kwargs: Any):
        """No-op title setter used in stub axes."""
        return None

    def set_xticks(self, *_args: Any, **_kwargs: Any):
        """No-op xticks setter used in stub axes."""
        return None

    def set_xticklabels(self, *_args: Any, **_kwargs: Any):
        """No-op xticklabels setter used in stub axes."""
        return None

    def legend(self, *_args: Any, **_kwargs: Any):
        """No-op legend call used in stub axes."""
        return None

    def grid(self, *_args: Any, **_kwargs: Any):
        """No-op grid call used in stub axes."""
        return None

    def axhline(self, *_args: Any, **_kwargs: Any):
        """No-op axhline used in stub axes."""
        return None

    def get_legend_handles_labels(self):
        """Return empty legend handles/labels for stub axes."""
        return [], []

    def fill_between(self, *_args: Any, **_kwargs: Any):
        """No-op fill_between used in stub axes."""
        return None

    def plot(self, *_args: Any, **_kwargs: Any):
        """No-op plot used in stub axes."""
        return None


def ensure_switch_backend(plt_mod: Any) -> Any:
    """Add a no-op switch_backend to a stubbed pyplot if missing."""
    if not hasattr(plt_mod, "switch_backend"):
        plt_mod.switch_backend = lambda *_a, **_k: None
    return plt_mod


def coerce_axes_sequence(axes: Any, expected: int | None = None) -> list[Any]:
    """
    Normalize matplotlib axes containers into a list for downstream iteration.

    Matplotlib may return a NumPy array of axes while stubs may return a single
    axis; this helper always returns a list and optionally repeats a single axis
    to match an expected length.
    """
    try:
        axes_seq = list(axes)
    except TypeError:
        axes_seq = [axes]
    if not isinstance(axes_seq, list):
        axes_seq = [axes_seq]
    if expected and expected > 1 and len(axes_seq) == 1:
        axes_seq = axes_seq * expected
    return axes_seq
