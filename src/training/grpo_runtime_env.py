"""Compatibility shim for :mod:`src.training.runtime.env`."""

from __future__ import annotations

from .runtime import env as _runtime_env


_public_exports = getattr(
    _runtime_env,
    "__all__",
    [name for name in dir(_runtime_env) if not name.startswith("_")],
)

__all__ = list(_public_exports)
globals().update({name: getattr(_runtime_env, name) for name in __all__})
