"""Optional dependency handling for hierarchical GRPO training."""

from __future__ import annotations

import types
from contextlib import nullcontext
from typing import Any


try:  # pragma: no cover - optional dependency
    import torch as _torch
except ImportError:  # pragma: no cover - type-check / lint environment
    _torch = None

if _torch is None:  # pragma: no cover - exercised in minimal envs
    import numpy as _np

    class _Tensor:
        def __init__(self, data, device=None, dtype=None):
            self.data = _np.array(data)
            self.device = device
            self.dtype = dtype

        def tolist(self):
            """Return the underlying data as a Python list."""
            return self.data.tolist()

        def size(self, dim=None):
            """Mirror torch.Tensor.size."""
            if dim is None:
                return self.data.size
            return self.data.shape[dim]

        @property
        def shape(self):
            """Mirror torch.Tensor.shape."""
            return self.data.shape

        def int(self):
            """Return an int-typed tensor wrapper."""
            return _Tensor(self.data.astype(int), device=self.device)

        def __int__(self):
            return int(self.data)

        def float(self):
            """Return a float-typed tensor wrapper."""
            return _Tensor(self.data.astype(float), device=self.device)

        def clone(self):
            """Return a shallow copy preserving device/dtype."""
            return _Tensor(self.data.copy(), device=self.device, dtype=self.dtype)

        def item(self):
            """Return the underlying scalar."""
            return self.data.item()

        def sum(self, dim=None):
            """Sum elements along the given dimension."""
            return _Tensor(self.data.sum(axis=dim))

        def nansum(self, dim=None):
            """Sum while ignoring NaNs along the given dimension."""
            return _Tensor(_np.nansum(self.data, axis=dim))

        def mean(self, dim=None):
            """Mean along the given dimension."""
            return _Tensor(self.data.mean(axis=dim))

        def std(self, dim=None, unbiased=False):
            """Return standard deviation; mirrors torch.std for stubs."""
            ddof = 1 if unbiased else 0
            return _Tensor(self.data.std(axis=dim, ddof=ddof))

        def any(self, dim=None):
            """Return boolean any along dimension."""
            return _Tensor(self.data.any(axis=dim))

        def argmax(self, dim=None):
            """Return argmax along the given dimension."""
            return _Tensor(self.data.argmax(axis=dim))

        def unsqueeze(self, dim):
            """Return a view with a new singleton dimension at ``dim``."""
            return _Tensor(_np.expand_dims(self.data, axis=dim), device=self.device)

        def expand_as(self, other):
            """Broadcast this tensor to the shape of ``other``."""
            return _Tensor(_np.broadcast_to(self.data, other.shape), device=self.device)

        def view(self, *shape):
            """Reshape the underlying array; mirrors torch.view for stubs."""
            return _Tensor(self.data.reshape(*shape), device=self.device)

        def repeat_interleave(self, repeats, dim=0):
            """Repeat elements along ``dim`` similar to torch.repeat_interleave."""
            data_rep = _np.repeat(self.data, repeats, axis=dim)
            return _Tensor(data_rep, device=self.device)

        def equal(self, other):
            """Elementwise equality against another tensor-like or scalar."""
            return _Tensor(self.data == (other.data if isinstance(other, _Tensor) else other))

        eq = equal

        def all(self):
            """Return True if all elements are truthy."""
            return bool(self.data.all())

        def __invert__(self):
            return _Tensor(~self.data, device=self.device, dtype=self.dtype)

        def __eq__(self, other):
            return self.equal(other)

        def __add__(self, other):
            return _Tensor(self.data + (other.data if isinstance(other, _Tensor) else other))

        def __sub__(self, other):
            return _Tensor(self.data - (other.data if isinstance(other, _Tensor) else other))

        def __mul__(self, other):
            return _Tensor(self.data * (other.data if isinstance(other, _Tensor) else other))

        def __truediv__(self, other):
            return _Tensor(self.data / (other.data if isinstance(other, _Tensor) else other))

        def __iter__(self):
            for item in self.data:
                yield _Tensor(item, device=self.device)

        def __getitem__(self, idx):
            if isinstance(idx, _Tensor):
                idx = idx.data
            return _Tensor(self.data[idx], device=self.device)

        def __bool__(self):
            return bool(self.data.any() if hasattr(self.data, "any") else self.data)

        def __setitem__(self, idx, value):
            self.data[idx] = value.data if isinstance(value, _Tensor) else value

        def __len__(self):
            return len(self.data)

        def __le__(self, other):
            return _Tensor(
                self.data <= (other.data if isinstance(other, _Tensor) else other),
                device=self.device,
            )

    def _tensor(data, device=None, dtype=None):
        return _Tensor(data, device=device, dtype=dtype)

    def _zeros(shape, device=None, dtype=None):
        return _Tensor(_np.zeros(shape, dtype=dtype), device=device, dtype=dtype)

    def _zeros_like(tensor, device=None):
        return _Tensor(_np.zeros_like(tensor.data), device=device)

    def _full(shape, fill_value, device=None, dtype=None):
        return _Tensor(_np.full(shape, fill_value, dtype=dtype), device=device, dtype=dtype)

    def _arange(end, device=None):
        return _Tensor(_np.arange(end), device=device)

    def _cat(tensors, dim=0):
        arrays = [_t.data if isinstance(_t, _Tensor) else _t for _t in tensors]
        return _Tensor(_np.concatenate(arrays, axis=dim))

    def _isclose(lhs, rhs, rtol=1e-05, atol=1e-08):
        a_data = lhs.data if isinstance(lhs, _Tensor) else lhs
        b_data = rhs.data if isinstance(rhs, _Tensor) else rhs
        return _Tensor(_np.isclose(a_data, b_data, rtol=rtol, atol=atol))

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, traceback):
            return False

        def __call__(self, func=None):
            return func

    def _no_grad():
        return _NoGrad()

    torch = types.SimpleNamespace(
        tensor=_tensor,
        zeros=_zeros,
        zeros_like=_zeros_like,
        full=_full,
        arange=_arange,
        cat=_cat,
        isclose=_isclose,
        no_grad=_no_grad,
        inference_mode=_no_grad,
        long="long",
        bfloat16="bfloat16",
        float16="float16",
    )
    FSDP = None

    def broadcast_object_list(*_args: Any, **_kwargs: Any) -> Any:
        """Stub broadcast_object_list that raises when accelerate is missing."""
        msg = "accelerate is required for hierarchical GRPO utilities."
        raise ImportError(msg)

    def gather_object(*_args: Any, **_kwargs: Any) -> Any:
        """Stub gather_object that raises when accelerate is missing."""
        msg = "accelerate is required for hierarchical GRPO utilities."
        raise ImportError(msg)

    Trainer = object
    TrainerCallback = object
    PaddingStrategy = object

    def is_flash_attn_2_available() -> bool:
        """Stub flash-attn-2 check that always returns False."""
        return False
else:  # pragma: no cover - primary runtime path
    torch = _torch
    try:
        from accelerate.utils import broadcast_object_list, gather_object
    except ImportError:
        # Provide no-op fallbacks so downstream helpers keep working in minimal environments.
        def broadcast_object_list(objs, *args, **kwargs):
            """Return objects unchanged when accelerate is unavailable."""
            _ = (args, kwargs)  # satisfy linters for unused passthrough args
            return objs

        def gather_object(obj, *args, **kwargs):
            """Return object unchanged when accelerate is unavailable."""
            _ = (args, kwargs)
            return obj

    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    except ImportError:
        FSDP = None
    try:
        from transformers import Trainer, TrainerCallback
        from transformers.tokenization_utils_base import PaddingStrategy
        from transformers.utils import is_flash_attn_2_available
    except ImportError:
        Trainer = object
        TrainerCallback = object
        PaddingStrategy = object

        def is_flash_attn_2_available() -> bool:
            """Stub flash-attn-2 availability check when transformers is missing."""
            return False


try:  # pragma: no cover - optional dependencies at runtime
    from trl import GRPOTrainer
    from trl.data_utils import is_conversational, maybe_apply_chat_template
    from trl.extras.profiling import profiling_context
    from trl.trainer.grpo_trainer import pad, unwrap_model_for_generation
except (ImportError, RuntimeError):  # pragma: no cover - type-check / lint environment
    GRPOTrainer = Trainer

    def is_conversational(*_args: Any, **_kwargs: Any) -> bool:
        """Stub conversational check that always returns False."""
        return False

    def maybe_apply_chat_template(example: Any, _processor: Any) -> dict:
        """Minimal passthrough: expect dicts with a 'prompt' key."""
        if isinstance(example, dict) and "prompt" in example:
            return {"prompt": example["prompt"]}
        return {"prompt": example}

    def profiling_context(*_args: Any, **_kwargs: Any):
        """Stub profiling context that acts as a no-op context manager."""
        return nullcontext()

    def pad(*_args: Any, **_kwargs: Any) -> Any:
        """Stub pad that raises when trl is missing."""
        msg = "trl is required for hierarchical GRPO training (pad)."
        raise ImportError(msg)

    def unwrap_model_for_generation(*_args: Any, **_kwargs: Any) -> Any:
        """Stub unwrap_model_for_generation that raises when trl is missing."""
        msg = "trl is required for hierarchical GRPO training (unwrap_model_for_generation)."
        raise ImportError(msg)


def resolve_dependency(instance: Any, name: str, default: Any):
    """
    Return an attribute from ``instance`` if present, otherwise ``default``.

    This helper avoids cross-module imports when hierarchical trainer mixins
    need to fall back to attributes supplied by the trainer.
    """
    return getattr(instance, name, default)


def canonicalize_device(device: Any) -> Any:
    """
    Convert accelerator-style device objects into something torch accepts.

    Torch APIs expect ``torch.device`` (or a string). Accelerate sometimes
    provides a SimpleNamespace with a ``type`` attribute, which causes type
    errors when passed through directly. This helper strips out the ``type`` or
    wraps it in ``torch.device`` when available.
    """
    if device is None:
        return None
    torch_device = getattr(torch, "device", None)
    try:
        if torch_device and isinstance(device, torch_device):
            return device
    except TypeError:
        # Some stub torch.device implementations raise TypeError on isinstance
        pass

    # Derive a string-like candidate (for example, "cpu") when given a SimpleNamespace.
    candidate = getattr(device, "type", device)
    # Try multiple fallbacks to coerce the device into something torch understands.
    for attempt in (candidate, getattr(candidate, "type", candidate)):
        if torch_device:
            try:
                return torch_device(attempt)
            except (TypeError, ValueError, RuntimeError):
                # Some stubs or unexpected device objects will fail here; keep trying.
                continue
        if isinstance(attempt, str):
            return attempt
    return candidate


_EXPORTED_SYMBOLS = {
    "torch": torch,
    "FSDP": FSDP,
    "broadcast_object_list": broadcast_object_list,
    "gather_object": gather_object,
    "resolve_dependency": resolve_dependency,
    "Trainer": Trainer,
    "TrainerCallback": TrainerCallback,
    "PaddingStrategy": PaddingStrategy,
    "is_flash_attn_2_available": is_flash_attn_2_available,
    "GRPOTrainer": GRPOTrainer,
    "is_conversational": is_conversational,
    "maybe_apply_chat_template": maybe_apply_chat_template,
    "profiling_context": profiling_context,
    "pad": pad,
    "unwrap_model_for_generation": unwrap_model_for_generation,
    "canonicalize_device": canonicalize_device,
}

__all__ = list(_EXPORTED_SYMBOLS)
