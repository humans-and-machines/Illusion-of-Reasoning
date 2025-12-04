"""Two-stage hierarchical rollout helper and trainer re-exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Optional, Tuple

from . import hierarchical_grpo_trainer as _trainer


# Optional numpy support for dtype shims.
try:  # pragma: no cover - optional dependency resolution
    import numpy as _np  # type: ignore
except ImportError:  # pragma: no cover - extremely minimal env
    _np = None

# Prefer a real torch install when available; fall back to the trainer stub otherwise.
try:  # pragma: no cover - optional dependency resolution
    import torch as _real_torch
except ImportError:  # pragma: no cover - environments without torch
    _real_torch = None


def _ensure_minimal_torch_apis(mod: Any) -> Any:
    """
    Backfill a handful of tensor helpers on lightweight torch stubs.

    Some unit tests install very small torch stand-ins that only expose
    ``Tensor`` and ``tensor``. Hierarchical rollout helpers (and a few tests)
    rely on: ``full``, ``ones``, ``cat``, and a ``long`` dtype marker.
    """
    if mod is None:
        return mod

    # Dtype marker used in tests when building integer tensors.
    if not hasattr(mod, "long") and _np is not None:
        try:
            setattr(mod, "long", _np.int64)
        except (AttributeError, TypeError):  # pragma: no cover - defensive
            pass

    # Helper to normalize arbitrary tensor-like inputs into Python lists.
    def _to_list(value: Any) -> Any:
        if hasattr(value, "tolist"):
            try:
                return value.tolist()
            except (TypeError, ValueError):
                pass
        try:
            return list(value)
        except TypeError:
            return value

    if not hasattr(mod, "full"):

        def _full(shape, fill_value, dtype=None, device=None):  # type: ignore[override]
            try:
                dims = list(shape)
            except TypeError:
                dims = [int(shape)]

            def _build(dimensions):
                if not dimensions:
                    return fill_value
                return [_build(dimensions[1:]) for _ in range(int(dimensions[0]))]

            data = _build(dims)
            return mod.tensor(data, dtype=dtype, device=device)

        try:
            setattr(mod, "full", _full)
        except (AttributeError, TypeError):  # pragma: no cover - defensive
            pass

    if not hasattr(mod, "ones"):

        def _ones(shape, dtype=None, device=None):  # type: ignore[override]
            return mod.full(shape, 1, dtype=dtype, device=device)

        try:
            setattr(mod, "ones", _ones)
        except (AttributeError, TypeError):  # pragma: no cover - defensive
            pass

    if not hasattr(mod, "cat"):

        def _cat(tensors, dim=0):  # type: ignore[override]
            seqs = [_to_list(t) for t in tensors]
            if not seqs:
                return mod.tensor([], device=None)

            # 1D concatenation: simple list extension.
            if dim == 0 and (not seqs or not isinstance(seqs[0][0], list)):
                merged = []
                for seq in seqs:
                    merged.extend(seq)
                return mod.tensor(merged, device=getattr(tensors[0], "device", None))

            # Assume 2D rows for dim==1 (the only case exercised in tests).
            rows = []
            for parts in zip(*seqs):
                row = []
                for part in parts:
                    row.extend(part)
                rows.append(row)
            return mod.tensor(rows, device=getattr(tensors[0], "device", None))

        try:
            setattr(mod, "cat", _cat)
        except (AttributeError, TypeError):  # pragma: no cover - defensive
            pass

    return mod


# Trainer stub exposes ``tensor`` but not the full Tensor API; swap in real torch when present.
if _real_torch is not None:
    torch = _ensure_minimal_torch_apis(_real_torch)
else:
    trainer_torch = getattr(_trainer, "torch", None)
    if trainer_torch is None:
        raise ImportError("hierarchical_grpo_trainer must expose a torch-like stub")
    torch = _ensure_minimal_torch_apis(trainer_torch)

try:  # pragma: no cover - optional dependency
    pad_sequence = import_module("torch.nn.utils.rnn").pad_sequence
except ImportError:  # pragma: no cover - type-check / lint environment

    def pad_sequence(*_args: Any, **_kwargs: Any) -> Any:
        """Stub pad_sequence that raises when torch is missing."""
        msg = "torch is required for hierarchical rollout utilities (pad_sequence)."
        raise ImportError(msg)


try:  # pragma: no cover - optional dependencies at runtime
    from transformers import PreTrainedTokenizerBase
    from transformers.generation.utils import GenerationMixin
except ImportError:  # pragma: no cover - type-check / lint environment
    PreTrainedTokenizerBase = object
    GenerationMixin = object

GenerationBatch = _trainer.GenerationBatch
RewardStatistics = _trainer.RewardStatistics
HierarchicalGRPOTrainer = _trainer.HierarchicalGRPOTrainer

__all__ = [
    "GenerationBatch",
    "RewardStatistics",
    "HierarchicalGRPOTrainer",
    "HierarchicalRollout",
]


def _is_full_torch_mod(mod: Any) -> bool:
    """
    Return True when ``mod`` behaves like a full torch install or the
    high-fidelity stubs used in tests.

    Lightweight fakes (for example, ``_FakeTorch`` in unit tests) only expose
    ``tensor``/``no_grad`` and should not trigger tensor coercion.
    """
    return bool(
        mod is not None
        and hasattr(mod, "tensor")
        and (hasattr(mod, "Tensor") or hasattr(mod, "cat") or getattr(mod, "__name__", "") == "torch")
    )


def _get_torch():
    """
    Return a torch-like object, preferring the module-level binding when set.

    Tests sometimes monkeypatch ``hr.torch`` to lightweight stubs; avoid
    importing or overriding torch in that case and fall back to the trainer's
    stub only when no usable binding exists.
    """
    torch_mod = globals().get("torch")
    if torch_mod is None or not hasattr(torch_mod, "tensor"):
        trainer_torch_mod = getattr(_trainer, "torch", None)
        torch_mod = trainer_torch_mod
    globals()["torch"] = torch_mod
    return torch_mod


def _device_from_input(obj: Any) -> Any:
    """Return a best-effort device spec from ``obj``, defaulting to CPU."""
    return getattr(obj, "device", "cpu")


def _coerce_sequence_data(seq: Any) -> list[Any]:
    """Return a list representation of ``seq`` even when ``len`` is unsupported."""
    if hasattr(seq, "tolist"):
        try:
            data = seq.tolist()
            if isinstance(data, list):
                return data
        except (TypeError, ValueError):
            pass
    try:
        return list(seq)
    except (TypeError, ValueError):
        return [seq]


def _try_pad_sequence(seq_batch: list[Any], pad_value: int) -> Any:
    """Attempt to pad ``seq_batch`` with ``pad_sequence``; return None on failure."""
    try:
        candidate = pad_sequence(
            seq_batch,
            batch_first=True,
            padding_value=pad_value,
        )
        len(candidate)  # ensure sequence-like
        return candidate
    except (RuntimeError, TypeError, ValueError, ImportError, AttributeError):
        return None


def _pad_rows_to_uniform_length(
    sequences: list[Any],
    pad_value: int,
    torch_mod: Any,
    device: Any,
) -> list[Any]:
    """Pad each sequence to the maximum length, preserving torch tensors when possible."""
    normalized = [_coerce_sequence_data(seq) for seq in sequences]
    max_len = max((len(seq) for seq in normalized), default=0)
    padded_rows = []
    for data in normalized:
        pad_needed = max_len - len(data)
        if pad_needed > 0:
            data = data + [pad_value] * pad_needed
        try:
            padded_rows.append(torch_mod.tensor(data, device=device))
        except (RuntimeError, TypeError, ValueError, AttributeError):
            padded_rows.append(data)
    return padded_rows


def _stack_padded_rows(
    padded_rows: list[Any],
    torch_mod: Any,
    device: Any,
) -> Any:
    """Convert padded rows into a single tensor; return None when conversion fails."""
    try:
        stacked = [row.tolist() if hasattr(row, "tolist") else list(row) for row in padded_rows]
        return torch_mod.tensor(stacked, device=device)
    except (RuntimeError, TypeError, ValueError, AttributeError):
        return None


def _safe_first_element(candidate: Any) -> Any:
    """Best-effort retrieval of the first element, returning None on failure."""
    try:
        return candidate[0]
    except (TypeError, IndexError, KeyError):
        return None


def _tensorize_rows(candidate: Any, torch_mod: Any, device: Any) -> Any | None:
    """Convert a sequence-like object to a list of tensors; return None on failure."""
    try:
        rows = candidate.tolist() if hasattr(candidate, "tolist") else list(candidate)
        return [torch_mod.tensor(row, device=device) for row in rows]
    except (RuntimeError, TypeError, ValueError, AttributeError):
        return None


def _normalize_candidate(
    candidate: Any,
    padded_rows: list[Any],
    torch_mod: Any,
    device: Any,
) -> Any:
    """
    Ensure the padded candidate is sequence-like and convertible to torch tensors.
    """
    try:
        len(candidate)
    except TypeError:
        return padded_rows

    first_elem = _safe_first_element(candidate)
    if first_elem is not None and not hasattr(first_elem, "tolist"):
        tensors = _tensorize_rows(candidate, torch_mod, device)
        if tensors is not None:
            return tensors
        return padded_rows

    if not hasattr(candidate, "tolist"):
        try:
            candidate = torch_mod.tensor(candidate, device=device)
        except (RuntimeError, TypeError, ValueError, AttributeError):
            return padded_rows
    return candidate


def _pad_sequences_for_batch(
    sequences: list[Any],
    pad_value: int,
    device: Any,
) -> Any:
    """
    Pad a batch of token id sequences using ``pad_sequence`` when compatible,
    falling back to a simple tensor stack when torch stubs are in play.
    """
    torch_mod = _get_torch()
    candidate = _try_pad_sequence(sequences, pad_value)
    if candidate is not None:
        if isinstance(candidate, list) and hasattr(torch_mod, "Tensor"):
            try:
                return torch_mod.tensor(candidate, device=device)
            except (RuntimeError, TypeError, ValueError, AttributeError):
                return candidate
        return candidate

    if not sequences:
        return torch_mod.tensor([], device=device)

    padded_rows = _pad_rows_to_uniform_length(
        sequences,
        pad_value,
        torch_mod,
        device,
    )

    candidate = _try_pad_sequence(padded_rows, pad_value)
    if candidate is None:
        candidate = _stack_padded_rows(padded_rows, torch_mod, device)
        if candidate is None:
            return padded_rows
    elif isinstance(candidate, list) and hasattr(torch_mod, "Tensor"):
        try:
            candidate = torch_mod.tensor(candidate, device=device)
        except (RuntimeError, TypeError, ValueError, AttributeError):
            pass

    return _normalize_candidate(candidate, padded_rows, torch_mod, device)


def _stack_row_list_if_possible(candidate: Any, torch_mod: Any, device: Any) -> Any:
    """
    Handle the special case where ``candidate`` is a list of row tensors.

    When successful, this returns a 2D tensor batch; otherwise, it returns the
    original candidate so the generic path can attempt coercion.
    """
    if not isinstance(candidate, list):
        return candidate

    try:
        rows = [row.tolist() if hasattr(row, "tolist") else list(row) for row in candidate]
        tensor = torch_mod.tensor(rows, device=device)
        if hasattr(tensor, "dim"):
            try:
                if tensor.dim() == 1:  # type: ignore[attr-defined]
                    tensor = tensor.unsqueeze(0)  # type: ignore[attr-defined]
            except (AttributeError, TypeError, RuntimeError):
                pass
        return tensor
    except (RuntimeError, TypeError, ValueError, AttributeError):
        return candidate


def _coerce_single_candidate(candidate: Any, torch_mod: Any, device: Any) -> Any:
    """Coerce a non-list candidate to a tensor batch when possible."""
    try:
        tensor_cls = getattr(torch_mod, "Tensor", None)
    except (AttributeError, TypeError):  # pragma: no cover - very defensive
        tensor_cls = None

    try:
        if tensor_cls is not None and isinstance(candidate, tensor_cls):
            tensor = candidate
        else:
            if hasattr(candidate, "tolist") and not isinstance(candidate, list):
                data = candidate.tolist()
            else:
                data = candidate
            tensor = torch_mod.tensor(data, device=device)
        if hasattr(tensor, "dim"):
            try:
                if tensor.dim() == 1:  # type: ignore[attr-defined]
                    tensor = tensor.unsqueeze(0)  # type: ignore[attr-defined]
            except (AttributeError, TypeError, RuntimeError):  # pragma: no cover - best-effort
                pass
        return tensor
    except (RuntimeError, TypeError, ValueError, AttributeError):
        return candidate


def _ensure_tensor(candidate: Any, device: Any) -> Any:
    """
    Convert ``candidate`` to a batched tensor when a full torch (or high
    fidelity stub) is available; otherwise, return ``candidate`` unchanged.
    """
    torch_mod = _get_torch()
    if not _is_full_torch_mod(torch_mod):
        return candidate

    stacked = _stack_row_list_if_possible(candidate, torch_mod, device)
    if stacked is not candidate:
        return stacked
    return _coerce_single_candidate(candidate, torch_mod, device)


class HierarchicalRollout:
    """
    Two-stage generation:
      1) Generate until </think> (or first <answer>), append the tags.
      2) Feed that full sequence back in to finish the answer.
    """

    def __init__(
        self,
        model: GenerationMixin,
        tokenizer: PreTrainedTokenizerBase,
        vllm_client: Optional[Any] = None,
        max_reason_tokens: int = 800,
    ):
        self.model = model
        self.tok = tokenizer
        self.vllm_client = vllm_client
        self.max_reason_tokens = max_reason_tokens

        # your tags
        self.think_close_ids = tokenizer.encode("</think>", add_special_tokens=False)
        self.answer_tag_ids = tokenizer.encode("<answer>", add_special_tokens=False)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: Optional[int] = None,
        **gen_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate completions using the two-stage hierarchical rollout."""
        return self(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            **gen_kwargs,
        )

    def get_tag_ids(self) -> Tuple[list[int], list[int]]:
        """Return the token ids used to detect </think> and <answer> tags."""
        return self.think_close_ids, self.answer_tag_ids

    def __call__(
        self, input_ids: torch.Tensor, max_new_tokens: Optional[int] = None, **gen_kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the two-stage rollout with an optional ``torch.no_grad`` guard.

        When torch is unavailable (for example, in minimal test environments),
        we simply skip installing the no-grad context.
        """

        def _run():
            device = _device_from_input(input_ids)
            reason_ids = self._run_stage1_reasoning(
                input_ids,
                device,
                **gen_kwargs,
            )
            full_ids = self._run_stage2_answer(
                reason_ids,
                device,
                max_new_tokens=max_new_tokens,
                **gen_kwargs,
            )
            return reason_ids, full_ids

        torch_mod = _get_torch()
        if torch_mod is None or not hasattr(torch_mod, "no_grad"):
            return _run()
        with torch_mod.no_grad():
            return _run()

    def _run_stage1_reasoning(
        self,
        input_ids: torch.Tensor,
        device: torch.device,
        **gen_kwargs,
    ) -> torch.Tensor:
        """Run the first-stage reasoning step and return padded reason ids."""
        if self.vllm_client:
            prompts = self.tok.batch_decode(
                input_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            reason_lists = self.vllm_client.generate(
                prompts=prompts,
                n=1,
                max_tokens=self.max_reason_tokens,
                **gen_kwargs,
            )
        else:
            reason_tensor = self.model.generate(
                input_ids,
                max_new_tokens=self.max_reason_tokens,
                eos_token_id=self.think_close_ids[-1],
                do_sample=True,
                **gen_kwargs,
            )
            reason_lists = [tensor.tolist() for tensor in reason_tensor]

        torch_mod = _get_torch()
        padded_reason_ids = []
        for sequence in reason_lists:
            if sequence[-len(self.think_close_ids) :] != self.think_close_ids:
                sequence = sequence + self.think_close_ids + self.answer_tag_ids
            else:
                sequence = sequence + self.answer_tag_ids
            padded_reason_ids.append(torch_mod.tensor(sequence, device=device))

        padded = _pad_sequences_for_batch(
            padded_reason_ids,
            pad_value=self.tok.pad_token_id,
            device=device,
        )
        padded = self._ensure_answer_tail(padded, torch_mod, device)
        return _ensure_tensor(padded, device)

    def _ensure_answer_tail(self, padded, torch_mod, device):  # pylint: disable=too-many-statements
        """Ensure padded reason sequences end with the <answer> tag ids."""
        answer_tail = list(self.answer_tag_ids)
        if not answer_tail:
            return padded

        tail_len = len(answer_tail)

        def _assign_tail(seq_obj, new_tail):
            try:
                seq_obj[-tail_len:] = new_tail  # type: ignore[index]
                return seq_obj, True
            except (IndexError, TypeError):  # pragma: no cover - defensive fallback
                try:
                    data = seq_obj.tolist()
                except (AttributeError, TypeError):  # pragma: no cover - defensive fallback
                    try:
                        data = list(seq_obj)
                    except (TypeError, ValueError):  # pragma: no cover - defensive fallback
                        return seq_obj, False
                if len(data) >= tail_len:
                    data[-tail_len:] = new_tail
                else:
                    data = (data + new_tail)[-tail_len:]
                try:
                    return torch_mod.tensor(data, device=device), True
                except (TypeError, ValueError):  # pragma: no cover - defensive fallback
                    try:
                        seq_obj[:] = data  # type: ignore[index]
                        return seq_obj, True
                    except (TypeError, ValueError):  # pragma: no cover - defensive fallback
                        return seq_obj, False

        try:
            seq_count = len(padded)  # type: ignore[arg-type]
        except (TypeError, AttributeError):  # pragma: no cover - defensive fallback
            return padded

        for idx in range(seq_count):
            updated = False
            try:
                padded[idx, -tail_len:] = torch_mod.tensor(  # type: ignore[index]
                    answer_tail,
                    device=device,
                )
                updated = True
            except (TypeError, IndexError):  # pragma: no cover - defensive fallback
                try:
                    seq = padded[idx]  # type: ignore[index]
                except (TypeError, IndexError):  # pragma: no cover - defensive fallback
                    continue
                new_seq, updated = _assign_tail(seq, answer_tail)
                if updated and isinstance(padded, list):
                    try:
                        padded[idx] = new_seq  # type: ignore[index]
                    except (TypeError, IndexError):  # pragma: no cover - defensive fallback
                        pass
        try:
            if not isinstance(padded, list) and hasattr(padded, "tolist"):
                raw = padded.tolist()

                def _coerce(val):
                    if hasattr(val, "tolist") and not isinstance(val, (str, bytes)):
                        return _coerce(val.tolist())
                    try:
                        return int(val)
                    except (TypeError, ValueError):  # pragma: no cover - defensive fallback
                        try:
                            return float(val)
                        except (TypeError, ValueError):  # pragma: no cover - defensive fallback
                            return val

                if isinstance(raw, list):
                    cleaned = [[_coerce(v) for v in row] if isinstance(row, list) else _coerce(row) for row in raw]
                    try:
                        return torch_mod.tensor(cleaned, device=device)
                    except (  # pragma: no cover - defensive fallback
                        TypeError,
                        ValueError,
                        RuntimeError,
                    ):
                        pass
        except (TypeError, ValueError, AttributeError):  # pragma: no cover - defensive fallback
            pass
        return padded

    def _run_stage2_answer(  # pylint: disable=too-many-locals
        self,
        reason_ids: torch.Tensor,
        device: torch.device,
        max_new_tokens: Optional[int] = None,
        **gen_kwargs,
    ) -> torch.Tensor:
        """Run the second-stage answer generation step."""
        torch_mod = _get_torch()
        full_torch = _is_full_torch_mod(torch_mod)

        # Normalize ``reason_ids`` into a batch representation suitable for
        # downstream generation. For full torch installs (or their high
        # fidelity stubs) we convert to a proper tensor batch; for lightweight
        # fakes we leave the input unchanged.
        normalized_reason = _ensure_tensor(reason_ids, device) if full_torch else reason_ids

        # Ensure a leading batch dimension for full-torch style backends when
        # given a 1D tensor-like object. Some unit-test stubs only expose a
        # ``shape`` attribute rather than ``dim``.
        if full_torch and not isinstance(normalized_reason, list):
            try:
                shape = getattr(normalized_reason, "shape", None)
                if shape is not None and len(shape) == 1:
                    row = (
                        normalized_reason.tolist() if hasattr(normalized_reason, "tolist") else list(normalized_reason)
                    )
                    normalized_reason = torch_mod.tensor([row], device=device)
            except (TypeError, ValueError, RuntimeError, AttributeError):
                # Best-effort normalization; fall back to original object on error.
                pass

        if self.vllm_client:
            # vLLM backend: decode the reasoning tokens and ask the client to
            # generate answer continuations.
            reason_texts = self.tok.batch_decode(
                normalized_reason,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            answer_lists = self.vllm_client.generate(
                prompts=reason_texts,
                n=1,
                max_tokens=max_new_tokens or 100,
                **gen_kwargs,
            )

            # Build full token sequences by concatenating original reason ids
            # (preserving their underlying container type) with the vLLM
            # answer continuations.
            source_reason = reason_ids if isinstance(reason_ids, list) else normalized_reason
            full_lists = []
            for reason_row, answer in zip(source_reason, answer_lists):
                if hasattr(reason_row, "tolist"):
                    base = reason_row.tolist()
                else:
                    base = list(reason_row)
                full_lists.append(list(base) + list(answer))
        else:
            # HF / model backend: delegate to the model's own generate method.
            full_tensor = self.model.generate(
                normalized_reason,
                max_new_tokens=max_new_tokens or 100,
                eos_token_id=self.tok.eos_token_id,
                do_sample=True,
                **gen_kwargs,
            )
            if hasattr(full_tensor, "tolist") and not isinstance(full_tensor, list):
                full_lists = full_tensor.tolist()
            else:
                full_lists = [tensor.tolist() if hasattr(tensor, "tolist") else list(tensor) for tensor in full_tensor]

        # Convert the Python lists back into a batched representation.
        if full_torch:
            full_tensors = [torch_mod.tensor(seq, device=device) for seq in full_lists]
            padded_full = _pad_sequences_for_batch(
                full_tensors,
                pad_value=self.tok.pad_token_id,
                device=device,
            )
            return _ensure_tensor(padded_full, device)

        # Lightweight torch stubs (for example, unit-test fakes) only require
        # a list-of-tensors; if tensor construction itself fails, fall back to
        # pure Python lists.
        try:
            return [torch_mod.tensor(seq, device=device) for seq in full_lists]
        except (RuntimeError, TypeError, ValueError, AttributeError):  # pragma: no cover
            return full_lists
