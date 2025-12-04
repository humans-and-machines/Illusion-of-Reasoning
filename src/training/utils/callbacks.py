"""Trainer callbacks for hub uploads, success caching, and replay buffer."""

from __future__ import annotations

import logging
import subprocess
import sys
from importlib import import_module
from types import SimpleNamespace
from typing import TYPE_CHECKING, List, Optional

from .replay_buffer import ReplayBuffer


if TYPE_CHECKING:  # pragma: no cover - used only for static typing
    from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
else:  # pragma: no cover - runtime optional dependency
    try:
        from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
    except ImportError:

        class TrainerCallback:
            """
            Minimal runtime stub for :class:`transformers.TrainerCallback`.

            The real callback interface exposes many hook methods; this stub only
            provides a tiny, no-op subset so that subclasses can be defined and
            instantiated when ``transformers`` is not installed.
            """

            def on_train_begin(self, *_, **__):
                """No-op hook mirroring the real callback API."""

            def on_train_end(self, *_, **__):
                """No-op hook mirroring the real callback API."""

        class TrainerControl:
            """
            Lightweight stand-in for :class:`transformers.TrainerControl`.

            The stub only exposes a minimal surface used by callbacks and always
            reports that no special trainer actions are required.
            """

            def should_save(self) -> bool:
                """Return ``False`` in the stub implementation."""
                return False

            def should_evaluate(self) -> bool:
                """Return ``False`` in the stub implementation."""
                return False

        class TrainerState:
            """
            Minimal trainer state stub used when :mod:`transformers` is missing.

            Only the attributes touched in this module are defined; additional
            state should be provided by the real ``TrainerState`` class.
            """

            is_world_process_zero: bool = True
            global_step: int = 0

            def copy(self) -> "TrainerState":
                """Return ``self``, mimicking the behaviour of a dataclass."""
                return self

            def to_dict(self) -> dict:
                """Return a small dictionary representation of the state."""
                return {
                    "is_world_process_zero": self.is_world_process_zero,
                    "global_step": self.global_step,
                }

        class TrainingArguments:
            """
            Minimal stand-in for :class:`transformers.TrainingArguments`.

            Only the fields accessed in this module are modelled so that
            callbacks can run in environments without the full Transformers
            dependency installed.
            """

            hub_model_id: Optional[str] = None
            hub_model_revision: str = "main"
            output_dir: str = "."
            system_prompt: Optional[str] = None
            benchmarks: Optional[List[str]] = None

            def to_dict(self) -> dict:
                """Return a minimal dictionary view of the arguments."""
                return {
                    "hub_model_id": self.hub_model_id,
                    "hub_model_revision": self.hub_model_revision,
                    "output_dir": self.output_dir,
                    "system_prompt": self.system_prompt,
                    "benchmarks": self.benchmarks,
                }

            def clone(self) -> "TrainingArguments":
                """Create a shallow copy of the stub arguments."""
                clone_obj = TrainingArguments()
                clone_obj.hub_model_id = self.hub_model_id
                clone_obj.hub_model_revision = self.hub_model_revision
                clone_obj.output_dir = self.output_dir
                clone_obj.system_prompt = self.system_prompt
                clone_obj.benchmarks = list(self.benchmarks) if self.benchmarks else None
                return clone_obj

# ---------------------------------------------------------------------------
#  SLURM helper --------------------------------------------------------------
# ---------------------------------------------------------------------------


def _slurm_available() -> bool:
    try:
        subprocess.run(
            ["sinfo"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return True
    except FileNotFoundError:
        return False


def _import_utils_module(submodule: str):
    """
    Import ``training.utils.<submodule>`` with a fallback to the ``src.*`` path.

    Some test and packaging environments alias the package under ``src.training``.
    """
    last_exc: Exception | None = None
    module_names = (
        f"training.utils.{submodule}",
        f"src.training.utils.{submodule}",
    )
    for mod_name in module_names:
        try:
            return import_module(mod_name)
        except (ImportError, ModuleNotFoundError, AttributeError) as exc:
            # pragma: no cover - exercised in fallback tests
            last_exc = exc
    for mod_name in module_names:
        cached = sys.modules.get(mod_name)
        if cached is not None:
            return cached
    if last_exc:
        raise last_exc
    raise ImportError(f"Unable to import training utils submodule: {submodule}")


# ---------------------------------------------------------------------------
#  Push-to-hub callback ------------------------------------------------------
# ---------------------------------------------------------------------------


class PushToHubRevisionCallback(TrainerCallback):
    """Callback that pushes a checkpoint to the hub under a step-specific tag."""

    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        self.log = logging.getLogger("PushToHub")

    def get_model_config(self):
        """Public accessor for the associated model configuration."""
        return self.model_cfg

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        _control: TrainerControl,  # unused but kept for Trainer API
        **_kwargs,
    ):
        """Push a new revision to the hub when the checkpoint is saved."""
        if not state.is_world_process_zero:
            return

        step_tag = f"step-{state.global_step:09d}"
        dummy = SimpleNamespace(
            hub_model_id=args.hub_model_id,
            hub_model_revision=f"{args.hub_model_revision}-{step_tag}",
            output_dir=f"{args.output_dir}/checkpoint-{state.global_step}",
            system_prompt=args.system_prompt,
        )

        # lazy import – avoids circular deps if huggingface_hub absent
        hub_mod = _import_utils_module("hub")
        push_to_hub_revision = getattr(hub_mod, "push_to_hub_revision")
        fut = push_to_hub_revision(dummy, extra_ignore_patterns=["*.pt"])

        # (optional) spawn benchmark job when the upload finishes
        if _slurm_available():

            def _after(_):
                eval_mod = _import_utils_module("evaluation")
                run_benchmark_jobs = getattr(eval_mod, "run_benchmark_jobs")
                self.log.info("Upload done – submitting benchmark job.")
                dummy.benchmarks = args.benchmarks
                run_benchmark_jobs(dummy, self.model_cfg)

            callback = getattr(fut, "add_done_callback", None)
            if callback is not None:
                callback(_after)


# ---------------------------------------------------------------------------
#  Success-caching callback (text-log scraper) -------------------------------
# ---------------------------------------------------------------------------


class SuccessCachingCallback(TrainerCallback):
    """
    Scrapes `trainer._textual_logs` after every log step and pushes any prompt
    whose accuracy ≥ `acc_threshold` into `ReplayBuffer`.

    NOTE: Transformers never passes `trainer` via **kwargs → use `set_trainer`.
    """

    def __init__(self, replay_buffer: ReplayBuffer, acc_threshold: float = 0.999):
        self.buf = replay_buffer
        self.thr = acc_threshold
        self._trainer = None  # will be set later
        self.log = logging.getLogger("SuccessCache")

        # ---------- lifecycle hooks ------------------------------------------

    def set_trainer(self, trainer):  # called once at start
        """Register the underlying Trainer instance."""
        self._trainer = trainer

    # ---------- main hook -------------------------------------------------
    def on_log(
        self,
        _args: TrainingArguments,
        _state: TrainerState,
        _control: TrainerControl,
        _logs: Optional[dict[str, float]] = None,
        **_kwargs,
    ):
        """Scrape textual logs and cache successful prompts into the buffer."""
        # nothing to do if trainer not yet registered or textual logs unavailable
        if self._trainer is None or not hasattr(self._trainer, "textual_logs"):
            return

        txt_logs = self._trainer.textual_logs  # exposed by HierarchicalGRPOTrainer
        if not txt_logs["prompt"]:  # empty until first eval step
            return

        # pick the accuracy reward head (name may differ in your config)
        acc_key = next((k for k in txt_logs["rewards"] if "accuracy" in k), None)
        if acc_key is None:
            return

        for prompt, acc in zip(txt_logs["prompt"], txt_logs["rewards"][acc_key]):
            if acc >= self.thr:
                self.buf.add(prompt)


# ---------------------------------------------------------------------------
#  Replay-buffer callback (fast path – uses training_step outputs) ----------
# ---------------------------------------------------------------------------


class ReplayBufferCallback(TrainerCallback):
    """Callback that pushes high-accuracy prompts into the replay buffer."""

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        tokenizer,
        accuracy_key: str = "crossword_accuracy_reward",
        threshold: float = 1.0,
    ):
        self.buf = replay_buffer
        self.tok = tokenizer
        self.key = accuracy_key
        self.thr = threshold
        print("[ReplayBufferCallback] registered ✔️", flush=True)

    def buffer_size(self) -> int:
        """Return the current replay buffer size."""
        return len(self.buf)

    # ←–––– this fires AFTER loss.backward() and BEFORE scheduler/step().
    # It always receives both `inputs` and `outputs`.
    def on_train_batch_end(self, args, *_unused, **kwargs):
        """Inspect per-batch rewards and add successful prompts to the buffer."""
        rewards = kwargs.get("outputs", {}).get("rewards", {})
        if self.key not in rewards:
            return  # key mismatch → nothing to do

        acc_vec = rewards[self.key].detach().cpu()
        ids_batch = kwargs["inputs"]["input_ids"]
        is_replay_flags = kwargs["inputs"].get("is_replay")

        added = 0
        for accuracy, token_ids in zip(acc_vec.tolist(), ids_batch):
            if accuracy >= self.thr:
                prompt_text = self.tok.decode(token_ids, skip_special_tokens=True)
                self.buf.add(prompt_text)
                added += 1

        local_rank = args.local_rank if args.local_rank != -1 else 0
        num_replay = int(is_replay_flags.sum().item()) if is_replay_flags is not None else 0

        print(
            f"[ReplayBufferCallback][rank{local_rank}] added {added} new • "
            f"{num_replay}/{len(ids_batch)} replay • buffer = {len(self.buf)}",
            flush=True,
        )


# ---------------------------------------------------------------------------
#  Registry ------------------------------------------------------------------
# ---------------------------------------------------------------------------

CALLBACKS = {
    "push_to_hub_revision": PushToHubRevisionCallback,
    "caching_callback": SuccessCachingCallback,
    "replay_buffer_callback": ReplayBufferCallback,
}

# ---------------------------------------------------------------------------
#  Factory -------------------------------------------------------------------
# ---------------------------------------------------------------------------


def get_callbacks(
    train_cfg,
    model_cfg,
    *,
    replay_buffer: Optional[ReplayBuffer] = None,
    tokenizer=None,
):
    """
    Build the callbacks requested in `train_cfg.callbacks`.
    """
    cb_list: List[TrainerCallback] = []

    for name in train_cfg.callbacks:
        if name not in CALLBACKS:
            raise ValueError(f"Unknown callback '{name}'")

        cls = CALLBACKS[name]

        if name == "push_to_hub_revision":
            cb_list.append(cls(model_cfg))

        elif name == "caching_callback":
            if replay_buffer is None:
                raise ValueError("SuccessCachingCallback requires `replay_buffer`.")
            # ↓↓↓ pass the lower threshold here
            cb_list.append(cls(replay_buffer, acc_threshold=0.0))

        elif name == "replay_buffer_callback":
            if replay_buffer is None or tokenizer is None:
                raise ValueError("ReplayBufferCallback requires `replay_buffer` and `tokenizer`.")
            cb_list.append(cls(replay_buffer=replay_buffer, tokenizer=tokenizer))

        else:
            cb_list.append(cls())

    return cb_list
