# open_r1/utils/callbacks.py
from __future__ import annotations

import logging
import subprocess
from typing import Dict, List, Optional

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from open_r1.utils.replay_buffer import ReplayBuffer

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

# ---------------------------------------------------------------------------
#  Push-to-hub callback ------------------------------------------------------
# ---------------------------------------------------------------------------

class _DummyCfg:
    def __init__(self, **kw):  # convenience holder for hub + benchmark helpers
        for k, v in kw.items():
            setattr(self, k, v)

class PushToHubRevisionCallback(TrainerCallback):
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        self.log = logging.getLogger("PushToHub")

    def on_save(self, args: TrainingArguments, state: TrainerState,
                control: TrainerControl, **kwargs):
        if not state.is_world_process_zero:
            return

        step_tag = f"step-{state.global_step:09d}"
        dummy = _DummyCfg(
            hub_model_id    = args.hub_model_id,
            hub_model_revision = f"{args.hub_model_revision}-{step_tag}",
            output_dir      = f"{args.output_dir}/checkpoint-{state.global_step}",
            system_prompt   = args.system_prompt,
        )

        # lazy import – avoids circular deps if huggingface_hub absent
        from .hub import push_to_hub_revision
        fut = push_to_hub_revision(dummy, extra_ignore_patterns=["*.pt"])

        # (optional) spawn benchmark job when the upload finishes
        if _slurm_available():
            def _after(_):
                from .evaluation import run_benchmark_jobs
                self.log.info("Upload done – submitting benchmark job.")
                dummy.benchmarks = args.benchmarks
                run_benchmark_jobs(dummy, self.model_cfg)
            fut.add_done_callback(_after)

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
        self._trainer = None                         # will be set later
        self.log = logging.getLogger("SuccessCache")

    # ---------- lifecycle hooks ------------------------------------------
    def set_trainer(self, trainer):                  # called once at start
        self._trainer = trainer

    # ---------- main hook -------------------------------------------------
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict[str, float]] = None,
        **kwargs,
    ):
        # nothing to do if trainer not yet registered or no textual logs
        if self._trainer is None or not hasattr(self._trainer, "_textual_logs"):
            return

        txt_logs = self._trainer._textual_logs
        if not txt_logs["prompt"]:                  # empty until first eval step
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
    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        tokenizer,
        accuracy_key: str = "crossword_accuracy_reward",
        threshold: float = 1.0,
    ):
        self.buf  = replay_buffer
        self.tok  = tokenizer
        self.key  = accuracy_key
        self.thr  = threshold
        print("[ReplayBufferCallback] registered ✔️", flush=True)

    # ←–––– this fires AFTER loss.backward() and BEFORE scheduler/step().
    # It always receives both `inputs` and `outputs`.
    def on_train_batch_end(self, args, state, control, **kw):
        outs    = kw["outputs"]             # dict from training_step
        inputs  = kw["inputs"]              # the batch fed forward

        rewards = outs.get("rewards", {})
        if self.key not in rewards:
            return                           # key mismatch → nothing to do

        acc_vec = rewards[self.key].detach().cpu()   # tensor (B,)
        print("accuracy vector", acc_vec)
        ids_vec = inputs["input_ids"]                 # tensor (B, seq)
        is_rep  = inputs.get("is_replay")             # tensor (B,) or None

        added = 0
        for acc, ids in zip(acc_vec.tolist(), ids_vec):
            if acc >= self.thr:
                prompt = self.tok.decode(ids, skip_special_tokens=True)
                self.buf.add(prompt)
                added += 1

        # diagnostics
        rank      = args.local_rank if args.local_rank != -1 else 0
        buf_size  = len(self.buf)
        num_rep   = int(is_rep.sum().item()) if is_rep is not None else 0
        batch_sz  = len(ids_vec)

        print(
            f"[ReplayBufferCallback][rank{rank}] added {added} new • "
            f"{num_rep}/{batch_sz} replay • buffer = {buf_size}",
            flush=True,
        )
# ---------------------------------------------------------------------------
#  Registry ------------------------------------------------------------------
# ---------------------------------------------------------------------------

CALLBACKS = {
    "push_to_hub_revision": PushToHubRevisionCallback,
    "caching_callback"    : SuccessCachingCallback,
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
                raise ValueError("ReplayBufferCallback requires both `replay_buffer` and `tokenizer`.")
            cb_list.append(cls(replay_buffer=replay_buffer, tokenizer=tokenizer))

        else:          
            cb_list.append(cls())

    return cb_list
