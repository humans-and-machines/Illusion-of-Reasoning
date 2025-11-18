#!/usr/bin/env python
"""
Stable-GRPO               (drop-in replacement for open-r1 `grpo.py`)
────────────────────────────────────────────────────────────────────
 • reward / advantage / gradient clipping  • adaptive-KL
 • prompt-length truncation for vLLM       • robust reward wrapper
 • DeepSpeed-ZeRO pickle patch (PT ≥ 2.6)  • dtype fix for ids/masks
 • COSINE reward legacy-factory shim       ← ✦ NEW unconditional wrap
"""
# ════════════════════════════════════════════════════════════════════════════
#  Imports & one-shot global patches
# ════════════════════════════════════════════════════════════════════════════
import logging, os, sys, functools, inspect
from dataclasses import dataclass
import torch
from packaging import version

# ── DeepSpeed ZeRO pickle patch ──────────────────────────────────────────────
if version.parse(torch.__version__) >= version.parse("2.6.0"):
    from torch.serialization import add_safe_globals
    from deepspeed.runtime.zero.config import ZeroStageEnum
    from deepspeed.runtime.fp16.loss_scaler import LossScaler
    add_safe_globals([ZeroStageEnum, LossScaler])

# ── Force ids / masks to .long() inside GRPOTrainer ──────────────────────────
import trl.trainer.grpo_trainer as _grpo_mod
from transformers import PreTrainedTokenizerBase

_orig_prepare = _grpo_mod.GRPOTrainer._prepare_inputs
def _patched_prepare(self, inputs):
    inputs = _orig_prepare(self, inputs)
    if "input_ids" in inputs:
        inputs["input_ids"]   = inputs["input_ids"].long()
        inputs["attention_mask"] = inputs["attention_mask"].long()
    return inputs
_grpo_mod.GRPOTrainer._prepare_inputs = _patched_prepare

# ── Side-effect patch (vLLM hooks) ───────────────────────────────────────────
import open_r1.utils.vllm_patch  # noqa: F401

# Standard Open-R1 / HF imports
import datasets, transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOTrainer, ModelConfig, TrlParser, get_peft_config
from transformers import EarlyStoppingCallback

from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.utils import get_dataset, get_model, get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
import open_r1.rewards as _rewards
from open_r1.rewards import get_reward_funcs

# ════════════════════════════════════════════════════════════════════════════
#  Robust reward wrapper  (keeps names for W&B)
# ════════════════════════════════════════════════════════════════════════════
def _robustify(fn):
    if getattr(fn, "_robust_wrapped_", False):
        return fn

    sig = inspect.signature(fn)
    wants_p = "prompts"     in sig.parameters
    wants_c = "completions" in sig.parameters

    @functools.wraps(fn)
    def wrapped(prompts, completions, **kw):
        fixed = []
        for c in completions:
            if isinstance(c, list) and c and isinstance(c[0], dict):
                fixed.append(c)
            elif isinstance(c, list) and c and isinstance(c[0], str):
                fixed.append([{"role": "assistant", "content": c[0]}])
            else:
                fixed.append([{"role": "assistant", "content": str(c)}])

        call_kw = {}
        if wants_p: call_kw["prompts"] = prompts
        if wants_c: call_kw["completions"] = fixed
        call_kw.update({k: v for k, v in kw.items() if k in sig.parameters})
        return fn(**call_kw)

    wrapped._robust_wrapped_ = True
    return wrapped

# ─── Patch every “real” reward once ─────────────────────────────────────
for _n, _obj in inspect.getmembers(_rewards, inspect.isfunction):
    if "completions" not in inspect.signature(_obj).parameters:
        continue                        # ← skip factories like get_cosine_scaled_reward
    setattr(_rewards, _n, _robustify(_obj))

# ════════════════════════════════════════════════════════════════════════════
#  Stable-GRPO extras
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class StableGRPOConfig(GRPOConfig):
    reward_clip_min: float = -1.0
    reward_clip_max: float =  1.0
    adv_clip_range:  float =  5.0
    max_grad_norm:   float =  1.0
    kl_target:       float =  0.02
    kl_horizon:      int   = 256

MAX_CTX, RESERVED = 2048, 512
from torch.nn.utils import clip_grad_norm_

class StableGRPOTrainer(GRPOTrainer):
    # -- reward / advantage clipping --
    def _compute_rewards(self, m_out, qs, rs):
        return torch.clamp(
            super()._compute_rewards(m_out, qs, rs),
            self.args.reward_clip_min,
            self.args.reward_clip_max,
        )

    def compute_advantages(self, rewards, values, **kw):
        adv = super().compute_advantages(rewards, values, **kw)
        return torch.clamp(
            adv,
            -self.args.adv_clip_range,
            self.args.adv_clip_range,
        )

    # -- gradient clipping --
    def training_step(self, *a, **k):
        loss = super().training_step(*a, **k)
        if self.args.max_grad_norm and self.args.max_grad_norm > 0:
            clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        return loss

    def _safely_free_ds_params(self):
        """Workaround for DeepSpeed ZeRO param free assert during vLLM transition."""
        for param in self.model.parameters():
            if hasattr(param, 'ds_active_sub_modules'):
                param.ds_active_sub_modules.clear()

    def _generate_and_score_completions(self, inputs):
        tok: PreTrainedTokenizerBase = self.tokenizer
        max_prompt = MAX_CTX - RESERVED
        for sample in inputs:
            raw = sample.get("prompt")
            if raw is None:
                continue
            txt = ("\n".join(f"{m['role']}: {m['content']}" for m in raw)
                if isinstance(raw, list) else raw)
            ids = tok(txt, return_tensors="pt", truncation=False).input_ids[0]
            if ids.size(0) > max_prompt:
                txt = tok.decode(ids[-max_prompt:], skip_special_tokens=True)
            sample["prompt"] = txt

        # 🧯 Patch: workaround DeepSpeed ZeRO assertion when transitioning to vLLM
        self._safely_free_ds_params()

        return super()._generate_and_score_completions(inputs)

# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════
logger = logging.getLogger(__name__)

def main(s_args, t_args, m_args):
    set_seed(t_args.seed)
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(t_args.get_process_log_level())

    last_ckpt = (get_last_checkpoint(t_args.output_dir)
                 if os.path.isdir(t_args.output_dir) else None)
    if last_ckpt and t_args.resume_from_checkpoint is None:
        logger.info("Resuming from checkpoint %s", last_ckpt)

    if "wandb" in t_args.report_to:
        init_wandb_training(t_args)

    dataset   = get_dataset(s_args)
    tokenizer = get_tokenizer(m_args, t_args)
    model     = get_model(m_args, t_args)
    reward_funcs = [_robustify(f) for f in get_reward_funcs(s_args)]

    # → conversation format
    def make_conv(ex, col=s_args.dataset_prompt_column):
        msgs = []
        if t_args.system_prompt:
            msgs.append({"role":"system","content":t_args.system_prompt})
        msgs.append({"role":"user","content":ex[col]})
        return {"prompt":msgs}
    dataset = dataset.map(make_conv)
    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    trainer = StableGRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=t_args,
        train_dataset=dataset[s_args.dataset_train_split],
        eval_dataset=(dataset[s_args.dataset_test_split]
                      if t_args.eval_strategy != "no" else None),
        peft_config=get_peft_config(m_args),
        callbacks=get_callbacks(t_args, m_args)
                  + [EarlyStoppingCallback(early_stopping_patience=2)],
        processing_class=tokenizer,
    )

    trainer.train(resume_from_checkpoint=t_args.resume_from_checkpoint or last_ckpt)
    trainer.save_model(t_args.output_dir)

    if t_args.do_eval:
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[s_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if t_args.push_to_hub:
        trainer.push_to_hub(dataset_name=s_args.dataset_name, tags=["open-r1"])

# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, StableGRPOConfig, ModelConfig))
    s_args, t_args, m_args = parser.parse_args_and_config()
    main(s_args, t_args, m_args)
