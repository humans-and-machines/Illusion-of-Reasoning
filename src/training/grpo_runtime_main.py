"""Entry-point for GRPO training with replay and EASY/CRYPTIC mixing.

This module was extracted from :mod:`training.grpo_runtime_impl_full` so that
the original file can stay as a thin compatibility shim.
"""

from __future__ import annotations

import logging
import os
import sys
from types import SimpleNamespace
from typing import Any

from .grpo_runtime_env import (
    datasets,
    transformers,
    get_last_checkpoint,
    get_peft_config,
    set_seed,
)
from .rewards import get_reward_funcs
from .utils import get_dataset, get_model, get_tokenizer
from .utils.callbacks import get_callbacks
from .utils.replay_buffer import ReplayBuffer
from .utils.replay_dataset import ReplayMixDataset
from .grpo_dataset import _make_conversation, _load_easy_pool
from .grpo_rewards_router import _wrap_reward_for_nested
from .grpo_trainer_replay_impl import GRPOTrainerReplay
from .grpo_trainer_replay_support import LossLoggingCallback

logger = logging.getLogger(__name__)


def _configure_logging(training_args: Any) -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def _init_model_and_tokenizer(model_args: Any, training_args: Any):
    tokenizer = get_tokenizer(model_args, training_args)
    model = get_model(model_args, training_args)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            if hasattr(model, "resize_token_embeddings"):
                model.resize_token_embeddings(len(tokenizer))
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "left"

    model.generation_config.sort_inputs = False
    model.generation_config.return_dict_in_generate = True
    model.generation_config.output_scores = False
    model.generation_config.do_sample = True
    model.config.return_dict_in_generate = True
    model.config.output_scores = False
    return model, tokenizer


def _maybe_load_easy_pool(script_args: Any, tokenizer: Any, training_args: Any):
    want_easy = bool(
        getattr(script_args, "easy_dataset_name", None)
        or os.environ.get("EASY_DATASET_NAME")
        or os.environ.get("EASY_DATASET")
    )
    if not want_easy:
        return None
    return _load_easy_pool(script_args, tokenizer, training_args)


def _build_reward_funcs(
    script_args: Any,
    model_args: Any,
    training_args: Any,
    tokenizer: Any,
):
    ref_model = get_model(model_args, training_args).eval().requires_grad_(False)
    reward_funcs = get_reward_funcs(script_args, ref_model=ref_model, tokenizer=tokenizer)

    if isinstance(reward_funcs, dict):
        return {
            key: _wrap_reward_for_nested(fn)
            for key, fn in dict(reward_funcs).items()
        }
    if isinstance(reward_funcs, (list, tuple)):
        return [_wrap_reward_for_nested(fn) for fn in reward_funcs]
    return _wrap_reward_for_nested(reward_funcs)


def _prepare_datasets(
    script_args: Any,
    training_args: Any,
    tokenizer: Any,
):
    dataset = get_dataset(script_args)
    dataset = dataset.map(
        lambda ex: _make_conversation(
            ex,
            script_args.dataset_prompt_column,
            script_args.dataset_solution_column,
            tokenizer,
            training_args.system_prompt,
        )
    ).filter(lambda x: x is not None)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.padding_side = "left"

    for split in list(dataset.keys()):
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")
        if "task" not in dataset[split].column_names:
            dataset[split] = dataset[split].add_column("task", ["MATH"] * len(dataset[split]))
        if "is_replay" not in dataset[split].column_names:
            dataset[split] = dataset[split].add_column("is_replay", [False] * len(dataset[split]))

    if training_args.do_eval and script_args.dataset_test_split in dataset:
        full_eval = dataset[script_args.dataset_test_split]
        n_total = len(full_eval)
        n_keep = max(1, int(n_total * 0.1))
        eval_ds = full_eval.shuffle(seed=training_args.seed).select(range(n_keep))
    else:
        eval_ds = None

    train_ds = ReplayMixDataset(
        base_dataset=dataset[script_args.dataset_train_split],
        tokenizer=tokenizer,
    )
    return dataset, train_ds, eval_ds


def _build_trainer(context: Any):
    training_args = context.training_args
    model_args = context.model_args
    model = context.model
    tokenizer = context.tokenizer
    reward_funcs = context.reward_funcs
    easy_pool = context.easy_pool
    train_ds = context.train_ds
    eval_ds = context.eval_ds

    replay_buffer = ReplayBuffer(capacity=4000, C=1.0, debug_steps=3)
    callback_objects = get_callbacks(
        training_args,
        model_args,
        replay_buffer=replay_buffer,
        tokenizer=tokenizer,
    )
    callback_objects.append(LossLoggingCallback(training_args.output_dir))
    trainer = GRPOTrainerReplay(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=get_peft_config(model_args),
        callbacks=callback_objects,
        replay_buffer=replay_buffer,
        mix_exploit_ratio=0.7,
        constant_test_reward=4.0,
        processing_class=tokenizer,
        temp_start=1.0,
        temp_end=0.3,
        anneal_steps=3000,
        high_temp_period=5000,
        easy_pool=easy_pool,
        mix_schedule=None,
    )
    trainer.data_collator.sort_by_length = False
    return trainer


def main(script_args, training_args, model_args):
    """Entry point for GRPO training with replay buffer and EASY/CRYPTIC mixing."""
    set_seed(training_args.seed)
    training_args.return_reward = True        #  ← THE ONE-LINE SWITCH
    training_args.steps_per_generation = 8
    training_args.num_iterations       = 5

    _configure_logging(training_args)

    logger.warning(
        "Process rank %s — device %s — n_gpu %s — distributed %s — bf16 %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.bf16,
    )

    model, tokenizer = _init_model_and_tokenizer(model_args, training_args)
    easy_pool = _maybe_load_easy_pool(script_args, tokenizer, training_args)
    reward_funcs = _build_reward_funcs(
        script_args=script_args,
        model_args=model_args,
        training_args=training_args,
        tokenizer=tokenizer,
    )
    _, train_ds, eval_ds = _prepare_datasets(
        script_args=script_args,
        training_args=training_args,
        tokenizer=tokenizer,
    )
    context = SimpleNamespace(
        training_args=training_args,
        model_args=model_args,
        model=model,
        tokenizer=tokenizer,
        reward_funcs=reward_funcs,
        easy_pool=easy_pool,
        train_ds=train_ds,
        eval_ds=eval_ds,
    )
    trainer = _build_trainer(context)

    # -------- Train (FORCED resume path) --------
    last_checkpoint = training_args.resume_from_checkpoint
    if last_checkpoint is None and os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    # -------- Save model & card --------
    trainer.save_model(training_args.output_dir)
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(dataset_name=script_args.dataset_name, tags=["open-r1"])
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    # -------- Evaluate --------
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # -------- Hub upload --------
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name, tags=["open-r1"])
