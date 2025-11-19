"""Utilities for configuring Weights & Biases logging."""

import os


def init_wandb_training(training_args):
    """
    Configure environment variables for Weights & Biases runs.

    Expects `training_args` to provide `wandb_entity`, `wandb_project`,
    and `wandb_run_group` attributes (all optional).
    """
    if training_args.wandb_entity is not None:
        os.environ["WANDB_ENTITY"] = training_args.wandb_entity
    if training_args.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project
    if training_args.wandb_run_group is not None:
        os.environ["WANDB_RUN_GROUP"] = training_args.wandb_run_group
