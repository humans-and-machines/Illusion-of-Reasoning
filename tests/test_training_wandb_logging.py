import os
from types import SimpleNamespace

import src.training.utils.wandb_logging as wl


def test_init_wandb_training_sets_env(monkeypatch):
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.delenv("WANDB_RUN_GROUP", raising=False)

    args = SimpleNamespace(
        wandb_entity="entity",
        wandb_project="project",
        wandb_run_group="group",
    )
    wl.init_wandb_training(args)
    assert os.environ["WANDB_ENTITY"] == "entity"
    assert os.environ["WANDB_PROJECT"] == "project"
    assert os.environ["WANDB_RUN_GROUP"] == "group"


def test_init_wandb_training_skips_none(monkeypatch):
    # Seed env values; None should leave them unchanged.
    monkeypatch.setenv("WANDB_ENTITY", "prev_entity")
    args = SimpleNamespace(wandb_entity=None, wandb_project=None, wandb_run_group=None)
    wl.init_wandb_training(args)
    assert os.environ["WANDB_ENTITY"] == "prev_entity"
