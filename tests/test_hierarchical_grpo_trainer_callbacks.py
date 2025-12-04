import src.training.utils.hierarchical_grpo_trainer as trainer_mod


def test_callback_instances_and_factories(monkeypatch):
    added = []

    class MyCallback:
        pass

    def factory(trainer):
        assert isinstance(trainer, trainer_mod.HierarchicalGRPOTrainer)
        return "factory_cb"

    class StubGRPO:
        def __init__(self, *args, callbacks=None, **kwargs):
            self.callbacks = callbacks or []

        def add_callback(self, cb):
            added.append(cb)

    original_bases = trainer_mod.HierarchicalGRPOTrainer.__bases__
    monkeypatch.setattr(trainer_mod, "GRPOTrainer", StubGRPO)
    trainer_mod.HierarchicalGRPOTrainer.__bases__ = (
        trainer_mod.HierarchicalGenerationMixin,
        trainer_mod.HierarchicalRewardMixin,
        StubGRPO,
    )
    monkeypatch.setattr(trainer_mod, "TrainerCallback", MyCallback)

    cb_instance = MyCallback()
    try:
        trainer = trainer_mod.HierarchicalGRPOTrainer(callbacks=[cb_instance, factory])
    finally:
        trainer_mod.HierarchicalGRPOTrainer.__bases__ = original_bases

    assert trainer.callbacks == [cb_instance]
    assert added == ["factory_cb"]
    assert trainer.mask_truncated_completions is True
