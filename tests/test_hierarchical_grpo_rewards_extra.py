import types

import src.training.utils.hierarchical_grpo_rewards as rewards_mod


class _DummyRewardMixin(rewards_mod.HierarchicalRewardMixin):
    def __init__(self, num_generations: int):
        self.num_generations = num_generations


def test_build_reward_views_reshape_fallback(monkeypatch):
    mixin = _DummyRewardMixin(num_generations=2)

    rewards = types.SimpleNamespace(
        reshape=lambda *_a, **_k: "reshaped",
        view=lambda *_a, **_k: "viewed",
        device="cpu",
        dtype=None,
    )
    flat = [1, 2, 3]  # not divisible by num_generations -> triggers ValueError path

    prompt_major, generation_major = mixin._build_reward_views(rewards, flat, prompt_count=2)

    assert prompt_major == "reshaped"
    assert generation_major == "reshaped"


def test_build_reward_views_view_fallback_when_no_reshape():
    mixin = _DummyRewardMixin(num_generations=2)

    rewards = types.SimpleNamespace(
        view=lambda *_a, **_k: "viewed",
        device="cpu",
        dtype=None,
    )
    flat = [1, 2, 3]

    prompt_major, generation_major = mixin._build_reward_views(rewards, flat, prompt_count=2)

    assert prompt_major == "viewed"
    assert generation_major == "viewed"


def test_build_reward_views_uses_torch_reshape(monkeypatch):
    mixin = _DummyRewardMixin(num_generations=2)

    class TorchWithReshape:
        def tensor(self, *_a, **_k):
            raise TypeError("tensor fails")

        def reshape(self, rewards_obj, shape):
            return ("reshaped", rewards_obj, shape)

    monkeypatch.setattr(rewards_mod, "torch", TorchWithReshape())
    rewards = object()  # lacks reshape/view

    prompt_major, generation_major = mixin._build_reward_views(rewards, [1, 2], prompt_count=1)

    assert prompt_major[0] == "reshaped"
    assert generation_major[2] == (-1, mixin.num_generations)


def test_build_reward_views_uses_view_when_no_torch_reshape(monkeypatch):
    mixin = _DummyRewardMixin(num_generations=2)

    class TorchNoReshape:
        def tensor(self, *_a, **_k):
            raise TypeError("tensor fails")

    class RewardsWithView:
        def view(self, *_a, **_k):
            return "viewed-fallback"

    monkeypatch.setattr(rewards_mod, "torch", TorchNoReshape())

    prompt_major, generation_major = mixin._build_reward_views(RewardsWithView(), [1, 2], prompt_count=1)

    assert prompt_major == "viewed-fallback"
    assert generation_major == "viewed-fallback"
